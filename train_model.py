import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load and preprocess data
df = pd.read_csv('ndtv_data_final.csv')

# Convert categorical variables to codes
categorical_columns = ['Name', 'Brand', 'Model', 'Touchscreen', 'Operating system', 
                     'Wi-Fi', 'Bluetooth', 'GPS', '3G', '4G/ LTE']
for col in categorical_columns:
    df[col] = df[col].astype('category').cat.codes

# Prepare features and target
X = df.drop(['Price', 'Unnamed: 0'], axis=1)
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
numerical_columns = ['Battery capacity (mAh)', 'Screen size (inches)', 'Resolution x', 
                    'Resolution y', 'Processor', 'RAM (MB)', 'Internal storage (GB)', 
                    'Rear camera', 'Front camera']
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'models/price_predictor_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Print model performance
print(f"Training score: {model.score(X_train, y_train):.2f}")
print(f"Test score: {model.score(X_test, y_test):.2f}")
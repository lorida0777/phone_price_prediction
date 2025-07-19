import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Loading the dataset
df = pd.read_csv('ndtv_data_final.csv')

# Cleaning data: Handle missing values and select relevant features
df = df.dropna()
features = ['Brand', 'Battery capacity (mAh)', 'Screen size (inches)', 'Processor', 
            'RAM (MB)', 'Internal storage (GB)', 'Rear camera', 'Front camera']
X = df[features]
y = df['Price']

# Encoding categorical variables
brand_encoder = LabelEncoder()
processor_encoder = LabelEncoder()
X['Brand'] = brand_encoder.fit_transform(X['Brand'])
X['Processor'] = processor_encoder.fit_transform(X['Processor'])

# Normalizing numerical features
scaler = StandardScaler()
numerical_features = ['Battery capacity (mAh)', 'Screen size (inches)', 'RAM (MB)', 
                      'Internal storage (GB)', 'Rear camera', 'Front camera']
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Saving the encoders and scaler
joblib.dump(brand_encoder, 'brand_encoder.pkl')
joblib.dump(processor_encoder, 'processor_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the model with optimized hyperparameters
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42
)

# Training the model
model.fit(X_train, y_train)

# Saving the model
joblib.dump(model, 'phone_price_model.pkl')

# Evaluating the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training R^2 Score: {train_score:.4f}")
print(f"Test R^2 Score: {test_score:.4f}")

# Feature importance
feature_importance = pd.Series(model.feature_importances_, index=features)
print("\nFeature Importance:")
print(feature_importance.sort_values(ascending=False))
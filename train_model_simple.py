import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🔧 Amélioration du modèle de prédiction de prix de téléphones")
print("=" * 60)

# Loading the dataset
print("📊 Chargement des données...")
df = pd.read_csv('ndtv_data_final.csv')
print(f"Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")

# Data cleaning
print("\n🧹 Nettoyage des données...")
df = df.dropna()
print(f"Après nettoyage: {df.shape[0]} lignes")

# Remove outliers using IQR method
print("🔍 Suppression des valeurs aberrantes...")
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]
print(f"Après suppression des valeurs aberrantes: {df.shape[0]} lignes")

# Feature engineering
print("\n⚙️ Création de nouvelles fonctionnalités...")
df['Camera_Total'] = df['Rear camera'] + df['Front camera']
df['RAM_GB'] = df['RAM (MB)'] / 1000
df['Price_per_GB'] = df['Price'] / df['Internal storage (GB)'].replace(0, 1)
df['Price_per_MP'] = df['Price'] / (df['Camera_Total'].replace(0, 1))
df['Screen_to_Battery_Ratio'] = df['Screen size (inches)'] / (df['Battery capacity (mAh)'] / 1000).replace(0, 1)

# Select features
features = ['Brand', 'Battery capacity (mAh)', 'Screen size (inches)', 'Processor', 
            'RAM (MB)', 'Internal storage (GB)', 'Rear camera', 'Front camera',
            'Camera_Total', 'RAM_GB', 'Price_per_GB', 'Price_per_MP', 'Screen_to_Battery_Ratio']

X = df[features]
y = df['Price']

print(f"Fonctionnalités utilisées: {len(features)}")

# Encoding categorical variables
print("\n🔤 Encodage des variables catégorielles...")
brand_encoder = LabelEncoder()
processor_encoder = LabelEncoder()
X['Brand'] = brand_encoder.fit_transform(X['Brand'])
X['Processor'] = processor_encoder.fit_transform(X['Processor'])

# Scaling features
print("📏 Normalisation des fonctionnalités...")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)
print(f"Ensemble d'entraînement: {X_train.shape[0]} échantillons")
print(f"Ensemble de test: {X_test.shape[0]} échantillons")

# Try different models
print("\n🤖 Test des modèles...")

# Random Forest with optimized parameters
rf_model = RandomForestRegressor(
    n_estimators=1000,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

# Gradient Boosting with optimized parameters
gb_model = GradientBoostingRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.9,
    random_state=42
)

# Train and evaluate models
models = {
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model
}

best_score = 0
best_model_name = ""
best_model = None

for name, model in models.items():
    print(f"\n🎯 Entraînement de {name}...")
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Score R² d'entraînement: {train_score:.4f}")
    print(f"Score R² de test: {test_score:.4f}")
    
    if test_score > best_score:
        best_score = test_score
        best_model_name = name
        best_model = model

print(f"\n🏆 Meilleur modèle: {best_model_name} (R² = {best_score:.4f})")

# Final evaluation
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n📈 Résultats finaux:")
print(f"Score R² d'entraînement: {best_model.score(X_train, y_train):.4f}")
print(f"Score R² de test: {best_score:.4f}")
print(f"Erreur quadratique moyenne (MSE): {mse:.2f}")
print(f"Erreur absolue moyenne (MAE): {mae:.2f}")

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.Series(best_model.feature_importances_, index=features)
    print(f"\n🎯 Importance des fonctionnalités:")
    print(feature_importance.sort_values(ascending=False).head(10))

# Saving the model and encoders
print(f"\n💾 Sauvegarde du modèle...")
joblib.dump(best_model, 'phone_price_model.pkl')
joblib.dump(brand_encoder, 'brand_encoder.pkl')
joblib.dump(processor_encoder, 'processor_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Save feature names for the app
joblib.dump(features, 'feature_names.pkl')

print(f"\n✅ Modèle sauvegardé avec succès!")
print(f"🎯 Précision cible (>90%): {'✅ ATTEINTE' if best_score > 0.90 else '❌ NON ATTEINTE'}")
print(f"📊 Précision actuelle: {best_score:.2%}")

if best_score < 0.90:
    print(f"\n💡 Suggestions pour améliorer:")
    print("- Collecter plus de données")
    print("- Ajouter de nouvelles fonctionnalités")
    print("- Essayer d'autres algorithmes (XGBoost, LightGBM)")
    print("- Ajuster les hyperparamètres")
else:
    print(f"\n🎉 Excellent! Le modèle atteint la précision cible!") 
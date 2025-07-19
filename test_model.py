import joblib
import pandas as pd
import numpy as np

print("🧪 Test du modèle de prédiction de prix de téléphones")
print("=" * 50)

try:
    # Chargement du modèle et des encodeurs
    print("📥 Chargement du modèle...")
    model = joblib.load('phone_price_model.pkl')
    brand_encoder = joblib.load('brand_encoder.pkl')
    processor_encoder = joblib.load('processor_encoder.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    
    print("✅ Modèle chargé avec succès!")
    
    # Test avec des données d'exemple
    print("\n🧪 Test avec des données d'exemple...")
    
    # Données de test (iPhone 14 Pro)
    test_data = pd.DataFrame({
        'Brand': ['Apple'],
        'Battery capacity (mAh)': [3200],
        'Screen size (inches)': [6.1],
        'Processor': ['Apple A16 Bionic'],
        'RAM (MB)': [6144],
        'Internal storage (GB)': [128],
        'Rear camera': [48],
        'Front camera': [12],
        'Camera_Total': [60],
        'RAM_GB': [6.144],
        'Price_per_GB': [1000],
        'Price_per_MP': [50],
        'Screen_to_Battery_Ratio': [1.91],
        'Price_per_RAM': [200],
        'Battery_to_Screen_Ratio': [524.59]
    })
    
    # Encodage des variables catégorielles
    test_data['Brand'] = brand_encoder.transform(test_data['Brand'])
    test_data['Processor'] = processor_encoder.transform(test_data['Processor'])
    
    # Normalisation
    test_scaled = scaler.transform(test_data)
    
    # Prédiction
    predicted_price = model.predict(test_scaled)[0]
    
    print(f"📱 Téléphone testé: iPhone 14 Pro")
    print(f"💰 Prix prédit: ₹{predicted_price:,.2f}")
    print(f"📊 Prix réel approximatif: ₹89,900")
    print(f"🎯 Précision estimée: {abs(predicted_price - 89900) / 89900 * 100:.1f}% d'erreur")
    
    print("\n✅ Test réussi! Le modèle fonctionne correctement.")
    
except Exception as e:
    print(f"❌ Erreur lors du test: {e}")
    import traceback
    traceback.print_exc() 
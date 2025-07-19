# 📱 Prédiction de Prix de Téléphones

Application Streamlit pour prédire le prix d'un téléphone à partir de ses caractéristiques techniques.

---

## 🚀 Lancer l'application

1. **Installer les dépendances** :

   ```bash
   pip install -r requirements.txt
   ```

2. **Lancer l'application Streamlit** :

   ```bash
   streamlit run app_streamlit.py
   ```

3. **Ouvrir le navigateur** sur l'adresse indiquée (généralement http://localhost:8501)

---

## 📂 Fichiers nécessaires

- `app_streamlit.py` : Application principale
- `ndtv_data_final.csv` : Dataset utilisé pour la prédiction
- `phone_price_model.pkl` : Modèle de machine learning entraîné
- `brand_encoder.pkl` : Encodeur pour la marque
- `processor_encoder.pkl` : Encodeur pour le processeur
- `scaler.pkl` : Scaler utilisé pour la normalisation
- `feature_names.pkl` : Liste des features utilisées lors de l'entraînement
- `requirements.txt` : Liste des dépendances Python

---

## 📝 Exemple d'utilisation

1. Remplir le formulaire avec les caractéristiques du téléphone (marque, processeur, batterie, écran, RAM, stockage).
2. Cliquer sur "Prédire le prix".
3. Le prix prédit s'affiche instantanément.

---

## 💡 Conseils

- Assurez-vous que tous les fichiers `.pkl` et le dataset `.csv` sont présents dans le même dossier que `app_streamlit.py`.
- Si vous rencontrez des erreurs de version, vérifiez que les versions de scikit-learn, numpy, etc. sont compatibles (voir `requirements.txt`).

---

## 🤝 Contribuer

Les contributions sont les bienvenues !

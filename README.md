# 📱 Prédiction de Prix de Téléphones

Application Streamlit pour prédire le prix d'un téléphone à partir de ses caractéristiques techniques.

---

## 🚀 Lancer l'application
# 📱 Prédiction de Prix de Téléphones

Bienvenue dans cette application intelligente de prédiction de prix de smartphones !  
Développée avec **Streamlit** et **scikit-learn**, cette app vous permet d’estimer le prix d’un téléphone à partir de ses spécifications techniques.

🎯 **Objectif** : Fournir une estimation rapide et visuelle du prix d’un téléphone pour aider à la comparaison, à l’achat ou à l’évaluation marché.

---

## 🌐 Démo en ligne

👉 **[Accéder à l’application ici](https://phonepriceprediction-a9uxwcr48ebeakdavgfrng.streamlit.app/)**  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://phonepriceprediction-a9uxwcr48ebeakdavgfrng.streamlit.app/)

---

## 🚀 Lancer l’application en local

### 1. Cloner le dépôt

```bash
git clone https://github.com/ton-utilisateur/phone_price_prediction.git
cd phone_price_prediction
```

> Remplace `ton-utilisateur` par ton vrai nom d'utilisateur GitHub.

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Exécuter l’application

```bash
streamlit run app_streamlit.py
```

Ouvre ensuite ton navigateur sur : [http://localhost:8501](http://localhost:8501)

---

## 📁 Contenu du projet

| Fichier | Description |
|--------|-------------|
| `app_streamlit.py` | Code principal de l'application Streamlit |
| `ndtv_data_final.csv` | Jeu de données de référence |
| `phone_price_model.pkl` | Modèle de prédiction sauvegardé |
| `brand_encoder.pkl` | Encodeur des marques |
| `processor_encoder.pkl` | Encodeur des processeurs |
| `scaler.pkl` | Normalisateur (scaler) utilisé à l'entraînement |
| `feature_names.pkl` | Liste des colonnes/features attendues |
| `requirements.txt` | Liste des dépendances Python à installer |

---

## 🧠 Fonctionnement

1. L’utilisateur entre les caractéristiques du téléphone via la sidebar (RAM, écran, stockage, etc.).
2. L’application encode et met à l’échelle les données d’entrée.
3. Le modèle prédictif retourne le **prix estimé** du téléphone.
4. Des visualisations comparent la saisie aux moyennes de marché.

---

## 📝 Exemple d'utilisation

- Sélectionnez la **marque** et le **processeur**
- Entrez les valeurs : **batterie, écran, RAM, stockage, caméras**
- Cliquez sur **"🚀 Prédire le Prix"**
- Résultats, graphiques, comparaisons et recommandations s’affichent automatiquement

---

## 💡 Conseils utiles

- 📂 Placez tous les fichiers `.pkl` et `.csv` dans le **même dossier** que `app_streamlit.py`.
- ⚠️ En cas d’erreurs, vérifiez les versions de `scikit-learn`, `numpy`, etc. dans `requirements.txt`.

---

## 🤝 Contribuer

Les contributions sont les bienvenues !  
N'hésitez pas à ouvrir une **issue**, proposer une **pull request**, ou suggérer une amélioration.

---

## 👨‍💻 Auteur

Développé dans le cadre d’un projet Data Science.  
📫 Contact : *ajoutez ici votre email ou votre profil LinkedIn*

---


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

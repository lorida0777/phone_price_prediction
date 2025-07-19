
# 📱 Prédiction de Prix de Téléphones

Bienvenue dans ce projet de **Machine Learning** déployé avec **Streamlit**.  
Cette application permet de **prédire le prix estimé d'un smartphone** à partir de ses caractéristiques techniques.

[🚀 **Accéder à l'application en ligne**](https://phonepriceprediction-a9uxwcr48ebeakdavgfrng.streamlit.app/)  
[![Streamlit Badge](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://phonepriceprediction-a9uxwcr48ebeakdavgfrng.streamlit.app/)

---

## 🎯 Objectif

Ce projet vise à fournir un outil interactif permettant de :

- Prédire le **prix approximatif** d’un téléphone selon ses spécifications
- Visualiser les données sous forme de **graphiques dynamiques**
- Exploiter un modèle de **machine learning** pré-entraîné

---

## 🖥️ Lancer l'application en local

### 🔧 Prérequis

- Python 3.8+ recommandé
- `pip` ou `conda` pour la gestion des packages

### ⚙️ Installation

```bash
git clone https://github.com/ton-utilisateur/phone_price_prediction.git
cd phone_price_prediction
pip install -r requirements.txt
```

### ▶️ Exécution

```bash
streamlit run app_streamlit.py
```

Ouvrir ensuite le navigateur à l'adresse [http://localhost:8501](http://localhost:8501)

---

## 📁 Fichiers du projet

| Fichier | Description |
|--------|-------------|
| `app_streamlit.py` | Script principal de l'application |
| `ndtv_data_final.csv` | Dataset utilisé pour la prédiction |
| `phone_price_model.pkl` | Modèle de régression pré-entraîné |
| `brand_encoder.pkl` | Encodeur pour la marque |
| `processor_encoder.pkl` | Encodeur pour le processeur |
| `scaler.pkl` | Scaler pour la normalisation |
| `feature_names.pkl` | Liste des variables utilisées |
| `requirements.txt` | Dépendances Python nécessaires |

---

## 🧪 Utilisation de l'app

1. Renseignez les **caractéristiques techniques** du téléphone :
   - Marque, Processeur, RAM, Stockage, Écran, Batterie, Caméra, etc.
2. Cliquez sur le bouton **"Prédire le prix"**
3. Le modèle prédit instantanément le **prix estimé**

Des **graphiques interactifs** s'affichent également pour illustrer les valeurs par rapport à la moyenne.

---

## 📦 Bibliothèques principales

- `streamlit`
- `pandas`, `numpy`
- `scikit-learn`
- `joblib`
- `plotly`

---

## 🤝 Contributions

Les contributions sont les bienvenues !  
Merci de bien vouloir créer une *pull request* ou ouvrir une *issue* pour toute suggestion.

---

## 👨‍💻 Auteur

Développé dans le cadre d’un projet d’apprentissage en **Data Science**.  
N’hésitez pas à me contacter pour toute collaboration ou remarque.

---

## 📸 Aperçu
![Aperçu de l'application](./screenshot.png)



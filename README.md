# 📘 README — Projet de Scoring PME

## 🏦 1. Présentation du projet

Ce projet propose une **application de scoring de crédit pour PME**, développée en Python avec **Streamlit**.  
L’objectif est d’estimer la **probabilité de défaut** d’une entreprise et de classifier son niveau de risque crédit.

L'application repose sur :

- un **modèle de régression logistique** entraîné sur un jeu de données simulées ;
- une **pipeline de prétraitement** robuste ;
- un **tableau de bord Streamlit** permettant d’entrer des caractéristiques d’entreprise et d’obtenir immédiatement un score crédit ;
- des **explications de type SHAP** pour comprendre la contribution des variables.

---

## 📂 2. Structure du projet

```
scoring-pme-project/
├── app.py                      
├── requirements.txt            
├── scoring_pme_dataset.csv     
│
├── model_logistic.pkl          
├── preprocess.pkl              
│
└── Notebook_Data&Modeling.ipynb
```

---

## 🧪 3. Les données utilisées

### 🔹 Jeu de données : scoring_pme_dataset.csv
Le dataset contient des informations financières, structurelles et comportementales de PME.

### ⚠️ Données générées automatiquement
Les données ont été **générées automatiquement via Python** dans le notebook :

👉 `Notebook_Data&Modeling.ipynb`

---

## ⚙️ 4. Pipeline de prétraitement

Le fichier `preprocess.pkl` contient une pipeline scikit-learn incluant :

- OneHotEncoder  
- StandardScaler  
- Création de variables dérivées  

---

## 🤖 5. Modèle de scoring

Modèle utilisé : **Régression Logistique**

---

## 📊 6. Explicabilité (SHAP)

L’application utilise **SHAP LinearExplainer** pour afficher les variables contributives au score.

---

## 🌐 7. Application Streamlit

Interface permettant :

- saisie des données PME  
- calcul du score  
- visualisation des SHAP  
- dashboard simple  

---

## 🔧 Installation

```
pip install -r requirements.txt
```

---

## ▶️ Lancement

```
streamlit run app.py
```
---
## 📝 Auteurs

Projet réalisé par **Alex DARGA, Analyste statisticien** dans un but pédagogique.

# ==================================================
#     STREAMLIT APP — SCORING PME COMPLET (VISUEL++)
# ==================================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap

st.set_page_config(
    page_title="Scoring PME",
    page_icon="💼",
    layout="wide"
)

# Style global → amélioration visuelle
st.markdown("""
<style>
/* Titres */
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
}

/* Cards */
.card {
    padding: 18px;
    border-radius: 12px;
    background: #ffffff10;
    border: 1px solid #e0e0e055;
    margin-bottom: 10px;
}

/* SHAP table */
table {
    border-collapse: collapse;
    width: 100%;
}
th {
    background-color: #f7f7f7;
    padding: 10px;
}
td {
    padding: 8px;
}

/* Score box */
.score-box {
    padding: 20px;
    text-align: center;
    border-radius: 12px;
    color: white;
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# Chargement du modèle + preprocessing
# ---------------------------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("model_logistic.pkl", "rb"))
    preprocess = pickle.load(open("preprocess.pkl", "rb"))
    return model, preprocess

model, preprocess = load_model()

# ---------------------------------------------
# Interface utilisateur
# ---------------------------------------------
st.title("💼 Scoring Crédit — PME")
st.write("Analyse du risque de défaut d'une PME grâce à un modèle statistique (modèle logistique).")
st.write("Réalisé par Alex DARGA, Analyste statisticien.")
st.write("Pour plus de détails sur la conception du projet : https://github.com/DataOdyssee/Credit_Scoring_PME.")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        secteur = st.selectbox("Secteur", ["Commerce", "Services", "Agroalimentaire", "Transport", "Industrie", "BTP"])
        forme = st.selectbox("Forme juridique", ["SARL", "SA", "SAS", "Entreprise Individuelle"])
        region = st.selectbox("Région", ["Ouaga", "Bobo", "Koudougou", "Ouahigouya", "Fada", "Banfora"])
        anciennete = st.number_input("Ancienneté (années)", 0, 50)
        nb_employes = st.number_input("Nombre d'employés", 0, 2000)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        experience_dirigeant = st.number_input("Expérience du dirigeant (années)", 0, 50)
        ca = st.number_input("Chiffre d'affaires (FCFA)", 0.0)
        resultat_net = st.number_input("Résultat net (FCFA)", -1e9, 1e9)
        fonds_propres = st.number_input("Fonds propres (FCFA)", 0.0)
        dettes = st.number_input("Dettes totales (FCFA)", 0.0)
        st.markdown("</div>", unsafe_allow_html=True)

with col3:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        tresorerie = st.number_input("Trésorerie (FCFA)", -1e9, 1e9)
        incidents = st.number_input("Incidents bancaires (12 mois)", 0, 50)
        utilisation_decouvert = st.slider("Utilisation du découvert", 0.0, 1.0)
        anciennete_banque = st.number_input("Ancienneté bancaire (années)", 0, 40)
        montant_credit = st.number_input("Montant du crédit demandé (FCFA)", 0.0)
        duree = st.selectbox("Durée du crédit (mois)", [12, 24, 36, 48, 60])
        st.markdown("</div>", unsafe_allow_html=True)

# Variables dérivées
ratio_endettement = dettes / ca if ca > 0 else 0
marge = resultat_net / ca if ca > 0 else 0
ratio_credit_ca = montant_credit / ca if ca > 0 else 0
garantie = montant_credit * 0.8
ltv = montant_credit / garantie if garantie > 0 else 0

# ---------------------------------------------
# Prédiction
# ---------------------------------------------
if st.button("🧮 Calculer le Score"):

    input_data = pd.DataFrame([[secteur, forme, region, anciennete, nb_employes, experience_dirigeant,
                                ca, resultat_net, fonds_propres, dettes, tresorerie,
                                ratio_endettement, marge, incidents, utilisation_decouvert, 
                                anciennete_banque, montant_credit, duree, ratio_credit_ca,
                                garantie, ltv]],
                              columns=["secteur", "forme", "region", "anciennete", "nb_employes",
                                       "experience_dirigeant", "ca", "resultat_net", "fonds_propres",
                                       "dettes", "tresorerie", "ratio_endettement", "marge",
                                       "incidents", "utilisation_decouvert", "anciennete_banque",
                                       "montant_credit", "duree", "ratio_credit_ca", "garantie", "ltv"])

    # Préprocessing
    Xp = preprocess.transform(input_data)
    proba_default = model.predict_proba(Xp)[0][1]

    # Niveau de risque
    if proba_default < .20:
        niveau, color = "Très faible", "#0f9d58"
    elif proba_default < .40:
        niveau, color = "Faible", "#34a853"
    elif proba_default < .60:
        niveau, color = "Modéré", "#fbbc04"
    elif proba_default < .80:
        niveau, color = "Élevé", "#ea4335"
    else:
        niveau, color = "Très élevé", "#b00020"

    # Display
    st.markdown("## 🔍 Résultat du Scoring")

    st.markdown(
        f"<div class='score-box' style='background:{color};'>"
        f"Probabilité de défaut : {proba_default:.2%}<br>{niveau}"
        f"</div>",
        unsafe_allow_html=True
    )

    # SHAP values
    st.markdown("### 📊 Top Variables Contributives (SHAP)")
    explainer = shap.LinearExplainer(model, Xp, feature_perturbation="interventional")
    shap_values = explainer.shap_values(Xp)

    # Récupération noms features
    feature_names = []
    for name, transformer, cols in preprocess.transformers_:
        if name != 'remainder':
            if hasattr(transformer, 'get_feature_names_out'):
                feature_names.extend(transformer.get_feature_names_out(cols))
            else:
                feature_names.extend(cols)
        else:
            feature_names.extend(cols)

    shap_df = pd.DataFrame({
        "Variable": feature_names,
        "SHAP_value": shap_values[0]
    }).sort_values(by="SHAP_value", key=abs, ascending=False).head(10)

    st.dataframe(shap_df, use_container_width=True)

    # Graphique
    st.markdown("### 📈 Indicateur visuel du score")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Probabilité de défaut"], [proba_default])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probabilité")

    st.pyplot(fig)




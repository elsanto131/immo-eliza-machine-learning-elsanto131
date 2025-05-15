import pandas as pd
import streamlit as st
#import joblib
from joblib import dump, load # EN : Importing the dump and load functions from Joblib - FR : Importation des fonctions dump et load de Joblib
from PIL import Image

# --- Fonction pour le chargement du modèle ---
@st.cache_resource
def load_model():
    with open("model_random_forest_regressor_building_santo.pkl", "rb") as f:
        model = load(f)
    return model

# --- Ajout du style et de la police ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .stButton button {
        background-color: #0F5FA6;
        color: white;
        font-size: 16px;
        padding: 12px 20px;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        width: 170px;  /* Élasticité du bouton */
    }

    .stButton button:hover {
        background-color: #0A4775;
        cursor: pointer;
    }

    .stButtonNext {
        display: flex;
        justify-content: flex-end;  /* Place le bouton à droite */
        padding-top: 20px;
    }

    .stButtonStartOver {
        display: flex;
        justify-content: center;
        padding-top: 20px;
    }

    .stMarkdown {
        margin-top: 20px;
    }

    .stSuccess {
        font-size: 24px;
        font-weight: 600;
        color: #0A4775;
    }

    .back-link-container {
        display: flex;
        justify-content: flex-start;  /* Aligner le lien à gauche */
        align-items: center;
        width: 100%;
        margin-bottom: 10px;
    }

    .back-link {
        font-size: 16px;
        color: #0F5FA6;
        cursor: pointer;
        text-decoration: none;
    }

    .back-link:hover {
        text-decoration: underline;
    }

    .logo-title-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .logo-title-container .logo {
        margin-right: 15px;
    }

    .logo-title-container h2 {
        margin: 15 px;
    }
    </style>
""", unsafe_allow_html=True)

# Charger ton logo
logo = Image.open("logo.png")  # Chemin vers ton logo

# Ligne contenant le logo et le titre
st.markdown('<div class="logo-title-container">', unsafe_allow_html=True)
col1, col2 = st.columns([2, 5])  # Ajuste les proportions selon la taille du logo

with col1:
    st.image(logo, width=200)  # Ajuste la taille du logo ici

with col2:
    st.title("Real estate appraisal")
st.markdown('</div>', unsafe_allow_html=True)

# -- Chargement des données nettoyées ---
df = pd.read_csv("data/Kangaroo_cleaned.csv")

# --- Chargement du modèle et des colonnes attendues ---
model = load_model()  # Chargement du modèle
expected_columns = load('model_features.pkl')  # fichier contenant la liste des colonnes d'entrée

# --- Utilisation de session_state pour gérer les étapes du formulaire ---
if 'step' not in st.session_state:
    st.session_state.step = 1

# --- Fonction pour chaque étape ---
def step_1():
    st.subheader("Step 1: Property Details")
    st.session_state.bedroom_count = st.number_input("Number of bedrooms", min_value=0, value=2)
    st.session_state.bathroom_count = st.number_input("Number of bathrooms", min_value=0, value=1)
    st.session_state.habitable_surface = st.number_input("Habitable surface (in m²)", min_value=10, value=100)
    st.session_state.has_garden = st.checkbox("Has a garden", value=False)
    st.session_state.has_terrace = st.checkbox("Has a terrace", value=False)
    st.session_state.has_fireplace = st.checkbox("Has a fireplace", value=False)
    st.session_state.has_air_conditioning = st.checkbox("Has air conditioning", value=False)
    st.session_state.construction_year = st.number_input("Construction year", min_value=1800, max_value=2023, value=2000)
    
    # Liste des provinces
    province_columns = [col.replace("province_", "") for col in df.columns if col.startswith("province_")]

    # Menu déroulant pour la province
    province_choice = st.selectbox("Choice a province", sorted(province_columns))
    
    col1, col2 = st.columns([3, 1])
    with col2:
        # Ligne contenant le lien "Retour" et le bouton "Next"
        with st.container():
            st.markdown('<div class="back-link-container"><a class="back-link" href="javascript:void(0);" onclick="window.history.back();">Retour</a></div>', unsafe_allow_html=True)
            st.session_state.province_choice = province_choice

            if st.button("Next", key="next_step_1"):
                st.session_state.step = 2
                st.rerun()  # Utilisation de st.rerun()
def step_2():
    st.subheader("Step 2: Energy and Property Condition")
    epc_score = st.selectbox("EPC Score (Energy Performance Certificate)", ["A+", "A", "B", "C", "D", "E", "F", "G"])
    epc_score_mapping = {
        "A+": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7
    }
    st.session_state.epc_score_value = epc_score_mapping[epc_score]
    
    st.session_state.property_type = st.selectbox("Property type", ["House", "Apartment", "Villa", "Chalet", "Others"])
    st.session_state.building_condition = st.selectbox("Building condition", ["Good", "Just Renovated", "To Be Done Up", "To Renovate", "To Restore"])
    
    col1, col2 = st.columns([3, 1])
    with col2:
        # Ligne contenant le lien "Retour" et le bouton "Next"
        with st.container():
            st.markdown('<div class="back-link-container"><a class="back-link" href="javascript:void(0);" onclick="window.history.back();">Retour</a></div>', unsafe_allow_html=True)
            
            if st.button("Next", key="next_step_2"):
                st.session_state.step = 3
                st.rerun()  # Utilisation de st.rerun()

def step_3():
    st.subheader("Step 3: Finalize and Predict")
    
    input_data = pd.DataFrame({
        'bedroomcount': [st.session_state.bedroom_count],
        'bathroomcount': [st.session_state.bathroom_count],
        'habitablesurface': [st.session_state.habitable_surface],
        'hasgarden': [st.session_state.has_garden],
        'hasterrace': [st.session_state.has_terrace],
        'hasfireplace': [st.session_state.has_fireplace],
        'constructionyear': [st.session_state.construction_year],
        'hasairconditioning': [st.session_state.has_air_conditioning],
        'type_HOUSE': [1 if st.session_state.property_type == 'House' else 0],
        'type_APARTMENT': [1 if st.session_state.property_type == 'Apartment' else 0],
        'type_VILLA': [1 if st.session_state.property_type == 'Villa' else 0],
        'type_CHALLET': [1 if st.session_state.property_type == 'Chalet' else 0],
        'epcscore': [st.session_state.epc_score_value],
        'buildingcondition_GOOD': [1 if st.session_state.building_condition == 'Good' else 0],
        'buildingcondition_JUST_RENOVATED': [1 if st.session_state.building_condition == 'Just Renovated' else 0],
        'buildingcondition_TO_BE_DONE_UP': [1 if st.session_state.building_condition == 'To Be Done Up' else 0],
        'buildingcondition_TO_RENOVATE': [1 if st.session_state.building_condition == 'To Renovate' else 0],
        'buildingcondition_TO_RESTORE': [1 if st.session_state.building_condition == 'To Restore' else 0],
        f'province_{st.session_state.province_choice}': [1],  # Utiliser seulement la province
        # Retirer la ligne suivante qui faisait référence à locality_choice
        # f'locality_{st.session_state.locality_choice}': [1],  
    })
    
    # --- Adapter input_data aux colonnes du modèle ---
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # Ajouter les colonnes manquantes avec valeur 0

    input_data = input_data[expected_columns]  # Réordonner les colonnes

    # --- Prédiction ---
    prediction = model.predict(input_data)
    st.success(f'Predicted Price: {prediction[0]:,.2f} €')

    col1, col2 = st.columns([3, 1])
    with col2:
        # Ligne contenant le lien "Retour" et le bouton "Restart"
        with st.container():
            st.markdown('<div class="back-link-container"><a class="back-link" href="javascript:void(0);" onclick="window.history.back();">Retour</a></div>', unsafe_allow_html=True)
            
            if st.button("Restart", key="start_over"):
                st.session_state.step = 1
                st.rerun()  # Utilisation de st.rerun()

# --- Affichage de la barre de progression ---
progress = (st.session_state.step - 1) / 2  # Trois étapes (de 1 à 3)
st.progress(progress)

# --- Affichage du formulaire en fonction de l'étape ---
if st.session_state.step == 1:
    step_1()
elif st.session_state.step == 2:
    step_2()
elif st.session_state.step == 3:
    step_3()

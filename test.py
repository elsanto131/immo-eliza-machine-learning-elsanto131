import pandas as pd
import streamlit as st

# Chargement du fichier CSV
df = pd.read_csv("data/Kangaroo_cleaned.csv")

# Optionnel : ajuste l'affichage de Pandas (utile pour print(df.head()) par exemple)
pd.set_option('display.max_columns', None)

# Affiche chaque nom de colonne sur une ligne
for col in df.columns:
    print(col)





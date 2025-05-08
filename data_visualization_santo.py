# EN : Importing libraries - FR : Importation des bibliothèques
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter

def create_visualizations(df, output_dir='figures'):

	import os
	os.makedirs(output_dir, exist_ok=True)  # Crée le dossier s'il n'existe pas

	# 2.1. EN : Prices distribution - FR : Distribution des prix

	plt.figure(figsize=(10, 3)) # EN Set the figure size - FR Définir la taille de la figure
	sns.histplot(df['price'], bins=100, kde=True) # EN Create a histogram with kernel density estimation - FR Créer un histogramme avec estimation de la densité du noyau
	plt.title('Distribution des prix') # EN Set the title of the plot - FR Définir le titre du tracé
	plt.xlabel('Prix (en €)') # EN Set the x-axis label - FR Définir l'étiquette de l'axe des x
	plt.ylabel('Nombre de biens') # EN Set the y-axis label - FR Définir l'étiquette de l'axe des y
	plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}€'))  # Ajout du séparateur de milliers et du symbole €
	plt.grid(True) # EN Add grid lines - FR Ajouter des lignes de grille
	plt.savefig(f'{output_dir}/distribution_prix.png', bbox_inches='tight')
	plt.show() # EN Show the plot - FR Afficher le tracé

	# 2.2. EN : Correlation heatmap - FR : Heatmap de corrélation

	# Calcul de la matrice de corrélation
	corr_matrix = df.corr(numeric_only=True)

	# Affichage de la heatmap
	plt.figure(figsize=(18, 12))  # Taille de la figure
	sns.heatmap(corr_matrix, 
				annot=True, 
				fmt='.2f', 
				cmap='coolwarm', 
				linewidths=0.5, 
				vmin=-1, 
				vmax=1,
				annot_kws={"size": 6})  # Réduction de la taille de la police

	# Titre et mise en page
	plt.title('Matrice de Corrélation des Variables', fontsize=16)
	plt.xticks(rotation=45, ha='right', fontsize=8)
	plt.yticks(fontsize=8)
	plt.tight_layout()  # Évite le chevauchement
	plt.savefig(f'{output_dir}/heatmap_correlation.png', bbox_inches='tight')
	plt.show()

	# # 2.3. FR : Boxplot pour visualiser la distribution des prix par type de bien
	# Filtrer les données pour ne garder que 'Appartement' et 'Maison'
	df_filtered = df[df['type'].isin(['Appartement', 'Maison'])]

	# Boxplot de la distribution des prix par type de bien
	plt.figure(figsize=(10, 6))
	sns.boxplot(x='type', y='price', data=df_filtered)
	plt.title('Distribution des prix par type de bien')
	plt.xlabel('Type de bien')
	plt.ylabel('Prix')

	# Rotation des labels sur l'axe X pour une meilleure lisibilité
	plt.xticks(rotation=45)

	# Appliquer le formattage des valeurs de l'axe Y en €
	plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}€'))

	# Afficher la grille
	plt.grid(True)

	# Afficher le graphique
	plt.savefig(f'{output_dir}/boxplot_prix_par_type.png', bbox_inches='tight')
	plt.show()


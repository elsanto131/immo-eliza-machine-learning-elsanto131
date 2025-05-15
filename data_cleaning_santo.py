# --- 1.1. EN : Importing libraries - FR : Importation des bibliothèques ---

import pandas as pd # EN : Importing the Pandas library - FR : Importation de la bibliothèque Pandas
import os # EN : Importing the OS library - FR : Importation de la bibliothèque OS
from sklearn.preprocessing import LabelEncoder # EN : Importing the LabelEncoder from sklearn - FR : Importation de LabelEncoder de sklearn
import logging  # EN: For logging errors and information - FR: Pour enregistrer les erreurs et informations

# EN : DataManager class to handle data operations - FR : Classe DataManager pour gérer les opérations sur les données
class DataManager:
    @staticmethod
    def merge_columnsFrom(main_df, path_to_csv, id_col, from_id_col, from_columns_to_merge, verbose=0):
        """Load columns from other dataset."""
        if verbose:
            print(f"DataManager::merge_columnsFrom -> Columns to merge : {from_columns_to_merge}")
            print(f"DataManager::merge_columnsFrom -> Columns before merge : {main_df.columns.to_list()}")
        
        main_df[id_col] = pd.to_numeric(main_df[id_col], errors='coerce').astype('Int64')

        # Vérification de l'existence du fichier avant de le charger
        if not os.path.exists(path_to_csv):
            logging.error(f"File not found: {path_to_csv}")
            return None  # Arrêter l'exécution si le fichier n'existe pas
        
        from_df = pd.read_csv(path_to_csv)
        
        # Vérification de l'existence des colonnes dans le fichier source
        if from_id_col not in from_df.columns:
            logging.error(f"Column '{from_id_col}' not found in source file.")
            return None  # Arrêter l'exécution si la colonne de fusion est manquante
        
        from_df = from_df[[from_id_col] + from_columns_to_merge]
        from_df = from_df.rename(columns={from_id_col: 'from_id'})
        from_df = from_df.drop_duplicates(subset=['from_id'], keep='first')

        main_df = main_df.merge(from_df, left_on=id_col, right_on='from_id', how='left')
        main_df = main_df.drop(columns=['from_id'])

        if verbose:
            print(f"DataManager::merge_columnsFrom -> columns merged successfully: {main_df.columns.to_list()}")
        return main_df

# EN : Function to clean the data - FR : Fonction pour nettoyer les données
def data_cleaning(filepath, output_path):
    # 1.2. EN : Importing the CSV file and reading it using Pandas - FR : Importation du fichier csv et lecture grâce à Pandas
    df = pd.read_csv(filepath)
    
    # Renommer les colonnes
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_') # EN Renaming columns - FR Renommer les colonnes

    # 1.2.2. EN : Check the structure of the DataFrame and merge with the main DataFrame - FR : Vérifier la structure du DataFrame et fusionner avec le DataFrame principal
    df = DataManager.merge_columnsFrom(
        main_df=df,
        path_to_csv="data/Giraffe.csv",  # Chemin relatif pour le fichier Giraffe
        id_col="id",
        from_id_col="propertyId",
        from_columns_to_merge=[
            'latitude',
            'longitude',
            'cadastralIncome',
            'primaryEnergyConsumptionPerSqm'
        ],
        verbose=1
    )
    if df is None:  # Si la fusion échoue, retourner None
        return None

    # 1.3. EN : Remove duplicates and unnecessary columns - FR : Supprimer les doublons et des colonnes inutiles
    df = df.drop_duplicates() # EN Remove duplicates - FR Supprimer les doublons
    df = df.drop(columns=['url', 'unnamed:_0', 'id','monthlycost', 'hasbalcony', 'accessibledisabledpeople', 
                          'roomcount', 'diningroomsurface', 'streetfacadewidth', 'kitchensurface', 
                          'floorcount', 'hasdiningroom', 'hasdressingroom', 'hasattic','diningroomsurface',
                          'haslivingroom','livingroomsurface','gardenorientation','hasbasement',
                          'streetfacadewidth','kitchensurface']) # EN Drop unnecessary columns - FR Supprimer les colonnes inutiles

    # 1.4. EN : Remove spaces in strings - FR : Supprimer les espaces dans les chaînes de caractères
    for column in df.select_dtypes(include=['object']).columns: # EN Remove leading and trailing spaces from string columns - FR Supprimer les espaces de début et de fin des colonnes de chaîne
        df[column] = df[column].astype(str).str.strip() # EN Convert all columns to string and remove leading and trailing spaces - FR Convertir toutes les colonnes en chaîne et supprimer les espaces de début et de fin

    # 1.5. EN : Remove columns with too many missing values - FR : Supprimer les colonnes avec trop de valeurs manquantes
    df = df.dropna(thresh=len(df)*0.8, axis=1) # EN Drop columns with more than 80% missing values - FR Supprimer les colonnes avec plus de 80 % de valeurs manquantes

    # 1.8. EN : Remove missing values in price, habitableSurface, and floodzonetype - FR : Supprimer les valeurs manquantes dans price, habitableSurface, et floodzonetype
    df = df.dropna(subset=['price', 'habitablesurface', 'floodzonetype', 'latitude', 'longitude'])  # Drop rows with missing critical values

    # 1.9. EN : Convert binary columns to 0/1 - FR : Convertir les colonnes binaires en 0/1
    binary_columns = [
        'haslift', 'hasheatpump', 
        'hasphotovoltaicpanels', 'hasthermicpanels', 'hasgarden', 
        'hasairconditioning', 'hasarmoreddoor', 'hasvisiophone',
        'hasoffice', 'hasswimmingpool', 'hasfireplace', 'hasterrace'
    ]  # EN Get the binary columns - FR Obtenir les colonnes binaires

    true_vals = ['True', 'true', True, '1', 1, 'yes', 'Yes', 'oui', 'Oui']
    false_vals = ['False', 'false', False, '0', 0, 'no', 'No', 'non', 'Non']

    for column in binary_columns:
        if column in df.columns:
            df[column] = df[column].apply(lambda x: 1 if x in true_vals else (0 if x in false_vals else 0)).astype(int)

    # 1.10.1. EN : Convert 'epcscore' to an ordered categorical and then to numeric codes - FR : Convertir 'epcscore' en catégorique ordonné puis en codes numériques
    if 'epcscore' in df.columns:
        epc_order = ['A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G']  # EN Define the order of the categories - FR Définir l'ordre des catégories
        df['epcscore'] = df['epcscore'].astype(pd.CategoricalDtype(categories=epc_order, ordered=True))  # EN Convert 'epcscore' to categorical with the defined order - FR Convertir 'epcscore' en catégorique avec l'ordre défini
        df['epcscore'] = df['epcscore'].cat.codes  # EN Convert categorical values to numerical codes - FR Convertir les valeurs catégorielles en codes numériques

    # 1.10.2. EN : Encode 'floodzonetype' as integers with LabelEncoder - FR : Encoder 'floodzonetype' en entiers avec LabelEncoder
    if 'floodzonetype' in df.columns:
        label_encoder = LabelEncoder()  # EN Create a LabelEncoder instance - FR Créer une instance de LabelEncoder
        df['floodzonetype'] = label_encoder.fit_transform(df['floodzonetype'])  # EN Fit and transform the 'floodzonetype' column - FR Ajuster et transformer la colonne 'floodzonetype'

    # Réduire le nombre de modalités dans 'locality' aux 50 plus fréquentes
    if 'locality' in df.columns:
        top_localities = df['locality'].value_counts().nlargest(50).index
        df['locality'] = df['locality'].apply(lambda x: x if x in top_localities else 'Other')

    # 1.11. EN : One-hot Encoding - FR : Encodage One-hot
    categorical_columns = [
        'type', 'subtype', 'province', 'locality',
        'buildingcondition', 'floodzonetype', 'heatingtype',
        'kitchentype', 'gardenorientation', 'terraceorientation'
    ]

    # Vérifie si ces colonnes existent encore avant de les encoder
    cols_to_encode = [col for col in categorical_columns if col in df.columns]

    # Appliquer get_dummies uniquement sur les colonnes encore présentes
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    # 1.12. EN : Filter out prices above 1.000.000€ - FR : Filtrer les prix au-delà de 1.000.000€
    df = df[(df['price'] >= 50000) & (df['price'] <= 1000000)]

    # 1.13. EN : Save the cleaned DataFrame to a CSV file - FR : Enregistrer le DataFrame nettoyé dans un fichier CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    row_count = len(df)
    print(f"Cleaned dataframe saved in: {output_path}")
    print(f"The DataFrame has {row_count} rows.")
    return df

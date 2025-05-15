# --- EN : Importation des bibliothèques - FR : Importation des bibliothèques ---
import logging # EN : Importing the logging library - FR : Importation de la bibliothèque de journalisation
#import joblib # EN : Importing the Joblib library - FR : Importation de la bibliothèque Joblib
from joblib import dump, load # EN : Importing the dump and load functions from Joblib - FR : Importation des fonctions dump et load de Joblib
from data_cleaning_santo import data_cleaning
#from data_visualization_santo import create_visualizations
#from model_linear_regression_building_santo import build_linear_regression_model
from model_random_forest_regressor_building_santo import build_random_forest_model
#from model_gradient_boosting_regressor_building_santo import build_gradient_boosting_model


# --- EN : Setting up logging - FR : Configuration de la journalisation 
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def main():

    logging.info("Start of the script execution.") # EN Log message indicating the start of the script execution - FR Message de log indiquant le début de l'exécution du script

    # --- 1. EN Cleaning the data - FR Nettoyage des données ---
    logging.info("Start of data cleaning...") # EN Log message indicating the start of data cleaning - FR Message de log indiquant le début du nettoyage des données
    
    cleaned_data = data_cleaning(
        filepath='data/Kangaroo.csv',
        output_path='data/Kangaroo_cleaned.csv'
    )

    logging.debug("Cleaning data...") # EN Log message indicating that the data is being cleaned - FR Message de log indiquant que les données sont en cours de nettoyage
    logging.debug(f"Data sample after cleaning:\n{cleaned_data.head()}") # EN Log message showing a sample of the cleaned data - FR Message de log affichant un échantillon des données nettoyées
    logging.info("Data cleaning finished") # EN Log message indicating that the data cleaning is finished - FR Message de log indiquant que le nettoyage des données est terminé

    '''
    print(df.shape)
    print(df.head()) 
    

    # --- 2. EN Visualize the data - FR Visualisation des données ---

    logging.info("Creating the visualizations... ") # EN Log message indicating the start of visualization creation - FR Message de log indiquant le début de la création de visualisations

    create_visualizations(cleaned_data, output_dir="figures")

    logging.info("Visualizations created and saved in the 'figures' folder.") # EN Log message indicating that the visualizations have been created and saved - FR Message de log indiquant que les visualisations ont été créées et enregistrées
    '''
    # --- 3. EN Training the model - FR Entraînement du modèle ---

    #model_name, model, train_r2, test_r2, train_rmse, test_rmse, train_mae, test_mae = build_linear_regression_model(cleaned_data)
    logging.info("Training the Random Forest model... ")
    model_name, model, train_r2, test_r2, train_rmse, test_rmse, train_mae, test_mae = build_random_forest_model(cleaned_data)
    logging.info(f"{model_name} model trained successfully.")
    #model_name, model, train_r2, test_r2, train_rmse, test_rmse, train_mae, test_mae = build_gradient_boosting_model(cleaned_data)
    
    # --- 4. EN Displaying the results - FR Affichage des résultats ---

    logging.info("-" * 58) # EN Log message indicating the start of the results display - FR Message de log indiquant le début de l'affichage des résultats
    logging.info(f"| {'Metric':<15} | {'TRAIN':>10} | {'TEST':>10} | {'DIFF':>10} |") # EN Log message showing the header of the results table - FR Message de log affichant l'en-tête du tableau des résultats
    logging.info("-" * 58) # EN Log message indicating the start of the results display - FR Message de log indiquant le début de l'affichage des résultats
    logging.info(f"| {'R2 Score':<15} | {train_r2:10.4f} | {test_r2:10.4f} | {(train_r2 - test_r2):10.4f} |") # EN Log message showing the R2 score - FR Message de log affichant le score R2
    logging.info(f"| {'R2 Score %':<15} | {(train_r2 * 100):10.2f} | {(test_r2 * 100):10.2f} | {((train_r2 - test_r2) * 100):10.2f} |") # EN Log message showing the R2 score percentage - FR Message de log affichant le pourcentage du score R2
    logging.info(f"| {'RMSE':<15} | {train_rmse:10.2f} | {test_rmse:10.2f} | {abs(train_rmse - test_rmse):10.2f} |") # EN Log message showing the RMSE - FR Message de log affichant le RMSE
    logging.info(f"| {'MAE':<15} | {train_mae:10.2f} | {test_mae:10.2f} | {abs(train_mae - test_mae):10.2f} |") # EN Log message showing the MAE - FR Message de log affichant le MAE
    logging.info("-" * 58) # EN Log message indicating the end of the results display - FR Message de log indiquant la fin de l'affichage des résultats

    # --- 5. EN Saving the model - FR Sauvegarde du modèle ---

    logging.info("Saving the trained model...") # EN Log message indicating the start of model saving - FR Message de log indiquant le début de la sauvegarde du modèle

    with open("model_random_forest_regressor_building_santo.pkl", "wb") as f:
        dump(model, 'model_random_forest_regressor_building_santo_compression.pkl', compress=('lzma', 9)) # EN Save the model to a file - FR Enregistrer le modèle dans un fichier

    logging.info("Model saved successfully.")

    logging.info("Script execution finished") # EN Log message indicating the end of the script execution - FR Message de log indiquant la fin de l'exécution du script

if __name__ == "__main__":
    main()


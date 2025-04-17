##############################################################################
### File for evaluating the topics and PCA/PCoA cluster with Random Forest ###
##############################################################################

# Load needed functions
from functions import *

# Defining variables according to the main function
dimensionality = 200
rf_method = 'regression'
input_rf_target_variable = 'salinity'
input_data = 'path/to/your/data'
input_data_clr = 'path/to/your/data_clr'
input_metadata = 'path/to/your/metadata'
topic_path = 'path/to/your/topic_data_location/'
cluster_path = 'path/to/your/pca_pcoa_data_location/'
output_rf_model = 'path/to/your/output/location_RF_models/'
output_rf_metrics = 'path/to/your/output/location_RF_models/'

# Run the Random Forest evaluation
for i in range(1, dimensionality, 10):
    print("This is k=", i)
    # No preprocession
    main_function_random_forest(i, 'none', 'lda', 'none', topic_path, cluster_path, rf_method, input_rf_target_variable,
                                input_data, input_data_clr, input_metadata, output_rf_model, output_rf_metrics)
    main_function_random_forest(i, 'none', 'nnmf', 'none', topic_path, cluster_path, rf_method, input_rf_target_variable,
                                input_data, input_data_clr,input_metadata, output_rf_model, output_rf_metrics)
    main_function_random_forest(i, 'none', 'none', 'pca', topic_path, cluster_path, rf_method, input_rf_target_variable,
                                input_data, input_data_clr, input_metadata, output_rf_model, output_rf_metrics)
    main_function_random_forest(i, 'none', 'none', 'pcoa', topic_path, cluster_path, rf_method, input_rf_target_variable,
                                input_data, input_data_clr, input_metadata, output_rf_model, output_rf_metrics)
    # Preprocession: clr
    main_function_random_forest(i, 'clr', 'lda', 'none', topic_path, cluster_path, rf_method, input_rf_target_variable,
                                input_data, input_data_clr, input_metadata, output_rf_model, output_rf_metrics)
    main_function_random_forest(i, 'clr', 'nnmf', 'none', topic_path, cluster_path, rf_method, input_rf_target_variable,
                                input_data,input_data_clr, input_metadata, output_rf_model, output_rf_metrics)
    main_function_random_forest(i, 'clr', 'none', 'pca', topic_path, cluster_path, rf_method, input_rf_target_variable,
                                input_data,input_data_clr, input_metadata, output_rf_model, output_rf_metrics)
    main_function_random_forest(i, 'clr', 'none', 'pcoa', topic_path, cluster_path, rf_method, input_rf_target_variable,
                                input_data, input_data_clr, input_metadata, output_rf_model, output_rf_metrics)
    # Preprocession: fractions
    main_function_random_forest(i, 'fractions', 'lda', 'none', topic_path, cluster_path, rf_method, input_rf_target_variable,
                                input_data,input_data_clr, input_metadata, output_rf_model, output_rf_metrics)
    main_function_random_forest(i, 'fractions', 'nnmf', 'none', topic_path, cluster_path, rf_method, input_rf_target_variable,
                                input_data, input_data_clr,input_metadata, output_rf_model, output_rf_metrics)
    main_function_random_forest(i, 'fractions', 'none', 'pca', topic_path, cluster_path, rf_method, input_rf_target_variable,
                                input_data, input_data_clr, input_metadata, output_rf_model, output_rf_metrics)
    main_function_random_forest(i, 'fractions', 'none', 'pcoa', topic_path, cluster_path, rf_method, input_rf_target_variable,
                                input_data, input_data_clr, input_metadata, output_rf_model, output_rf_metrics)
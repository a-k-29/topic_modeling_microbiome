##############################################################################
### File for evaluating the topics and PCA/PCoA cluster with Random Forest ###
##############################################################################

# Load needed functions
from functions import *

# Defining paths for input and output data
input_data_dir = 'path/to/your/data/'
input_data = input_data_dir + 'example_otu_data.csv'
input_data_clr = input_data_dir + 'example_otu_data_clr.csv'
input_metadata = input_data_dir + 'example_metadata.csv'
output_rf_model = 'path/to/your/data/'
output_rf_metrics = 'path/to/your/data/'

# Paths to directory with topic and PCA/PCoA csv files
topic_path = 'path/to/your/data/'
cluster_path = 'path/to/your/data/'

# Defining variables according to the main function (functions.py)
dimensionality = 10
rf_method = 'regression'
input_rf_target_variable = 'salinity'

# Run the Random Forest evaluation
for i in range(2, dimensionality, 10):
    print("This is k=", i)
    # No preprocession
    main_function_random_forest(i, 'none', 'lda', 'none', topic_path, cluster_path, rf_method, input_rf_target_variable,
                                input_data, input_data_clr, input_metadata, output_rf_model, output_rf_metrics)
    main_function_random_forest(i, 'none', 'nnmf', 'none', topic_path, cluster_path, rf_method, input_rf_target_variable,
                                input_data, input_data_clr,input_metadata, output_rf_model, output_rf_metrics)
    main_function_random_forest(i, 'none', 'none', 'pca', topic_path, cluster_path, rf_method, input_rf_target_variable,
                                input_data, input_data_clr, input_metadata, output_rf_model, output_rf_metrics)
    # Preprocession: clr
    main_function_random_forest(i, 'clr', 'lda', 'none', topic_path, cluster_path, rf_method, input_rf_target_variable,
                                input_data, input_data_clr, input_metadata, output_rf_model, output_rf_metrics)
    main_function_random_forest(i, 'clr', 'nnmf', 'none', topic_path, cluster_path, rf_method, input_rf_target_variable,
                                input_data,input_data_clr, input_metadata, output_rf_model, output_rf_metrics)
    main_function_random_forest(i, 'clr', 'none', 'pca', topic_path, cluster_path, rf_method, input_rf_target_variable,
                                input_data,input_data_clr, input_metadata, output_rf_model, output_rf_metrics)
    # Preprocession: fractions
    main_function_random_forest(i, 'fractions', 'lda', 'none', topic_path, cluster_path, rf_method, input_rf_target_variable,
                                input_data,input_data_clr, input_metadata, output_rf_model, output_rf_metrics)
    main_function_random_forest(i, 'fractions', 'nnmf', 'none', topic_path, cluster_path, rf_method, input_rf_target_variable,
                                input_data, input_data_clr,input_metadata, output_rf_model, output_rf_metrics)
    main_function_random_forest(i, 'fractions', 'none', 'pca', topic_path, cluster_path, rf_method, input_rf_target_variable,
                                input_data, input_data_clr, input_metadata, output_rf_model, output_rf_metrics)
    main_function_random_forest(i, 'fractions', 'none', 'pcoa', topic_path, cluster_path, rf_method, input_rf_target_variable,
                                input_data, input_data_clr, input_metadata, output_rf_model, output_rf_metrics)

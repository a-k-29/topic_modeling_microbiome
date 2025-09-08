############################################################################
### File for genearting topics and PCA/PCoA Cluster from microbiome data ###
############################################################################

# Load needed functions
from functions import *

# Defining variables and paths according to the main TM function (functions.py)
dimensionality = 10
input_data_dir = 'path/to/your/data/'
input_data = input_data_dir + 'example_otu_data.csv'
input_data_clr = input_data_dir + 'example_otu_data_clr.csv'
input_metadata = input_data_dir + 'example_metadata.csv'
output_trained_TM_models = 'path/to/your/data/'
output_tm_components = 'path/to/your/data/'
output_tm_topics = 'path/to/your/data/'
output_tm_metrics = 'path/to/your/data/'
output_rf_model = 'path/to/your/data/'
output_rf_metrics = 'path/to/your/data/'
output_clusters = 'path/to/your/data/'

# Generate topics and PCA/PCoA cluster
for i in range(2, dimensionality, 1):
    print("This is k=", i)
    # No preprocession
    main_function_topic_generation(i, 'none', 'lda', 'none',
               input_data, input_data_clr, input_metadata, output_trained_TM_models, output_tm_components, output_tm_topics,
               output_tm_metrics, output_clusters)
    main_function_topic_generation(i, 'none', 'nnmf', 'none',
               input_data, input_data_clr, input_metadata, output_trained_TM_models, output_tm_components, output_tm_topics,
               output_tm_metrics, output_clusters)
    main_function_topic_generation(i, 'none', 'none', 'pca',
                input_data, input_data_clr, input_metadata, output_trained_TM_models, output_tm_components, output_tm_topics,
                output_tm_metrics, output_clusters)
    # Preprocession: clr
    main_function_topic_generation(i, 'clr', 'lda', 'none', input_data, input_data_clr, input_metadata, output_trained_TM_models, output_tm_components,
                  output_tm_topics, output_tm_metrics, output_clusters)
    main_function_topic_generation(i, 'clr', 'nnmf', 'none',
                  input_data, input_data_clr, input_metadata, output_trained_TM_models, output_tm_components,
                  output_tm_topics, output_tm_metrics, output_clusters)
    main_function_topic_generation(i, 'clr', 'none', 'pca',
                  input_data, input_data_clr, input_metadata, output_trained_TM_models, output_tm_components,
                  output_tm_topics, output_tm_metrics, output_clusters)
    # Preprocession: fractions
    main_function_topic_generation(i, 'fractions', 'lda', 'none',
                input_data, input_data_clr, input_metadata, output_trained_TM_models, output_tm_components,
                  output_tm_topics, output_tm_metrics, output_clusters)
    main_function_topic_generation(i, 'fractions', 'nnmf', 'none',
                  input_data, input_data_clr, input_metadata, output_trained_TM_models, output_tm_components,
                  output_tm_topics, output_tm_metrics, output_clusters)
    main_function_topic_generation(i, 'fractions', 'none', 'pca',
                  input_data, input_data_clr, input_metadata, output_trained_TM_models, output_tm_components,
                  output_tm_topics, output_tm_metrics, output_clusters)
    main_function_topic_generation(i, 'fractions', 'none', 'pcoa',
                  input_data, input_data_clr, input_metadata, output_trained_TM_models, output_tm_components,
                  output_tm_topics, output_tm_metrics, output_clusters)

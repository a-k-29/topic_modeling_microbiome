############################################################################
### File for genearting topics and PCA/PCoA Cluster from microbiome data ###
############################################################################

# Load needed functions
from functions import *

# Defining variables according to the main TM funciton
dimensionality = 200
input_data = 'path/to/your/data'
input_data_clr = 'path/to/your/data_clr'
input_metadata = 'path/to/your/metadata'
output_trained_TM_models = 'path/to/your/output/location_TM_models'
output_tm_components = 'path/to/your/output/location_TM_components'
output_tm_topics = 'path/to/your/output/location_TM_models'
output_tm_metrics = 'path/to/your/output/location_TM_models'
output_rf_model = 'path/to/your/output/location_TM_models'
output_rf_metrics = 'path/to/your/output/location_TM_models'
output_clusters = 'path/to/your/output/location_TM_models'

# Generate topics and PCA/PCoA cluster
for i in range(1, dimensionality, 10):
    print("This is k=", i)
    # No preprocession
    main_function_topic_generation(i, 'none', 'lda', 'none',
               input_data, input_data_clr, input_metadata, input_as_list, output_trained_TM_models, output_tm_components, output_tm_topics,
               output_tm_metrics, output_clusters)
    main_function_topic_generation(i, 'none', 'nnmf', 'none',
               input_data, input_data_clr, input_metadata, input_as_list, output_trained_TM_models, output_tm_components, output_tm_topics,
               output_tm_metrics, output_clusters)
    main_function_topic_generation(i, 'none', 'none', 'pca',
                input_data, input_data_clr, input_metadata, input_as_list, output_trained_TM_models, output_tm_components, output_tm_topics,
                output_tm_metrics, output_clusters)
    main_function_topic_generation(i, 'none', 'none', 'pcoa',
               input_data, input_data_clr, input_metadata, input_as_list, output_trained_TM_models, output_tm_components, output_tm_topics,
               output_tm_metrics, output_clusters)
    # Preprocession: clr
    main_function_topic_generation(i, 'clr', 'lda', 'none', input_data, input_data_clr, input_metadata,
                  input_as_list, output_trained_TM_models, output_tm_components,
                  output_tm_topics, output_tm_metrics, output_clusters)
    main_function_topic_generation(i, 'clr', 'nnmf', 'none',
                  input_data, input_data_clr, input_metadata, input_as_list, output_trained_TM_models, output_tm_components,
                  output_tm_topics, output_tm_metrics, output_clusters)
    main_function_topic_generation(i, 'clr', 'none', 'pca',
                  input_data, input_data_clr, input_metadata, input_as_list, output_trained_TM_models, output_tm_components,
                  output_tm_topics, output_tm_metrics, output_clusters)
    main_function_topic_generation(i, 'clr', 'none', 'pcoa',
                 input_data, input_data_clr, input_metadata, input_as_list, output_trained_TM_models, output_tm_components,
                 output_tm_topics, output_tm_metrics, output_clusters)
    # Preprocession: fractions
    main_function_topic_generation(i, 'fractions', 'lda', 'none',
                input_data, input_data_clr, input_metadata, input_as_list, output_trained_TM_models, output_tm_components,
                  output_tm_topics, output_tm_metrics, output_clusters)
    main_function_topic_generation(i, 'fractions', 'nnmf', 'none',
                  input_data, input_data_clr, input_metadata, input_as_list, output_trained_TM_models, output_tm_components,
                  output_tm_topics, output_tm_metrics, output_clusters)
    main_function_topic_generation(i, 'fractions', 'none', 'pca',
                  input_data, input_data_clr, input_metadata, input_as_list, output_trained_TM_models, output_tm_components,
                  output_tm_topics, output_tm_metrics, output_clusters)
    main_function_topic_generation(i, 'fractions', 'none', 'pcoa',
                  input_data, input_data_clr, input_metadata, input_as_list, output_trained_TM_models, output_tm_components,
                  output_tm_topics, output_tm_metrics, output_clusters)
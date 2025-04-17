###################################
### File for defining functions ###
###################################

# Install packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from composition_stats import clr
import _pickle as cPickle
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import MDS
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from scipy.spatial.distance import squareform, pdist
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import wilcoxon
from itertools import combinations
import csv


# Functions for preprocessing of microbiome data
def data_transformation_clr(dataframe_in):
    """ Centered-log ratio transformation
    :param dataframe_in: The input microbiome data set (e.g. 16S, 18S,..)
    :return: The clr transformed data set
    """
    dataframe_in = dataframe_in + 0.5
    dataframe_in = dataframe_in.to_numpy()
    df_trans_clr = pd.DataFrame(clr(dataframe_in))
    df_trans_clr = abs(df_trans_clr.round())
    return df_trans_clr

def data_transformation_fractions(dataframe_in):
    """ Data transformation for obtaining fractionized counts
    :param dataframe_in: The input microbiome data set (e.g. 16S, 18S,..)
    :return: The fractionated data set
    """
    df_trans_fract = dataframe_in.div(dataframe_in.sum(axis=1), axis=0)
    return df_trans_fract

# Topic Modeling functions
def lda_on_microbiome_data(dataframe_in, dimensionality, N_lda):
    """ Latent Dirichlet Allocation (LDA) on microbiome data
    :param dataframe_in: The input microbiome data set (e.g. 16S, 18S,..)
    :param dimensionality: The number of topics (k)
    :param N_lda: The total number of samples in dataframe_in
    :return: The fitted LDA model, the LDA evaluation metrics lda_score and lda_perplexity,
    the LDA defined topics, the contributing components (OTUs) of each topic
    """
    lda_model = LatentDirichletAllocation(n_components= dimensionality, total_samples = N_lda,
         random_state=0)
    lda_topics = lda_model.fit_transform(dataframe_in)
    lda_score = lda_model.score(dataframe_in)
    lda_perplexity = lda_model.perplexity(dataframe_in)
    lda_components = lda_model.components_
    lda_topics = pd.DataFrame(lda_topics)
    lda_metrics_dict = {'lda score:': lda_score, 'lda perplexity': lda_perplexity}
    lda_metric_df = pd.DataFrame(lda_metrics_dict.items())
    lda_components = pd.DataFrame(lda_components, columns=dataframe_in.columns)
    return lda_model, lda_metric_df, lda_topics, lda_components

def NNMF_on_microbiome_data(dataframe_in, dimensionality):
    """ Non-negative Matrix Factorization (NNMF) on microbiome data
    :param dataframe_in: The input microbiome data set (e.g. 16S, 18S,..)
    :param dimensionality: The number of topics (k))
    :return: The fitted NNMF model, the NNMF defined topics, the contributing components (OTUs)
    of each topic
    """
    nnmf_model = NMF(n_components=dimensionality, init='random', random_state=0)
    nnmf_topics = nnmf_model.fit_transform(dataframe_in)
    nnmf_components = nnmf_model.components_
    nnmf_topics = pd.DataFrame(nnmf_topics)
    nnmf_components = pd.DataFrame(nnmf_components, columns=dataframe_in.columns)
    return nnmf_model, nnmf_topics, nnmf_components

# Alternative dimensionality reduction pproaches
def PCA_on_microbiome_data(dataframe_in, dimensionality):
    """ Principal Component Analysis (PCA) on microbiome data
    :param dataframe_in: The input microbiome data set (e.g. 16S, 18S,..)
    :param dimensionality: The number of components (k)
    :return: pca_model: the trained PCA model, pca_clusters: the PCA clusters per sample,
    pca_components: the features (OTU) per PCA cluster
    """
    pca = PCA(n_components=dimensionality)
    pca_model = pca.fit(dataframe_in)
    pca_components = pca_model.components_
    pca_components = pd.DataFrame(pca_components, columns=dataframe_in.columns)
    pca_clusters = pca_model.transform(dataframe_in)
    return pca_model, pca_clusters, pca_components

def pcoa_on_microbiome_data(dataframe_in, dimensionality):
    """ Principal Coordinates Analysis (PCoA) on microbiome data using Bray-Curtis distance
    :param dataframe_in: The input microbiome data set (e.g. 16S, 18S,..)
    :param dimensionality: The number of dimensions (k) for the PCoA projection
    :return: pcoa_model: the trained MDS model, pcoa_clusters: the PCoA clusters per sample
    """
    # Compute Bray-Curtis distance matrix
    distance_matrix = pd.DataFrame(squareform(pdist(dataframe_in, metric='braycurtis')),
                                   index=dataframe_in.index, columns=dataframe_in.index)
    # Perform PCoA using MDS (Multidimensional Scaling)
    mds = MDS(n_components=dimensionality, dissimilarity='precomputed', metric=True)
    pcoa_clusters = mds.fit_transform(distance_matrix)
    return mds, pcoa_clusters

# Dimensionality reduction function
def main_function_topic_generation(dimensionality, preprocessing_method, topic_modeling_method, clustering_method,
                input_data, input_data_clr, input_metadata, output_trained_TM_models, output_tm_components,
                output_tm_topics, output_tm_metrics, output_clusters):
    """ Topic Modeling on Microbiome Data
    :param dimensionality: The number of topics (k)
    :param preprocessing_method: 'clr', 'fractions', 'none'
    :param topic_modeling_method: 'lda', 'nnmf', 'none'
    :param clustering_method: 'pca', 'pcoa', 'none'
    :param input_data: Path to microbiome data set
    :param input_data_clr: Path to clr-transformed data set
    :param input_metadata: Path to metadata set
    :param output_trained_TM_models: Path to saving location
    :param output_tm_components: Path to saving location
    :param output_tm_topics: Path to saving location
    :param output_tm_metrics: Path to saving location
    :param output_clusters: Path to saving location
    :return:
    """
    # Load and prepare data: Ensuring same samples are processed
    df = pd.read_csv(input_data, sep=',', index_col=0, header=0)
    df = df.fillna(0)
    # clr transformed data
    df_clr = pd.read_csv(input_data_clr, sep=',', index_col=0, header=0)
    df_clr = df_clr.fillna(0)
    df_clr.index = df.index
    # Load metadata into workspace
    df_metadata = pd.read_csv(input_metadata, sep=',', index_col=0, header=0)
    # create lists with sample names in microbiome data and metadata
    n_ids = df.index.values.tolist()
    n_ids_metadata = df_metadata.index.values.tolist()
    # Identify the common sample ids of microbiome data and metadata
    common_ids = [x for x in n_ids if x in n_ids_metadata]
    df = df.loc[common_ids]  # keeping only the common samples
    df_metadata = df_metadata.loc[common_ids]
    df_clr = df_clr.loc[common_ids]
    df_clr.columns = df.columns
    # defining the total sample number
    N_sample_number = int(len(df.index))
    # Data preprocessing
    if preprocessing_method == 'clr':
        df_trans = df_clr
    elif preprocessing_method == 'fractions':
        df_trans = data_transformation_fractions(df)
    elif preprocessing_method == 'none':
        df_trans = df
    else:
        print('Invalid preprocessing_method')
    # Topic Modeling
    # LDA
    if topic_modeling_method == 'lda':
        lda_model, lda_metric_df, lda_topics, lda_components = lda_on_microbiome_data(
            df_trans, dimensionality, N_sample_number)
        # Save LDA topics, components, metrics and model
        lda_topics.index = df_metadata.index
        lda_topics.to_csv(output_tm_topics + 'lda_dim_'
                          + str(dimensionality) + '_topic_model_' + str(topic_modeling_method) + '_prepro_' +
                          str(preprocessing_method) + '_topics.csv')
        lda_components.to_csv(output_tm_components + 'lda_dim_'
                              + str(dimensionality) + '_topic_model_' + str(topic_modeling_method) + '_prepro_' +
                              str(preprocessing_method) + '_components.csv')
        lda_metric_df.to_csv(output_tm_metrics + 'lda_dim_'
                             + str(dimensionality) + '_topic_model_' + str(topic_modeling_method) + '_prepro_' +
                             str(preprocessing_method) + '_metrics.csv')
        with open(output_trained_TM_models + 'lda_dim_'
                  + str(dimensionality) + '_topic_model_' + str(topic_modeling_method) + '_prepro_' +
                  str(preprocessing_method) + '_model', "wb") as output_file:
            cPickle.dump(lda_model, output_file)
    ## NNMF
    elif topic_modeling_method == 'nnmf':
        nnmf_model, nnmf_topics, nnmf_components = NNMF_on_microbiome_data(df_trans, dimensionality)
        # Save the NNMF topics, components and model
        nnmf_topics.index = df_metadata.index
        nnmf_topics.to_csv(output_tm_topics + 'nnmf_dim_'
                              + str(dimensionality) + '_topic_model_' + str(topic_modeling_method) + '_prepro_' +
                              str(preprocessing_method) + '_topics.csv')
        nnmf_components.to_csv(output_tm_components + 'nnmf_dim_'
                              + str(dimensionality) + '_topic_model_' + str(topic_modeling_method) + '_prepro_' +
                              str(preprocessing_method) + '_components.csv')
        with open(output_trained_TM_models + 'nnmf_dim_'
                  + str(dimensionality) + '_topic_model_' + str(topic_modeling_method) + '_prepro_' +
                  str(preprocessing_method) + '_model', "wb") as output_file:
            cPickle.dump(nnmf_model, output_file)
    # Alternative Dimensionality Reduction Methods
    elif topic_modeling_method == 'none':
        if clustering_method == 'pca':
            pca_model, pca_clusters, pca_components = PCA_on_microbiome_data(df_trans, dimensionality)
            pca_clusters = pd.DataFrame(pca_clusters)
            pca_components = pd.DataFrame(pca_components)
            # Save the pca cluster and components
            pca_clusters.index = df_metadata.index
            pca_clusters.to_csv(output_clusters + 'pca_dim_'
                               + str(dimensionality) + '_topic_model_' + str(topic_modeling_method) + '_prepro_' +
                               str(preprocessing_method) + '_clusters.csv')
            # Save the pca components
            pca_components.to_csv(output_clusters + 'pca_dim_'
                                + str(dimensionality) + '_topic_model_' + str(topic_modeling_method) + '_prepro_' +
                                str(preprocessing_method) + '_components.csv')
        elif clustering_method == 'pcoa':
            print('pcoa in progress')
            pcoa_model, pcoa_clusters = pcoa_on_microbiome_data(df_trans, dimensionality)
            pcoa_clusters = pd.DataFrame(pcoa_clusters)

            # Save the pcoa cluster
            pcoa_clusters.index = df_metadata.index
            pcoa_clusters.to_csv(output_clusters + 'pcoa_dim_'
                                 + str(dimensionality) + '_topic_model_' + str(topic_modeling_method) + '_prepro_' +
                                 str(preprocessing_method) + '_clusters.csv')

        elif clustering_method == 'none':
            print('No Topic Modeling or Clustering done')
        else:
            print('Invalid Clustering method')
    else:
         print('Invalid Topic Modeling Method')
        
# Function for ecological evaluation with Random Forest
def main_function_random_forest(dimensionality, preprocessing_method, topic_modeling_method, clustering_method, topic_path, cluster_path,
                  rf_method, input_rf_target_variable, input_data, input_data_clr, input_metadata, output_rf_model, output_rf_metrics):
    """" Random Forest Evaluation of Dimensionality Reduction Methods
    :param dimensionality: The number of topics (k)
    :param preprocessing_method: 'clr', 'fractions', 'none'
    :param topic_modeling_method: 'lda', 'nnmf', 'none'
    :param clustering_method: 'pca', 'pcoa', 'none'
    :param topic_path: Path to input data set TM topics
    :param cluster_path: Path to input data set PCA/PCoA components
    :param rf_method: 'regression', 'classification'
    :param input_rf_target_variable: e.g. 'salinity', 'temperature', 'Chl_a'
    :param input_data: Path to microbiome data set
    :param input_data_clr: Path to clr-transformed data set
    :param input_metadata: Path to metadata set
    :param output_rf_model: Path for saving the RF model
    :param output_rf_metrics: Path for saving RF metrics
    """
    # Load microbiome data into workspace and preparing it
    df = pd.read_csv(input_data, sep=',', index_col=0, header=0)
    df = df.fillna(0)
    df_clr = pd.read_csv(input_data_clr, sep=',', index_col=0, header=0)
    df_clr.index = df.index
    # Load metadata and transform date column
    df_metadata = pd.read_csv(input_metadata, sep=',', index_col=0, header=0)
    df_metadata['date'] = pd.to_datetime(df_metadata['date'], format='%d/%m/%Y')
    # Create lists with sample names in microbiome data and metadata
    n_ids = df.index.values.tolist()
    n_ids_metadata = df_metadata.index.values.tolist()
    # Identify the common sample ids of microbiome data and metadata
    common_ids = [x for x in n_ids if x in n_ids_metadata]
    df = df.loc[common_ids]
    df_metadata = df_metadata.loc[common_ids]
    df_clr = df_clr.loc[common_ids]
    # Perform fractions preprocessing
    df_fractions = data_transformation_fractions(df)
    # Load TM/PCA/PCoA data based on input parameters
    # Preprocesseing: clr
    if preprocessing_method == 'clr' and topic_modeling_method == 'lda':
        data_for_rf = pd.read_csv(topic_path + 'lda_dim_' + str(dimensionality) + '_topic_model_lda_prepro_clr_topics.csv', 
                                  sep=',', index_col=0, header=0)
        data_for_rf.index = df.index
    elif preprocessing_method == 'clr' and topic_modeling_method == 'nnmf':
        data_for_rf = pd.read_csv(topic_path + 'nnmf_dim_' + str(dimensionality) + '_topic_model_nnmf_prepro_clr_topics.csv', 
                                  sep=',', index_col=0, header=0)
        data_for_rf.index = df.index
    elif preprocessing_method == 'clr' and topic_modeling_method == 'none' and clustering_method == 'pca':
        data_for_rf = pd.read_csv(cluster_path + 'pca_dim_' + str(dimensionality) + '_topic_model_none_prepro_clr_clusters.csv', 
                                  sep=',', index_col=0, header=0)
        data_for_rf.index = df.index
    elif preprocessing_method == 'clr' and topic_modeling_method == 'none' and clustering_method == 'pcoa':
        data_for_rf = pd.read_csv(cluster_path + 'pcoa_dim_' + str(dimensionality) + '_topic_model_none_prepro_clr_clusters.csv', 
                                  sep=',', index_col=0, header=0)
        data_for_rf.index = df.index
    elif preprocessing_method == 'clr' and topic_modeling_method == 'none' and clustering_method == 'none':
        data_for_rf = df_clr
        data_for_rf.index = df_clr.index
    # Preprocessing: fractions
    elif preprocessing_method == 'fractions' and topic_modeling_method == 'lda':
        data_for_rf = pd.read_csv(topic_path + 'lda_dim_' + str(dimensionality) + '_topic_model_lda_prepro_fractions_topics.csv', 
                                  sep=',', index_col=0, header=0)
        data_for_rf.index = df.index
    elif preprocessing_method == 'fractions' and topic_modeling_method == 'nnmf':
        data_for_rf = pd.read_csv(topic_path + 'nnmf_dim_' + str(dimensionality) + '_topic_model_nnmf_prepro_fractions_topics.csv', 
                                  sep=',', index_col=0, header=0)
        data_for_rf.index = df.index
    elif preprocessing_method == 'fractions' and topic_modeling_method == 'none' and clustering_method == 'pca':
        data_for_rf = pd.read_csv(cluster_path + 'pca_dim_' + str(dimensionality) + '_topic_model_none_prepro_fractions_clusters.csv', 
                                  sep=',', index_col=0, header=0)
        data_for_rf.index = df.index
    elif preprocessing_method == 'fractions' and topic_modeling_method == 'none' and clustering_method == 'pcoa':
        data_for_rf = pd.read_csv(cluster_path + 'pcoa_dim_' + str(dimensionality) + '_topic_model_none_prepro_fractions_clusters.csv', 
                                  sep=',', index_col=0, header=0)
        data_for_rf.index = df.index
    elif preprocessing_method == 'fractions' and topic_modeling_method == 'none' and clustering_method == 'none':
        data_for_rf = df_fractions
        data_for_rf.index = df_fractions.index
    # No preprocessing
    elif preprocessing_method == 'none' and topic_modeling_method == 'lda':
        data_for_rf = pd.read_csv(topic_path + 'lda_dim_' + str(dimensionality) + '_topic_model_lda_prepro_none_topics.csv', 
                                  sep=',', index_col=0, header=0)
        data_for_rf = data_for_rf.loc[common_ids]
        data_for_rf.index = df.index
    elif preprocessing_method == 'none' and topic_modeling_method == 'nnmf':
        data_for_rf = pd.read_csv(topic_path + 'nnmf_dim_' + str(dimensionality) + '_topic_model_nnmf_prepro_none_topics.csv', 
                                  sep=',', index_col=0, header=0)
        data_for_rf.index = df.index
    elif preprocessing_method == 'none' and topic_modeling_method == 'none' and clustering_method == 'pca':
        data_for_rf = pd.read_csv(cluster_path + 'pca_dim_' + str(dimensionality) + '_topic_model_none_prepro_none_clusters.csv', 
                                  sep=',', index_col=0, header=0)
        data_for_rf.index = df.index
    elif preprocessing_method == 'none' and topic_modeling_method == 'none' and clustering_method == 'pcoa':
        data_for_rf = pd.read_csv(cluster_path + 'pcoa_dim_' + str(dimensionality) + '_topic_model_none_prepro_none_clusters.csv', 
                                  sep=',', index_col=0, header=0)
        data_for_rf.index = df.index
    # Unprocessed evaluation
    elif preprocessing_method == 'none' and topic_modeling_method == 'none' and clustering_method == 'none':
        data_for_rf = df
        data_for_rf.index = df.index
    else:
        print('Invalid TM or Clustering Method')

    # DRM evaluation with Random Forest
    # Classification with Random Forest for categorical target variables
    if rf_method == 'classification':
        # Encoding the target variables into numeric classes
        prep = preprocessing.LabelEncoder()
        # Timesseries Split
        mask_y_train = (df_metadata['date'] >= '2022-04-21') & (df_metadata['date'] <= '2022-12-29')
        mask_y_test = (df_metadata['date'] >= '2023-01-02') & (df_metadata['date'] <= '2023-05-01')
        # Creating training and test subsets for y
        df_metadata_train = df_metadata.loc[mask_y_train]
        df_metadata_test = df_metadata.loc[mask_y_test]
        # fit_transform --> Label Encoder
        df_metadata_train[input_rf_target_variable] = prep.fit_transform(df_metadata_train[input_rf_target_variable])
        df_metadata_test[input_rf_target_variable] = prep.transform(df_metadata_test[input_rf_target_variable])
        # Based on the y values which based on the dates creating X training and X test subsets
        index_train = list(df_metadata_train.index)
        index_test = list(df_metadata_test.index)
        # Splitting the data based on dates
        X_train = data_for_rf.loc[index_train]
        X_test = data_for_rf.loc[index_test]
        y_train = df_metadata_train[input_rf_target_variable].dropna()
        y_test = df_metadata_test[input_rf_target_variable].dropna()

        # Building the RF model
        rf_model = RandomForestClassifier(random_state=44)

        # Fit the models to training data
        rf_model.fit(X_train, y_train)

        # Save the trained models
        with open(output_rf_model + 'classification/' + 'dim_' + str(dimensionality) + '_topic_model_' + str(
            topic_modeling_method) + '_clustering_method_' + str(clustering_method) + '_prepro_' + str(preprocessing_method)
                  + '_target_' + str(input_rf_target_variable) + '_classification_model', 'wb') as output_file:
            cPickle.dump(rf_model, output_file)

        # Making predictions with the trained model
        y_pred = rf_model.predict(X_test)

        # Evaluating the models prediction ability
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')

        # Joining metrics to dic and sve it as csv
        metric_dict = pd.DataFrame([{'Accuracy score': accuracy, 'F1_macro score': f1_macro}])
        metric_dict.to_csv(output_rf_metrics + 'classification/' + 'dim_' + str(dimensionality) + '_topic_model_' + str(
            topic_modeling_method) + '_clustering_method_' + str(clustering_method) + '_prepro_' + str(preprocessing_method)
                           + '_target_' + str(input_rf_target_variable) + '_classification_metrics.csv')

    # Regression with Random Forest for numeric target variables
    elif rf_method == 'regression':
        print('regression is performing')
        # Timesseries split
        # Splitting the data into training and test subsets based on time
        mask_y_train = (df_metadata['date'] >= '2022-04-21') & (df_metadata['date'] <= '2022-12-29')
        mask_y_test = (df_metadata['date'] >= '2023-01-02') & (df_metadata['date'] <= '2023-05-01')
        # Creating training and test subsets for y
        df_metadata_train = df_metadata.loc[mask_y_train].dropna()
        df_metadata_test = df_metadata.loc[mask_y_test].dropna()
        # Based on the y values which based on the dates creating X training and X test subsets
        index_train = list(df_metadata_train.index)
        index_test = list(df_metadata_test.index)
        # Splitting the data based on dates
        X_train = data_for_rf.loc[index_train]
        X_test = data_for_rf.loc[index_test]
        y_train = df_metadata_train[input_rf_target_variable].astype(int)
        y_test = df_metadata_test[input_rf_target_variable].astype(int)

        # Building the Random Forest Regressor
        rf_model = RandomForestRegressor() #random_state=0)
        # Fit the model to training data
        rf_model.fit(X_train, y_train)

         # Save the trained model
        with open(output_rf_model + 'regression/' + 'dim_' + str(dimensionality) + '_topic_model_' + str(topic_modeling_method)
                  + '_clustering_method_' + str(clustering_method) + '_prepro_' + str(preprocessing_method)
                  + '_target_' + str(input_rf_target_variable) + '_regression_model', 'wb') as output_file:
            cPickle.dump(rf_model, output_file)

        # Making predictions with the trained model
        y_pred = rf_model.predict(X_test)
        # Evaluating the models prediction ability
        # R^2 and others: Mean Absolute Error, Root Mean Squared Error
        rsquared = r2_score(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

        # Joining metrics to dic and save it
        metric_regr = pd.DataFrame([{'R^2': rsquared, 'Mean Absolute Error': mae, 'Root Mean Squared Error': rmse}])
        metric_regr.to_csv(output_rf_metrics + str(rf_method)+ '/' + 'dim_' + str(dimensionality) + '_topic_model_'
                           + str(topic_modeling_method) + '_clustering_method_' + str(clustering_method)
                           + '_prepro_' + str(preprocessing_method) + '_target_' + str(input_rf_target_variable)
                           + str(rf_method) + '_metrics.csv')
    else:
        print('Invalid rf_method')

# Functions for functional evaluation
# Perform Mantel Test
def mantel_test(matrix1, matrix2):
    matrix1_flat = squareform(matrix1)
    matrix2_flat = squareform(matrix2)
    correlation, p_value = spearmanr(matrix1_flat, matrix2_flat)
    return correlation, p_value

# Calculate Jaccard distance matrices
def compute_jaccard_distance(matrix):
    return squareform(pdist(matrix, metric='jaccard'))

# Calculate Eucledean distance matrices
def compute_euclidean_distance(matrix):
    return squareform(pdist(matrix, metric='euclidean'))

# Parser function for filename information (optimize based on your file format)
def parse_filename(filename):
    parts = filename.split('_')
    tm = parts[0]
    dim = int(parts[2])
    prepro = parts[7]
    return tm, dim, prepro
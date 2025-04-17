#######################################################################
### File for evaluating the topics and PCA compontents functionally ###
#######################################################################

# Load needed functions
from functions import *

# Directory containing the TM topic or PCA component x OTU matrices
directory = 'path/to/your/components/'

# List to store results
results = []

# Load the FAPROTAX OTU x Function matrix
otu_function_matrix = pd.read_csv('path/to/your/function_matrix', index_col=1, header=0)

# Load full microbiome data and find functional annotated OTUs (as an extra savety step)
otus_16 = pd.read_csv('path/to/your/data', header=0, sep=',', index_col=0).T
n_ids = otu_function_matrix.index.values.tolist()
n_ids_16 = otus_16.index.values.tolist()
# identify the common sample ids of microbiome data and function matrix
common_ids = [x for x in n_ids_16 if x in n_ids]
# Keep common ids
otu_function_matrix = otu_function_matrix.loc[common_ids]

# Modify function matrix: Replace strings with integers; remove columns with non-binary categories
otu_function_matrix = otu_function_matrix.replace('variable', 0.5)
otu_function_matrix = otu_function_matrix.replace('yes,no', 0.5)
otu_function_matrix = otu_function_matrix.replace('yes', 1)
otu_function_matrix = otu_function_matrix.replace('no', 0)
otu_function_matrix = otu_function_matrix.drop(otu_function_matrix[['elements', 'main_element', 'electron_donor', 'electron_acceptor']], axis=1)
# Convert to int
otu_function_matrix = otu_function_matrix.astype(int)
# Calculate Jaccard distance matrix for the OTU x function matrix
jaccard_distance_matrix = compute_jaccard_distance(otu_function_matrix)

# Loop through each topic or PCA Component x OTU file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        # Parse the filename to get tm, dim, prepro
        tm, dim, prepro = parse_filename(filename)
        # Load the topic or PCA-component x OTU matrix
        filepath = os.path.join(directory, filename)
        topic_otu_matrix = pd.read_csv(filepath, index_col=0).T
        # Normalize components
        topic_otu_matrix = topic_otu_matrix.div(topic_otu_matrix.sum(axis=1), axis=0)
        # Keep the common ids
        topic_otu_matrix = topic_otu_matrix.loc[common_ids]
        # Compute the Euclidean distance matrix for the topic or PCA Component x OTU matrix
        euclidean_distance_matrix = compute_euclidean_distance(topic_otu_matrix)
        # Perform Mantel test
        correlation, p_value = mantel_test(jaccard_distance_matrix, euclidean_distance_matrix)
        # Store results
        results.append([tm, dim, prepro, correlation, p_value])

# Put results in a dataframe
results_df = pd.DataFrame(results, columns=['tm', 'dim', 'prepro', 'mantel_correlation', 'p_value'])
results_df = results_df.sort_values(by='mantel_correlation', ascending= False)
# Save results
#results_df.to_csv('your/output/path', index=False)

# Rank the results
results_df = results_df[results_df['tm'] != 'none']
print('new', results_df)
# Get a rank value fo each dimension based on the approach with the maximal performance
results_df['ranked'] = results_df.groupby('dim')['mantel_correlation'].rank(method = 'max', ascending=False)
print('watch2', results_df)

# Group by 'tm' and 'prepro' and calculate the mean rank
mean_rank_per_approach = results_df.groupby(['tm', 'prepro'])['ranked'].mean().reset_index()

# Optionally, you can sort the result to see which combination has the best (lowest) mean rank
mean_rank_per_approach_sorted = mean_rank_per_approach.sort_values(by='ranked', ascending=True)
# Save results
#mean_rank_per_approach_sorted.to_csv('your/output/path')

# Test Significance between the ranks of the results
# Wilcoxon test to check if the differences between the approaches are significant
# List all unique (tm, prepro) combinations
unique_combinations = mean_rank_per_approach_sorted[['tm', 'prepro']].drop_duplicates()
# Generate all possible pairs of combinations for comparison
pairwise_combinations = list(combinations(unique_combinations.index, 2))
# Initialize a list to store the results
wilcoxon_results = []
# Perform Wilcoxon test for each pairwise combination
for pair in pairwise_combinations:
    comb1 = unique_combinations.loc[pair[0]]
    comb2 = unique_combinations.loc[pair[1]]
    # Get the ranks for the two combinations
    ranks_comb1 = results_df[(results_df['tm'] == comb1['tm']) & (results_df['prepro'] == comb1['prepro'])]
    ranks_comb2 = results_df[(results_df['tm'] == comb2['tm']) & (results_df['prepro'] == comb2['prepro'])]
    # Merge on the 'dim' column to ensure the ranks are compared for the same dimensions
    merged = pd.merge(ranks_comb1[['dim', 'ranked']], ranks_comb2[['dim', 'ranked']], on='dim', suffixes=('_comb1', '_comb2'))
    # If there's no overlap, skip the test
    if len(merged) == 0:
        continue
    # Perform Wilcoxon signed-rank test
    stat, p_value = wilcoxon(merged['ranked_comb1'], merged['ranked_comb2'])
    # Store the results
    wilcoxon_results.append({
        'comb1': f"{comb1['tm']}_{comb1['prepro']}",
        'comb2': f"{comb2['tm']}_{comb2['prepro']}",
        'p_value': p_value
    })
# Convert results to DataFrame
wilcoxon_df = pd.DataFrame(wilcoxon_results)
# Apply Bonferroni correction
bonferroni_alpha = 0.05
wilcoxon_df['bonferroni_corrected_p'] = wilcoxon_df['p_value'] * len(wilcoxon_df)
wilcoxon_df['significant'] = wilcoxon_df['bonferroni_corrected_p'] < bonferroni_alpha
# Save the results
#wilcoxon_df.to_csv('your/output/path')
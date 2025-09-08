# topic_modeling_on_microbiome_data
## Repository for storing Python scripts for evaluating and executing topic modeling on high dimensional microbiome data.

### Repository Overview

This repository contains scripts for applying and evaluating topic modeling methods on high-dimensional microbiome data. The implemented methods include Latent Dirichlet Allocation (LDA), Non-negative Matrix Factorization (NNMF), and dimensionality reduction via PCA and PCoA.

The repository is organized as follows:

```README.md
topic_generation.py       # main script for topic generation & dimensionality reduction
functions.py              # helper functions (preprocessing, modeling, evaluation)
ecological_evaluation.py  # Random Forest evaluation of ecological signal retention
functional_evaluation.py  # Functional evaluation (Mantel test, distance comparisons)
```
For a well-grounded application of topic modeling, we evaluated the impact of 
* data preprocessing
  * centered-log ratio transformation (clr)
  * fractionated feature counts (fractions)
* model dimensionality
  * $2 < k < 200$
* and the choice of dimensionality reduction method
  * LDA
  * NNMF
  * PCA
  * PCoA

This evaluation was built on two steps: 1) An **ecological evaluation** using Random Forest (ecological_evaluation.py), in which we assessed the extent to which the specified parameters influenced the predictability of various ecological variables — or, in other words, the preservation of ecological information. 2) A **functional evaluation** (functional_evaluation.py), in which we assessed how much functional information was retained in the topics or PCA component × OTU matrices compared to the full microbiome.

### Input Data Requirements
1. Microbiome Data (input_data)
   * CSV file, samples × features (OTUs/ASVs)
   * Rows = samples, Columns = OTU/ASV IDs
   * First column = sample ID
2. CLR-transformed data (input_data_clr)
   * Same format as above, but already CLR-transformed
   * If not available, use the provided function data_transformation_clr() in functions.py
3. Metadata (input_metadata)
   * CSV file, samples x environmental parameters (salinity, temperature, ..)
   * Rows = samples (same IDs as in microbiome data)
## How to Run
1. Topic Generation
Run topic_generation.py after updating the file paths:
```
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
```
2. Ecological Evaluation
Use ecological_evaluation.py to run Random Forest classification/regression against ecological metadata variables.
3. Functional Evaluation

## Outputs
* TM Models 
* Topics → CSV with topic loadings per sample
* Topic modeling components → CSV with OTU loadings per topic
* TM models → Pickled model object
* Metrics → CSV with model scores (LDA perplexity/score, RF metrics).
* Random Forest models → Pickled model objects
* Clusters → CSV with PCA/PCoA projections per sample

# topic_modeling_on_microbiome_data
## Repository for storing Python scripts for evaluating and executing topic modeling on high dimensional microbiome data.

This repository contains a set of scripts and functions used to apply topic modeling methods—Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NNMF)—in comparison to the dimensionality reduction techniques PCA and PCoA on high-dimensional 16S rRNA amplicon data (topic_generation.py). We evaluated the impact of 
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

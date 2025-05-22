#!/bin/bash

# Run the scripts to produce embeddings and save them in data/generated/embeddings
set -e

for algorithm in 'cebra_hybrid_optimised' # 'autoencoder_optimised' 'autoregressor_autoencoder_optimised' 'tsne_optimised' 'cebra_hybrid_optimised'
do
  python3 c_elegans_results/embedding_algorithms/algorithms_optimised_parameters/${algorithm}.py
done

for algorithm in  'tsne' 'bundlenet' #  'pca' 'lda'  'autoencoder' 'autoregressor_autoencoder' 'cebra_hybrid' 'autoencoder' 'autoregressor_autoencoder' 'cebra_hybrid'
do
  python3 c_elegans_results/embedding_algorithms/${algorithm}.py
done

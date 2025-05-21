#!/bin/bash

# Run the scripts to produce embeddings and save them in data/generated/embeddings
set -e

for algorithm in 'tsne' #  'lda' 'bundlenet'  'autoencoder' 'autoregressor_autoencoder' 'cebra_hybrid'
do
  python3 c_elegans_results/embedding_algorithms/${algorithm}.py
done

for algorithm in 'tsne_optimised' 'autoencoder_optimised' 'autoregressor_autoencoder_optimised' 'cebra_hybrid_optimised'
do
  python3 c_elegans_results/embedding_algorithms/${algorithm}.py
done

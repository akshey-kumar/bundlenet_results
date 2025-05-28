#!/bin/bash

# Run the scripts to produce embeddings and save them in data/generated/embeddings/rat_results
set -e

for algorithm in 'pca' 'cca' 'rrr'
do
  python3 rat_results/embedding_algorithms/${algorithm}.py
done

# Scripts with time delay embeddings
for algorithm in  'pca' 'cca' 'rrr' 'bundlenet'  'cebra_hybrid' #  'autoencoder' 'dynamics_autoencoder'
do
  python3 rat_results/embedding_algorithms/with_time_delay_embedding/${algorithm}.py
done

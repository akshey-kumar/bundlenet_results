#!/bin/bash
set -e

## Performing evaluation of the microvariable (neuronal level) for all worms
# Loop for microvariable_evaluation.py
for worm_num in 0 1 2 3 4
do
    python3 c_elegans_embedding_evaluation/evaluation_scripts/microvariable_evaluation.py $worm_num
done

## Performing evaluation of embeddings of various algorithms for all worms
# Loop for behaviour_decoding_analysis.py and dynamics_predictability.py

algorithms=(
  'PCA'
  'PCA_time_delay_embedding'
  'autoencoder_time_delay_embedding'
  'dynamics_autoencoder_time_delay_embedding'
  'BunDLeNet_linear'
  'BunDLeNet_win_1'
  'BunDLeNet'
  'tsne_time_delay_embedding'
  'LDA_time_delay_embedding'
  'LDA'
  'cebra_h'
  'rnn_autoencoder'
)

for worm_num in 0 1 2 3 4
do
  for algorithm in "${algorithms[@]}"
      do
          echo $worm_num $algorithm
          python3 c_elegans_embedding_evaluation/evaluation_scripts/behaviour_decoding.py $algorithm $worm_num
          python3 c_elegans_embedding_evaluation/evaluation_scripts/dynamics_predictability.py $algorithm $worm_num
      done
done


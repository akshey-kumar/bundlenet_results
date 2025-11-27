#!/bin/bash

set -e

## Performing evaluation of the microvariable (neuronal level) for all worms
# Loop for neuronal_decoding.py
# for worm_num in 0 1 2 3 4
# do
#     python3 c_elegans_results/quantitative_evaluation/evaluation_scripts/neuronal_decoding.py $worm_num
# done

## Performing evaluation of embeddings of various algorithms for all worms
# Loop for behaviour_decoding_analysis.py and dynamics_predictability.py
for worm_num in 0 1 2 3 4
do
    for algorithm in  'bundlenet_tf' # 'pca' 'lda' 'tsne' 'autoencoder' 'dynamics_autoencoder' 'rnn_autoencoder' 'cebra_h' # 'bundlenet_linear' 'bundlenet_win_1'
    do
        python3 c_elegans_results/quantitative_evaluation/evaluation_scripts/behaviour_decoding.py $algorithm  $worm_num
        python3 c_elegans_results/quantitative_evaluation/evaluation_scripts/dynamics_predictability.py $algorithm $worm_num
    done
done

# Old Loop for old behaviour_decoding_analysi
# for worm_num in 0 1 2 3 4
# do
#     for algorithm in 'PCA' 'tsne' 'autoencoder' 'ArAe' 'BunDLeNet' 'cebra_h' # 'PCA_time_delay_embedding' 'autoencoder_time_delay_embedding' 'ArAe_time_delay_embedding' 'tsne_time_delay_embedding' 'cebra_h_time_delay_embedding' 'LDA_time_delay_embedding' 'BunDLeNet_linear' 'BunDLeNet_win_1'
#     do
#         python3 c_elegans_results/quantitative_evaluation/evaluation_scripts/behaviour_decoding.py $algorithm $worm_num
#         # python3 c_elegans_results/quantitative_evaluation/evaluation_scripts/dynamics_predictability.py $algorithm $worm_num
#     done
# done

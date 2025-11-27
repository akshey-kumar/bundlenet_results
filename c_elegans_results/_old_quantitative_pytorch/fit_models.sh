#!/bin/bash

set -e


for algorithm in  'bundlenet_win_1' 'bundlenet_linear'# 'bundlenet_linear'  'bundlenet'   'pca' 'lda' 'tsne' 'autoencoder' 'dynamics_autoencoder' 'rnn_autoencoder' 'cebra_hybrid'
do
    python3 c_elegans_results/quantitative_evaluation/$algorithm.py
done


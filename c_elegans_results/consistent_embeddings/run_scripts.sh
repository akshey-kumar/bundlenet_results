#!/bin/bash
set -e

#python c_elegans_results/consistent_embeddings/first_embedding.py
python c_elegans_results/consistent_embeddings/comparable_embeddings.py
python c_elegans_results/consistent_embeddings/plot_embeddings.py



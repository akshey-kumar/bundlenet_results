# BunDLe-Net Paper – Results

This repository contains all code required to reproduce the results and figures from the BunDLe-Net paper.

---

## Setup

### Requirements
- Python 3.10.12
- Dependencies listed in `requirements.txt`

### External Dependency
This repository requires the NCMCM toolbox:

https://github.com/neuroinformatics-uni-vienna/NC-MCM

Ensure it is installed and accessible in your Python environment.

---

## Repository Structure

- `c_elegans_results/` – Analyses and figures for _C. elegans_ data  
- `rat_results/` – Analyses and figures for rat hippocampus data  
- `monkey_results/` – Analyses and figures for primate (monkey) data 
- `data/raw/` – Raw datasets used in the study  
- `data/generated/` – Generated data, embeddings, and trained models  

---

## Code for figures from paper

**Figure 2**
- a–h: `c_elegans_results/embedding_algorithms/`
- i–j: `c_elegans_results/c_elegans_embedding_evaluation/`
- k: `c_elegans_results/behaviour_vs_embedding/`
- l–m: `c_elegans_results/ablation_studies/`
- n–p: `c_elegans_results/shuffling_experiments/`

**Figure 3**
- a–b: `c_elegans_results/time_alignment/`

**Figure 4**
- `c_elegans_results/consistent_embeddings/`

**Figure 5**
- a: `rat_results/embedding_algorithms/`
- b: `rat_results/consistency_of_behaviour_aligned_embeddings/`
- c: `rat_results/behaviour_decoding_analysis/`

---

### Supplementary Figures

- **Fig S2**:  
  `c_elegans_results/embedding_algorithms/plot_embeddings/plotting_embeddings.ipynb`

- **Fig S3**:  
  `c_elegans_results/ablation_studies/`

- **Fig S4**:  
  `c_elegans_results/shuffling_experiments/`

- **Fig S5a**:  
  `c_elegans_results/c_elegans_embedding_evaluation/figures/confusion_matrix.pdf`

- **Fig S5b**:  
  `c_elegans_results/c_elegans_embedding_evaluation/figures/dynamical_performance.pdf`

- **Fig S6**:  
  `c_elegans_results/c_elegans_embedding_evaluation/figures/distinct_motifs/`

- **Fig S7**:  
  `c_elegans_results/c_elegans_embedding_evaluation/learning_process/`

- **Fig S8**:  
  `rat_results/ablation_studies/plotting.ipynb`

- **Fig S9**:  
  `monkey_results/embeddings_monkey_1.ipynb`

---

## Contact

If you have any questions or issues, please feel free to email me at `akshey.kumar@univie.ac.at`.
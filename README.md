# caRBP-Pred
Deep Learning-based Prediction of Chromatin-Associated RNA-Binding Proteins

Authors: Qiang Sun, Feng Yang, Hao Sun, Xiaona Chen, Huating Wang

This repository contains a deep learning framework designed to identify Chromatin-associated RNA-binding Proteins (caRBPs) using Protein Language Models (pLM) and advanced hybrid neural architectures.

Key Features
Multi-Level Prediction: Supports both full-length Protein (up to 2000aa) and Peptide (50aa) level identification.
pLM Integration: Leverages ProtT5-XL-UniRef50 embeddings to capture high-dimensional evolutionary and biophysical information.
Hybrid Loss Function: Implements a Hybrid Dice-Focal Loss to specifically address class imbalance in biological datasets.
Attention Mechanism: Incorporates custom Attention Layers within CNN-BiLSTM architectures to focus on critical sequence motifs.
Comprehensive Ablation: Includes scripts to evaluate the relative contributions of sequence, secondary structure (SS), and pLM features.

Dependency
Python 3.8+
PyTorch (用于特征提取)
TensorFlow 2.x (用于模型训练与推理)
Transformers (HuggingFace)
Scikit-learn, Pandas, Numpy, Matplotlib

Installation
# Clone the repository
git clone https://github.com/YourUsername/caRBP-ML.git
cd caRBP-ML
# Install dependencies
pip install -r requirements.txt

Usage Workflow
1. Feature Extraction
Generate residue-level embeddings using the ProtT5 model.
python scripts/gen_T5_feature.py

Input: all_fasta.csv
Output: protein_plm_residue.npy

2. Model Training & Ablation Studies
Run experiments for either proteins or peptides. These scripts execute 5-fold StratifiedGroupKFold cross-validation.
For Proteins (pLM Focus): python scripts/plm_protein_model.py.
For Peptides (Secondary Structure Focus): python scripts/peptide_model.py.

3. Inference
Apply a trained model to new data to generate probability distributions and classification results.
python scripts/inference.py




caRBP-Pred is written in python3 for GUN Linux/Unix platforms. 
The training script various_agorithms.py depends on the tensorflow=2.8.0 

To predict the potential caRBPs, users just download the prediction_with_sliding_window.py, positive_peptide.csv, and negative_peptide.csv files, and use the following command:
python3.9 prediction_with_sliding_window.py positive_peptide.csv negative_peptide.csv RBP_fasta.csv >test_result



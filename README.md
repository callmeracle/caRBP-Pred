# caRBP-Pred
Deep Learning-based Prediction of Chromatin-Associated RNA-Binding Proteins
Authors: Qiang Sun, Feng Yang, Hao Sun, Huating Wang, Xiaona Chen

This repository contains a deep learning framework designed to identify Chromatin-associated RNA-binding Proteins (caRBPs) using Protein Language Models (pLM) and advanced hybrid neural architectures.

**Key Features**  
Multi-Level Prediction: Supports both full-length Protein (up to 2000aa) and Peptide (50aa) level identification.    
pLM Integration: Leverages ProtT5-XL-UniRef50 embeddings to capture high-dimensional evolutionary and biophysical information.    
Hybrid Loss Function: Implements a Hybrid Dice-Focal Loss to specifically address class imbalance in biological datasets.    
Attention Mechanism: Incorporates custom Attention Layers within CNN-BiLSTM architectures to focus on critical sequence motifs.    
Comprehensive Ablation: Includes scripts to evaluate the relative contributions of sequence, secondary structure (SS), and pLM features.    

# 1. Introduction
RNA-binding proteins (RBPs) play a pivotal role in cellular processes ranging from RNA metabolism to 3D genome organization. A distinct subset, chromatin-associated RBPs (caRBPs), binds directly to chromatin to function as transcriptional regulators. However, identifying caRBPs via traditional methods like Chromatin Immunoprecipitation Sequencing (ChIP-seq) and Mass Spectrometry (MS) is labor-intensive and costly. While computational tools for DNA- and RNA-binding protein (DRBP) prediction are available, they often rely on outdated Gene Ontology annotations and fail to capture the unique characteristics of chromatin association. Here, we introduce caRBP-Pred, a deep learning framework that integrates a pre-trained protein language model (pLM) with Convolutional Neural Networks (CNN) and Bidirectional Long Short-Term Memory networks (BiLSTM) networks. By leveraging full-length protein sequences and evolutionary embeddings from ProtT5-XL, caRBP-Pred achieves an AUROC of 0.82 and AUPRC of 0.58, significantly outperforming existing DNA- and RNA-binding protein (DRBP) predictors. Application of our model to the mouse proteome identified 41 high-confidence caRBP candidates. Multi-dimensional validation using the COMPARTMENTS database and InterProScan confirmed that a proportion of these candidates possess experimentally verified chromatin-binding domains or high-confidence nuclear localization. Collectively, caRBP-Pred is the first tool specifically designed for caRBP prediction, offering a valuable resource for investigating the regulatory roles of caRBPs on transcription.

# 2.Code Details
The repository is organized into specialized scripts for each stage of the machine learning lifecycle:
gen_T5_feature.py: Extracts residue-level features from protein sequences using the T5EncoderModel.
plm_protein_model.py: The primary training script for full-length proteins, incorporating pLM features and a dual-pathway CNN-BiLSTM architecture.
peptide_model.py: A specialized model for short peptide fragments (50aa), used for classifying chromain-binding peptide .
inference.py: A deployment-ready script for high-throughput prediction on new FASTA sequences, including automated probability distribution plotting.

# 3.Dependency
Python 3.8+  
PyTorch   
TensorFlow 2.x   
Transformers (HuggingFace)  
Scikit-learn  
Pandas  
Numpy  
Matplotlib

# 4. Dataset
The data is structured to facilitate ablation studies between sequence-only and feature-enhanced models:
Sequences: Provided in CSV format containing protein_id and sequence columns.  
Secondary Structure: Predicted SS labels matched to protein IDs.  
Embeddings: .npy files containing the pre-computed ProtT5 residue-level features.  

# 5. Installation
git clone https://github.com/YourUsername/caRBP-ML.git  
cd caRBP-ML
conda create -n carbp_env python=3.8
conda activate carbp_env

# 6. Steps for Re-training the Model 
To retrain the model on your own large-scale dataset, follow these steps:
Prepare FASTA: Save your genome-scale sequences in a CSV file.  
Feature Extraction: Run gen_T5_feature.py to generate the .npy embedding matrix. This is a one-time computational cost.  
Configure Alpha: The scripts automatically calculate a dynamic alpha (negative sample ratio) to balance the Hybrid Loss.  
Execute Training:
python scripts/plm_protein_model.py  
Validation: The script will perform 5-Fold StratifiedGroupKFold cross-validation and save the best-performing model to saved_models_final_ablation/  

# 7. Steps for Identifying Potential caRBPs
Once you have a trained model (.h5 file), use the following steps for discovery:  
Extract Features: Generate pLM embeddings for the target sequences using the extraction script.  
Run Inference:  
python scripts/inference.py   
Thresholding: The default decision threshold is set to 0.7 to minimize false positives during genome-wide screening.  
Analyze Results: Review Final_BiLSTM_pLM_Inference.csv for gene names, probabilities, and classification decisions.  

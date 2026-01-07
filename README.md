# caRBP-Pred
Deep Learning-based Prediction of Chromatin-Associated RNA-Binding Proteins Using Short Peptide Sequences

Authors: Qiang Sun, Feng Yang, Hao Sun, Xiaona Chen, Huating Wang

caRBP-Pred is written in python3 for GUN Linux/Unix platforms. 
The training script various_agorithms.py depends on the tensorflow=2.8.0 

To predict the potential caRBPs, users just download the prediction_with_sliding_window.py, positive_peptide.csv, and negative_peptide.csv files, and use the following command:
python3.9 prediction_with_sliding_window.py positive_peptide.csv negative_peptide.csv RBP_fasta.csv >test_result



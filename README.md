# caRBPPred
Predicting chromatin-associated RNA-binding proteins from short sequencing using deep learning

Authors: Qiang Sun, Feng Yang, Xiaona Chen, Hao Sun, Huating Wang

caRBPPred is written in python3 for GUN Linux/Unix platforms. 
caRBPPred depends on the following tools:

Users just download the prediction_with_sliding_window.py, positive_peptide.csv, and negative_peptide.csv files.
To predict the potential caRBPs with sliding window (50bp), we can use the following command:
python3.9 prediction_with_sliding_window.py positive_peptide.csv negative_peptide.csv RBP_fasta.csv >test_result



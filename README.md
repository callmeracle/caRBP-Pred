# caRBPpred
Predicting chromatin-associated RNA-binding proteins from short sequencing using deep learning

To predict the potential caRBPs with sliding window (50bp), we can use the following command:
python3.9 prediction_with_sliding_window.py positive_peptide.csv negative_peptide.csv RBP_fasta.csv >test_result

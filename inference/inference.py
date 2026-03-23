import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging

# ==================== 0. ====================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU : {e}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 1. ====================

class AttentionLayer(tf.keras.layers.Layer):
    """复刻自 training (12).py"""
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name="att_w", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_b", shape=(input_shape[1], 1), initializer="zeros")
    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.expand_dims(K.softmax(et), axis=-1)
        return K.sum(x * at, axis=1)

def hybrid_dice_focal_loss(gamma=2.0, alpha=0.9):
    def loss(y_true, y_pred): return 0
    return loss

# ==================== 2.  ====================

def run_final_bilstm_inference(input_csv, ptm_npy_path, model_path, output_csv):
    maxlen = 2000  
    aa_map = {c: i + 1 for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    
    if not os.path.exists(model_path):
        logger.error(f": {model_path}")
        return

    custom_objects = {'AttentionLayer': AttentionLayer, 'loss': hybrid_dice_focal_loss()}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    df_input = pd.read_csv(input_csv)
    X_plm_raw = np.load(ptm_npy_path, allow_pickle=True)

    results = []
    all_probs = []
    threshold = 0.7

    pdf_report = PdfPages("Inference_Probability_Distribution.pdf")

    for idx, row in df_input.iterrows():
        gene_name = str(row[0])
        sequence = str(row[1]).upper()[:maxlen]
        
        idx_vec = [aa_map.get(aa, 0) for aa in sequence]
        idx_vec = idx_vec + [0] * (maxlen - len(idx_vec))
        X_idx_single = np.array(idx_vec).reshape(1, -1)
        
        ptm_feat = X_plm_raw[idx]
        X_plm_gap_single = np.mean(ptm_feat, axis=0).reshape(1, -1)
        
        prob = model.predict([X_idx_single, X_plm_gap_single], verbose=0)[0][0]
        all_probs.append(prob)
        
        decision = "Positive" if prob > threshold else "Negative"

        results.append({
            "Gene_Name": gene_name,
            "Prediction_Prob": f"{prob:.4f}",
            "Decision": decision,
            "Threshold": threshold
        })
        
        if (idx + 1) % 100 == 0:
            logger.info(f"已处理 {idx + 1}/{len(df_input)}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_probs, bins=50, color='skyblue', edgecolor='black')
    ax.axvline(0.7, color='red', linestyle='--', label='Threshold (0.7)')
    ax.set_title("Distribution of Prediction Probabilities (BiLSTM+pLM)")
    ax.set_xlabel("Probability"); ax.set_ylabel("Frequency"); ax.legend()
    pdf_report.savefig(fig); plt.close()
    pdf_report.close()

    pd.DataFrame(results).to_csv(output_csv, index=False)
    logger.info(f"Done! CSV : {output_csv}, PDF : Inference_Probability_Distribution.pdf")

# ==================== 3.  ====================
if __name__ == "__main__":
    task = {
        "input_csv": "inference/all_fasta.csv",
        "ptm_npy_path": "./inference_protein_plm_residue.npy", 
        "model_path": "../model/model.h5",
        "output_csv": "Final_BiLSTM_pLM_Inference.csv"
    }
    run_final_bilstm_inference(**task)


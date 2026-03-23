import os
import time
import logging
import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import *

# ==================== 0.  ====================
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_SAVE_DIR = "./saved_models_final_ablation"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ==================== 1.  ====================

class AttentionLayer(Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name="att_w", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_b", shape=(input_shape[1], 1), initializer="zeros")
    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.expand_dims(K.softmax(et), axis=-1)
        return K.sum(x * at, axis=1)

def hybrid_dice_focal_loss(gamma=2.0, alpha=0.9):

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        
        f_loss = -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - \
                 K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
        
        intersection = K.sum(y_true * y_pred)
        dice_loss = 1.0 - (2.0 * intersection + K.epsilon()) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + K.epsilon())
        return f_loss + dice_loss
    return loss

def build_cnnbilstm_plm_pure_seq_model(maxlen, alpha, plm_dim=None):
    in_seq = Input(shape=(maxlen,), name="Seq_Input")
    emb = Embedding(input_dim=21, output_dim=64, input_length=maxlen)(in_seq)
    
    bilstm = Bidirectional(LSTM(64, return_sequences=True))(emb)
    att_feat = AttentionLayer()(bilstm)
    cnn_feat = GlobalAveragePooling1D()(Conv1D(64, 3, activation='relu', padding='same')(emb))
    
    main_seq_feat = concatenate([att_feat, cnn_feat])
    inputs, feats = [in_seq], [main_seq_feat]
    
    if plm_dim is not None:
        in_plm = Input(shape=(plm_dim,), name="pLM_Input")
        plm_feat = Dense(64, activation='relu')(in_plm)
        inputs.append(in_plm)
        feats.append(plm_feat)
        
    merged = concatenate(feats) if len(feats) > 1 else main_seq_feat
    x = Dense(64, activation='relu')(merged)
    out = Dense(1, activation='sigmoid')(Dropout(0.4)(x))
    
    model = Model(inputs=inputs, outputs=out)
    model.compile(loss=hybrid_dice_focal_loss(alpha=alpha), 
                  optimizer='adam', 
                  metrics=[tf.keras.metrics.AUC(name='ap', curve='PR')])
    return model

# ==================== 2.  ====================

def calculate_all_metrics(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        "AUC": auc(fpr, tpr), "AP": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred), "ACC": accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Prec": precision_score(y_true, y_pred, zero_division=0),
        "Rec": recall_score(y_true, y_pred, zero_division=0)
    }

# ==================== 3.  ====================

def run_pure_sequence_comparison(task):
    name = task['name']; maxlen = task['maxlen']
    df = pd.read_csv(task['csv'])
    y = df['label'].values; groups = df['protein_id'].values
    
    aa_map = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    X_idx = np.array([[aa_map.get(a, 0) for a in str(s)[:maxlen]] + [0]*(maxlen-len(str(s)[:maxlen])) for s in df['sequence']])
    
    X_plm_raw = np.load(task['npy'])
    X_plm_gap = np.mean(X_plm_raw, axis=1) # (N, plm_dim)
    plm_dim = X_plm_gap.shape[1]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tv_idx, test_idx = next(gss.split(X_idx, y, groups=groups))
    y_true_test = y[test_idx]
    
    alpha_val = np.sum(y[tv_idx] == 0) / len(y[tv_idx])
    logger.info(f"Alpha: {alpha_val:.4f}")

    experiments = ["CNN-BiLSTM (Seq Only)", "CNN-BiLSTM (Seq + pLM)"]
    pdf_report = PdfPages(f"Report_{name}_Optimized.pdf")
    test_curve_data = {}; perf_records_cv = []; perf_records_test = []
    cv_metric_storage = {exp: {k: [] for k in ["AUC", "AP", "F1", "ACC", "MCC", "Prec", "Rec"]} for exp in experiments}

    for exp_name in experiments:
        logger.info(f"========  {exp_name} ========")
        is_plm = "pLM" in exp_name
        exp_tag = "with_plm" if is_plm else "seq_only"
        
        def get_inputs(indices):
            base = [X_idx[indices]]
            if is_plm: base.append(X_plm_gap[indices])
            return base

        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        best_ap_in_cv = -1; best_model_obj = None
        
        for fold, (t_idx, v_idx) in enumerate(skf.split(X_idx[tv_idx], y[tv_idx], groups=groups[tv_idx])):
            real_t_idx, real_v_idx = tv_idx[t_idx], tv_idx[v_idx]
            model = build_cnnbilstm_plm_pure_seq_model(maxlen, alpha_val, plm_dim if is_plm else None)
            
            fold_model_path = os.path.join(MODEL_SAVE_DIR, f"Best_{exp_tag}_fold{fold}.h5")
            checkpoint = ModelCheckpoint(
                filepath=fold_model_path, monitor='val_ap', mode='max', 
                save_best_only=True, verbose=0
            )
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            model.fit(get_inputs(real_t_idx), y[real_t_idx],
                      validation_data=(get_inputs(real_v_idx), y[real_v_idx]),
                      epochs=30, batch_size=512, verbose=0,
                      callbacks=[early_stop, checkpoint])
            
            model.load_weights(fold_model_path)
            probs = model.predict(get_inputs(real_v_idx), verbose=0).flatten()
            m = calculate_all_metrics(y[real_v_idx], probs)
            
            for k in m: cv_metric_storage[exp_name][k].append(m[k])
            if m["AP"] > best_ap_in_cv:
                best_ap_in_cv = m["AP"]; best_model_obj = model

        final_save_path = os.path.join(MODEL_SAVE_DIR, f"FINAL_BEST_{exp_tag}.h5")
        best_model_obj.save(final_save_path)
        logger.info(f"{exp_name} best model: {final_save_path}")

        test_p = best_model_obj.predict(get_inputs(test_idx), verbose=0).flatten()
        test_curve_data[exp_name] = {"y": y_true_test, "p": test_p}
        
        exam_m = calculate_all_metrics(y_true_test, test_p)
        row_test = {"Task": name, "Model": exp_name}
        for k, v in exam_m.items(): row_test[k] = f"{v:.4f}"
        perf_records_test.append(row_test)
        K.clear_session()

    for k in ["AUC", "AP", "F1", "ACC", "MCC", "Prec", "Rec"]:
        group_a = cv_metric_storage[experiments[0]][k]
        group_b = cv_metric_storage[experiments[1]][k]
        _, p_val = stats.ttest_rel(group_a, group_b)
        perf_records_cv.append({"Task": name, "Model": "P-Value Comparison", k: f"{p_val:.4e}"})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.grid(False); ax2.grid(False)
    for mn, data in test_curve_data.items():
        fpr, tpr, _ = roc_curve(data["y"], data["p"])
        ax1.plot(fpr, tpr, label=f"{mn} (AUC={auc(fpr, tpr):.3f})")
        prec, rec, _ = precision_recall_curve(data["y"], data["p"])
        ax2.plot(rec, prec, label=f"{mn} (AP={average_precision_score(data['y'], data['p']):.3f})")
    ax1.set_title("Test ROC"); ax1.legend(); ax2.set_title("Test PRC"); ax2.legend()
    pdf_report.savefig(fig); plt.close()
    pdf_report.close()

    pd.DataFrame(perf_records_cv).to_csv(f"Ablation_CV_Stability_{name}_Optimized.csv", index=False)
    pd.DataFrame(perf_records_test).to_csv(f"Ablation_Test_Performance_{name}_Optimized.csv", index=False)
    logger.info("Done")

if __name__ == "__main__":
    task = {
        "name": "Protein_pLM", 
        "maxlen": 2000, 
        "csv": "data/protein/protein_sequence.csv", 
        "npy": "data/protein/protein_plm_residue.npy"
    }
    if os.path.exists(task["csv"]):
        run_pure_sequence_comparison(task)

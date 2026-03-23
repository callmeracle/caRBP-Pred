import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as stats
import logging
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import *

# ==================== 1.  ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
SAVE_DIR = "./plm_ablation_final"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==================== 2. Hybrid Loss ====================

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

# ==================== 3. CNN-BiLSTM ====================

def build_model(maxlen, ptm_dim, alpha, use_ptm=False):
    idx_in = Input(shape=(maxlen,), name="Seq_Input")
    x = Embedding(21, 64)(idx_in)
    
    x = Conv1D(64, 3, padding='same', activation='relu')(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    
    if use_ptm:
        ptm_in = Input(shape=(maxlen, ptm_dim), name="pLM_Input")
        p = Dense(64, activation='relu')(ptm_in) 
        x = Concatenate()([x, p])
        x = Conv1D(128, 3, padding='same', activation='relu')(x)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        inputs = [idx_in, ptm_in]
    else:
        inputs = [idx_in]

    x = GlobalMaxPooling1D()(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, out)
    model.compile(optimizer='adam', 
                  loss=hybrid_dice_focal_loss(alpha=alpha), 
                  metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

# ==================== 4.  ====================

def run_rigorous_experiment(csv_path, npy_path):
    task_name = "pLM_Peptide"
    df = pd.read_csv(csv_path)
    X_ptm = np.load(npy_path) 
    
    aa_map = {aa: i+1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    maxlen = 50
    
    X_idx_list = []
    for seq in df['sequence']:
        indices = [aa_map.get(aa, 0) for aa in str(seq)]
        if len(indices) >= maxlen: indices = indices[:maxlen]
        else: indices = indices + [0] * (maxlen - len(indices))
        X_idx_list.append(indices)
    X_idx = np.array(X_idx_list)
    
    y = df['label'].values
    groups = df['protein_id'].values 
    
    pdf_report = PdfPages(f"Report_{task_name}.pdf")
    perf_records_cv = []
    perf_records_test = []

    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    dev_idx, te_idx = next(gss_test.split(X_idx, y, groups=groups))
    
    X_dev_idx, y_dev, groups_dev = X_idx[dev_idx], y[dev_idx], groups[dev_idx]
    y_true_test = y[te_idx]
    
    alpha_val = np.sum(y_dev == 0) / len(y_dev)
    logger.info(f" alpha : {alpha_val:.4f}")

    modes = ['Baseline', 'With_pLM']
    
    for mode in modes:
        logger.info(f"\n>>> mode: {mode}")
        use_ptm = (mode == 'With_pLM')
        cv_raw_stats = {k: [] for k in ["AUC", "AP", "F1", "ACC", "MCC", "Prec", "Rec"]}
        test_curve_store = None

        # --- B. 5-Fold StratifiedGroupKFold ---
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold, (tr_sub_idx, va_sub_idx) in enumerate(sgkf.split(X_dev_idx, y_dev, groups=groups_dev)):
            logger.info(f"  Fold {fold+1}/5...")
            idx_tr, idx_va = dev_idx[tr_sub_idx], dev_idx[va_sub_idx]
            
            def get_in(ids): return [X_idx[ids], X_ptm[ids]] if use_ptm else [X_idx[ids]]
            
            model = build_model(maxlen, X_ptm.shape[2], alpha_val, use_ptm)
            m_path = f"{SAVE_DIR}/{mode}_Fold{fold}.h5"
            cp = ModelCheckpoint(m_path, monitor='val_auc', save_best_only=True, mode='max', verbose=0)
            es = EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True)
            
            model.fit(
                get_in(idx_tr), y[idx_tr],
                validation_data=(get_in(idx_va), y[idx_va]),
                epochs=30, batch_size=32, callbacks=[cp, es], verbose=0
            )
            
            y_prob_va = model.predict(get_in(idx_va), verbose=0).flatten()
            y_pred_va = (y_prob_va > 0.5).astype(int)
            
            cv_raw_stats["AUC"].append(roc_auc_score(y[idx_va], y_prob_va))
            cv_raw_stats["AP"].append(average_precision_score(y[idx_va], y_prob_va))
            cv_raw_stats["F1"].append(f1_score(y[idx_va], y_pred_va))
            cv_raw_stats["ACC"].append(accuracy_score(y[idx_va], y_pred_va))
            cv_raw_stats["MCC"].append(matthews_corrcoef(y[idx_va], y_pred_va))
            cv_raw_stats["Prec"].append(precision_score(y[idx_va], y_pred_va, zero_division=0))
            cv_raw_stats["Rec"].append(recall_score(y[idx_va], y_pred_va, zero_division=0))

            if fold == 4:
                test_curve_store = model.predict(get_in(te_idx), verbose=0).flatten()

        # --- P value ---
        row_cv = {"Task": task_name, "Mode": mode, "Model": "CNN-BiLSTM"}
        for k in ["AUC", "AP", "F1", "ACC", "MCC", "Prec", "Rec"]:
            vals = cv_raw_stats[k]
            row_cv[k] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
            row_cv[f"raw_{k}"] = vals 
        perf_records_cv.append(row_cv)

        y_ts_pred = (test_curve_store > 0.5).astype(int)
        perf_records_test.append({
            "Task": task_name, "Mode": mode, "Model": "CNN-BiLSTM",
            "AUC": f"{roc_auc_score(y_true_test, test_curve_store):.4f}",
            "AP": f"{average_precision_score(y_true_test, test_curve_store):.4f}",
            "F1": f"{f1_score(y_true_test, y_ts_pred):.4f}",
            "ACC": f"{accuracy_score(y_true_test, y_ts_pred):.4f}",
            "MCC": f"{matthews_corrcoef(y_true_test, y_ts_pred):.4f}",
            "Prec": f"{precision_score(y_true_test, y_ts_pred, zero_division=0):.4f}",
            "Rec": f"{recall_score(y_true_test, y_ts_pred, zero_division=0):.4f}",
            "y_prob": test_curve_store
        })

    for k in ["AUC", "AP", "F1", "ACC", "MCC", "Prec", "Rec"]:
        v_base = perf_records_cv[0][f"raw_{k}"]
        v_ptm = perf_records_cv[1][f"raw_{k}"]
        _, p = stats.ttest_rel(v_base, v_ptm)
        perf_records_cv[1][f"P-Value_{k}"] = f"{p:.4e}"
        perf_records_cv[0][f"P-Value_{k}"] = "-"
        del perf_records_cv[0][f"raw_{k}"], perf_records_cv[1][f"raw_{k}"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.grid(False); ax2.grid(False)
    for d in perf_records_test:
        f, t, _ = roc_curve(y_true_test, d['y_prob'])
        pr, re, _ = precision_recall_curve(y_true_test, d['y_prob'])
        ax1.plot(f, t, label=f"{d['Mode']} (AUC={d['AUC']})")
        ax2.plot(re, pr, label=f"{d['Mode']} (AP={d['AP']})")
    ax1.set_title(f"ROC - {task_name} (Test Set)"); ax1.legend()
    ax2.set_title(f"PRC - {task_name} (Test Set)"); ax2.legend()
    pdf_report.savefig(fig); plt.close()
    pdf_report.close()

    pd.DataFrame(perf_records_cv).to_csv(f"Ablation_CV_Stability_{task_name}.csv", index=False)
    for d in perf_records_test: del d['y_prob']
    pd.DataFrame(perf_records_test).to_csv(f"Ablation_Test_Performance_{task_name}.csv", index=False)
    logger.info(f"Done! ")

if __name__ == "__main__":
    csv_file = "data/peptide/peptide_sequence.csv"
    npy_file = "data/peptide/peptide_plm_residue.npy"
    if os.path.exists(csv_file) and os.path.exists(npy_file):
        run_rigorous_experiment(csv_file, npy_file)

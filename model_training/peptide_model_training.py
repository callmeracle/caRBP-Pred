import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import os
import scipy.stats as stats

# ==================== 1. ====================
SAVE_DIR = "./peptide_final_results"
os.makedirs(SAVE_DIR, exist_ok=True)

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

# ==================== 2. ====================
def build_dl_model(name, input_mode, maxlen, alpha):
    in_seq = Input(shape=(maxlen,))
    emb_seq = Embedding(21, 32)(in_seq)

    if input_mode == 'dual':
        in_ss = Input(shape=(maxlen,))
        emb_ss = Embedding(9, 16)(in_ss)
        x = Concatenate()([emb_seq, emb_ss])
        inputs = [in_seq, in_ss]
    else:
        x = emb_seq
        inputs = in_seq

    name = name.lower()
    if name == 'cnn-bilstm':
        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        x = Bidirectional(LSTM(32, return_sequences=False))(x)
    elif name == 'cnn':
        x = Conv1D(64, 3, activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
    elif name == 'bilstm':
        x = Bidirectional(LSTM(32, return_sequences=True))(x)
        x = Conv1D(32, 3, activation='relu')(x)
        x = GlobalAveragePooling1D()(x)
    elif name == 'resnet-cnn':
        shortcut = Conv1D(64, 1, padding='same')(x)
        c1 = Conv1D(64, 3, padding='same', activation='relu')(x)
        c2 = Conv1D(64, 3, padding='same')(c1)
        x = Add()([c2, shortcut])
        x = Activation('relu')(x)
        x = GlobalMaxPooling1D()(x)

    x = Dropout(0.4)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, out)
    model.compile(optimizer='adam', loss=hybrid_dice_focal_loss(alpha=alpha))
    return model

# ==================== 3.  ====================
def run_experiment(task):
    name = task['name']
    maxlen = task['maxlen']
    df_seq = pd.read_csv(task['seq_file'])
    df_ss = pd.read_csv(task['ss_file'])

    aa_map = {aa: i+1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    ss_map = {s: i for i, s in enumerate("HECTSGIB")}
    X_seq_all = np.array([[aa_map.get(a, 0) for a in str(s)[:maxlen]] + [0]*(maxlen-len(str(s)[:maxlen])) for s in df_seq['sequence']])
    X_ss_all = np.array([[ss_map.get(a, 0) for a in str(s)[:maxlen]] + [0]*(maxlen-len(str(s)[:maxlen])) for s in df_ss['sequence']])
    y_all = df_seq['label'].values
    groups_all = df_seq['protein_id'].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    dev_idx, test_idx = next(gss.split(X_seq_all, y_all, groups=groups_all))

    X_dev = {"seq": X_seq_all[dev_idx], "ss": X_ss_all[dev_idx], "y": y_all[dev_idx], "g": groups_all[dev_idx]}
    X_test = {"seq": X_seq_all[test_idx], "ss": X_ss_all[test_idx], "y": y_all[test_idx]}
    y_true_test = X_test['y']
    
    alpha_val = np.sum(X_dev['y'] == 0) / len(X_dev['y'])

    model_names = ['CNN-BiLSTM', 'CNN', 'BiLSTM', 'ResNet-CNN', 'RF', 'XGB', 'SVM']
    input_modes = ['seq', 'dual']

    pdf = PdfPages(f"{SAVE_DIR}/Test_Set_Curves.pdf")
    test_perf_records, cv_perf_records = [], []

    for in_mode in input_modes:
        print(f"\n>>> Running Mode: {in_mode}")
        test_probs_ensemble = {m: [] for m in model_names}
        cv_raw_stats = {m: {k: [] for k in ["AUC", "AP", "F1", "ACC", "MCC", "Prec", "Rec"]} for m in model_names}

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_dev['seq'], X_dev['y'], groups=X_dev['g'])):
            print(f"  - Fold {fold+1}/5")
            y_tr, y_val = X_dev['y'][tr_idx], X_dev['y'][val_idx]

            for d_name in model_names[:4]:
                m = build_dl_model(d_name, in_mode, maxlen, alpha_val)
                tr_in = [X_dev['seq'][tr_idx], X_dev['ss'][tr_idx]] if in_mode == 'dual' else X_dev['seq'][tr_idx]
                val_in = [X_dev['seq'][val_idx], X_dev['ss'][val_idx]] if in_mode == 'dual' else X_dev['seq'][val_idx]
                ts_in = [X_test['seq'], X_test['ss']] if in_mode == 'dual' else X_test['seq']

                m.fit(tr_in, y_tr, validation_data=(val_in, y_val), epochs=20, batch_size=64, verbose=0, 
                      callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

                v_p = m.predict(val_in, verbose=0).ravel()
                y_v_pred = (v_p > 0.5).astype(int)
                
                for k, metric_fn in zip(["AUC", "AP", "F1", "ACC", "MCC", "Prec", "Rec"], 
                                      [roc_auc_score, average_precision_score, f1_score, accuracy_score, matthews_corrcoef, precision_score, recall_score]):
                    if k in ["AUC", "AP"]: cv_raw_stats[d_name][k].append(metric_fn(y_val, v_p))
                    else: cv_raw_stats[d_name][k].append(metric_fn(y_val, y_v_pred, zero_division=0) if "Prec" in k or "Rec" in k else metric_fn(y_val, y_v_pred))

                test_probs_ensemble[d_name].append(m.predict(ts_in, verbose=0).ravel())

            X_tr_ml = X_dev['seq'][tr_idx].reshape(len(tr_idx), -1)
            X_val_ml = X_dev['seq'][val_idx].reshape(len(val_idx), -1)
            X_ts_ml = X_test['seq'].reshape(len(X_test['seq']), -1)
            if in_mode == 'dual':
                X_tr_ml = np.hstack([X_tr_ml, X_dev['ss'][tr_idx].reshape(len(tr_idx), -1)])
                X_val_ml = np.hstack([X_val_ml, X_dev['ss'][val_idx].reshape(len(val_idx), -1)])
                X_ts_ml = np.hstack([X_ts_ml, X_test['ss'].reshape(len(X_test['seq']), -1)])

            ml_models = {
                'RF': RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1),
                'XGB': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=(len(y_tr)-sum(y_tr))/sum(y_tr)),
                'SVM': CalibratedClassifierCV(SVC(class_weight='balanced'))
            }

            for m_name in ['RF', 'XGB', 'SVM']:
                clf = ml_models[m_name]; clf.fit(X_tr_ml, y_tr)
                v_p = clf.predict_proba(X_val_ml)[:, 1]; y_v_pred = (v_p > 0.5).astype(int)
                for k, metric_fn in zip(["AUC", "AP", "F1", "ACC", "MCC", "Prec", "Rec"], 
                                      [roc_auc_score, average_precision_score, f1_score, accuracy_score, matthews_corrcoef, precision_score, recall_score]):
                    if k in ["AUC", "AP"]: cv_raw_stats[m_name][k].append(metric_fn(y_val, v_p))
                    else: cv_raw_stats[m_name][k].append(metric_fn(y_val, y_v_pred, zero_division=0) if "Prec" in k or "Rec" in k else metric_fn(y_val, y_v_pred))
                test_probs_ensemble[m_name].append(clf.predict_proba(X_ts_ml)[:, 1])

        mode_label = 'With_SS' if in_mode == 'dual' else 'Sequence_Only'
        for m_name in model_names:
            row_cv = {"Task": name, "Mode": mode_label, "Model": m_name}
            for k in ["AUC", "AP", "F1", "ACC", "MCC", "Prec", "Rec"]:
                vals = cv_raw_stats[m_name][k]
                row_cv[k] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
                if m_name != 'CNN-BiLSTM':
                    _, p = stats.ttest_rel(cv_raw_stats['CNN-BiLSTM'][k], vals)
                    row_cv[f"P-Value_{k}"] = f"{p:.4e}"
                else: row_cv[f"P-Value_{k}"] = "-"
            cv_perf_records.append(row_cv)
            avg_p = np.mean(test_probs_ensemble[m_name], axis=0); y_pred = (avg_p > 0.5).astype(int)
            test_perf_records.append({"Task": name, "Mode": mode_label, "Model": m_name, "AUC": f"{roc_auc_score(y_true_test, avg_p):.4f}", "AP": f"{average_precision_score(y_true_test, avg_p):.4f}", "F1": f"{f1_score(y_true_test, y_pred):.4f}", "ACC": f"{accuracy_score(y_true_test, y_pred):.4f}", "MCC": f"{matthews_corrcoef(y_true_test, y_pred):.4f}", "Prec": f"{precision_score(y_true_test, y_pred, zero_division=0):.4f}", "Rec": f"{recall_score(y_true_test, y_pred, zero_division=0):.4f}"})

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.grid(False); ax2.grid(False)
        for m_name in model_names:
            avg_p = np.mean(test_probs_ensemble[m_name], axis=0)
            fpr, tpr, _ = roc_curve(y_true_test, avg_p); ax1.plot(fpr, tpr, label=f"{m_name} ({auc(fpr, tpr):.3f})")
            pre, rec, _ = precision_recall_curve(y_true_test, avg_p); ax2.plot(rec, pre, label=f"{m_name} ({average_precision_score(y_true_test, avg_p):.3f})")
        ax1.set_title(f"{mode_label} ROC"); ax1.legend(); ax2.set_title(f"{mode_label} PRC"); ax2.legend(); pdf.savefig(fig); plt.close()

    pdf.close()
    pd.DataFrame(cv_perf_records).to_csv(f"{SAVE_DIR}/Ablation_CV_Stability_{name}.csv", index=False)
    pd.DataFrame(test_perf_records).to_csv(f"{SAVE_DIR}/Ablation_Test_Performance_{name}.csv", index=False)

if __name__ == "__main__":
    task = {"name": "Peptide", 
    "maxlen": 50, 
    "seq_file": "data/peptide/peptide_sequence.csv", 
    "ss_file": "data/peptide/peptide_secondary_structure.csv"}
    if os.path.exists(task["seq_file"]): run_experiment(task)

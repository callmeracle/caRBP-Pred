import os
import time
import logging
import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import *
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

# ==================== 0.  ====================
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 1.  ====================

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name="att_w", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_b", shape=(input_shape[1], 1), initializer="zeros")
    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.expand_dims(K.softmax(et), axis=-1)
        return K.sum(x * at, axis=1)

def residual_block(x, filters, kernel_size=3):
    shortcut = x
    x = Conv1D(filters, kernel_size, activation='relu', padding='same')(x)
    x = Conv1D(filters, kernel_size, padding='same')(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)
    return Activation('relu')(Add()([x, shortcut]))

def hybrid_dice_focal_loss(gamma=2.0, alpha=0.9):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        
        # Focal Loss
        f_loss = -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - \
                 K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
        
        # Dice Loss
        intersection = K.sum(y_true * y_pred)
        dice_loss = 1.0 - (2.0 * intersection + K.epsilon()) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + K.epsilon())
        return f_loss + dice_loss
    return loss

def build_deep_model(model_type, maxlen, alpha, mode):
    in_seq = Input(shape=(maxlen,), name="Sequence_Input")
    emb = Embedding(input_dim=21, output_dim=64, input_length=maxlen)(in_seq)
    
    if model_type == "CNN-BiLSTM":
        bilstm = Bidirectional(LSTM(64, return_sequences=True))(emb)
        att = AttentionLayer()(bilstm)
        cnn_feat = GlobalAveragePooling1D()(Conv1D(64, 3, activation='relu', padding='same')(emb))
        main_feat = concatenate([att, cnn_feat])
    elif model_type == "BiLSTM":
        bilstm = Bidirectional(LSTM(64, return_sequences=True))(emb)
        main_feat = AttentionLayer()(bilstm)
    elif model_type == "ResNet-CNN":
        x = Conv1D(64, 7, padding='same')(emb)
        for _ in range(3): x = residual_block(x, 64)
        main_feat = GlobalAveragePooling1D()(x)
    else: # CNN
        main_feat = GlobalAveragePooling1D()(Conv1D(128, 3, activation='relu')(emb))
        
    extra_feats, inputs = [main_feat], [in_seq]
    
    if mode == "With_SS":
        in_s = Input(shape=(maxlen,), name="SS_Input")
        s_feat = GlobalAveragePooling1D()(Conv1D(16, 3, activation='relu', padding='same')(Embedding(10, 16)(in_s)))
        extra_feats.append(s_feat); inputs.append(in_s)
        
    merged = concatenate(extra_feats) if len(extra_feats) > 1 else main_feat
    x = Dense(64, activation='relu')(merged)
    out = Dense(1, activation='sigmoid')(Dropout(0.4)(x))
    
    model = Model(inputs=inputs, outputs=out)
    model.compile(loss=hybrid_dice_focal_loss(alpha=alpha), 
                  optimizer='adam', 
                  metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

# ==================== 2.  ====================

def encode_sequence(seq, maxlen):
    mapping = {a: i+1 for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    return [mapping.get(a, 0) for a in str(seq)[:maxlen]] + [0] * max(0, maxlen - len(str(seq)))

def encode_ss(ss, maxlen):
    mapping = {a: i+1 for i, a in enumerate("HEC")} 
    return [mapping.get(a, 0) for a in str(ss)[:maxlen]] + [0] * max(0, maxlen - len(str(ss)))

def load_and_prepare_data(task):
    df_seq = pd.read_csv(task["seq_file"])
    df_ss = pd.read_csv(task["ss_file"])
    df = pd.merge(df_seq, df_ss[['protein_id', 'ss']], on='protein_id', how='inner')
    maxlen = task["maxlen"]
    df['seq_encoded'] = df['sequence'].apply(lambda x: encode_sequence(x, maxlen))
    df['ss_encoded'] = df['ss'].apply(lambda x: encode_ss(x, maxlen))
    return df

# ==================== 3.  ====================

def run_experiment(task):
    full_df = load_and_prepare_data(task)
    name = task["name"]
    maxlen = task["maxlen"]
    
    modes = ["Sequence_Only", "With_SS"]
    models = ["CNN-BiLSTM", "BiLSTM", "ResNet-CNN", "CNN", "RF", "SVM", "XGBoost"]
    
    perf_records_cv = []    
    perf_records_test = []  
    pdf_report = PdfPages(f"Report_{name}.pdf")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    dev_idx, test_idx = next(gss.split(full_df, groups=full_df['protein_id']))
    dev_df = full_df.iloc[dev_idx]
    test_df = full_df.iloc[test_idx]
    y_true_test = test_df['label'].values

    alpha_val = np.sum(dev_df['label'] == 0) / len(dev_df)
    logger.info(f"alpha: {alpha_val:.4f}")

    for mode in modes:
        logger.info(f"\n>>> model: {mode}")
        test_curve_store = {}
        cv_raw_results = {m: {k: [] for k in ["AUC", "AP", "F1", "ACC", "MCC", "Prec", "Rec"]} for m in models}

        for m_name in models:
            logger.info(f"---: {m_name}")
            skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            
            def get_x(df_in):
                x_seq = np.array(df_in['seq_encoded'].tolist())
                if mode == "With_SS":
                    return [x_seq, np.array(df_in['ss_encoded'].tolist())]
                return [x_seq]

            for fold, (tr_idx, val_idx) in enumerate(skf.split(dev_df, dev_df['label'], groups=dev_df['protein_id'])):
                d_tr = dev_df.iloc[tr_idx]
                d_val = dev_df.iloc[val_idx]
                
                if m_name in ["CNN-BiLSTM", "BiLSTM", "ResNet-CNN", "CNN"]:
                    model = build_deep_model(m_name, maxlen, alpha_val, mode)
                    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                    model.fit(get_x(d_tr), d_tr['label'], validation_data=(get_x(d_val), d_val['label']),
                              epochs=30, batch_size=32, callbacks=[es], verbose=0)
                    probs = model.predict(get_x(d_val), verbose=0).flatten()
                    if fold == 4: test_curve_store[m_name] = model.predict(get_x(test_df), verbose=0).flatten()
                else:
                    X_ml_tr = np.array(d_tr['seq_encoded'].tolist())
                    X_ml_val = np.array(d_val['seq_encoded'].tolist())
                    y_tr = d_tr['label'].values
                    
                    if m_name == "RF": 
                        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
                    elif m_name == "SVM": 
                        clf = CalibratedClassifierCV(SVC(class_weight='balanced', random_state=42))
                    elif m_name == "XGBoost": 
                        scale_pos = (len(y_tr) - sum(y_tr)) / sum(y_tr)
                        clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos, random_state=42)
                    
                    clf.fit(X_ml_tr, y_tr)
                    probs = clf.predict_proba(X_ml_val)[:, 1]
                    if fold == 4: 
                        X_ts = np.array(test_df['seq_encoded'].tolist())
                        test_curve_store[m_name] = clf.predict_proba(X_ts)[:, 1]

                y_v_true = d_val['label'].values
                y_v_pred = (probs > 0.5).astype(int)
                
                cv_raw_results[m_name]["AUC"].append(roc_auc_score(y_v_true, probs))
                cv_raw_results[m_name]["AP"].append(average_precision_score(y_v_true, probs))
                cv_raw_results[m_name]["F1"].append(f1_score(y_v_true, y_v_pred))
                cv_raw_results[m_name]["ACC"].append(accuracy_score(y_v_true, y_v_pred))
                cv_raw_results[m_name]["MCC"].append(matthews_corrcoef(y_v_true, y_v_pred))
                cv_raw_results[m_name]["Prec"].append(precision_score(y_v_true, y_v_pred, zero_division=0))
                cv_raw_results[m_name]["Rec"].append(recall_score(y_v_true, y_v_pred, zero_division=0))

            row_cv = {"Task": name, "Mode": mode, "Model": m_name}
            for k in ["AUC", "AP", "F1", "ACC", "MCC", "Prec", "Rec"]:
                vals = cv_raw_results[m_name][k]
                row_cv[k] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
                if m_name != "CNN-BiLSTM":
                    _, p = stats.ttest_rel(cv_raw_results["CNN-BiLSTM"][k], vals)
                    row_cv[f"P-Value_{k}"] = f"{p:.4e}"
                else: row_cv[f"P-Value_{k}"] = "-"
            perf_records_cv.append(row_cv)

            p_ts = test_curve_store[m_name]
            y_ts_pred = (p_ts > 0.5).astype(int)
            perf_records_test.append({
                "Task": name, "Mode": mode, "Model": m_name,
                "AUC": f"{roc_auc_score(y_true_test, p_ts):.4f}",
                "AP": f"{average_precision_score(y_true_test, p_ts):.4f}",
                "F1": f"{f1_score(y_true_test, y_ts_pred):.4f}",
                "ACC": f"{accuracy_score(y_true_test, y_ts_pred):.4f}",
                "MCC": f"{matthews_corrcoef(y_true_test, y_ts_pred):.4f}",
                "Prec": f"{precision_score(y_true_test, y_ts_pred, zero_division=0):.4f}",
                "Rec": f"{recall_score(y_true_test, y_ts_pred, zero_division=0):.4f}"
            })

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.grid(False); ax2.grid(False)
        for mn, p_test in test_curve_store.items():
            fpr, tpr, _ = roc_curve(y_true_test, p_test)
            ax1.plot(fpr, tpr, label=f"{mn} (AUC={auc(fpr, tpr):.3f})")
            prec, rec, _ = precision_recall_curve(y_true_test, p_test)
            ax2.plot(rec, prec, label=f"{mn} (AP={average_precision_score(y_true_test, p_test):.3f})")
        ax1.set_title(f"ROC - {mode}"); ax1.legend()
        ax2.set_title(f"PRC - {mode}"); ax2.legend()
        pdf_report.savefig(fig); plt.close()

    pdf_report.close()
    pd.DataFrame(perf_records_cv).to_csv(f"Ablation_CV_Stability_{name}.csv", index=False)
    pd.DataFrame(perf_records_test).to_csv(f"Ablation_Test_Performance_{name}.csv", index=False)
    logger.info(f"Done!")

if __name__ == "__main__":
    task = {
        "name": "Protein_Final", 
        "maxlen": 2000, 
        "seq_file": "data/proteinprotein_sequence.csv", 
        "ss_file": "data/protein/protein_secondary_structure.csv"
   }
    if os.path.exists(task["seq_file"]):
        run_experiment(task)

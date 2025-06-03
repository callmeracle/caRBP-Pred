import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, accuracy_score, precision_score, f1_score, matthews_corrcoef
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Bidirectional, LSTM, Dense, Input, concatenate, TimeDistributed, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
matplotlib.use('TkAgg')

# 数据编码函数
def encode(sequence, mapping):
    one_hot_encoded = []
    for char in sequence:
        one_hot_vector = [0] * len(mapping)
        one_hot_vector[mapping[char] - 1] = 1
        one_hot_encoded.append(one_hot_vector)
    return np.array(one_hot_encoded)

# 绘制ROC曲线和PRC曲线的函数
def plot_roc_and_prc(y_true, y_pred_prob, label, ax1, ax2):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc_score = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    ap_score = average_precision_score(y_true, y_pred_prob)
    optimal_threshold = 0.5  # 示例阈值
    prediction = (y_pred_prob > optimal_threshold).astype("int32")
    accuracy = accuracy_score(y_true, prediction)
    precision_val = precision_score(y_true, prediction)
    f1 = f1_score(y_true, prediction)
    mcc = matthews_corrcoef(y_true, prediction)
    ax1.plot(fpr, tpr, label=f"{label} (AUC={auc_score:.2f})")
    ax2.plot(recall, precision, label=f"{label} (AP={ap_score:.2f})")
    return auc_score, ap_score, accuracy, precision_val, f1, mcc

# 数据预处理函数
def preprocess_data(file_path, sequence_column, label_column, maxlen, mapping):
    data = pd.read_csv(file_path, header=None)
    data.columns = ["gene", sequence_column, label_column]
    encoded_sequences = [encode(row[sequence_column], mapping) for _, row in data.iterrows()]
    padded_sequences = pad_sequences(encoded_sequences, maxlen=maxlen, padding='post')
    return padded_sequences, data[label_column].values

# 定义模型训练和评估函数
def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type, class_weights, input_shape=None):
    if model_type == "CNN-LSTM":
        model = Sequential([
            Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Dropout(0.5),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.5),
            TimeDistributed(Flatten()),
            Bidirectional(LSTM(70, return_sequences=True)),
            Bidirectional(LSTM(70)),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif model_type == "SVM":
        model = SVC(kernel='linear', probability=True)
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    elif model_type == "RF":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    elif model_type == "XGBoost":
        model = xgb.XGBClassifier()
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    else:
        raise ValueError("Unsupported model type")
    
    if model_type == "CNN-LSTM":
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test), class_weight=class_weights, callbacks=[early_stopping])
        y_pred_prob = model.predict(X_test)
    else:
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    return y_pred_prob

# 主程序
if __name__ == "__main__":
    # 定义映射表
    mapping = {'A': 1, 'B': 22, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9,
               'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17,
               'V': 18, 'W': 19, 'Y': 20, 'X': 21, 'U': 23, 'J': 24, 'Z': 25, 'O': 26}

# 数据预处理
X_prot_seq, y_prot_seq = preprocess_data("/lustre/sunqiang/01_project/24_chrRBP/00_raw_data/CARBP/check_overlap/merged_caRBP_fasta.csv", "sequence", "label", maxlen=1000, mapping=mapping)
X_prot_ss, y_prot_ss = preprocess_data("/mnt/hwt2_data1/sunqiang/software/ps4-dataset/merged_caRBP_ss.csv", "ss", "label", maxlen=1000, mapping=mapping)
X_pep_seq, y_pep_seq = preprocess_data("/lustre/sunqiang/01_project/24_chrRBP/00_raw_data/CARBP/merged_peptide.csv", "sequence", "label", maxlen=50, mapping=mapping)
X_pep_ss, y_pep_ss = preprocess_data("/mnt/hwt2_data1/sunqiang/software/ps4-dataset/merged_peptide_ss.csv", "ss", "label", maxlen=50, mapping=mapping)

# 拆分数据集
X_train_prot_seq, X_test_prot_seq, y_train_prot_seq, y_test_prot_seq = train_test_split(X_prot_seq, y_prot_seq, test_size=0.2, random_state=42)
X_train_prot_ss, X_test_prot_ss, y_train_prot_ss, y_test_prot_ss = train_test_split(X_prot_ss, y_prot_ss, test_size=0.2, random_state=42)
X_train_pep_seq, X_test_pep_seq, y_train_pep_seq, y_test_pep_seq = train_test_split(X_pep_seq, y_pep_seq, test_size=0.2, random_state=42)
X_train_pep_ss, X_test_pep_ss, y_train_pep_ss, y_test_pep_ss = train_test_split(X_pep_ss, y_pep_ss, test_size=0.2, random_state=42)

# 定义模型类型和数据集
models = ["CNN-LSTM", "SVM", "RF", "XGBoost"]
datasets = [
        (X_train_prot_seq, y_train_prot_seq, X_test_prot_seq, y_test_prot_seq, (1000, 26), "prot_seq"),
        (X_train_prot_ss, y_train_prot_ss, X_test_prot_ss, y_test_prot_ss, (1000, 26), "prot_ss"),
        (X_train_pep_seq, y_train_pep_seq, X_test_pep_seq, y_test_pep_seq, (50, 26), "pep_seq"),
        (X_train_pep_ss, y_train_pep_ss, X_test_pep_ss, y_test_pep_ss, (50, 26), "pep_ss"),
        (np.concatenate((X_train_prot_seq, X_train_prot_ss)), y_train_prot_seq, np.concatenate((X_test_prot_seq, X_test_prot_ss)), y_test_prot_seq, (1000, 52), "prot_seq_ss"),
        (np.concatenate((X_train_pep_seq, X_train_pep_ss)), y_train_pep_seq, np.concatenate((X_test_pep_seq, X_test_pep_ss)), y_test_pep_seq, (50, 52), "pep_seq_ss")
    ]

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
ax1.plot([0, 1], [0, 1], 'k--')
ax2.plot([0, 1], [1, 0], 'k--')

# 训练和评估模型
results = []
for X_train, y_train, X_test, y_test, input_shape, dataset_name in datasets:
        class_weights = {0: 1, 1: 4 if dataset_name in ["prot_seq", "prot_ss"] else 10}
        for model_type in models:
            y_pred_prob = train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type, class_weights, input_shape)
            auc_score, ap_score, accuracy, precision, f1, mcc = plot_roc_and_prc(y_test, y_pred_prob, f"{dataset_name}_{model_type}", ax1, ax2)
            results.append((dataset_name, model_type, auc_score, ap_score, accuracy, precision, f1, mcc))

# 显示图表
ax1.set_xlabel("FPR")
ax1.set_ylabel("TPR")
ax1.set_title('Receiver Operating Characteristic')
ax1.legend(loc="lower right")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title('Precision-Recall Curve')
ax2.legend(loc="best")
plt.show()

# 打印结果
for result in results:
    print(result)

#######################  plot multiple  ROC curve
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import matplotlib.pyplot as plt
from keras.layers import Dense ,LSTM,concatenate,Input,Flatten, Dropout, Conv1D, MaxPooling1D, TimeDistributed, Bidirectional  
import tensorflow as tf
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay

def encode(s1): 
    one_hot_encoded = []
    nucletide_to_index= {'A':1,'B':22,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,
           'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,
           'V':18,'W':19,'Y':20,'X':21,'U':23,'J':24,'Z':25,'O':26}
    for nucleotide in s1:
        one_hot_vector = [0] * 26
        one_hot_vector[nucletide_to_index[nucleotide]-1] = 1
        one_hot_encoded.append(one_hot_vector)
    return np.array(one_hot_encoded)	
    
#########  protein sequence
prot_seq=pd.read_csv("merged_caRBP_fasta.csv",header=None)
prot_seq.columns=["gene","sequence","label"]   
df_one_hot=[]
for i, row in prot_seq.iterrows():
    df_one_hot.append(encode(row["sequence"]))
	
input_data_1 = pad_sequences(df_one_hot, maxlen=1000, padding='post')
df_one_hot_array = np.array(input_data_1,dtype=np.ndarray)
trainSeq,testSeq,trainLabels,testLabels = train_test_split(df_one_hot_array,prot_seq['label'],test_size=0.2,random_state=101)
seqTrain1 = np.array(trainSeq)
seqTest1 = np.array(testSeq)
X_train = np.reshape(seqTrain1, (seqTrain1.shape[0], seqTrain1.shape[1], 26))  
X_test =  np.reshape(seqTest1, (seqTest1.shape[0], seqTest1.shape[1], 26))
X_train_format = np.asarray(X_train).astype(np.float32)
X_test_format = np.asarray(X_test).astype(np.float32)
class_weights = {0: 1 ,1: 10 }
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_format.shape[1], 26)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(LSTM(70,return_sequences=True)))
model.add(Bidirectional(LSTM(70)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train_format, trainLabels, epochs=60, validation_data=(X_test_format, testLabels), class_weight=class_weights, callbacks=[early_stopping])
y_pred_prob = model.predict(X_test_format)
fpr1, tpr1, thresholds1 = roc_curve(testLabels, y_pred_prob)
auc_roc1=metrics.roc_auc_score(testLabels,y_pred_prob)
ap1 = average_precision_score(testLabels, y_pred_prob)
precision1, recall1, thresholds1 = precision_recall_curve(testLabels, y_pred_prob)

######### protein secondary structure
prot_ss=pd.read_csv("merged_caRBP_ss.csv",header=None)
prot_ss.columns=["gene","ss","label"]
df_one_hot_ss=[]
for i, row in prot_ss.iterrows():
    df_one_hot_ss.append(encode(row["ss"]))	
    
input_data_2 = pad_sequences(df_one_hot_ss, maxlen=1000, padding='post')
df_one_hot_ss_array = np.array(input_data_2,dtype=np.ndarray)
df_one_hot_ss_array=np.asarray(df_one_hot_ss_array).astype(np.float32)
trainSeq,testSeq,trainLabels,testLabels = train_test_split(df_one_hot_ss_array,prot_ss['label'],test_size=0.2,random_state=101)
seqTrain1 = np.array(trainSeq)
seqTest1 = np.array(testSeq)
X_train = np.reshape(seqTrain1, (seqTrain1.shape[0], seqTrain1.shape[1], 26))  
X_test =  np.reshape(seqTest1, (seqTest1.shape[0], seqTest1.shape[1], 26))
X_train_format = np.asarray(X_train).astype(np.float32)
X_test_format = np.asarray(X_test).astype(np.float32)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
class_weights = {0: 1 ,1: 10 }
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_format.shape[1], 26)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(LSTM(70,return_sequences=True)))
model.add(Bidirectional(LSTM(70)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train_format, trainLabels, epochs=60, validation_data=(X_test_format, testLabels), class_weight=class_weights, callbacks=[early_stopping])
y_pred_prob = model.predict(X_test_format)
fpr2, tpr2, thresholds2 = roc_curve(testLabels, y_pred_prob)
auc_roc2=metrics.roc_auc_score(testLabels,y_pred_prob)
ap2 = average_precision_score(testLabels, y_pred_prob)
precision2, recall2, thresholds2 = precision_recall_curve(testLabels, y_pred_prob)

###### protein sequence + ss
df_one_hot_array=np.asarray(df_one_hot_array).astype(np.float32)
df_one_hot_ss_array=np.asarray(df_one_hot_ss_array).astype(np.float32)
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(df_one_hot_array, df_one_hot_ss_array, prot_seq['label'], test_size=0.2, random_state=42)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Define the first input
input1 = Input(shape=(1000, 26))
x1 = Conv1D(filters=32, kernel_size=3, activation='relu')(input1)
x1 = MaxPooling1D(pool_size=2)(x1)
x1 = Dropout(0.5)(x1)
x1 = Conv1D(filters=32, kernel_size=3, activation='relu')(input1)
x1 = MaxPooling1D(pool_size=2)(x1)
x1 = Dropout(0.5)(x1)
x1 = Bidirectional(LSTM(70,return_sequences=True))(x1)
x1 = Bidirectional(LSTM(70))(x1)
# Define the second input
input2 = Input(shape=(1000, 26))
x2 = Conv1D(filters=32, kernel_size=3, activation='relu')(input2)
x2 = MaxPooling1D(pool_size=2)(x2)
x2 = Dropout(0.5)(x2)
x2 = Conv1D(filters=32, kernel_size=3, activation='relu')(input2)
x2 = MaxPooling1D(pool_size=2)(x2)
x2 = Dropout(0.5)(x2)
x2 = Bidirectional(LSTM(70,return_sequences=True))(x2)
x2 = Bidirectional(LSTM(70))(x2)
# Concatenate the outputs of the two LSTMs
merged = concatenate([x1, x2])
# Define the output layer
output = Dense(1, activation='sigmoid')(merged)
# Create the model
model = Model(inputs=[input1, input2], outputs=output)
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Print the model summary
model.summary()
history = model.fit([X1_train, X2_train],y_train, validation_data=([X1_test, X2_test], y_test), batch_size=32, epochs=60, callbacks=[early_stopping])
y_pred_prob = model.predict([X1_test, X2_test])
fpr3, tpr3, thresholds3 = roc_curve(y_test, y_pred_prob)
auc_roc3=metrics.roc_auc_score(y_test,y_pred_prob)
ap3 = average_precision_score(y_test, y_pred_prob)
precision3, recall3, thresholds3 = precision_recall_curve(y_test, y_pred_prob)

########  peptide sequence
pep_seq=pd.read_csv("merged_peptide.csv",header=None)
pep_seq.columns=["name","sequence","label"]
df_one_hot=[]
for i, row in pep_seq.iterrows():
    df_one_hot.append(encode(row["sequence"]))
	
input_data_1 = pad_sequences(df_one_hot, maxlen=50, padding='post')
df_one_hot_array = np.array(input_data_1,dtype=np.ndarray)
trainSeq,testSeq,trainLabels,testLabels = train_test_split(df_one_hot_array,pep_seq['label'],test_size=0.2,random_state=101)
seqTrain1 = np.array(trainSeq)
seqTest1 = np.array(testSeq)
X_train = np.reshape(seqTrain1, (seqTrain1.shape[0], seqTrain1.shape[1], 26))  
X_test =  np.reshape(seqTest1, (seqTest1.shape[0], seqTest1.shape[1], 26))
X_train_format = np.asarray(X_train).astype(np.float32)
X_test_format = np.asarray(X_test).astype(np.float32)
class_weights = {0: 1 ,1: 10 }
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_format.shape[1], 26)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(LSTM(70,return_sequences=True)))
model.add(Bidirectional(LSTM(70)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train_format, trainLabels, epochs=60, validation_data=(X_test_format, testLabels), class_weight=class_weights, callbacks=[early_stopping])
y_pred_prob = model.predict(X_test_format)
fpr4, tpr4, thresholds4 = roc_curve(testLabels, y_pred_prob)
auc_roc4=metrics.roc_auc_score(testLabels,y_pred_prob)
ap4 = average_precision_score(testLabels, y_pred_prob)
precision4, recall4, thresholds4 = precision_recall_curve(testLabels, y_pred_prob)

######## predict secondary structure
pep_ss=pd.read_csv("merged_peptide_ss.csv",header=None)
pep_ss.columns=["gene","ss","label"]
df_one_hot_ss=[]
for i, row in pep_ss.iterrows():
    df_one_hot_ss.append(encode(row["ss"]))	
    
input_data_2 = pad_sequences(df_one_hot_ss, maxlen=50, padding='post')
df_one_hot_ss_array = np.array(input_data_2,dtype=np.ndarray)
df_one_hot_ss_array=np.asarray(df_one_hot_ss_array).astype(np.float32)
trainSeq,testSeq,trainLabels,testLabels = train_test_split(df_one_hot_ss_array,pep_ss['label'],test_size=0.2,random_state=101)
seqTrain1 = np.array(trainSeq)
seqTest1 = np.array(testSeq)
X_train = np.reshape(seqTrain1, (seqTrain1.shape[0], seqTrain1.shape[1], 26))  
X_test =  np.reshape(seqTest1, (seqTest1.shape[0], seqTest1.shape[1], 26))
X_train_format = np.asarray(X_train).astype(np.float32)
X_test_format = np.asarray(X_test).astype(np.float32)
class_weights = {0: 1 ,1: 10 }
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_format.shape[1], 26)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(LSTM(70,return_sequences=True)))
model.add(Bidirectional(LSTM(70)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train_format, trainLabels, epochs=60, validation_data=(X_test_format, testLabels), class_weight=class_weights, callbacks=[early_stopping])
y_pred_prob = model.predict(X_test_format)
fpr5, tpr5, thresholds5 = roc_curve(testLabels, y_pred_prob)
auc_roc5=metrics.roc_auc_score(testLabels,y_pred_prob)
ap5 = average_precision_score(testLabels, y_pred_prob)
precision5, recall5, thresholds5 = precision_recall_curve(testLabels, y_pred_prob)

######### integrate peptide sequence and secondary structure to train and predict 
df_one_hot_array=np.asarray(df_one_hot_array).astype(np.float32)
df_one_hot_ss_array=np.asarray(df_one_hot_ss_array).astype(np.float32)
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(df_one_hot_array, df_one_hot_ss_array, pep_seq['label'], test_size=0.2, random_state=42)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Define the first input
input1 = Input(shape=(50, 26))
x1 = Conv1D(filters=32, kernel_size=3, activation='relu')(input1)
x1 = MaxPooling1D(pool_size=2)(x1)
x1 = Dropout(0.5)(x1)
x1 = Conv1D(filters=32, kernel_size=3, activation='relu')(input1)
x1 = MaxPooling1D(pool_size=2)(x1)
x1 = Dropout(0.5)(x1)
#x1 = TimeDistributed(Flatten())
x1 = Bidirectional(LSTM(70,return_sequences=True))(x1)
x1 = Bidirectional(LSTM(70))(x1)
# Define the second input
input2 = Input(shape=(50, 26))
x2 = Conv1D(filters=32, kernel_size=3, activation='relu')(input2)
x2 = MaxPooling1D(pool_size=2)(x2)
x2 = Dropout(0.5)(x2)
x2 = Conv1D(filters=32, kernel_size=3, activation='relu')(input2)
x2 = MaxPooling1D(pool_size=2)(x2)
x2 = Dropout(0.5)(x2)
#x2 = TimeDistributed(Flatten())
x2 = Bidirectional(LSTM(70,return_sequences=True))(x2)
x2 = Bidirectional(LSTM(70))(x2)
# Concatenate the outputs of the two LSTMs
merged = concatenate([x1, x2])
# Define the output layer
output = Dense(1, activation='sigmoid')(merged)
# Create the model
model = Model(inputs=[input1, input2], outputs=output)
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Print the model summary
model.summary()
history = model.fit([X1_train, X2_train],y_train, validation_data=([X1_test, X2_test], y_test), batch_size=32, epochs=60, callbacks=[early_stopping])
y_pred_prob = model.predict([X1_test, X2_test])
fpr6, tpr6, thresholds6 = roc_curve(y_test, y_pred_prob)
auc_roc6=metrics.roc_auc_score(y_test,y_pred_prob)
ap6 = average_precision_score(y_test, y_pred_prob)
precision6, recall6, thresholds6 = precision_recall_curve(y_test, y_pred_prob)

ap1,ap2,ap3,ap4,ap5,ap6
auc_roc1,auc_roc2,auc_roc3,auc_roc4,auc_roc5,auc_roc6

plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr1, tpr1, linewidth=4, label= "prot seq")
plt.plot(fpr2, tpr2, linewidth=4,label= "prot ss")
plt.plot(fpr3, tpr3, linewidth=4,label= "protein seq + ss")
plt.plot(fpr4, tpr4, linewidth=4,label= "pep seq")
plt.plot(fpr5, tpr5, linewidth=4,label= "pep ss")
plt.plot(fpr6, tpr6, linewidth=4,label= "pep seq + ss")
plt.legend(fontsize=16)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title('Receiver Operating Characteristic')
plt.show()

plt.plot([0,1],[0,1], 'k--')
plt.plot(recall1, precision1, linewidth=4, label= "prot seq")
plt.plot(recall2, precision2, linewidth=4,label= "prot ss")
plt.plot(recall3, precision3, linewidth=4,label= "protein seq + ss")
plt.plot(recall4, precision4, linewidth=4,label= "pep seq")
plt.plot(recall5, precision5, linewidth=4,label= "pep ss")
plt.plot(recall6, precision6, linewidth=4,label= "pep seq + ss")
plt.legend(fontsize=16)
plt.xlabel("recall")
plt.ylabel("precision")
plt.title('PRC')
plt.show()




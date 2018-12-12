import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

import sklearn.model_selection as skms
import sklearn.metrics as skme
import sklearn.preprocessing as skpp
import sklearn.decomposition as skde

# Import Dataset (Data_train)
data_train = pd.read_csv('data_input/npl_train.csv')
dataX = data_train[['jumlah_kartu', 'outstanding', 'limit_kredit', 'tagihan',
       'total_pemakaian_tunai', 'total_pemakaian_retail',
       'sisa_tagihan_tidak_terbayar','rasio_pembayaran',
       'persentasi_overlimit', 'rasio_pembayaran_3bulan',
       'rasio_pembayaran_6bulan', 'skor_delikuensi',
       'jumlah_tahun_sejak_pembukaan_kredit', 'total_pemakaian',
       'sisa_tagihan_per_jumlah_kartu', 'sisa_tagihan_per_limit',
       'total_pemakaian_per_limit', 'pemakaian_3bln_per_limit',
       'pemakaian_6bln_per_limit', 'utilisasi_3bulan', 'utilisasi_6bulan']]
scale1 = skpp.Normalizer()
dataX = scale1.fit_transform(dataX)

dataY = data_train['flag_kredit_macet']

# Train Test Split
X_train, X_test, y_train, y_test = skms.train_test_split(dataX, dataY.values, 
                                                         test_size = 0.2,
                                                         random_state = 13)

# Oversample Data
smote = SMOTE(n_jobs = 8, k_neighbors = 3)
X_smote, y_smote = smote.fit_sample(X_train, y_train)

# Neural Network
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD

col_input = X_smote.shape[1]
col_output = 1
num_neurons = (256,128,100,100,100)
num_layers = len(num_neurons)
learning_rt = 0.0005
num_epochs = 50
num_batchs = 100

model = Sequential()
model.add(Dense(units = num_neurons[0], activation = 'relu', 
                input_shape = (col_input,)))
model.add(Dense(units = num_neurons[1], activation = 'relu',))
model.add(Dense(units = num_neurons[2], activation = 'relu',))
model.add(Dense(units = num_neurons[3], activation = 'relu',))
model.add(Dense(units = num_neurons[4], activation = 'relu',))
model.add(Dense(units = col_output, activation = 'sigmoid'))

import tensorflow as tf
from keras import backend as K

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

model.compile(loss="binary_crossentropy", optimizer=Adam(lr = learning_rt), 
              metrics=[auc])
run_hist = model.fit(X_smote, y_smote, batch_size = num_batchs, epochs = num_epochs,
          validation_data = (X_test, y_test))

score = model.evaluate(X_test, y_test)
y_test_prob = model.predict_proba(X_test)
y_test_pred = model.predict(X_test)

# Plot Precision and Recall Trade off
precision, recall, threesold = skme.precision_recall_curve(y_true = y_test, 
                                                           probas_pred = y_test_prob)
plt.plot(threesold, precision[:-1], label = 'precision')        #precision dan recall memiliki kelebihan 1 elemen
plt.plot(threesold, recall[:-1], label = 'recall')
plt.xlabel('Threesold')
plt.legend(); plt.grid(); plt.show()
plt.plot(recall, precision)
plt.xlabel('recall'); plt.ylabel('precision')
plt.grid(); plt.show()

# ROC (Receiver Operating Curve) and AUC (Area Under Curve)
fpr, tpr, threesold = skme.roc_curve(y_test, y_test_prob)
AUC = skme.auc(x = fpr, y = tpr)

plt.title('ROC (Receiver Operating Curve)')
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.grid(); plt.show()
    
print('AUC : ', AUC)

y_test_predict = []
for i in range(len(y_test_prob)):
    if (y_test_prob[i] >= 0.5):
        y_test_predict.append(1)
    else:
        y_test_predict.append(0)

def plotting(run_hist):
    plt.title('Loss Function')
    plt.plot(run_hist.history['loss'], label = 'Data Train')
    plt.plot(run_hist.history['val_loss'], label = 'Data Test')
    plt.legend(); plt.grid()
    plt.show()

    plt.title('AUC')    
    plt.plot(run_hist.history['auc'], label = 'Data Train')
    plt.plot(run_hist.history['val_auc'], label = 'Data Test')
    plt.legend(); plt.grid()
    plt.show()
    
plotting(run_hist)
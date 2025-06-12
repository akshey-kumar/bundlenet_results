"""
import sys
sys.path.append("../..")
sys.path.append("..")
print(sys.path)
from c_elegans_embedding_evaluation.functions import *

"""
import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn import metrics

algorithm = sys.argv[1]
worm_num = int(sys.argv[2])
print(algorithm, ' worm_num: ', worm_num)

file_pattern = f'data/generated/saved_Y/{{}}__{algorithm}_worm_{worm_num}'
Y0_tr = np.loadtxt(file_pattern.format('Y0_tr'))
Y1_tr = np.loadtxt(file_pattern.format('Y1_tr'))
Y0_tst = np.loadtxt(file_pattern.format('Y0_tst'))
Y1_tst = np.loadtxt(file_pattern.format('Y1_tst'))
B_train_1 = np.loadtxt(file_pattern.format('B_train_1'))
B_test_1 = np.loadtxt(file_pattern.format('B_test_1'))

Y0_tr = Y0_tr.reshape(Y0_tr.shape[0],-1)
Y1_tr = Y1_tr.reshape(Y1_tr.shape[0],-1)
Y0_tst = Y0_tst.reshape(Y0_tst.shape[0],-1)
Y1_tst = Y1_tst.reshape(Y1_tst.shape[0],-1)
Ydiff_tr = Y1_tr - Y0_tr
Ydiff_tst = Y1_tst - Y0_tst

### Behaviour decodability evaluation
acc_list = []
for i in tqdm(np.arange(10)):
    b_predictor = tf.keras.Sequential([tf.keras.layers.Dense(8, activation='linear')])
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
    b_predictor.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = b_predictor.fit(Y1_tr,
                          B_train_1,
                          epochs=100,
                          batch_size=100,
                          validation_data=(Y1_tst, B_test_1),
                          verbose=0
                          )

    B1_tst_pred = b_predictor(Y1_tst).numpy().argmax(axis=1)
    acc_list.append(metrics.accuracy_score(B1_tst_pred, B_test_1))

# Saving the metrics
acc_list = np.array(acc_list)
os.makedirs('data/generated/c_elegans_evaluation_metrics', exist_ok=True)
np.savetxt(f'data/generated/c_elegans_evaluation_metrics/acc_list_{algorithm}_worm_{worm_num}', acc_list)

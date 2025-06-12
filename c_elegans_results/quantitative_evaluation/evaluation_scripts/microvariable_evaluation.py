import os
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

worm_num = int(sys.argv[1])

current_dir = os.getcwd()
# os.chdir('../')
# Loading data 
b_neurons = [
	'AVAR',
	'AVAL',
	'SMDVR',
	'SMDVL',
	'SMDDR',
	'SMDDL',
	'RIBR',
	'RIBL',]

data = Database(data_set_no=worm_num)
data.exclude_neurons(b_neurons)
X = data.neuron_traces.T
B = data.states
time, X = preprocess_data(X, data.fps)
X_, B_ = prep_data(X, B, win=15)
X_train, X_test, B_train_1, B_test_1 = timeseries_train_test_split(X_, B_)
X1_tr = X_train[:,1,:,:]
X1_tst = X_test[:,1,:,:]
os.chdir(current_dir)

### Behavioural prediction accuracy directly from neuronal data
### Behavioural decodablity from neuronal level (microvariable)
acc_list = []
for i in tqdm(np.arange(50)):
	b_predictor = tf.keras.Sequential([
		layers.Flatten(),
		layers.Dense(8, activation='linear')
	]) 
	opt = tf.keras.optimizers.Adam(learning_rate=0.01)
	b_predictor.compile(optimizer='adam',
				  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
				  metrics=['accuracy'])
	history = b_predictor.fit(X1_tr,
						  B_train_1,
						  epochs=100,
						  batch_size=100,
						  validation_data=(X1_tst, B_test_1),
						  verbose=0
						  )
		# Summarize history for accuracy
	#plt.plot(history.history['accuracy'])
	#plt.plot(history.history['val_accuracy'])
	#plt.show()
	B1_tst_pred = b_predictor(X1_tst).numpy().argmax(axis=1)
	acc_list.append(accuracy_score(B1_tst_pred, B_test_1))

acc_list = np.array(acc_list)
np.savetxt('data/generated/evaluation_metrics/acc_list_X_worm_' +  str(worm_num), acc_list)



### Estimating the chance accuracy of behaviour decoding
chance_acc = np.zeros(500)
for i, _ in enumerate(chance_acc):
	B_perm = np.random.choice(B_test_1, size=B_test_1.shape)
	chance_acc[i] = accuracy_score(B_perm, B_test_1)
print('Chance prediction accuracy: ', chance_acc.mean().round(3), ' pm ', chance_acc.std().round(3))
np.savetxt('data/generated/evaluation_metrics/acc_list_chance_worm_' +  str(worm_num), chance_acc)
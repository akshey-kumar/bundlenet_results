import sys
sys.path.append(r'../')
import numpy as np
from functions import *

algorithm = 'ArAe'

### Load Data (and excluding behavioural neurons)
for worm_num in range(5):
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
    state_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing', 'Ventral turn']

    ### Preprocess and prepare data for BundLe Net
    time, X = preprocess_data(X, data.fps)
    X_, B_ = prep_data(X, B, win=1)

    ### Train test split 
    X_train, X_test, B_train_1, B_test_1 = timeseries_train_test_split(X_, B_)
    X0_tr = X_train[:,0,:,:]
    X1_tr = X_train[:,1,:,:]
    Xdiff_tr = X1_tr - X0_tr
    X0_tst = X_test[:,0,:,:]
    X1_tst = X_test[:,1,:,:]
    Xdiff_tst = X1_tst - X0_tst

    ### Scaling input and output data
    Xdmax = (np.abs(Xdiff_tr)).max() # Parameters for scaling
    Xdiff_tr, Xdiff_tst = Xdiff_tr/Xdmax, Xdiff_tst/Xdmax

    ### ArAe architecture (autoregressor with autoencoder architecture)
    class Autoencoder(Model):
        def __init__(self, latent_dim):
            super(Autoencoder, self).__init__()
            self.latent_dim = latent_dim   
            self.encoder = tf.keras.Sequential([
              layers.Flatten(),
              layers.Dense(50, activation='relu'),
              layers.Dense(30, activation='relu'),
              layers.Dense(25, activation='relu'),
              layers.Dense(10, activation='relu'),
              layers.Dense(latent_dim, activation='linear'),
            ])
            self.decoder = tf.keras.Sequential([
              layers.Dense(latent_dim, activation='relu'),
              layers.Dense(10, activation='relu'),
              layers.Dense(25, activation='relu'),
              layers.Dense(30, activation='relu'),
              layers.Dense(50, activation='relu'),
              layers.Dense(X1_tr.shape[-1]*X1_tr.shape[-2], activation='linear'),
              layers.Reshape(X1_tr.shape[1:])
            ])
        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    ArAe = Autoencoder(latent_dim = 3)
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    ArAe.compile(optimizer=opt, loss='mse', metrics=['mse'])
    ### Deploy ArAe
    history = ArAe.fit(X0_tr,
                            Xdiff_tr,
                            epochs=50,
                            batch_size=100,
                            validation_data=(X0_tst, Xdiff_tst),
                            verbose=0,
                            )
    ### Predictions
    Xdiff_tr_pred = ArAe(X0_tr).numpy()
    Xdiff_tst_pred = ArAe(X0_tst).numpy()

    # Inverse scaling the data
    Xdiff_tr_pred, Xdiff_tr = Xdiff_tr_pred*Xdmax, Xdiff_tr*Xdmax
    Xdiff_tst_pred, Xdiff_tst = Xdiff_tst_pred*Xdmax, Xdiff_tst*Xdmax

    X1_tr_pred = X0_tr + Xdiff_tr_pred
    X1_tst_pred = X0_tst + Xdiff_tst_pred
    
    baseline_tst  = mean_squared_error(flat_partial(X1_tst), flat_partial(X0_tst))
    modelmse_tst = mean_squared_error(flat_partial(X1_tst), flat_partial(X1_tst_pred))

    print('\nOn test set \n')
    print('Baseline mse', baseline_tst.round(8), 'Model mse:', modelmse_tst.round(8))

    ### Projecting into latent space
    Y0_tr = ArAe.encoder(X0_tr).numpy()
    Y1_tr = ArAe.encoder(X1_tr).numpy()           

    Y0_tst = ArAe.encoder(X0_tst).numpy()
    Y1_tst = ArAe.encoder(X1_tst).numpy()    

    # Save the weights
    # model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
    np.savetxt('data/generated/saved_Y/Y0_tr__'+algorithm+'_worm_'+ str(worm_num), Y0_tr)
    np.savetxt('data/generated/saved_Y/Y1_tr__'+algorithm+'_worm_'+ str(worm_num), Y1_tr)
    np.savetxt('data/generated/saved_Y/Y0_tst__'+algorithm+'_worm_'+ str(worm_num), Y0_tst)
    np.savetxt('data/generated/saved_Y/Y1_tst__'+algorithm+'_worm_'+ str(worm_num), Y1_tst)
    np.savetxt('data/generated/saved_Y/B_train_1__'+algorithm+'_worm_'+ str(worm_num), B_train_1)
    np.savetxt('data/generated/saved_Y/B_test_1__'+algorithm+'_worm_'+ str(worm_num), B_test_1)






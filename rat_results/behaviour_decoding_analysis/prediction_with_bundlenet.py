import os
import numpy as np
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model, project_into_latent_space
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
import matplotlib.pyplot as plt

algorithm = 'bundlenet'
for rat_name in ['achilles', 'gatsby', 'cicero', 'buddy']:
    data = np.load(f'data/raw/rat_hippocampus/{rat_name}.npz')
    x, b = data['x'], data['b']
    x = x - np.min(x)  # cebra doesn't work otherwise if there are negative values
    np.where(x < 0)
    x_, b_ = prep_data(x, b, win=20)

    # Train test split
    x_train, x_test, b_train_1, b_test_1 = timeseries_train_test_split(x_, b_)

    # Deploy BunDLe Net
    model = BunDLeNet(latent_dim=3, num_behaviour=b_.shape[1], input_shape=x_.shape)

    train_history, test_history = train_model(
        x_train,
        b_train_1,
        model,
        b_type='continuous',
        gamma=0.9,
        learning_rate=0.001,
        n_epochs=500,
        initialisation=None,
        validation_data=(x_test, b_test_1),
    )

    plt.figure()
    for i, label in enumerate([
        r"$\mathcal{L}_{\mathrm{Markov}}$",
        r"$\mathcal{L}_{\mathrm{Behavior}}$",
        r"Train loss $\mathcal{L}$"
    ]):
        plt.plot(train_history[:, i], label=label)
    plt.plot(test_history[:, -1], label='Test loss', linestyle='--')
    plt.legend()

    # Projecting into latent space
    y0_tr = project_into_latent_space(x_train[:,0], model)
    y1_tr = project_into_latent_space(x_train[:,1], model)
    y0_tst = project_into_latent_space(x_test[:,0], model)
    y1_tst = project_into_latent_space(x_test[:,1], model)

    import torch
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        y1_tr_tensor = torch.from_numpy(y1_tr).float().to(device)
        y1_tst_tensor = torch.from_numpy(y1_tst).float().to(device)
        b_train_1_pred = model.predictor(y1_tr_tensor).cpu().numpy()
        b_test_1_pred = model.predictor(y1_tst_tensor).cpu().numpy()

    # Save the weights
    # model.save_weights(f'data/generated/BunDLeNet_model_rat_{rat_name}')
    os.makedirs('data/generated/predicted_and_true_behaviours', exist_ok=True)
    np.savetxt(f'data/generated/predicted_and_true_behaviours/b_train_1_pred__{algorithm}_rat_{rat_name}',
               b_train_1_pred)
    np.savetxt(f'data/generated/predicted_and_true_behaviours/b_test_1_pred__{algorithm}_rat_{rat_name}', b_test_1_pred)
    np.savetxt(f'data/generated/predicted_and_true_behaviours/b_train_1__{algorithm}_rat_{rat_name}', b_train_1)
    np.savetxt(f'data/generated/predicted_and_true_behaviours/b_test_1__{algorithm}_rat_{rat_name}', b_test_1)

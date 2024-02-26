import torch
from time import time
from RNN_Class import RNN
import numpy as np
import datetime

start_time = time()

'''~~~      DMS Task            ~~~'''
# DMS Data
stimuli = torch.tensor([[[0,1], [0,1]],                         # Match
                        [[0,1], [1,0]],                         # Mismatch
                        [[1,0], [0,1]],                         # Mismatch
                        [[1,0], [1,0]]], dtype=torch.float)     # Match
labels = torch.tensor([[[0,1], [1,0]],
                       [[0,1], [0,1]],
                       [[1,0], [0,1]],
                       [[1,0], [1,0]]], dtype=torch.float)


'''~~~      Fixed Model Params        ~~~'''
# Model Hyperparameters
activation = 'relu'
reg = 0.0001

# Training Hyperparameters
N_EPOCHS = 2500
LEARNING_RATE = 0.003


'''~~~      Varying Model Params        ~~~'''
w_var_arr = np.logspace(0.001, 0.01, 50)         # Input Weight variance, 10x-100x larger than 0.0001 (rec_weight variance)


'''~~~      RNN Training        ~~~'''
for i in range(len(w_var_arr)):
    print(f"\nTraining Model {i}/{len(w_var_arr)}...")
    # Model name
    model_name = f'W_init - Model {i+1} of {len(w_var_arr)}'
    # Initialize the RNN model
    model = RNN(dir='Models', name=model_name)
    model.hyp('dms', activation=activation, lr=LEARNING_RATE, num_epochs=N_EPOCHS, reg=reg, w_var=w_var_arr[i])

    # Train the model
    model.train_model(stimuli, labels, p=False)

    '''~~~      Model Evaluation    ~~~'''
    # Load Saved Best Model
    model = RNN(dir='Models', name=model_name, p=False)

    # Test Model
    accuracy = model.test(stimuli, labels, p=False)
    with open(f'Models/{model_name}/{"Successful" if accuracy == 1.0 else "Failed"}.txt', 'w') as file:
        file.write(f"Description: \n{model.description}")
        
    # Model Evaluation
    model.eval()
    model.forward(stimuli)

    # Analysis
    model.plot_pca_trajectories_2D(stimuli, f"Trajectories in PC1-PC2 space for DMS task stages")
    model.plot_abs_activity(stimuli)
    model.plot_drs(stimuli)

    # Time Analysis
    time_elapsed = time() - start_time
    ETA = time_elapsed * (len(w_var_arr) - i - 1) / (i + 1)
    print(f"\nETA: {datetime.timedelta(seconds=np.round(ETA))} / {datetime.timedelta(seconds=np.round(time_elapsed+ETA))}")

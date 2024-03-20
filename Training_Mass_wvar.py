import torch
from time import time
from RNN_Class import RNN
import numpy as np
import datetime
import os

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
reg = 1e-4

# Training Hyperparameters
N_EPOCHS = 3000
LEARNING_RATE = 0.002
N_MODELS = 100


'''~~~      Varying Model Params        ~~~'''
w_var_arr = np.logspace(-3, 0, num=50)         # Input Weight variance, 10x-100x larger than 0.0001 (rec_weight variance)

repeats = 1
'''~~~      RNN Training        ~~~'''
for i in range(len(w_var_arr)):
    print(f"\nTraining Model {i+1}/{len(w_var_arr)}...")
    # Model name
    model_name = f'W_init - Model {i+1} of {len(w_var_arr)}'
    model = RNN(dir='Models/wvar', name=model_name)

    # Repeat Training for 3 times unless early sucess
    for trial in range(repeats):
        # Check if Model is already successfully trained
        try:
            with open(f'Models/wvar/{model_name}/Successful.txt', 'r') as file:
                # Delete Failed.txt if it exists
                try:
                    os.remove(f'Models/wvar/{model_name}/Failed.txt')
                except FileNotFoundError:
                    pass
                continue
        except FileNotFoundError:
            # Clear all files in the directory
            for file in os.listdir(f'Models/wvar/{model_name}'):
                os.remove(f'Models/wvar/{model_name}/{file}')


        '''~~~      Model Training    ~~~'''

        # Initialize the RNN model
        model.hyp('dms', N_Models=N_MODELS, activation=activation, lr=LEARNING_RATE, num_epochs=N_EPOCHS, reg=reg, w_var=w_var_arr[i])

        # Train the model
        model.train_model(stimuli, labels, p=False)

        '''~~~      Model Evaluation    ~~~'''
        # Load Saved Best Model
        model = RNN(dir='Models/wvar', name=model_name, p=False)

        # Test Model
        accuracy = model.test(stimuli, labels, p=False)
        with open(f'Models/wvar/{model_name}/{"Successful" if accuracy == 1.0 else "Failed"}.txt', 'w') as file:
            file.write(f"Description: \n{model.description}")
            
        # Model Evaluation
        model.eval()
        # Test Model
        acc = model.test(stimuli, labels)
        indices = torch.nonzero(acc.eq(1), as_tuple=True)[0]
        indices = indices[0]
        model.forward(stimuli)

        # Analysis
        model.plot_pca_trajectories_2D(indices, stimuli)
        model.plot_abs_activity(indices, stimuli)
        model.plot_drs(indices, stimuli)

        # Time Analysis
        time_elapsed = time() - start_time
        ETA = time_elapsed * (len(w_var_arr) - i - 1) / (i + 1)
        print(f"\nETA: {datetime.timedelta(seconds=np.round(ETA))} / {datetime.timedelta(seconds=np.round(time_elapsed+ETA))}")

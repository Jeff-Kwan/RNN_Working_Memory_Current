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
w_var = 0.001

# Training Hyperparameters
N_EPOCHS = 2500
LEARNING_RATE = 0.003


'''~~~      Varying Model Params        ~~~'''
reg_arr = np.linspace(0.00001, 0.001, 50)         # Input Weight variance, 10x-100x larger than 0.0001 (rec_weight variance)


repeats = 1
'''~~~      RNN Training        ~~~'''
for i in range(len(reg_arr)):
    print(f"\nTraining Model {i+1}/{len(reg_arr)}...")
    # Model name
    model_name = f'Reg - Model {i+1} of {len(reg_arr)}'
    model = RNN(dir='Models', name=model_name)

    # Repeat Training for 3 times unless early sucess
    for trial in range(repeats):
        # Check if Model is already successfully trained
        try:
            with open(f'Models/{model_name}/Successful.txt', 'r') as file:
                # Delete Failed.txt if it exists
                try:
                    os.remove(f'Models/{model_name}/Failed.txt')
                except FileNotFoundError:
                    pass
                continue
        except FileNotFoundError:
            # Clear all files in the directory
            for file in os.listdir(f'Models/{model_name}'):
                os.remove(f'Models/{model_name}/{file}')


        '''~~~      Model Training    ~~~'''

        # Initialize the RNN model
        model.hyp('dms', activation=activation, lr=LEARNING_RATE, num_epochs=N_EPOCHS, reg=reg_arr[i], w_var=w_var)

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
        ETA = time_elapsed * (len(reg_arr) - i - 1) / (i + 1)
        print(f"\nETA: {datetime.timedelta(seconds=np.round(ETA))} / {datetime.timedelta(seconds=np.round(time_elapsed+ETA))}")

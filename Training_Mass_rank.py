import torch
from time import time
from RNN_Class import RNN
import numpy as np
import datetime
import os
import shutil

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
w_var = 0.01
reg = 0.001
N_CELL = 10

# Training Hyperparameters
N_EPOCHS = 5000
LEARNING_RATE = 0.002
N_MODELS = 100


'''~~~      Varying Model Params        ~~~'''
rank_arr = np.linspace(1, 10, num=10).astype(int)        
print("Training on ranks:")
print(rank_arr)
varying_var = 'rank'

repeats = 1 
'''~~~      RNN Training        ~~~'''
for i in range(len(rank_arr)):
    print(f"\nTraining Model {i+1}/{len(rank_arr)}...")
    # Model name
    model_name = f'{varying_var} - Model {i+1} of {len(rank_arr)}'
    model = RNN(dir=f'Models/{varying_var}', name=model_name)

    # Repeat Training for 3 times unless early sucess
    for trial in range(repeats):
        # Check if Model is already successfully trained
        try:
            with open(f'Models/{varying_var}/{model_name}/Successful.txt', 'r') as file:
                # Delete Failed.txt if it exists
                try:
                    os.remove(f'Models/{varying_var}/{model_name}/Failed.txt')
                except FileNotFoundError:
                    pass
                continue
        except FileNotFoundError:
            # Recreate folder
            shutil.rmtree(f'Models/{varying_var}/{model_name}')
            os.makedirs(f'Models/{varying_var}/{model_name}')


        '''~~~      Model Training    ~~~'''

        # Initialize the RNN model
        model.hyp('dms', N_Models=N_MODELS, activation=activation, lr=LEARNING_RATE, num_epochs=N_EPOCHS, reg=reg, w_var=w_var, N_CELL=N_CELL, rank=rank_arr[i])

        # Train the model
        model.train_model(stimuli, labels, p=False)

        '''~~~      Model Evaluation    ~~~'''
        # Test Model
        model.eval()
        acc = model.test(stimuli, labels, p=False)
        indices = torch.nonzero(acc.eq(1), as_tuple=True)[0]
        if len(indices) > 0:
            indices = indices[torch.randint(len(indices), size=(1,))] # Random Correct Model(s)
            model.plot_PCAs(indices, stimuli)
            model.plot_PCAs_2(indices, stimuli)
            model.forward(stimuli)
            model.plot_abs_activity(indices, stimuli)
            model.plot_drs(indices, stimuli)
            
            if sum(acc.eq(1))/N_MODELS >= 0.5:
                with open(f'Models/{varying_var}/{model_name}/Successful.txt', 'w') as file:
                    file.write(model.description)
            else:
                with open(f'Models/{varying_var}/{model_name}/Failed.txt', 'w') as file:
                    file.write(model.description)

        # Time Analysis
        time_elapsed = time() - start_time
        ETA = time_elapsed * (len(rank_arr) - i - 1) / (i + 1)
        print(f"\nETA: {datetime.timedelta(seconds=np.round(ETA))} / {datetime.timedelta(seconds=np.round(time_elapsed+ETA))}")

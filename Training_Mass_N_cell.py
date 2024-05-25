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

# Training Hyperparameters
N_EPOCHS = 5000
LEARNING_RATE = 0.002
N_MODELS = 500


'''~~~      Varying Model Params        ~~~'''
cell_arr = np.linspace(5, 40, num=8).astype(int)        
print("Training on N_CELL values:")
print(cell_arr)
varying_var = 'N_cell'

repeats = 1 
'''~~~      RNN Training        ~~~'''
for i in range(len(cell_arr)):
    print(f"\nTraining Model {i+1}/{len(cell_arr)}...")
    # Model name
    model_name = f'{varying_var} - Model {i+1} of {len(cell_arr)}'
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
        model.hyp('dms', N_Models=N_MODELS, activation=activation, lr=LEARNING_RATE, num_epochs=N_EPOCHS, reg=reg, w_var=w_var, N_CELL=cell_arr[i])

        # Train the model
        model.train_model(stimuli, labels, p=False)

        '''~~~      Model Evaluation    ~~~'''
        # Test Model
        model.eval()
        acc = model.test(stimuli, labels, p=False)
        indices = torch.nonzero(acc.eq(1), as_tuple=True)[0]
        if len(indices) >= 3:
            model.participation_ratio(stimuli, labels, t=1)
            
            if sum(acc.eq(1))/N_MODELS >= 0.5:
                with open(f'Models/{varying_var}/{model_name}/Successful.txt', 'w') as file:
                    file.write(model.description)
            else:
                with open(f'Models/{varying_var}/{model_name}/Failed.txt', 'w') as file:
                    file.write(model.description)

        # Time Analysis
        time_elapsed = time() - start_time
        ETA = time_elapsed * (len(cell_arr) - i - 1) / (i + 1)
        print(f"\nETA: {datetime.timedelta(seconds=np.round(ETA))} / {datetime.timedelta(seconds=np.round(time_elapsed+ETA))}")

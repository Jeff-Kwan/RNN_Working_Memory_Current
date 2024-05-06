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
w_var = 0.001

# Training Hyperparameters
N_EPOCHS = 3000
LEARNING_RATE = 0.002
N_MODELS = 100


'''~~~      Varying Model Params        ~~~'''
reg_arr = np.logspace(-5, 0, num=11)        
print("Training on regularization values:")
print(reg_arr)

repeats = 3
'''~~~      RNN Training        ~~~'''
for i in range(len(reg_arr)):
    print(f"\nTraining Model {i+1}/{len(reg_arr)}...")
    # Model name
    model_name = f'W_init - Model {i+1} of {len(reg_arr)}'
    model = RNN(dir='Models/reg', name=model_name)

    # Repeat Training for 3 times unless early sucess
    for trial in range(repeats):
        # Check if Model is already successfully trained
        try:
            with open(f'Models/reg/{model_name}/Successful.txt', 'r') as file:
                # Delete Failed.txt if it exists
                try:
                    os.remove(f'Models/reg/{model_name}/Failed.txt')
                except FileNotFoundError:
                    pass
                continue
        except FileNotFoundError:
            # Recreate folder
            shutil.rmtree(f'Models/reg/{model_name}')
            os.makedirs(f'Models/reg/{model_name}')


        '''~~~      Model Training    ~~~'''

        # Initialize the RNN model
        model.hyp('dms', N_Models=N_MODELS, activation=activation, lr=LEARNING_RATE, num_epochs=N_EPOCHS, reg=reg_arr[i], w_var=w_var)

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

        # Time Analysis
        time_elapsed = time() - start_time
        ETA = time_elapsed * (len(reg_arr) - i - 1) / (i + 1)
        print(f"\nETA: {datetime.timedelta(seconds=np.round(ETA))} / {datetime.timedelta(seconds=np.round(time_elapsed+ETA))}")

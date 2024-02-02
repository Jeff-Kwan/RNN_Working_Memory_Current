import torch
from time import time
from RNN_Class import RNN

start_time = time()

# Define hyperparameters
N_STIM = 2
N_CELL = 10
N_EPOCHS = 5000
LEARNING_RATE = 0.002
activation = 'relu'
reg = 0.0001

# Model name and description
model_name = 'Model_7'
if reg > 0:
    loss_comb = 'CE + Aux + Reg'
else:
    loss_comb = 'CE + Aux'
description = f'LOSS: {loss_comb}, {N_CELL}_cell / {activation}_activation / {N_EPOCHS}_epochs / {LEARNING_RATE}_rate / {N_STIM}D_DMS / {reg}_reg'

# DMS Data
stimuli = torch.tensor([[[0,1], [0,1]],                         # Match
                        [[0,1], [1,0]],                         # Mismatch
                        [[1,0], [0,1]],                         # Mismatch
                        [[1,0], [1,0]]], dtype=torch.float)     # Match
labels = torch.tensor([[[0,1], [1,0]],
                       [[0,1], [0,1]],
                       [[1,0], [0,1]],
                       [[1,0], [1,0]]], dtype=torch.float)

# Initialize the RNN model
model = RNN(dir='Models', name=model_name)
model.hyp('dms', activation=activation, lr=LEARNING_RATE, num_epochs=N_EPOCHS, reg=reg)

# Train the model
model.train_model(stimuli, labels)

# Test Model
accuracy = model.test(stimuli, labels)

# Save (or Delete) Model 
if accuracy == 1:
    model.save_model('Models', model_name, description)
else:
    print(f'\nLow Accuracy, failed training of {model_name}')
    model.del_model('Models', model_name)

# Time Elapsed
hours, remainder = divmod(time() - start_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Time elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
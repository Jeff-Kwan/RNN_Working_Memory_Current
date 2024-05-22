import torch
from time import time
from RNN_Class import RNN

start_time = time()


'''~~~      Model Params        ~~~'''
# Model name
model_name = 'Baseline Model'
N_Models = 100
N_CELL = 10

# Model Hyperparameters
activation = 'relu'
reg = 0.001
w_var = 0.01         # Input Weight variance, 10x-100x larger than 0.0001 (rec_weight variance)
rank = None

# Training Hyperparameters
N_EPOCHS = 5000
LEARNING_RATE = 0.002


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


'''~~~      RNN Training        ~~~'''
# Initialize the RNN model
model = RNN(dir='Models', name=model_name)
model.hyp('dms', N_Models=N_Models, N_CELL=N_CELL, activation=activation, lr=LEARNING_RATE, num_epochs=N_EPOCHS, reg=reg, w_var=w_var, rank=rank)

# Train the model
model.train_model(stimuli, labels)

'''~~~      Model Evaluation    ~~~'''
# Load Saved Best Model
model = RNN(dir='Models', name=model_name)

# Test Model
model.eval()
acc = model.test(stimuli, labels, p=False)
indices = torch.nonzero(acc.eq(1), as_tuple=True)[0]
indices = indices[torch.randint(len(indices), size=(2,))] # Random Correct Model(s)

# PCA Plots
model.plot_PCAs(indices, stimuli)

# Activity Plots
model.forward(stimuli)
model.plot_abs_activity(indices, stimuli)
model.plot_drs(indices, stimuli)



'''~~~      End of File         ~~~'''
# Time Elapsed
hours, remainder = divmod(time() - start_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Time elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
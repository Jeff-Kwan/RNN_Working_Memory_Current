import torch
from time import time
from RNN_Class import RNN

start_time = time()


'''~~~      Model Params        ~~~'''
# Model name
model_name = 'Model_1'

# Model Hyperparameters
activation = 'relu'
reg = 0.0002
w_var = 0.1

# Training Hyperparameters
N_EPOCHS = 2000
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
model.hyp('dms', activation=activation, lr=LEARNING_RATE, num_epochs=N_EPOCHS, reg=reg, w_var=w_var)

# Train the model
model.train_model(stimuli, labels)

# Test Model
model.test(stimuli, labels)

'''~~~      Model Evaluation    ~~~'''
# Model Evaluation
model.eval()
model.forward(stimuli)

# Analysis
model.plot_pca_trajectories_2D(stimuli, f"Trajectories in PC1-PC2 space for DMS task stages")
model.plot_abs_activity(stimuli)
model.plot_drs(stimuli)


'''~~~      End of File         ~~~'''
# Time Elapsed
hours, remainder = divmod(time() - start_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Time elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
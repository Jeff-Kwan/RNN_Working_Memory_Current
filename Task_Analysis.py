import torch
from time import time
from RNN_Class import RNN

start_time = time()

# Load Model
model_name = 'Model_1'
model = RNN('Models', model_name)

# DMS Data & Run Model
stimuli = torch.tensor([[[0,1], [0,1]],                         # Match
                        [[0,1], [1,0]],                         # Mismatch
                        [[1,0], [0,1]],                         # Mismatch
                        [[1,0], [1,0]]], dtype=torch.float)     # Match
labels = torch.tensor([[[0,1], [1,0]],
                       [[0,1], [0,1]],
                       [[1,0], [0,1]],
                       [[1,0], [1,0]]], dtype=torch.float)

# Test Model
model.test(stimuli, labels)

# Model Evaluation
model.eval()
model.forward(stimuli)

# Analysis
model.plot_pca_trajectories_2D(stimuli, f"Trajectories in PC1-PC2 space for DMS task stages")
model.plot_abs_activity(stimuli)
model.plot_drs(stimuli)
#model.plot_gradient_field(stimuli)


# Time Elapsed
hours, remainder = divmod(time() - start_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Time elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
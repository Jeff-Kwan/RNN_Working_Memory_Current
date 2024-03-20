import torch
from time import time
from RNN_Class import RNN

start_time = time()

# Load Model
model_name = 'Models_1'
model = RNN('Models', model_name, p=True)

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
acc = model.test(stimuli, labels)
indices = torch.nonzero(acc.eq(1), as_tuple=True)[0]
indices = indices[[0, -1]] # First and last correct models by index

# Model Evaluation
model.eval()
model.forward(stimuli)

# Analysis
# model.plot_pca_trajectories_2D(indices, stimuli)
# model.plot_abs_activity(indices, stimuli)
# model.plot_drs(indices, stimuli)
model.plot_gradient_flow(indices, stimuli)


# Time Elapsed
hours, remainder = divmod(time() - start_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Time elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
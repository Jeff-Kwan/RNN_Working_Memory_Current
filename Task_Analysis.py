import torch
from time import time
from RNN_Class import RNN

start_time = time()

# Load Model
model_name = 'Models_4'
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
model.eval()
acc = model.test(stimuli, labels, p=True)
indices = torch.nonzero(acc.eq(1), as_tuple=True)[0]
indices = indices[torch.randint(len(indices), size=(1,))] # Random Correct Model(s)
indices = torch.tensor([60])

# PCA Plots
model.plot_PCAs(indices, stimuli)
model.plot_PCAs_2(indices, stimuli)

# Activity Plots
model.forward(stimuli)
model.plot_abs_activity(indices, stimuli)
model.plot_drs(indices, stimuli)

# Time Elapsed
hours, remainder = divmod(time() - start_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Time elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
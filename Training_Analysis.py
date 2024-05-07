import torch
from time import time
from RNN_Class import RNN, progress_bar
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


dir = 'Models/reg'
varying_param = 'reg'
subfolders_num = len(os.listdir(dir))
for i in range(subfolders_num):
    # Model name
    model_name = f'{varying_param} - Model {i+1} of {subfolders_num}'

    '''~~~      Model Evaluation    ~~~'''
    try:
        with open(f'Models/{varying_param}/{model_name}/Successful.txt', 'r') as file:
            model = RNN(dir=f'Models/{varying_param}', name=model_name)
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

    except FileNotFoundError:
        pass

    # Time Analysis
    progress_bar(subfolders_num, i, start_time)
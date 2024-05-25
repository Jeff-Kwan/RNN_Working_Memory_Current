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


dir = 'Models/Batch 2 Data - Gaussian'
params = ['w_init', 'reg', 'rank', 'N_cell']
#varying_param = 'w_init'

for varying_param in params:
    subfolders_num = len(os.listdir(dir+f'/{varying_param}'))
    for i in range(subfolders_num):
        # Model name
        model_name = f'{varying_param} - Model {i+1} of {subfolders_num}'
        '''~~~      Model Evaluation    ~~~'''
        try:
            with open(dir+f'/{varying_param}/{model_name}/Successful.txt', 'r') as file:
                model = RNN(dir=dir+f'/{varying_param}', name=model_name)
                model.eval()
                model.participation_ratio(stimuli, labels)

        except FileNotFoundError:
                pass

        # Time Analysis
        progress_bar(subfolders_num, i, start_time)
import torch
from time import time
from RNN_Class import RNN, progress_bar
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

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
params = ['w_init', 'reg', 'N_cell', 'rank']
param_ranges = {'w_init': np.logspace(-4, 0, num=5),
                'reg': np.logspace(-5, 0, num=6),
                'rank': np.linspace(1, 10, num=10).astype(int),
                'N_cell': np.linspace(5, 40, num=8).astype(int)  }

for varying_param in params:
    subfolders_num = len(param_ranges[varying_param])#len(os.listdir(dir+f'/{varying_param}'))

    PR_mean = np.zeros(subfolders_num)
    PR_var = np.zeros(subfolders_num)
    for i in range(subfolders_num):
        # Model name
        model_name = f'{varying_param} - Model {i+1} of {subfolders_num}'
        '''~~~      Model Evaluation    ~~~'''
        try:
            model = RNN(dir=dir+f'/{varying_param}', name=model_name)
            model.eval()
            PR_mean[i], PR_var[i] = model.participation_ratio(stimuli, labels, p=False, t=2)
        except:
            pass

        # Time Analysis
        progress_bar(subfolders_num, i, start_time, f'{varying_param}')

    '''~~~ Participation Ratio Statistics ~~~'''
    mask = PR_mean != 0
    PR_mean_nonzero = PR_mean[mask]
    param_range_nonzero = np.array(param_ranges[varying_param])[mask]
    def plot_with_error_bars(x, mean, var, ax, color):
        std_error = np.sqrt(var) / np.sqrt(len(x))
        ax.errorbar(x, mean, yerr=std_error, color=color, fmt='-o')

    fig, ax1 = plt.subplots()
    plot_with_error_bars(np.array(param_ranges[varying_param]), PR_mean, PR_var, ax1, 'blue')
    ax1.set_xlabel(varying_param)
    ax1.set_ylabel('PR Mean', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.spines['top'].set_visible(False) 
    ax1.spines['right'].set_visible(False)
    if varying_param == 'w_init' or varying_param == 'reg':
        ax1.set_xscale('log')
    plt.tight_layout()
    plt.savefig(f'{dir}/{varying_param}/PR_Analysis_{varying_param}.svg')
    plt.close()

    # Calculate correlation and p-value for each parameter
    if varying_param == 'w_init' or varying_param == 'reg':
        corr, p_value = stats.pearsonr(PR_mean, np.log(param_ranges[varying_param]))
    else:
        corr, p_value = stats.pearsonr(PR_mean, param_ranges[varying_param])
    #print(f'\nCorrelation with {varying_param}: {format(corr, ".3g")}, p-value: {format(p_value, ".3g")}')
    with open(f'{dir}/correlation_analysis.txt', 'a') as f:
        f.write(f'\nCorrelation with {varying_param}: {format(corr, ".3g")}, p-value: {format(p_value, ".3g")}')


# Time Elapsed
hours, remainder = divmod(time() - start_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"\nTime elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
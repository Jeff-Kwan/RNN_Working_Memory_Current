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
params = ['w_init', 'reg', 'rank', 'N_cell']
param_ranges = {'w_init': np.logspace(-4, 0, num=5),
                'reg': np.logspace(-5, 0, num=6),
                'rank': np.linspace(1, 10, num=10).astype(int),
                'N_cell': np.linspace(5, 40, num=8).astype(int)}
param_str = {'w_init': 'w_in init variance',
                'reg': 'regularisation strength \u03bb',
                'rank': 'rank',
                'N_cell': 'number of neurons'}

for varying_param in params:
    subfolders_num = len(param_ranges[varying_param])#len(os.listdir(dir+f'/{varying_param}'))
    N_models = 100
    PR_mean = np.zeros(subfolders_num)
    PR_var = np.zeros(subfolders_num)
    PRs = np.zeros((subfolders_num, N_models))
    for i in range(subfolders_num):
        if varying_param == 'rank' and i == subfolders_num-1:
            continue
        # Model name
        model_name = f'{varying_param} - Model {i+1} of {subfolders_num}'
        '''~~~      Model Evaluation    ~~~'''
        try:
            model = RNN(dir=dir+f'/{varying_param}', name=model_name)
            model.eval()
            PR_mean[i], PR_var[i], PRs[i] = model.participation_ratio(stimuli, labels, p=False, t=10)
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
    ax1.set_xlabel(param_str[varying_param])
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
    if varying_param == 'rank':
        t_stat, p_value_t_test = stats.ttest_ind(PRs[0][(PRs[0] != 0) & (PRs[-2] != 0)], PRs[-2][(PRs[0] != 0) & (PRs[-2] != 0)])
    else:
        t_stat, p_value_t_test = stats.ttest_ind(PRs[0][(PRs[0] != 0) & (PRs[-1] != 0)], PRs[-1][(PRs[0] != 0) & (PRs[-1] != 0)])
    param = np.ravel(np.tile(param_ranges[varying_param][:, np.newaxis], (1, N_models)))
    PRs = np.ravel(PRs)
    mask = PRs != 0
    PRs_nonzero = PRs[mask]
    param_nonzero = param[mask]
    if varying_param == 'w_init' or varying_param == 'reg':
        corr, p_value = stats.pearsonr(PRs_nonzero, np.log(param_nonzero))
    else:
        corr, p_value = stats.pearsonr(PRs_nonzero, param_nonzero)
    with open(f'{dir}/correlation_analysis.txt', 'a') as f:
        f.write(f'\nCorrelation with {varying_param}: {format(corr, ".3g")}, p-value: {format(p_value, ".3g")}')
        f.write(f'\nT-test with {varying_param}: {format(t_stat, ".3g")}, p-value: {format(p_value_t_test, ".3g")}')


# Time Elapsed
hours, remainder = divmod(time() - start_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"\nTime elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
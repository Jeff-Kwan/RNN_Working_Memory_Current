'''
Draft for RNN class for training delayed match to sample task
'''

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrow
from sklearn.decomposition import PCA
import os, shutil
from time import time


class RNN(nn.Module):
    '''Class for training and analysing RNNs.
       Automatically loads model if model name exists in directory.
       N_CELL and N_STIM need to be intially defined.'''
    def __init__(self, dir, name, p=False):
        # Inheritance from torch.nn
        super(RNN,self).__init__()

        # Utility
        self.name = name
        self.dir = os.path.join(dir, name)
        try:
            os.makedirs(self.dir, exist_ok=False)
        except FileExistsError:
            if os.path.isfile(os.path.join(self.dir, f'{name}.pt')):
                if p:
                    print(f"Model {name} already exists. Loading model...")
                self.load_model(name, p=p)
            else:
                if p:
                    print(f"Directory {name} exists but no model found. Proceeding with new model?")


    def hyp(self, task='dms', N_Models=1, activation='relu', lr=0.001, num_epochs=1000, reg=0.0001, N_CELL=10, N_STIM=2, w_var=0.1, rank=None):
        '''Set hyperparameters'''
        # Model parameters
        self.N_cell = N_CELL
        self.N_stim = N_STIM
        self.w_var = w_var
        w_std = np.sqrt(w_var)
        self.N_Models = N_Models

        # Recurrent parameters
        self.rank = rank
        if rank:
            self.rec_m = torch.nn.Parameter(torch.FloatTensor(N_Models, N_CELL, rank).normal_(0,np.sqrt(0.1)))
            self.rec_n = torch.nn.Parameter(torch.FloatTensor(N_Models, rank, N_CELL).normal_(0,np.sqrt(0.1)))
        else:
            self.rec_weights = torch.nn.Parameter(torch.FloatTensor(N_Models, N_CELL, N_CELL).normal_(0, 0.1))
        self.rec_biases  = torch.nn.Parameter(torch.FloatTensor(N_Models, N_CELL, 1).normal_(0, 0.1))

        # Input and output weights
        self.inp_weights = torch.nn.Parameter(torch.FloatTensor(N_Models, N_CELL, N_STIM).normal_(0, w_std))
        self.out_weights = torch.nn.Parameter(torch.FloatTensor(N_Models, 2, N_CELL).normal_(0, 0.1))
        self.mem_weights = torch.nn.Parameter(torch.FloatTensor(N_Models, 2, N_CELL).normal_(0, 0.1))

        # Hyperparameters
        self.activation = activation
        self.task = task

        # Constants
        self.dt = 0.02      # 20 ms
        self.tau = 0.1      # 100 ms
        self.noise = 0.1
        self.T_cycle = 50   # int(0.5/self.dt)

        # Training and Analysis
        self.training_losses = []       
        self.learning_rate = lr
        self.num_epochs = num_epochs
        self.reg_hyp = reg      # Regularisation hyperparameter
        self.activities = []        # r(t)
        self.drs = []               # dr(t)

        # Set task
        self.set_task()

        # Move to GPU 
        self.to_gpu()




    '''Task Infrastructure'''
    def set_task(self, task='dms'):
        assert type(task) == str, "Definition of task must be a string."
        self.task = task
        if task == 'dms':
            self.dms_task_epochs(rand=False)

    def dms_task_epochs(self, rand=False):
        '''Set epochs for delayed match to sample task'''
        self.fixation_len = self.get_epoch(self.T_cycle, rand=False, interval=20)
        self.sample_len =self.get_epoch(self.T_cycle, rand=False, interval=20)
        self.delay_len = self.get_epoch(self.T_cycle*2, rand, interval=40)
        self.test_len = self.get_epoch(self.T_cycle, rand=False, interval=20)
        self.response_len = self.get_epoch(self.T_cycle, rand=False, interval=20)
        self.run_len = self.fixation_len + self.sample_len + self.delay_len + self.test_len + self.response_len

    def get_epoch(self, time, rand, interval=20):
        '''Get the number of timesteps of an epochs for a task'''
        if rand:
            return np.random.randint(time-interval, time+interval)
        else:
            return time

    def phi(self, r, t=True):
        '''Activation function selection'''
        if t:
            if self.activation == 'relu':
                return torch.relu(r)
            elif self.activation == 'tanh':
                return torch.tanh(r)
            elif self.activation == 'ssn':
                return (torch.relu(r))**2
            else:
                raise ValueError(f"Unknown activation function: {self.activation}")
        else:
            if self.activation == 'relu':
                return np.maximum(0, r)
            elif self.activation == 'tanh':
                return np.tanh(r)
            elif self.activation == 'ssn':
                return (np.maximum(0, r))**2
            else:
                raise ValueError(f"Unknown activation function: {self.activation}")


    def forward(self, x):
        '''Forward pass, directing to all tasks.'''
        x = x.float()
        if self.rank:
            self.rec_weights = self.rec_m @ self.rec_n
        if self.task == 'dms':
            return self.forward_dms(x)
        else:
            raise ValueError("Task not defined.")

    def forward_dms(self, x):
        '''Forward pass for delayed match to sample task.
        Expected input x to be of dimensions: [2, N_stim]'''
        # Check input dimensions, Extract sample and test stimuli
        assert x.shape == (4, 2, self.N_stim), "Wrong input dimensions."
        x_sample = x[:, 0, :].unsqueeze(-1).unsqueeze(1) #[4, 1, self.N_stim,1]
        x_test = x[:, 1, :].unsqueeze(-1).unsqueeze(1) #[4, 1, self.N_stim,1]

        # Init firing rates matrix, dt/tau
        r = torch.zeros([4, self.N_Models, self.N_cell, 1], device=self.device)
        dr = torch.zeros([4, self.N_Models, self.N_cell, 1], device=self.device)
        c = self.dt / self.tau

        # Reset Lists to record activity metrics
        self.activities = []             # r(t)
        self.drs = []                    # dr(t)

        # Record function
        def record_data():
            self.activities.append(r.clone())
            self.drs.append(dr.clone())

        def noise():
            return self.noise*torch.randn([self.N_Models, self.N_cell,1], device=self.device)

        # Fixation
        for t in range(self.fixation_len):
            dr = (-r + self.phi(self.rec_weights@r + self.rec_biases) + noise())*c # Noise
            r = r + dr
            record_data()

        # Training on Sample Stimulus
        for t in range(self.sample_len):
            dr = (-r + self.phi(self.rec_weights@r + self.rec_biases + self.inp_weights@x_sample) + noise())*c
            r = r + dr
            record_data()

        rm = torch.zeros([self.delay_len, 4, self.N_Models, 2], dtype=float, device=self.device)
        # Delay
        for t in range(self.delay_len):
            dr = (-r + self.phi(self.rec_weights@r + self.rec_biases) + noise())*c # Noise
            r = r + dr
            rm[t] = torch.squeeze(self.mem_weights@r, dim=-1)
            record_data()

        # Training on Test Stimulus
        for t in range(self.test_len):
            dr = (-r + self.phi(self.rec_weights@r + self.rec_biases + self.inp_weights@x_test) + noise())*c
            r = r + dr
            record_data()

        # Response
        rs = torch.zeros([self.response_len, 4, self.N_Models, 2], dtype=float, device=self.device)
        for t in range(self.response_len):
            dr = (-r + self.phi(self.rec_weights@r + self.rec_biases) + noise())*c
            r = r + dr
            rs[t] = torch.squeeze(self.out_weights@r, dim=-1)
            record_data()


        # Average response over time is returned
        self.activities = torch.squeeze(torch.stack(self.activities))
        self.drs = torch.squeeze(torch.stack(self.drs))

        return rm, rs   # dim: [timesteps, cases, N_models, label logits]




    '''Training and Analysis Methods'''
    def train_model(self, train_data, train_labels, p=True):
        if p:
            print("Initializing model training...")

        # Optimizer and Loss
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        best_loss = np.inf
        save_delay = 20

        # Move to GPU
        train_data = train_data.to(self.device)
        train_labels = train_labels.to(self.device)
        memory_labels = train_labels[:,0,0].long()
        task_labels = train_labels[:,1,0].long()
        self.train()
 
        self.acc = []
        start_time = time()
        for epoch in range(self.num_epochs):
            # Randomize the task epochs
            self.dms_task_epochs(rand=True)

            optimizer.zero_grad()
            rm, rs = self.forward(train_data)
            loss_mem = loss_fn(rm.permute(1, 3, 0, 2), memory_labels.view(4, 1, 1).expand(-1, rm.shape[0], self.N_Models))
            loss_pred = loss_fn(rs.permute(1, 3, 0, 2), task_labels.view(4, 1, 1).expand(-1, rs.shape[0], self.N_Models))
            loss = loss_mem + loss_pred
            loss = loss + self.reg_hyp*torch.mean(self.activities ** 2)
            loss.backward()
            optimizer.step()
            total_loss = loss.item()

            self.training_losses.append(total_loss)
            self.acc.append(torch.mean(self.test(train_data, train_labels, p=False)).cpu().numpy())
            save_delay -= 1

            # Progress bar
            progress_bar(self.num_epochs, epoch, start_time, f"Total: {total_loss:.4f} Mem: {loss_mem.item():.4f} Task:{loss_pred.item():.4f}")

            # Save Better Models (with delay)
            if (total_loss<best_loss) and (save_delay<0):
                best_loss = total_loss
                self.save_model()
                save_delay = 20

                # Early exit
                # if self.acc[-1] == 1.0 and total_loss < 0.2:
                #     break

        # Reset task epochs
        self.dms_task_epochs(rand=False)
        print()
        if p:
            print('\nTraining Complete. \n')


    def test(self, test_data, test_labels, p=True, t=1):
        """Test the model and print accuracy and confusion matrix."""
        if test_labels.shape == torch.Size([4, 2, 2]):
            labels = test_labels[:,1,0].long()
        self.eval()
        trials = t
        rs_all = torch.zeros([trials, self.response_len, 4, self.N_Models, 2], device=self.device)
        for i in range(trials):
            if p:
                progress_bar(trials, i, time(), f"Trialing (Test)...")
            _, rs_all[i] = self.forward(test_data)
            

        with torch.no_grad():
            # Get predictions
            predictions = torch.argmax(torch.softmax(rs_all, dim=-1), dim=-1).float() # [trials, timesteps, cases, models]
            predictions = torch.mean(predictions, axis=0)
            labels = labels.view(1, 4, 1).repeat(self.response_len, 1, self.N_Models)

            # Calculate accuracy
            correct_predictions = torch.eq(predictions.int(), labels).sum(dim=(0,1))
            accuracy = correct_predictions / (4*self.response_len)

            # Print results
            labels = test_labels[:,1,0].long()
            if p:
                print(f'Successfully trained models = {torch.count_nonzero(accuracy == 1)}/{len(accuracy)}')
                print(f"Average accuracy: {torch.round(torch.mean(accuracy) * 100).item()}%")
                print("Correct Label | Average Predicted Label")
                print("--------------|------------------------")
                for i in range(4):
                    print(f"{int(labels[i])}             | {round(torch.mean(predictions[:,i,:]).item(),2)}")
        return accuracy


    def pca(self, activities):
        '''Performs PCA on the activities, assumes forward pass is completed'''
        # Concatenate activities along different stimuli iterations
        activities = np.concatenate([activities[:, i, :].detach().numpy() 
                                     for i in range(activities.shape[1])], axis=0)
        # PCA
        pca = PCA()
        principalComponents = pca.fit_transform(activities)

        return principalComponents



    '''Plotting Methods'''
    def plot_training_loss(self):
        fig, ax1 = plt.subplots()
        ax1.plot(np.arange(len(self.training_losses)), self.training_losses, 'b-')
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss", color='b')
        ax1.tick_params('y', colors='b')
        ax1.set_ylim(0, 1.5)
        ax2 = ax1.twinx()
        ax2.plot(np.arange(len(self.acc)), self.acc, linestyle='--', color='orange')
        ax2.set_ylabel("accuracy", color='orange')
        ax2.tick_params('y', colors='orange')
        ax1.spines['top'].set_visible(False) 
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False) 
        ax2.spines['left'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir, f"{self.name}_training_loss.png"), format='png')
        plt.close()


    def plot_abs_activity(self, inds, stimuli):
        """Plot the absolute value of neural activities across time for each task."""
        for ind in inds:
            try:
                os.makedirs(os.path.join(self.dir,f'Index_{ind}'), exist_ok=False)
            except FileExistsError:
                pass
            # Activities [Time, 4, N_Models, N_cell]
            if self.N_Models > 1:
                activities = torch.squeeze(self.activities[:, :, ind, :])
            else:
                activities = self.activities

            # Compute the absolute value of the activities
            abs_activities = torch.abs(activities).detach().numpy()

            # Define the splits
            splits = [0, self.fixation_len, self.sample_len, self.delay_len, self.test_len, self.response_len]
            cumulative_splits = np.cumsum(splits)

            # Create a 2x2 grid of subplots
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            fig.subplots_adjust(hspace=0.4, wspace=0.2)
            stim = self.stim_AB(stimuli)

            # Plot the absolute activities for each task
            for task in range(abs_activities.shape[1]):
                ax = axs[task // 2, task % 2]  # Select the subplot
                for neuron in range(abs_activities.shape[2]):
                    ax.plot(abs_activities[:, task, neuron], label=f"Neuron {neuron}")
                ax.set_title(f'Combination {stim[task]}', fontsize=14)
                ax.set_xlabel('Time', fontsize=14)
                ax.set_ylabel('Absolute Activity', fontsize=14)

                # Add vertical lines to split the task stages
                for split in cumulative_splits:
                    ax.axvline(x=split, color='lightgrey', linestyle='--')

            # Create a custom legend
            lines = [mlines.Line2D([], [], color='C'+str(i), label=f'Neuron {i}') for i in range(abs_activities.shape[2])]
            fig.legend(handles=lines, loc='lower right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.dir,f'Index_{ind}',f"{self.name}_abs_activities_index_{ind}.png"), format='png')
            plt.close()


    def plot_drs(self, inds, stimuli):
        """Plot the absolute value of neural activities across time for each task."""
        for ind in inds:
            try:
                os.makedirs(os.path.join(self.dir,f'Index_{ind}'), exist_ok=False)
            except FileExistsError:
                pass
            if self.N_Models > 1:
                drs = torch.squeeze(self.drs[:, :, ind, :])
            else:
                drs = self.drs

            # Compute the absolute value of the activities
            drs = drs.detach().numpy()

            # Define the splits
            splits = [0, self.fixation_len, self.sample_len, self.delay_len, self.test_len, self.response_len]
            cumulative_splits = np.cumsum(splits)

            # Create a 2x2 grid of subplots
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            fig.subplots_adjust(hspace=0.4, wspace=0.2)
            stim = self.stim_AB(stimuli)

            # Plot the absolute activities for each task
            for task in range(drs.shape[1]):
                ax = axs[task // 2, task % 2]  # Select the subplot
                for neuron in range(drs.shape[2]):
                    ax.plot(drs[:, task, neuron], label=f"Neuron {neuron}")
                ax.set_title(f'Combination {stim[task]}', fontsize=14)
                ax.set_xlabel('Time', fontsize=14)
                ax.set_ylabel('Neural Activity Gradients', fontsize=14)

                # Add vertical lines to split the task stages
                for split in cumulative_splits:
                    ax.axvline(x=split, color='lightgrey', linestyle='--')

            # Create a custom legend
            lines = [mlines.Line2D([], [], color='C'+str(i), label=f'Neuron {i}') for i in range(drs.shape[2])]
            fig.legend(handles=lines, loc='lower right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.dir, f'Index_{ind}', f"{self.name}_drs_index_{ind}.png"), format='png')
            plt.close()

    def plot_PCAs(self, inds, stimuli):
        '''Plot the gradient flow in PC1-PC2 space.'''
        # Constants
        cs = ['red', 'cyan', 'palegreen', 'plum']
        self.dms_task_epochs(rand=False)
        trials = 100
        uall_models = np.zeros([self.run_len, 4, self.N_Models, self.N_cell, trials])
        for i in range(trials):
            self.forward(stimuli)
            uall_models[:,:,:,:,i] = torch.squeeze(self.activities).cpu().detach().numpy()

        for ind in inds:
            try:
                os.makedirs(os.path.join(self.dir,f'Index_{ind}'), exist_ok=False)
            except FileExistsError:
                pass
            # Model-specific weights and biases
            c = self.dt / self.tau
            W = np.squeeze(self.rec_weights[ind,:,:].cpu().detach().numpy())
            b = np.squeeze(self.rec_biases[ind,:,:].cpu().detach().numpy())
            Win = np.squeeze(self.inp_weights[ind,:,:].cpu().detach().numpy())
            Wout = np.squeeze(self.out_weights[ind,:,:].cpu().detach().numpy())
            uall = uall_models[:,:,ind,:,:]
           
            # PCA
            cov = np.cov(np.transpose(uall,[0,1,3,2]).reshape(-1,self.N_cell).T)
            w, v = np.linalg.eig(cov)
            vinv = np.linalg.inv(v) #to convert from neuron space to PC space
            pcspace = (vinv@(uall))
            pcspace = np.mean(pcspace, axis=3) #(time, cases, PCs)
            pcmean = np.mean(pcspace,axis=(0,1))

            ubase = np.zeros([self.N_cell])
            for pc in range(2,self.N_cell):
                ubase += pcmean[pc] * v[:,pc]

            # Compute min and max for axes limits
            pc1_min, pc1_max = np.min(pcspace[:,:, 0])-0.5, np.max(pcspace[:,:, 0])+0.5
            pc2_min, pc2_max = np.min(pcspace[:,:, 1])-0.5, np.max(pcspace[:,:, 1])+0.5
            pc1axis = np.linspace(pc1_min,pc1_max,100)
            pc2axis = np.linspace(pc2_min,pc2_max,100)

            '''% variance explained'''
            explained_variance = np.round(w/np.sum(w), decimals=3)*100
            plt.plot(np.arange(1, self.N_cell+1), explained_variance, 'o-')
            ax = plt.gca()
            ax.spines['top'].set_visible(False) 
            ax.spines['right'].set_visible(False)
            plt.xlabel('PC Index')
            plt.ylabel(r'% Explained Variance')
            plt.savefig(os.path.join(self.dir,f'Index_{ind}',f"{self.name}_explained_variance_{ind}.png"), format='png')
            plt.clf()

            def log_abs_grad(h):
                absgradplot = np.zeros([100,100])
                for i in range(100):
                    for j in range(100):
                        pc1 = pc1axis[i]
                        pc2 = pc2axis[j]
                        u = ubase + pc1*v[:,0] + pc2*v[:,1]
                        dudt = c*(-u + self.phi(W@u + b, t=False)+ Win@h)
                        absdudt = np.sqrt(np.sum(np.square(dudt)))
                        absgradplot[i,j] = absdudt
                return np.log(absgradplot)
            abs_grad_cases = np.array([log_abs_grad(np.array([0,1])),      # A
                                       log_abs_grad(np.array([1,0])),      # B
                                       log_abs_grad(np.array([0,0])),])    # -
            
            # '''Decision Boundary'''
            # predictions = np.zeros([100,100])
            # for i in range(100):
            #     for j in range(100):
            #         pc1 = pc1axis[i]
            #         pc2 = pc2axis[j]
            #         u = pc1*v[:,0] + pc2*v[:,1]
            #         output = Wout@u
            #         predictions[i,j] = np.argmax(output)


            '''PCA Plot'''
            # Create figure with 4 rows and 5 columns
            fig, axes = plt.subplots(4, 5, figsize=(12, 8), sharex=True, sharey=True)

            # Plotting parameters
            stages = ["Fixation", "Sample", "Delay", "Test", "Response"]
            splits = [0, self.fixation_len, self.sample_len, self.delay_len, self.test_len, self.response_len]
            cumulative_splits = np.cumsum(splits)

            # Task titles
            tasks = self.stim_AB(stimuli)

            for trial in range(4):
                trial_pc = pcspace[:,trial,:]
                for idx, stage in enumerate(stages):
                    ax = axes[trial, idx]
                    stage_start = cumulative_splits[idx]
                    stage_end = cumulative_splits[idx+1]
                    x = trial_pc[stage_start:stage_end, 0]
                    y = trial_pc[stage_start:stage_end, 1]
                    ax.plot(x, y, c=cs[trial], linewidth=2)
                    arrow = FancyArrow(x[-2], y[-2], x[-1]-x[-2], y[-1]-y[-2], color=cs[trial],
                           shape='full', lw=1, length_includes_head=True, head_width=1.2, fill=False)
                    ax.add_patch(arrow)
                    ax.set_xlim(pc1_min, pc1_max)
                    ax.set_ylim(pc2_min, pc2_max)
                    ax.set_title(stage if trial == 0 else "")

                    if stage == 'Fixation' or stage == 'Delay' or stage == 'Response':
                        absgradplot=abs_grad_cases[2]; s = '-'
                    elif stage == 'Sample' and trial < 2:
                        absgradplot=abs_grad_cases[0]; s = "A"
                    elif stage == 'Sample' and trial >= 2:
                        absgradplot=abs_grad_cases[1]; s = "B"
                    elif stage == 'Test' and trial%2 == 0:
                        absgradplot=abs_grad_cases[0]; s = "A"
                    elif stage == 'Test' and trial%2 == 1:
                        absgradplot=abs_grad_cases[1]; s = "B"
                  
                    im = ax.imshow(np.flipud(absgradplot.T), extent=[pc1_min, pc1_max, pc2_min, pc2_max], 
                              aspect='auto', vmin=np.min(abs_grad_cases), 
                              vmax=np.max(abs_grad_cases))
                    ax.set_xlabel(str(s)) 

                    # # Decision boundary
                    # if stage == 'Response':
                    #     xx, yy = np.meshgrid(pc1axis, pc2axis)
                    #     ax.contour(xx, yy, predictions, levels=[0.5], colors='black')


                # Set stimulus labels
                axes[trial, 0].set_ylabel(tasks[trial], rotation=0, labelpad=20, fontsize=16)

            # Set common labels
            fig.text(0.5, 0.02, 'PC1', ha='center', fontsize=15)
            fig.text(0.02, 0.5, 'PC2', va='center', fontsize=15)

            # Add legend
            handles, labels = axes[0, -1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), location='right')
            cbar.set_label('log|dr/dt|', rotation=0, labelpad=20, loc='center', fontsize=15)
            plt.savefig(os.path.join(self.dir,f'Index_{ind}',f"{self.name}_pca_trajectories_index_av_{ind}.png"), format='png')
            plt.close()

            '''Scatter Plot to justify approximation'''
            approx_pc = np.zeros([self.run_len, 4, self.N_cell])
            approx_pc[:,:,0:2] = pcspace[:,:,0:2]
            approx_pc[:,:,2:] = pcmean[None, None, 2:]
            approx_u = approx_pc @ v.T
            plt.scatter(approx_u[self.fixation_len:,:,:], np.mean(uall, axis=3)[self.fixation_len:,:,:], s=1)
            ax = plt.gca()
            ax.spines['top'].set_visible(False) 
            ax.spines['right'].set_visible(False)
            axis_range = [0, max(np.max(approx_u[self.fixation_len:,:,:]), np.max(np.mean(uall, axis=3)[self.fixation_len:,:,:]))]
            plt.plot(axis_range, axis_range, 'k--')
            plt.xlabel('approximated neuron activities by PC averaging')
            plt.ylabel('recorded neuron activities')
            plt.axis('equal')
            plt.savefig(os.path.join(self.dir,f'Index_{ind}',f"{self.name}_averaged_pc_approx_{ind}.png"), format='png')
            plt.close()

            '''Difference of true vs approx over time'''
            diff_av = np.mean(uall, axis=3) - approx_u
            diff_av2 = np.mean(np.abs(diff_av), axis=(1,2))
            plt.scatter(np.arange(self.run_len), diff_av2, s=1)
            ax = plt.gca()
            ax.spines['top'].set_visible(False) 
            ax.spines['right'].set_visible(False)
            for split in cumulative_splits:
                ax.axvline(x=split, color='lightgrey', linestyle='--')
            plt.xlabel('timesteps')
            plt.ylabel('averaged estimation error')
            plt.savefig(os.path.join(self.dir,f'Index_{ind}',f"{self.name}_averaged_pc_over_time_{ind}.png"), format='png')
            plt.close()

            '''Plot the gradient flow in PC1-PC2 space. Truncated PC approximation.'''
            def log_abs_grad2(h):
                absgradplot = np.zeros([100,100])
                for i in range(100):
                    for j in range(100):
                        pc1 = pc1axis[i]
                        pc2 = pc2axis[j]
                        u = pc1*v[:,0] + pc2*v[:,1]
                        dudt = c*(-u + self.phi(W@u + b, t=False)+ Win@h)
                        absdudt = np.sqrt(np.sum(np.square(dudt)))
                        absgradplot[i,j] = absdudt
                return np.log(absgradplot)
            abs_grad_cases = np.array([log_abs_grad2(np.array([0,1])),      # A
                                       log_abs_grad2(np.array([1,0])),      # B
                                       log_abs_grad2(np.array([0,0])),])    # -


            '''PCA Plot'''
            # Create figure with 4 rows and 5 columns
            fig, axes = plt.subplots(4, 5, figsize=(12, 8), sharex=True, sharey=True)

            for trial in range(4):
                trial_pc = pcspace[:,trial,:]
                for idx, stage in enumerate(stages):
                    ax = axes[trial, idx]
                    stage_start = cumulative_splits[idx]
                    stage_end = cumulative_splits[idx+1]
                    x = trial_pc[stage_start:stage_end, 0]
                    y = trial_pc[stage_start:stage_end, 1]
                    ax.plot(x, y, c=cs[trial], linewidth=2)
                    arrow = FancyArrow(x[-2], y[-2], x[-1]-x[-2], y[-1]-y[-2], color=cs[trial],
                           shape='full', lw=1, length_includes_head=True, head_width=1.2, fill=False)
                    ax.add_patch(arrow)
                    ax.set_xlim(pc1_min, pc1_max)
                    ax.set_ylim(pc2_min, pc2_max)
                    ax.set_title(stage if trial == 0 else "")

                    if stage == 'Fixation' or stage == 'Delay' or stage == 'Response':
                        absgradplot=abs_grad_cases[2]; s = '-'
                    elif stage == 'Sample' and trial < 2:
                        absgradplot=abs_grad_cases[0]; s = "A"
                    elif stage == 'Sample' and trial >= 2:
                        absgradplot=abs_grad_cases[1]; s = "B"
                    elif stage == 'Test' and trial%2 == 0:
                        absgradplot=abs_grad_cases[0]; s = "A"
                    elif stage == 'Test' and trial%2 == 1:
                        absgradplot=abs_grad_cases[1]; s = "B"
                  
                    im = ax.imshow(np.flipud(absgradplot.T), extent=[pc1_min, pc1_max, pc2_min, pc2_max], 
                              aspect='auto', vmin=np.min(abs_grad_cases), 
                              vmax=np.max(abs_grad_cases))
                    ax.set_xlabel(str(s)) 

                # Set stimulus labels
                axes[trial, 0].set_ylabel(tasks[trial], rotation=0, labelpad=20, fontsize=16)

            # Set common labels
            fig.text(0.5, 0.02, 'PC1', ha='center', fontsize=15)
            fig.text(0.02, 0.5, 'PC2', va='center', fontsize=15)

            # Add legend
            handles, labels = axes[0, -1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), location='right')
            cbar.set_label('log|dr/dt|', rotation=0, labelpad=20, loc='center', fontsize=15)
            plt.savefig(os.path.join(self.dir,f'Index_{ind}',f"{self.name}_pca_trajectories_index_tr_{ind}.png"), format='png')
            plt.close()

            '''Scatter Plot to justify approximation'''
            approx_pc = np.zeros([self.run_len, 4, self.N_cell])
            approx_pc[:,:,0:2] = pcspace[:,:,0:2]
            approx_u = approx_pc @ v.T
            plt.scatter(approx_u[self.fixation_len:,:,:], np.mean(uall, axis=3)[self.fixation_len:,:,:], s=1)
            ax = plt.gca()
            ax.spines['top'].set_visible(False) 
            ax.spines['right'].set_visible(False)
            axis_range = [0, max(np.max(approx_u[self.fixation_len:,:,:]), np.max(np.mean(uall, axis=3)[self.fixation_len:,:,:]))]
            plt.plot(axis_range, axis_range, 'k--')
            plt.xlabel('approximated neuron activities by PC truncation')
            plt.ylabel('recorded neuron activities')
            plt.axis('equal')
            plt.savefig(os.path.join(self.dir,f'Index_{ind}',f"{self.name}_truncated_pc_approx_{ind}.png"), format='png')
            plt.close()

            '''Difference of true vs approx over time'''
            diff_tr = np.mean(uall, axis=3) - approx_u
            diff_tr2 = np.mean(np.abs(diff_tr), axis=(1,2))
            plt.scatter(np.arange(self.run_len), diff_tr2, s=1)
            ax = plt.gca()
            ax.spines['top'].set_visible(False) 
            ax.spines['right'].set_visible(False)
            for split in cumulative_splits:
                ax.axvline(x=split, color='lightgrey', linestyle='--')
            plt.xlabel('timesteps')
            plt.ylabel('averaged estimation error')
            plt.savefig(os.path.join(self.dir,f'Index_{ind}',f"{self.name}_truncated_pc_over_time_{ind}.png"), format='png')
            plt.close()

            '''Scatter of truncate vs averaged error'''
            plt.scatter(np.abs(diff_av[self.fixation_len:]), np.abs(diff_tr[self.fixation_len:]), s=1)
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            axis_range = [0, max(np.max(diff_av[self.fixation_len:]), np.max(diff_tr[self.fixation_len:]))]
            plt.plot(axis_range, axis_range, 'k--')
            plt.xlabel('averaged error')
            plt.ylabel('truncated error')
            plt.axis('equal')
            plt.savefig(os.path.join(self.dir,f'Index_{ind}',f"{self.name}_averaged_vs_truncated_{ind}.png"), format='png')
            plt.close()
            
    
    def participation_ratio(self, stimuli, labels, p=False):
        '''Calculate the Participation Ratio for each model and plots histogram.
        Only for successfully trained models.'''
        # Test model
        acc = self.test(stimuli, labels, p=p, t=100)
        indices = torch.nonzero(acc.eq(1), as_tuple=True)[0]

        # Collect trial data
        self.dms_task_epochs(rand=False)
        trials = 100
        uall_models = np.zeros([self.run_len, 4, self.N_Models, self.N_cell, trials])
        for i in range(trials):
            if p:
                progress_bar(trials, i, time(), f"Trialing (Participation Ratio)...")
            self.forward(stimuli)
            uall_models[:,:,:,:,i] = torch.squeeze(self.activities).cpu().detach().numpy()

        # Calculate PCA and Participation Ratio (PR)
        PRs = np.zeros(self.N_Models)
        for ind in indices:
            uall = np.squeeze(uall_models[:,:,ind,:,:])
            cov = np.cov(np.transpose(uall,[0,1,3,2]).reshape(-1,self.N_cell).T)
            w, v = np.linalg.eig(cov)
            PRs[ind] = (np.sum(w) ** 2) / np.sum(w ** 2)
        
        # Plot Participation ratios by index
        plt.bar(np.arange(self.N_Models), PRs)
        plt.xlabel('Model Index')
        plt.ylabel('Participation Ratio')
        plt.savefig(os.path.join(self.dir, f"{self.name}_participation_ratio.png"), format='png')
        plt.close()

        # Histogram of participation ratios
        non_zero_PRs = PRs[PRs != 0]
        plt.hist(non_zero_PRs, bins=10)
        plt.xlabel('Participation Ratio')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.dir, f"{self.name}_participation_ratio_hist.png"), format='png')
        plt.close()

        # Min, Median, Max participation ratio plots
        sorted_indices = np.argsort(non_zero_PRs)
        inds = [sorted_indices[0], sorted_indices[len(sorted_indices) // 2], sorted_indices[-1]]
        self.plot_PCAs(inds, stimuli)
        # Activity Plots
        self.plot_abs_activity(inds, stimuli)
        self.plot_drs(inds, stimuli)

            

    '''Utility Functions'''
    def to_gpu(self):
        # At this training scale CPU > GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for name, param in self.named_parameters():
            param.data = param.data.to(device)
        for name, buffer in self.named_buffers():
            buffer.data = buffer.data.to(device)
        self.device = device
        print(f'Running on {device}')

    def save_model(self):
        """Save the model's state_dict and a description at the specified path."""
        description = (
        f'{self.w_var}_wvar / {self.reg_hyp}_reg / {self.activation}_activation / '
        f'{self.num_epochs}_epochs / {self.learning_rate}_rate / {self.N_stim}D_DMS / '
        f'{self.N_cell}_cells / {self.N_Models}_models / {self.rank}_rank / '
        f'\n@ {len(self.training_losses)}/{self.num_epochs} training steps, loss = {self.training_losses[-1]}')
        torch.save({
            'model_state_dict': self.state_dict(),
            'description': description,
        }, os.path.join(self.dir, f'{self.name}.pt'))
        self.description = description
        self.plot_training_loss()

    def load_model(self, name, p=False):
        """Load the model's state_dict and a description from the specified path.
        Load n_cell, activation, n_stim, N_Models, and rank."""
        checkpoint = torch.load(os.path.join(self.dir, f'{name}.pt'))
        self.description = checkpoint.get('description', '')
        if p:
            print(f"\nLoading {name}...")
            print(f"Accessing {self.dir}...")
            print(f"Description: \n{self.description}\n")
        parts = self.description.split('/')
        parts = [part.strip() for part in parts]
        w_var = float(parts[0].split('_')[0])
        reg_hyp = float(parts[1].split('_')[0])
        activation = parts[2].split('_')[0]
        num_epochs = int(parts[3].split('_')[0])
        learning_rate = float(parts[4].split('_')[0])
        self.N_stim = int(parts[5].split('D')[0])
        self.N_cell = int(parts[6].split('_')[0])
        N_Models = int(parts[7].split('_')[0])
        rank = None if parts[8].split('_')[0] == 'None' else int(parts[8].split('_')[0])
        self.hyp(N_Models=N_Models, N_CELL=self.N_cell, activation=activation, lr=learning_rate, num_epochs=num_epochs, reg=reg_hyp, w_var=w_var, rank=rank)
        self.load_state_dict(checkpoint['model_state_dict'])

    def del_model(self, dir, name, p=True):
        """Delete the model's directory."""
        assert self.dir == os.path.join(dir, name), "Model directory does not match."
        shutil.rmtree(self.dir)
        if p:
            print(f"{name} deleted from {dir}.")

    def print_model_parameters(self):
        """Print the model's parameters."""
        for name, param in self.named_parameters():
            print(f"{name}:")
            print(param.data)

    def stim_AB(self, stim):
        '''Returns the stimulus A or B'''
        label = []
        for i in range(stim.shape[0]):
            pair = []
            for j in range(stim.shape[1]):
                if stim[i,j,0] == 0:
                    pair.append('A')
                elif stim[i,j,0] == 1:
                    pair.append('B')
            label.append("".join(pair))
        return label





def progress_bar(total, current, start_time, info=""):
    elapsed_time = time() - start_time
    bar_length = 40
    progress = int(current+1) / int(total)
    arrow = '-' * int(round(progress * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    total_time = elapsed_time / progress
    hours, remainder = divmod(total_time - elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_string = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    print('\r[{}] {}/{} - ETA: {}. {}'.format(arrow + spaces, current+1, total, time_string, info), end='', flush=True)

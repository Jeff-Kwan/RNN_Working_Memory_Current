'''
Draft for RNN class for training delayed match to sample task
'''

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics import accuracy_score
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


    def hyp(self, task='dms', activation='relu', lr=0.001, num_epochs=1000, reg=0.0001, N_CELL=10, N_STIM=2, w_var=0.1):
        '''Set hyperparameters'''
        self.N_cell = N_CELL
        self.N_stim = N_STIM
        self.w_var = w_var
        w_std = np.sqrt(w_var)

        # Recurrent parameters
        self.rec_weights = torch.nn.Parameter(torch.FloatTensor(N_CELL, N_CELL).uniform_(-0.01,0.01))
        self.rec_biases  = torch.nn.Parameter(torch.FloatTensor(N_CELL, 1).uniform_(-0.01,0.01))

        # Input and output weights
        self.inp_weights = torch.nn.Parameter(torch.FloatTensor(N_CELL, N_STIM).uniform_(-w_std,w_std))
        self.out_weights = torch.nn.Parameter(torch.FloatTensor(2, N_CELL).uniform_(-0.01,0.01))
        self.mem_weights = torch.nn.Parameter(torch.FloatTensor(2, N_CELL).uniform_(-0.01,0.01))

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
        self.fixation_len = self.get_epoch(self.T_cycle, rand)
        self.sample_len =self.get_epoch(self.T_cycle, rand)
        self.delay_len = self.get_epoch(self.T_cycle, rand)
        self.test_len = self.get_epoch(self.T_cycle, rand)
        self.response_len = self.get_epoch(self.T_cycle, rand)

    def get_epoch(self, time, rand):
        '''Get the number of timesteps of an epochs for a task'''
        if rand:
            return np.random.randint(time-20, time+20)
        else:
            return time

    def phi(self, r):
        '''Activation function selection'''
        if self.activation == 'relu':
            return torch.relu(r)
        elif self.activation == 'tanh':
            return torch.tanh(r)
        elif self.activation == 'ssn':
            return (torch.relu(r))**2
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")


    def forward(self, x):
        '''Forward pass, directing to all tasks.'''
        x = x.float()
        if self.task == 'dms':
            return self.forward_dms(x)
        else:
            raise ValueError("Task not defined.")

    def forward_dms(self, x):
        '''Forward pass for delayed match to sample task with a residual network.
        Expected input x to be of dimensions: [2, N_stim]'''
        # Check input dimensions, Extract sample and test stimuli
        assert x.shape == (4, 2, self.N_stim), "Wrong input dimensions."
        x_sample = x[:, 0, :].unsqueeze(2) #[4,self.N_stim,1]
        x_test = x[:, 1, :].unsqueeze(2) #[4,self.N_stim,1]

        # Init firing rates matrix, dt/tau
        r = torch.zeros([4, self.N_cell, 1])
        r = r.to(self.device)
        c = self.dt / self.tau

        # Reset Lists to record activity metrics
        self.activities = []             # r(t)
        self.drs = []                    # dr(t)

        # Record function
        def record_data():
            self.activities.append(r.clone())
            self.drs.append(dr.clone())

        # Fixation
        for t in range(self.fixation_len):
            dr = (-r + self.phi(self.rec_weights@r + self.rec_biases) + self.noise*torch.rand([self.N_cell,1]))*c # Noise
            r = r + dr
            record_data()

        # Training on Sample Stimulus
        for t in range(self.sample_len):
            dr = (-r + self.phi(self.rec_weights@r + self.rec_biases + self.inp_weights@x_sample) + self.noise*torch.rand([self.N_cell,1]))*c
            r = r + dr
            record_data()

        rm = torch.zeros([self.delay_len, 4, 2], dtype=float)
        # Delay
        for t in range(self.delay_len):
            dr = (-r + self.phi(self.rec_weights@r + self.rec_biases) + self.noise*torch.rand([self.N_cell,1]))*c # Noise
            r = r + dr
            rm[t] = torch.squeeze(self.mem_weights@r, dim=-1)
            record_data()

        # Training on Test Stimulus
        for t in range(self.test_len):
            dr = (-r + self.phi(self.rec_weights@r + self.rec_biases + self.inp_weights@x_test) + self.noise*torch.rand([self.N_cell,1]))*c
            r = r + dr
            record_data()

        # Response
        rs = torch.zeros([self.response_len, 4, 2], dtype=float)
        for t in range(self.response_len):
            dr = (-r + self.phi(self.rec_weights@r + self.rec_biases) + self.noise*torch.rand([self.N_cell,1]))*c
            r = r + dr
            rs[t] = torch.squeeze(self.out_weights@r, dim=-1)
            record_data()


        # Average response over time is returned
        # Output 
        output = torch.stack([torch.mean(rm, 0), torch.mean(rs, 0)])
        self.activities = torch.squeeze(torch.stack(self.activities))
        self.drs = torch.squeeze(torch.stack(self.drs))

        return output




    '''Training and Analysis Methods'''
    def train_model(self, train_data, train_labels, p=True):
        if p:
            print("Initializing model training...")

        # Optimizer and Loss
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        best_loss = np.inf
        save_delay = 10
        best_acc = 0

        # Move to GPU
        train_data = train_data.to(self.device)
        train_labels = train_labels.to(self.device)
        optimizer = optimizer.to(self.device)
        self.train()

        self.acc = []
        start_time = time()
        for epoch in range(self.num_epochs):
            total_loss = 0

            # Randomize the task epochs
            self.dms_task_epochs(rand=True)

            optimizer.zero_grad()
            output = self.forward(train_data)
            loss_mem = loss_fn(output[0], train_labels[:,0,:])
            loss_pred = loss_fn(output[1], train_labels[:,1,:])
            loss = loss_mem + loss_pred
            loss += self.reg_hyp*torch.sum(self.activities ** 2)
            loss.backward()
            optimizer.step()
            total_loss = loss.item()

            self.training_losses.append(total_loss)
            self.acc.append(self.test(train_data, train_labels, p=False))
            save_delay -= 1

            # Progress bar
            progress_bar(self.num_epochs, epoch, start_time, f"Loss: {total_loss:.4f}")

            # Save Better Models (with delay)
            if (total_loss<best_loss) and (save_delay<0) and (self.acc[-1]>=best_acc):
                best_loss = total_loss
                best_acc = self.acc[-1]
                self.save_model()
                save_delay = 20

                # Early exit
                if self.acc[-1] == 1.0 and total_loss < 0.2:
                    break

        # Reset task epochs
        self.dms_task_epochs(rand=False)

        if p:
            print('\nTraining Complete. \n')


    def pca(self):
        '''Performs PCA on the activities, assumes forward pass is completed'''
        # Concatenate activities along different stimuli iterations
        activities = np.concatenate([self.activities[:, i, :].detach().numpy() 
                                     for i in range(self.activities.shape[1])], axis=0)
        # PCA
        pca = PCA()
        principalComponents = pca.fit_transform(activities)

        return principalComponents


    def plot_pca_trajectories_2D(self, stimuli, title):
        '''Assumes 2D DMS task'''
        # PCA
        principalComponents = self.pca()

        # Create figure with 4 rows and 5 columns
        fig, axes = plt.subplots(4, 5, figsize=(12, 8), sharex=True, sharey=True)
        fig.suptitle(f"Trajectories in PC1-PC2 space for {title} task stages", fontsize=16)

        # Plotting parameters
        stages = ["Fixation", "Sample", "Delay", "Test", "Response"]
        splits = [0, self.fixation_len, self.sample_len, self.delay_len, self.test_len, self.response_len]
        cumulative_splits = np.cumsum(splits)
        div = np.cumsum([0] + [len(a) for a in [self.activities[:,0,:], self.activities[:,1,:],
                                                self.activities[:,2,:], self.activities[:,3,:]]])
        
        # Compute global min and max for axes limits
        pc1_min, pc1_max = np.min(principalComponents[:, 0]), np.max(principalComponents[:, 0])
        pc2_min, pc2_max = np.min(principalComponents[:, 1]), np.max(principalComponents[:, 1])

        # Task titles
        tasks = self.stim_AB(stimuli)

        for trial in range(4):
            start = div[trial]
            end = div[trial + 1]
            trial_pc = principalComponents[start:end]

            for idx, stage in enumerate(stages):
                ax = axes[trial, idx]
                stage_start = cumulative_splits[idx]
                stage_end = cumulative_splits[idx+1]
                ax.plot(trial_pc[stage_start:stage_end, 0], trial_pc[stage_start:stage_end, 1])
                ax.set_xlim(pc1_min, pc1_max)
                ax.set_ylim(pc2_min, pc2_max)
                ax.set_title(stage if trial == 0 else "")
                ax.scatter(trial_pc[stage_start, 0], trial_pc[stage_start, 1], color='green', label='Start', s=50, marker='o')
                ax.scatter(trial_pc[stage_end-1, 0], trial_pc[stage_end-1, 1], color='red', label='End', s=50, marker='x')

            # Set stimulus labels
            axes[trial, 0].set_ylabel(tasks[trial], rotation=0, labelpad=20, fontsize=16)

        # Set common labels
        fig.text(0.5, 0.02, 'PC1', ha='center', fontsize=15)
        fig.text(0.02, 0.5, 'PC2', va='center', rotation='vertical', fontsize=15)

        # Add legend
        handles, labels = axes[0, -1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)

        plt.savefig(os.path.join(self.dir, f"{self.name}_pca_trajectories.png"))


    def test(self, test_data, test_labels, p=True):
        """Test the model and print accuracy and confusion matrix."""
        self.eval()
        with torch.no_grad():
            # Forward pass
            output = self.forward(test_data)

            # The second item in output is used for evaluation
            predictions = np.round(torch.softmax(output[1], dim=1).numpy()).astype(int)[:,0]
            labels = test_labels[:,1,0].numpy().astype(int)

            # Calculate accuracy
            accuracy = accuracy_score(labels.flatten(), predictions.flatten())
            
            # Print results
            if p:
                print(f"Accuracy: {accuracy * 100:.2f}%")
                print("Correct Label | Predicted Label")
                print("--------------|----------------")
                for i in range(4):
                    print(f"{labels[i]}             | {predictions[i]}")
        return accuracy




    '''Plotting Methods'''
    def plot_training_loss(self):
        fig, ax1 = plt.subplots()
        ax1.plot(np.arange(len(self.training_losses)), self.training_losses, 'b-')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color='b')
        ax1.tick_params('y', colors='b')
        ax1.set_yscale('log')  # Set y-axis to log scale
        ax2 = ax1.twinx()
        ax2.plot(np.arange(len(self.acc)), self.acc, linestyle='--', color='orange')
        ax2.set_ylabel("Accuracy", color='orange')
        ax2.tick_params('y', colors='orange')
        plt.title("Training Loss and Accuracy over Epochs")
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir, f"{self.name}_training_loss.png"))
        plt.close()


    def plot_abs_activity(self, stimuli):
        """Plot the absolute value of neural activities across time for each task."""
        # Compute the absolute value of the activities
        abs_activities = torch.abs(self.activities).detach().numpy()

        # Define the splits
        splits = [0, self.fixation_len, self.sample_len, self.delay_len, self.test_len, self.response_len]
        cumulative_splits = np.cumsum(splits)

        # Create a 2x2 grid of subplots
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Absolute Neural Activities Across Time', fontsize=16)
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
        plt.savefig(os.path.join(self.dir, f"{self.name}_abs_activities.png"))
        plt.close()


    def plot_drs(self, stimuli):
        """Plot the absolute value of neural activities across time for each task."""
        # Compute the absolute value of the activities
        drs = self.drs.detach().numpy()

        # Define the splits
        splits = [0, self.fixation_len, self.sample_len, self.delay_len, self.test_len, self.response_len]
        cumulative_splits = np.cumsum(splits)

        # Create a 2x2 grid of subplots
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Neural Activity Gradients Across Time', fontsize=16)
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
        plt.savefig(os.path.join(self.dir, f"{self.name}_drs.png"))
        plt.close()


    def plot_gradient_field(self):
        """Plot the gradient field of the neural activity in PC1-PC2 space."""
        # PCA
        principalComponents = self.pca()
        pc1 = principalComponents[:, 0]
        pc2 = principalComponents[:, 1]

        # Compute the gradient of r in the PC1-PC2 space
        dr_dt = np.gradient(self.activities.detach().numpy(), axis=0)

        # Project the gradient onto the PC1-PC2 space
        dr_dt_pc1 = np.dot(dr_dt, pc1)
        dr_dt_pc2 = np.dot(dr_dt, pc2)

        # Compute the magnitude of the gradient
        magnitude = np.sqrt(dr_dt_pc1**2 + dr_dt_pc2**2)

        # Plot the gradient field
        plt.figure()
        plt.quiver(pc1, pc2, dr_dt_pc1, dr_dt_pc2, magnitude, cmap='viridis')
        plt.colorbar(label='dr/dt')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Gradient field of neural activity in PC1-PC2 space')
        plt.savefig(os.path.join(self.dir, f"{self.name}_gradient_field.png"))
        plt.close()


    def plot_gradient_flow(self):
        pass




    '''Utility Functions'''
    def to_gpu(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for name, param in self.named_parameters():
            param.data = param.data.to(device)
        for name, buffer in self.named_buffers():
            buffer.data = buffer.data.to(device)
        self.device = device
        #print(f'Running on {device}')


    def save_model(self):
        """Save the model's state_dict and a description at the specified path."""
        description = (
        f'{self.w_var}_wvar / {self.reg_hyp}_reg / {self.activation}_activation / '
        f'{self.num_epochs}_epochs / {self.learning_rate}_rate / {self.N_stim}D_DMS / '
        f'{self.N_cell}_cells / \n@ {len(self.training_losses)}/{self.num_epochs} '
        f'training steps, loss = {self.training_losses[-1]}')
        torch.save({
            'model_state_dict': self.state_dict(),
            'description': description,
        }, os.path.join(self.dir, f'{self.name}.pt'))
        self.plot_training_loss()

    def load_model(self, name, p=False):
        """Load the model's state_dict and a description from the specified path.
           Load n_cell, activation, n_stim."""
        checkpoint = torch.load(os.path.join(self.dir, f'{name}.pt'))
        self.description = checkpoint.get('description', '')
        if p:
            print(f"\nLoading {name}...")
            print(f"Accessing {self.dir}...")
            print(f"Description: \n{self.description}\n")
        parts = self.description.split('/')
        parts = [part.strip() for part in parts]
        self.w_var = float(parts[0].split('_')[0])
        self.reg_hyp = float(parts[1].split('_')[0])
        self.activation = parts[2].split('_')[0]
        self.num_epochs = int(parts[3].split('_')[0])
        self.learning_rate = float(parts[4].split('_')[0])
        self.N_stim = int(parts[5].split('D')[0])
        self.N_cell = int(parts[6].split('_')[0])
        self.hyp(task=self.N_stim, activation=self.activation, lr=self.learning_rate, num_epochs=self.num_epochs, reg=self.reg_hyp, w_var=self.w_var)
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





def progress_bar(total, current, start_time, info):
    elapsed_time = time() - start_time
    bar_length = 50
    progress = (current+1) / total
    arrow = '-' * int(round(progress * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    total_time = elapsed_time / progress
    hours, remainder = divmod(total_time - elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_string = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    print('\r[{}] {}/{} - ETA: {}. {}'.format(arrow + spaces, current+1, total, time_string, info), end='', flush=True)

import csv
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

def plot_random_trajectory(x1_trajectories, x2_trajectories, x3_trajectories, d_trajectories):
    trajectory_id = random.choice(range(len(x1_trajectories)))

    x1_values = x1_trajectories[trajectory_id]
    x2_values = x2_trajectories[trajectory_id]
    x3_values = x3_trajectories[trajectory_id]
    d_values = d_trajectories[trajectory_id]

    fig, axs = plt.subplots(4, 1, figsize=(10, 12))

    axs[0].plot(range(len(x1_values)), x1_values, label='x1', color='b')
    axs[0].set_title(f"Trajectory {trajectory_id} - x1")
    axs[0].set_xlabel("Transition")
    axs[0].set_ylabel("x1 Value")
    axs[0].grid(True)

    axs[1].plot(range(len(x2_values)), x2_values, label='x2', color='g')
    axs[1].set_title(f"Trajectory {trajectory_id} - x2")
    axs[1].set_xlabel("Transition")
    axs[1].set_ylabel("x2 Value")
    axs[1].grid(True)

    axs[2].plot(range(len(x3_values)), x3_values, label='x3', color='r')
    axs[2].set_title(f"Trajectory {trajectory_id} - x3")
    axs[2].set_xlabel("Transition")
    axs[2].set_ylabel("x3 Value")
    axs[2].grid(True)

    axs[3].plot(range(len(d_values)), d_values, label='d', color='purple')
    axs[3].set_title(f"Trajectory {trajectory_id} - d")
    axs[3].set_xlabel("Transition")
    axs[3].set_ylabel("d Value (True/False)")
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()


def load_dataset(file_path):
    trajectories = defaultdict(lambda: {'x1': [], 'x2': [], 'x3': [], 'd': []})

    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            trajectory_id = int(row['trajectory_id'])
            transition = int(row['transition'])
            x1_value = float(row['x1'])
            x2_value = float(row['x2'])
            x3_value = float(row['x3'])
            d_value = row['d'].lower() == 'true' 

            trajectories[trajectory_id]['x1'].append(x1_value)
            trajectories[trajectory_id]['x2'].append(x2_value)
            trajectories[trajectory_id]['x3'].append(x3_value)
            trajectories[trajectory_id]['d'].append(d_value)

    x1_trajectories = [trajectories[tid]['x1'] for tid in sorted(trajectories)]
    x2_trajectories = [trajectories[tid]['x2'] for tid in sorted(trajectories)]
    x3_trajectories = [trajectories[tid]['x3'] for tid in sorted(trajectories)]
    d_trajectories = [trajectories[tid]['d'] for tid in sorted(trajectories)]

    return x1_trajectories, x2_trajectories, x3_trajectories, d_trajectories

def create_next_state_sequences(x1_trajectories, x2_trajectories, x3_trajectories, n):
    inputs = []
    targets = []

    for x1_traj, x2_traj, x3_traj in zip(x1_trajectories, x2_trajectories, x3_trajectories):
        trajectory_length = len(x1_traj)
        for i in range(trajectory_length - n):
            # Input: window of size n
            window_x1 = x1_traj[i:i+n]
            window_x2 = x2_traj[i:i+n]
            window_x3 = x3_traj[i:i+n]

            # Combine x1, x2, x3 into a feature vector for each time step
            sequence = [[x1, x2, x3] for x1, x2, x3 in zip(window_x1, window_x2, window_x3)]
            inputs.append(sequence)

            # Target: next state after the window
            next_state = [x1_traj[i+n], x2_traj[i+n], x3_traj[i+n]]
            targets.append(next_state)

    return inputs, targets

class SigmaDetlaDataset(Dataset):
    """ """
    def __init__(self,X,Y):
        super().__init__()
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class SigmaDeltaLSTM(nn.Module):
    """ """
    def __init__(self,HIDDEN_SIZE,LAYER_NUM):
        super(SigmaDeltaLSTM,self).__init__()
        self.lstm = nn.LSTM(input_size = 3, hidden_size=HIDDEN_SIZE, num_layers=LAYER_NUM)
        self.fc = nn.Linear(HIDDEN_SIZE , 3)

    def forward(self,x):
        out, (hn, cn) = self.lstm(x)    
        last_hidden = hn[-1]
        output = self.fc(last_hidden)
        
        return output

    def fit_model(self,csv_path,seq_len,epochs=10,batch_size=32, save=True, val_split = 0.2, lr = 0.001):
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        print(f"Device Chosen: {device}")
        self.to(device)
        
        x1_trajectories, x2_trajectories, x3_trajectories, d_trajectories = load_dataset(file_path)
        inputs, targets = create_next_state_sequences(x1_trajectories, x2_trajectories, x3_trajectories, seq_len)

        combined_inputs = np.array([[[x1, x2, x3] for x1, x2, x3 in zip(traj_x1, traj_x2, traj_x3)] for traj_x1, traj_x2, traj_x3 in zip(x1_trajectories, x2_trajectories, x3_trajectories)])
        combined_inputs_flat = combined_inputs.reshape(-1, 3)
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_inputs = scaler.fit_transform(combined_inputs_flat)
        normalized_inputs = normalized_inputs.reshape(combined_inputs.shape)
        targets_flat = np.array(targets)
        normalized_targets = scaler.fit_transform(targets_flat)

        X_train, X_val, Y_train, Y_val = train_test_split(normalized_inputs, normalized_targets, test_size=val_split, random_state=69)

        train_dataset = SigmaDetlaDataset(X_train, Y_train)
        val_dataset = SigmaDetlaDataset(X_val, Y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = Adam(self.parameters(),lr)
        criterion = nn.MSELoss()

        


SEQUENCE_LENGTH = 5
HIDDEN_SIZE = 3
LAYER_NUM = 1
LR = 0.001

file_path = "state_trajectories.csv"
model = SigmaDeltaLSTM(HIDDEN_SIZE=HIDDEN_SIZE,LAYER_NUM=LAYER_NUM)
model.fit_model(file_path,SEQUENCE_LENGTH,lr=LR)
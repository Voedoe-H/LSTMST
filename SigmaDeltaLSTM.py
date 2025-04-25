import csv
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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

def create_sequences():
    inputs = []
    targets = []


file_path = "state_trajectories.csv"
x1_trajectories, x2_trajectories, x3_trajectories, d_trajectories = load_dataset(file_path)

plot_random_trajectory(x1_trajectories, x2_trajectories, x3_trajectories, d_trajectories)

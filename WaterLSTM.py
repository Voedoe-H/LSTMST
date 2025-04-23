import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch.optim as optim

SEQ_LEN = 5 
CSV_PATH = 'WaterLevelTrace.csv'


def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]
        y = data[i+seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


class WaterLevelDataset(Dataset):
    def __init__(self, sequences, targets):
        self.X = torch.from_numpy(sequences)
        self.y = torch.from_numpy(targets)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class WaterLevelLSTM(nn.Module):

    def __init__(self, input_size = 1, hidden_size = 64, num_layers = 1):
        super(WaterLevelLSTM,self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers= num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size,1)

    def forward(self, x):
        out, (hn,cn) = self.lstm(x)
        last_hidden = hn[-1]
        output = self.fc(last_hidden)
        return output

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        df = pd.read_csv(CSV_PATH, header=None)
        water_levels = df.values.astype(np.float32)  

        scaler = MinMaxScaler()
        normalized_levels = scaler.fit_transform(water_levels)

        X, y = create_sequences(normalized_levels, SEQ_LEN)

        dataset = WaterLevelDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = WaterLevelLSTM()
test_inp = torch.randn(1,5,1)
output = model(test_inp)

print("Input shape:", test_inp.shape)
print("Output shape:", output.shape)
print("Predicted value:", output.item())



#num_samples = 5 
#plt.figure(figsize=(12, 6))

#for i in range(num_samples):
#    seq = X[i].flatten()
#    target = y[i].item()

#    plt.plot(range(len(seq)), seq, label=f'Seq {i+1}')
#    plt.scatter(len(seq), target, marker='x', color='red')

#plt.title("Sample Water Level Sequences (normalized)")
#plt.xlabel("Time Step")
#plt.ylabel("Normalized Water Level")
#plt.legend()
#plt.grid(True)
#plt.show()
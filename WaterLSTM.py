import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch.optim as optim
import joblib

SEQ_LEN = 5 
CSV_PATH = 'WaterLevelTrace.csv'
MODEL_PATH = 'water_level_lstm.pth'
SCALER_PATH = 'scaler.save'


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
        self.X = torch.from_numpy(sequences).float()
        self.y = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class WaterLevelLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(WaterLevelLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        last_hidden = hn[-1]
        output = self.fc(last_hidden)
        return output

    def fit_model(self, csv_path=CSV_PATH, seq_len=SEQ_LEN, epochs=10, batch_size=32, save=True):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        df = pd.read_csv(csv_path, header=None)
        water_levels = df.values.astype(np.float32)

        scaler = MinMaxScaler()
        normalized_levels = scaler.fit_transform(water_levels)

        if save:
            joblib.dump(scaler, SCALER_PATH)

        X, y = create_sequences(normalized_levels, seq_len)
        dataset = WaterLevelDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.0005)

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0

            for batch_x, batch_y in loader:
                if batch_x.ndim == 2:
                    batch_x = batch_x.unsqueeze(-1)
                batch_x = batch_x.to(device)
                batch_y = batch_y.unsqueeze(-1).to(device)

                optimizer.zero_grad()
                predictions = self(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(loader):.6f}")

        if save:
            torch.save(self.state_dict(), MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")

    def load_model(self, model_path=MODEL_PATH):
        self.load_state_dict(torch.load(model_path))
        self.eval()
        print(f"Model loaded from {model_path}")

    def predict(self, input_sequence):
        self.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        if isinstance(input_sequence, list):
            input_sequence = np.array(input_sequence, dtype=np.float32)
        input_sequence = input_sequence.reshape(1, -1, 1)
        sequence = torch.from_numpy(input_sequence).float().to(device)

        with torch.no_grad():
            prediction = self(sequence)
        
        return prediction.cpu().numpy().flatten()[0]

model = WaterLevelLSTM()

model.fit_model(epochs=30)

model.load_model()

scaler = joblib.load(SCALER_PATH)
input_sequence = scaler.transform([[0.4], [0.5], [0.45], [0.47], [0.49]])
predicted = model.predict(input_sequence)
print(f"Predicted normalized water level: {predicted}")

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
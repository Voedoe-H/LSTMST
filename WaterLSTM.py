import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch.optim as optim
import joblib

SEQ_LEN = 125
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
    def __init__(self, input_size=1, hidden_size=64, num_layers=4):
        super(WaterLevelLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        last_hidden = hn[-1]
        output = self.fc(last_hidden)
        return output

    def fit_model(self, csv_path=CSV_PATH, seq_len=SEQ_LEN, epochs=30, batch_size=32, save=True, val_split=0.2, patience=5):
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        df = pd.read_csv(csv_path, header=None)
        water_levels = df.values.astype(np.float32)
        scaler = MinMaxScaler()
        normalized_levels = scaler.fit_transform(water_levels)

        if save:
            joblib.dump(scaler, SCALER_PATH)

        X, y = create_sequences(normalized_levels, seq_len)

        val_size = int(len(X) * val_split)
        train_X, val_X = X[:-val_size], X[-val_size:]
        train_y, val_y = y[:-val_size], y[-val_size:]

        train_dataset = WaterLevelDataset(train_X, train_y)
        val_dataset = WaterLevelDataset(val_X, val_y)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.0005)

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                if batch_x.ndim == 2:
                    batch_x = batch_x.unsqueeze(-1)
                batch_x = batch_x.to(device)
                batch_y = batch_y.unsqueeze(-1).to(device)

                optimizer.zero_grad()
                predictions = self(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_x, val_y in val_loader:
                    if val_x.ndim == 2:
                        val_x = val_x.unsqueeze(-1)
                    val_x = val_x.to(device)
                    val_y = val_y.unsqueeze(-1).to(device)

                    val_predictions = self(val_x)
                    val_batch_loss = criterion(val_predictions, val_y)
                    val_loss += val_batch_loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                # Optionally save the best model during training
                if save:
                    torch.save(self.state_dict(), MODEL_PATH)
                    print(f"Best model saved to {MODEL_PATH}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered. Stopping training after {epoch+1} epochs.")
                break

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


# --- Test Phase ---
model = WaterLevelLSTM()
model.fit_model(epochs=30)
model.load_model()

scaler = joblib.load(SCALER_PATH)

df = pd.read_csv(CSV_PATH, header=None)
water_levels = df.values.astype(np.float32)

normalized_levels = scaler.transform(water_levels)

test_X, test_y = create_sequences(normalized_levels, SEQ_LEN)

predicted_values = []
actual_values = []

for i in range(len(test_X)):
    input_seq = test_X[i]
    true_value = test_y[i]
    
    predicted_value = model.predict(input_seq)
    
    # Debug: Check scaled predictions and true values
    #print(f"Predicted (scaled): {predicted_value}, True (scaled): {true_value}")
    
    # Ensure predicted_value is reshaped correctly for inverse_transform
    predicted_value_denormalized = scaler.inverse_transform(np.array([[predicted_value]]).reshape(1, -1))[0][0]
    
    # True value also needs to be reshaped for inverse_transform
    true_value_denormalized = scaler.inverse_transform(np.array([[true_value]]).reshape(1, -1))[0][0]
    
    predicted_values.append(predicted_value_denormalized)
    actual_values.append(true_value_denormalized)

# Plotting the results
plt.figure(figsize=(10,6))
plt.plot(np.arange(len(predicted_values)), predicted_values, label='Predicted', color='r')
plt.plot(np.arange(len(actual_values)), actual_values, label='Actual', color='b')
plt.title('Predicted vs Actual Water Levels (Denormalized)')
plt.xlabel('Time Step')
plt.ylabel('Water Level')
plt.legend()
plt.show()

# Diagnostic Plot: Check both scaled and denormalized predictions vs. true values
plt.figure(figsize=(10, 6))

# Plot the scaled predictions and true values
plt.plot(np.arange(len(predicted_values)), predicted_values, label='Predicted (Scaled)', color='r', alpha=0.6)
plt.plot(np.arange(len(actual_values)), actual_values, label='True (Scaled)', color='b', alpha=0.6)

# Denormalize the predicted and true values
denormalized_pred = scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1)).flatten()
denormalized_true = scaler.inverse_transform(np.array(actual_values).reshape(-1, 1)).flatten()

# Plot the denormalized predictions and true values
plt.plot(np.arange(len(denormalized_pred)), denormalized_pred, label='Predicted (Denormalized)', color='orange', linestyle='--')
plt.plot(np.arange(len(denormalized_true)), denormalized_true, label='True (Denormalized)', color='green', linestyle='--')

# Labels and title
plt.title('Scaled and Denormalized Predictions vs. True Values')
plt.xlabel('Time Step')
plt.ylabel('Water Level')
plt.legend()
plt.show()
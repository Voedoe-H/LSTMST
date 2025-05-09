import pandas as pd
import torch
import torch.nn as nn
import os
from sklearn.model_selection import train_test_split

class PredictorDNN(nn.Module):

    def __init__(self, input_size=30):
        super(PredictorDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(32, 1)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.output(x))
        return x

def load_csv_to_dataframe(file_path):
    """
    Loads a CSV file and converts its contents into a pandas DataFrame.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"CSV file loaded successfully with {len(df)} rows and {len(df.columns)} columns.")
        return df
    except Exception as e:
        print(f"An error occurred while loading the CSV file: {e}")
        return None

def normalize_data(df):
    """
    Preprocesses the stock DataFrame:
    - Converts 'Gmt time' to datetime
    - Removes rows with Volume == 0
    - Sorts by time
    - Keeps only the 'Close' column

    Parameters:
        df (pd.DataFrame): Raw stock data

    Returns:
        pd.Series: Normalized close prices
    """
    df['Gmt time'] = pd.to_datetime(df['Gmt time'], format="%d.%m.%Y %H:%M:%S.%f")

    df = df[df['Volume'] > 0]

    df.sort_values('Gmt time', inplace=True)

    df.reset_index(drop=True, inplace=True)

    close_prices = df['Close'].values

    mean = close_prices.mean()
    std = close_prices.std()
    normalized = (close_prices - mean) / std

    print(f"Normalized {len(normalized)} closing prices.")

    return normalized

def create_windows(normalized_prices, window_size=30):
    """
    Converts a 1D array of normalized prices into a 2D tensor of sliding windows.

    Parameters:
        normalized_prices (np.ndarray): The 1D array of normalized prices.
        window_size (int): Number of days to look back.

    Returns:
        torch.Tensor: 2D tensor of shape [num_samples, window_size]
    """
    X = []
    for i in range(len(normalized_prices) - window_size):
        window = normalized_prices[i:i + window_size]
        X.append(window)
    return torch.tensor(X, dtype=torch.float32)

def create_labels(normalized_prices, window_size=30):
    """
    Creates binary labels indicating if the price went up the next day.

    Parameters:
        normalized_prices (np.ndarray): Normalized price data
        window_size (int): Length of input window

    Returns:
        torch.Tensor: Labels [num_samples, 1]
    """
    labels = []
    for i in range(window_size, len(normalized_prices)):
        today = normalized_prices[i - 1]
        tomorrow = normalized_prices[i]
        label = 1.0 if tomorrow > today else 0.0
        labels.append(label)
    return torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  #

def train_model(model, X, y, epochs=20, batch_size=64, lr=0.001,
                model_path="predictor_dnn.pt", val_split=0.2, early_stopping_patience=5):
    """
    Trains or continues training a model with validation tracking to avoid overfitting.

    Parameters:
        model (nn.Module): The model to train
        X (Tensor): Input data [N, window_size]
        y (Tensor): Target labels [N, 1]
        epochs (int): Training epochs
        batch_size (int): Batch size
        lr (float): Learning rate
        model_path (str): File path to save/load model weights
        val_split (float): Fraction of data to use for validation
        early_stopping_patience (int): Stop if val loss doesn't improve for this many epochs

    Returns:
        nn.Module: Trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load existing model
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded existing model from {model_path}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, shuffle=True)

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        train_loss = 0.0

        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / (X_train.size(0) / batch_size)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Check improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Validation loss improved. Model saved to {model_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    # Load best model before returning
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def test_model(model_path="predictor_dnn.pt", test_csv="AAPL.USUSD_Candlestick_1_D_BID_01.01.2019-31.12.2019.csv"):
    if not os.path.exists(model_path):
        print(f'Model not found at {model_path}')
        return

    # Load model
    model = PredictorDNN(input_size=30)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load and normalize test data
    df = load_csv_to_dataframe(test_csv)
    normalized = normalize_data(df)

    # Create features and labels from test set
    X_test = create_windows(normalized, window_size=30)  # [N, 30]
    y_test = create_labels(normalized, window_size=30)   # [N, 1]

    X_test, y_test = X_test.to(device), y_test.to(device)

    with torch.no_grad():
        predictions = model(X_test)
        predicted_labels = (predictions > 0.5).float()

        correct = (predicted_labels == y_test).sum().item()
        total = y_test.size(0)
        accuracy = correct / total

    print(f"Test Accuracy (directional correctness): {accuracy * 100:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    # Load and normalize
    #df = load_csv_to_dataframe("AAPL.USUSD_Candlestick_1_D_BID_01.01.2020-30.04.2025.csv")
    #normalized = normalize_data(df)

    # Create input features and labels
    #X = create_windows(normalized, window_size=30)              # [N, 30]
    #y = create_labels(normalized, window_size=30)               # [N, 1]

    # Initialize and train model
    #model = PredictorDNN(input_size=30)
    #train_model(model, X, y, epochs=10000)
    test_model()
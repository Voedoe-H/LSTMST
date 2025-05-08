import pandas as pd
import torch
import torch.nn as nn


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


# Load and normalize
df = load_csv_to_dataframe("AAPL.USUSD_Candlestick_1_D_BID_01.01.2020-30.04.2025.csv")
normalized = normalize_data(df)

# Create model input windows
X = create_windows(normalized, window_size=30)  # shape: [samples, 30]

# Initialize model
model = PredictorDNN(input_size=30)
model.eval()

# Forward pass on first 5 windows
with torch.no_grad():
    output = model(X[:5])
    print("Model output (up probability):", output.squeeze())
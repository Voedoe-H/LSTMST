import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEQUENCE_LENGTH = 3

def load():
    df = pd.read_csv(
        "AAPL.USUSD_Ticks_14.04.2025-14.04.2025.csv",
        parse_dates=["Gmt time"],
        dayfirst=True,            
        dtype={
            "Ask": "float32",
            "Bid": "float32",
            "AskVolume": "float32",
            "BidVolume": "float32"
        }
    )
    return df

def prep_data(df,n):
    """ df is the pandas data frame, n is sequence length for lstm"""
    rows = []
    for row in df.itertuples(index=False):
        rows.append(np.array([row.Ask,row.Bid,row.AskVolume,row.BidVolume]))

    sequences = []

    for i in range(0,len(rows)-n+1):
        sequence = []
        for j in range(0,n):
            sequence.append(rows[i+j])
        sequences.append(np.array(sequence))

    return np.array(sequences)

class APPLSet(Dataset):
    
    def __init__(self,sequences):
        super().__init__()
        self.sequences = torch.from_numpy(sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]

df = load()
seq = prep_data(df,SEQUENCE_LENGTH)
dtSet = APPLSet(seq)
print(dtSet[0])
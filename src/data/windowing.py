import numpy as np
import torch
from torch.utils.data import Dataset


def extract_windows(df, feature_cols, seq_len, is_test=False):
    x, y = [], []
    for unit_id in df['unit'].unique():
        unit_data = df[df['unit'] == unit_id]
        data_matrix = unit_data[feature_cols].values
        rul_array = unit_data['RUL'].values

        if len(data_matrix) >= seq_len:
            if is_test:
                x.append(data_matrix[-seq_len:])
                y.append(rul_array[-1])
            else:
                for i in range(len(data_matrix) - seq_len + 1):
                    x.append(data_matrix[i:i + seq_len])
                    y.append(rul_array[i + seq_len - 1])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


class CMAPSSDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

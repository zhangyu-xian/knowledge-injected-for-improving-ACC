
"""
get the code text and prologText as dictionary format
"""

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from tqdm.auto import tqdm
class shieldTunnelDataset(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    def load_data(self, data_file):
        df = pd.read_csv(data_file, encoding='utf-8')
        df = df.dropna(axis=0, how='any')  # 去掉nan所在行的数据
        codeText = df["Text"].tolist()
        prologText = df["Translation"].tolist()
        Data = {}
        assert len(codeText) == len(prologText)
        for idx in range(0, len(codeText)):
            Data[idx] = {
                'prologText': prologText[idx],
                'codeText': codeText[idx]
            }
        return Data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
def getDataset(dataPath):
    train_data = shieldTunnelDataset(dataPath)
    return train_data

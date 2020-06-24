# @Author: chargerKong
# @Time: 20-6-22 下午3:18
# @File: data_generator.py

import torch
from torch.utils.data import DataLoader
from utils.image_process import LaneDataset
from utils.config import train_csv_path

kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
training_dataset = LaneDataset(train_csv_path, transform=None)

data_gen = DataLoader(training_dataset, batch_size=2)


for batch in data_gen:
    print(22)
    import pdb
    pdb.set_trace()


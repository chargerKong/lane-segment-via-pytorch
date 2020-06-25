# @Author: chargerKong
# @Time: 20-6-22 下午3:18
# @File: data_generator.py

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.image_process import LaneDataset, ToTensor, CutOut, ImageAug, DeformAug, ScaleAug
from utils.config import train_csv_path
from torchvision import transforms

kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {
}
training_dataset = LaneDataset(train_csv_path, transform=transforms.Compose(
    [ImageAug(), DeformAug(), ScaleAug(), CutOut(32, 0.5), ToTensor()]))

data_gen = DataLoader(training_dataset, batch_size=2, drop_last=True, **kwargs)


for batch in tqdm(data_gen):

    image, mask = batch['image'], batch['mask']
    if torch.cuda.is_available():
        image, mask = image.cuda(), mask.cuda()

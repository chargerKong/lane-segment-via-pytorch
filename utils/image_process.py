# @Author: chargerKong
# @Time: 20-6-22 下午3:37
# @File: image_process.py
import pdb
import cv2
import pandas as pd
from utils.config import train_csv_path
from utils.process_label import encode_labels


def crop_data(img, label, new_size=(1024, 384), offset=690):
    img, label = img[offset:, :], label[offset:, :]
    img_rs = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    label_rs = cv2.resize(label, new_size, interpolation=cv2.INTER_NEAREST)
    return img_rs, label_rs


class LaneDataset:
    def __init__(self, csv_file, transform):
        self.train = pd.read_csv(csv_file)
        self.image = self.train['image']
        self.label = self.train['label']
        self.transform = transform

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, item):
        # read image data
        print(11)
        img_data = cv2.imread(self.image[item])
        gray_data = cv2.imread(self.label[item], cv2.IMREAD_GRAYSCALE)

        train_img, train_mask = crop_data(img_data, gray_data)

        # Encode
        train_mask = encode_labels(train_mask)

        sample = [train_img, train_mask]

        pdb.set_trace()
        return sample


if __name__ == '__main__':

    data = pd.read_csv(train_csv_path)
    img_path = data['image']
    img = img_path[:2]
    # img = '/home/kong/study/CV/exer/5_data_process/lane-segment-via-tensorflow/data/img_data/Road02/Record004/Camera 5/170927_064247017_Camera_5.jpg'
    # data = cv2.imread(img)
    cv2.imshow()
    import pdb
    pdb.set_trace()
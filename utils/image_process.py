# @Author: chargerKong
# @Time: 20-6-22 下午3:37
# @File: image_process.py
import pdb
import cv2
import pandas as pd
import torch
import numpy as np
from imgaug import augmenters as iaa
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
        img_data = cv2.imread(self.image[item])
        gray_data = cv2.imread(self.label[item], cv2.IMREAD_GRAYSCALE)

        train_img, train_mask = crop_data(img_data, gray_data)

        # Encode
        train_mask = encode_labels(train_mask)

        sample = [train_img, train_mask]

        if self.transform:
            sample = self.transform(sample)

        return sample


class ImageAug(object):
    def __call__(self, sample):
        image, mask = sample
        if np.random.uniform(0, 1) > 0.5:
            seq = iaa.Sequential([iaa.OneOf([
                iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),
                iaa.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3)),
                iaa.GaussianBlur(sigma=(0, 1.0))])])
            image = seq.augment_image(image)
        return image, mask


# deformation augmentation
class DeformAug(object):
    """
    从图像四周裁剪并且填充
    """
    def __call__(self, sample):
        image, mask = sample
        seq = iaa.Sequential([iaa.CropAndPad(percent=(-0.05, 0.1))])
        seg_to = seq.to_deterministic()
        image = seg_to.augment_image(image)
        mask = seg_to.augment_image(mask)
        return image, mask


class ScaleAug(object):
    """
    按比例缩小或者放大图片
        1、缩小则填充为原来大小
        2、放大后剪裁到原来大小
    """
    def __call__(self, sample):
        image, mask = sample
        scale = np.random.uniform(0.7, 1.5)
        h, w, _ = image.shape
        aug_image = image.copy()
        aug_mask = mask.copy()
        aug_image = cv2.resize(aug_image, (int(scale * w), int(scale * h)))
        aug_mask = cv2.resize(aug_mask, (int(scale * w), int(scale * h)))
        if scale < 1.0:
            new_h, new_w, _ = aug_image.shape
            pre_h_pad = int((h - new_h) / 2)
            pre_w_pad = int((w - new_w) / 2)
            pad_list = [[pre_h_pad, h - new_h - pre_h_pad],
                        [pre_w_pad, w - new_w - pre_w_pad], [0, 0]]
            aug_image = np.pad(aug_image, pad_list, mode="constant")
            aug_mask = np.pad(aug_mask, pad_list[:2], mode="constant")
        if scale > 1.0:
            new_h, new_w, _ = aug_image.shape
            pre_h_crop = int((new_h - h) / 2)
            pre_w_crop = int((new_w - w) / 2)
            post_h_crop = h + pre_h_crop
            post_w_crop = w + pre_w_crop
            aug_image = aug_image[pre_h_crop:post_h_crop,
                                  pre_w_crop:post_w_crop]
            aug_mask = aug_mask[pre_h_crop:post_h_crop, pre_w_crop:post_w_crop]
        return aug_image, aug_mask


class CutOut:
    """
    随机截取一个mask_size大小的框, 截取概率为p
    """
    def __init__(self, mask_size, p):
        self.mask_size = mask_size
        self.p = p

    def __call__(self, sample):
        image, mask = sample
        if np.random.uniform(0, 1) < self.p:
            mask_size_half = self.mask_size // 2
            offset = 1 if self.mask_size % 2 == 0 else 0
            h, w = image.shape[:2]
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
            cx = np.random.randint(cxmin, cxmax)
            cy = np.random.randint(cymin, cymax)
            xmin, ymin = cx - mask_size_half, cy - mask_size_half
            xmax, ymax = xmin + self.mask_size, ymin + self.mask_size
            xmin, ymin, xmax, ymax = max(
                0, xmin), max(
                0, ymin), min(
                w, xmax), min(
                h, ymax)
            image[ymin:ymax, xmin:xmax] = (0, 0, 0)
        return image, mask


class ToTensor(object):
    def __call__(self, sample):

        image, mask = sample
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.int32)
        mask = mask.astype(np.uint8)
        return {'image': torch.from_numpy(image.copy()),
                'mask': torch.from_numpy(mask.copy())}


if __name__ == '__main__':

    data = pd.read_csv(train_csv_path)
    img_path = data['image']
    img = img_path[:2]
    # img = '/home/kong/study/CV/exer/5_data_process/lane-segment-via-tensorflow/data/img_data/Road02/Record004/Camera 5/170927_064247017_Camera_5.jpg'
    # data = cv2.imread(img)
    cv2.imshow()
    import pdb
    pdb.set_trace()

# @Author: chargerKong
# @Time: 20-6-21 上午12:29
# @File: config.py
import pathlib
import os
root = pathlib.Path(__file__).parent.parent

img_dir = os.path.join(root, 'data', 'img_data')
label_dir = os.path.join(root, 'data', 'Labels_Fixed')

train_csv_path = os.path.join(root, 'data', 'train.csv')
val_csv_path = os.path.join(root, 'data', 'valid.csv')
test_csv_path = os.path.join(root, 'data', 'test.csv')


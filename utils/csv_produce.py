# @Author: chargerKong
# @Time: 20-6-19 下午4:06
# @File: csv_produce.py

import os
import pathlib
import pandas as pd
root = pathlib.Path(__file__).parent.parent

img_dir = os.path.join(root, 'data', 'img_data')
label_dir = os.path.join(root, 'data', 'Gray_Label')

img_list = []
label_list = []

for ob in os.listdir(img_dir):
    img_sub_dir1 = os.path.join(img_dir, ob)
    label_sub_dir1 = os.path.join(label_dir, 'Label_' + ob.lower(), 'Label')

    for ob2 in os.listdir(img_sub_dir1):
        img_sub_dir2 = os.path.join(img_sub_dir1, ob2)
        label_sub_dir2 = os.path.join(label_sub_dir1, ob2)
        # print(label_sub_dir2)

        for ob3 in os.listdir(img_sub_dir2):
            img_sub_dir3 = os.path.join(img_sub_dir2, ob3)
            label_sub_dir3 = os.path.join(label_sub_dir2, ob3)
            # print(label_sub_dir3)
            for ob4 in os.listdir(img_sub_dir3):
                file_name = os.path.join(img_sub_dir3, ob4)
                # label_sub_dir4 = os.path.join(label_sub_dir3, ob4)
                # print(label_sub_dir4)
                # for file_name in os.listdir(img_sub_dir4):

                # img_name = os.path.join(img_sub_dir3, file_name)

                img_list.append(file_name)
                label_list.append(file_name.replace('jpg', 'bin.png'))

file_name_csv = pd.DataFrame({'image': img_list, 'label': label_list})
file_name_csv.to_csv('../data/train_csv.csv', index=None)


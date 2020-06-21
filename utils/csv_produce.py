# @Author: chargerKong
# @Time: 20-6-19 下午4:06
# @File: csv_produce.py
from utils.config import *
import pandas as pd

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
                ob44 = ob4.replace('.jpg', '_bin.png')
                file_name = os.path.join(img_sub_dir3, ob4)
                label_sub_dir4 = os.path.join(label_sub_dir3, ob44)
                # print(label_sub_dir4)
                # for file_name in os.listdir(img_sub_dir4):

                # img_name = os.path.join(img_sub_dir3, file_name)
                if not os.path.exists(file_name):
                    print(file_name)
                    continue
                if not os.path.exists(label_sub_dir4):
                    print(label_sub_dir4)
                    continue

                img_list.append(file_name)
                label_list.append(label_sub_dir4)

assert len(img_list) == len(label_list)
n = len(img_list)
print('rows of file is {}'.format(n))

file_name_csv = pd.DataFrame({'image': img_list, 'label': label_list})
six_part = int(n * 0.6)
eight_part = int(n * 0.8)

file_name_csv[: six_part].to_csv(train_csv_path, index=None)
file_name_csv[six_part: eight_part].to_csv(val_csv_path, index=None)
file_name_csv[eight_part:].to_csv(test_csv_path, index=None)

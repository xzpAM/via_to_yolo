# 这一段代码是将via转化为yolo格式
# python via2yolo3.py --via_Dataset  ../detect_frames/ --yolo_Dataset ../0.71k_university_yolo_Dataset --tain_r 0.8
# 注意这是 via 中多个动作（以看书写字举手为例）转化为yolo格式
# 并且可以检查没有标注的框，并给出没有标注的图片

import os
import json
import cv2
import random
import argparse
import shutil

parser = argparse.ArgumentParser()

parser.add_argument('--via_Dataset', default = './via_Dataset', type = str)
parser.add_argument('--yolo_Dataset', default = './yolo_Dataset', type = str)
parser.add_argument('--tain_r', default = 0.8, type = float)

arg = parser.parse_args()

via_Dataset = arg.via_Dataset

# yolo数据集的存放位置 Dataset_dir
yolo_Dataset = arg.yolo_Dataset

# 删除 Dataset_dir 文件夹
if os.path.exists(yolo_Dataset):
    shutil.rmtree(yolo_Dataset)
# 在 Dataset_dir 下创建 labels images train val
os.makedirs(yolo_Dataset + '/labels/train')
os.makedirs(yolo_Dataset + '/labels/val')
os.makedirs(yolo_Dataset + '/images/train')
os.makedirs(yolo_Dataset + '/images/val')

# 训练集与验证集的比例，tain_r代表训练集的比例，1-tain_r 代表验证集的比例
tain_r = arg.tain_r
# 设置 labels images train val 的路径
train_label_dir = yolo_Dataset + '/labels/train/'
val_label_dir = yolo_Dataset + '/labels/val/'
train_image_dir = yolo_Dataset + '/images/train/'
val_image_dir = yolo_Dataset + '/images/val/'


# via2yolo 函数是将via信息转化为yolo的格式·
def via2yolo(json, root):
    # 循环读出每一个框的信息
    for i in json['metadata']:
        # 从 i 中获取 vid
        vid = i.split('_')[0]
        if 'image' in vid:
            vid = vid.split('image')[-1]
        # 获取对应的图片名字
        image_name = json['file'][vid]['fname']

        # 获取图片的路径
        image_dir = os.path.join(root, image_name.split('/')[-1])

        # 读取图片的高和宽
        img = cv2.imread(image_dir)

        if img is None:
            print(f"Warning: Failed to read image {image_name}. Skipping this image.")
            continue  # 跳过当前图片的处理

        h = img.shape[0]
        w = img.shape[1]

        # 读取 via 中的 坐标 xywh，框的左上角坐标 x y，与框的宽高
        xywh = json['metadata'][i]['xy'][1:]

        # 获取动作的id
        try:
            action_id = json['metadata'][i]['av']['1']
        except KeyError:
            print(json['metadata'][i]['av'])
            print("下面的图片中有没有标注的框")
            print(f"image_name: {image_name}, xywh: {xywh}")
            # input()

        '''
        将xywh 转化为 yolo 的格式
        第一个参数是0，代表举手这一个分类
        第二个参数是，框的中心坐标 x，归一化处理
        第三个参数是，框的中心坐标 y，归一化处理
        第四个参数是，框的宽度 w，归一化处理
        第五个参数是，框的高度 h，归一化处理
        第六个参数是，换行
        '''
        xcyc_wh = action_id + ' ' + str((xywh[0] + xywh[2] / 2) / w) + ' ' + str((xywh[1] + xywh[3] / 2) / h) +\
                  ' ' + str(xywh[2] / w) + ' ' + str(xywh[3] / h) + '\n'

        # 设置 txt 的 train 和 val 的路径
        temp_train_txt_path = os.path.join(train_label_dir, image_name.split('/')[-1].split('.')[0] + '.txt')
        temp_val_txt_path = os.path.join(val_label_dir, image_name.split('/')[-1].split('.')[0] + '.txt')

        if not (os.path.isfile(temp_train_txt_path) or os.path.isfile(temp_val_txt_path)):
            if random.random() <= tain_r:
                txt_path = temp_train_txt_path
                image_path = os.path.join(train_image_dir, image_name.split('/')[-1])
            else:
                txt_path = temp_val_txt_path
                image_path = os.path.join(val_image_dir, image_name.split('/')[-1])
            shutil.copy(image_dir, image_path)
        else:
            if image_name.split('.')[0] == image_name.split('/')[-1].split('.')[0]:
                if os.path.isfile(temp_train_txt_path):
                    txt_path = temp_train_txt_path
                else:
                    txt_path = temp_val_txt_path

        with open(txt_path, "a") as file:
            file.write(xcyc_wh)


# 循环 读出所有json文件
for root, dirs, files in os.walk(via_Dataset, topdown = False):
    for name in files:
        file_dir = os.path.join(root, name)
        if '.json' in name:
            try:
                file_json = open(file_dir, 'r')
                file_json_content = json.loads(file_json.read())
            except:
                file_json = open(file_dir, 'r', encoding = 'gb18030', errors = 'ignore')
                file_json_content = json.loads(file_json.read())

            via2yolo(file_json_content, root)
            file_json.close()
            print(file_dir)
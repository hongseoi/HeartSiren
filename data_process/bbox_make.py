import torchvision.io as io
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch
import numpy as np
import json
import os


class Bbox:
    def __init__(self, f_path, max_second=10):
        self.folder_path = f_path
        self.max_second = max_second

        self.file_name = None
        self.image_path = None
        self.tsv_path = None
        self.image = None

        self.output_path = ''

    def __tsv_load(self):
        self.tsv_path = self.image_path.split('.png')[0] + '.tsv'
        t = open(self.tsv_path, 'r')
        anno_data = list()
        for i in t.readlines():
            i = i.strip()
            if i and i[-1] in ['1', '3']:
                anno_data.append(i.split('\t'))
        return anno_data

    def bbox(self, image_name):
        self.file_name = image_name
        self.image_path = self.folder_path + image_name + '.png'
        self.image = io.read_image(self.folder_path + self.image_path)
        channel, height, width = self.image.size()

        anno_data = self.__tsv_load()
        ratio = width / self.max_second
        # anno_dict = {1: list(), 3: list()}
        anno_list = list()
        for anno in anno_data:
            min_x, max_x, l = map(lambda a: float(a), anno)

            min_x = int(min_x * ratio)
            max_x = int(max_x * ratio)
            l = int(l)

            y_list = list()
            for x in range(min_x, max_x):
                min_height = height // 2

                for y in range(height // 2, height):
                    if self.image[2, y, x] <= 50:
                        if y > min_height:
                            min_height = y

                y_list.append(min_height)

            # 라벨: 해당 박스 리스트
            # anno_dict[l].append([min_x, min(y_list), max_x, height])

            anno_list.append([min_x, min(y_list), max_x, height, l])
        return anno_list

    def anno_save_file(self, json_data):
        save_folder = '../bbox/'
        with open(save_folder + self.file_name + '.json', 'w') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)


folder_path = '../data/'
bbox = Bbox(folder_path, max_second=5)

file_list = os.listdir(folder_path)[6222:]
for fdx, f in enumerate(file_list):
    if '.png' in f:
        f_name = f.split('.')[0]
        try:
            r = bbox.bbox(f_name)
        except ValueError:
            continue
        bbox.anno_save_file(r)
        print(fdx)


# def show(imgs):
#     if not isinstance(imgs, list):  # 하나의 이미지일때
#         imgs = [imgs]
#     fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)  # 총 사진의 개수만큼 plot
#     for i, img in enumerate(imgs):
#         img = img.detach()  # 학습 그래프에서 제외
#         img = F.to_pil_image(img)  # torch.tensor 에서 pil 이미지로 변환
#         axs[0, i].imshow(np.asarray(img))  # numpy 배열로 변경후, 가로로 이미지를 나열
#         axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
#     plt.show()
#
# image = bbox.image[:3]
# for k, v in r.items():
#     boxes = torch.tensor(v, dtype=torch.float)
#     colors = 'red'
#     print(k)
#     if k == 3:
#         colors = 'blue'
#     image = draw_bounding_boxes(image, boxes, colors=colors, width=2)
#
# show(image)

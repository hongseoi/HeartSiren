import torchvision.io as io
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch
import numpy as np
import json


class Bbox:
    def __init__(self, f_path, m_s=10):
        self.folder_path = f_path
        self.max_second = m_s

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
        self.image = io.read_image(self.image_path)
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
            for x in range(min_x, max_x + 1):
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
        save_folder = f'{self.folder_path}/'
        with open(save_folder + self.file_name + '.json', 'w') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

    def labeled_save_image(self, anno_data, figsize):
        image = self.image[:3]
        anno_dict = {1: list(), 3: list()}
        for x_min, y_min, x_max, y_max, label in anno_data:
            box = [x_min, y_min, x_max, y_max]
            anno_dict[label].append(box)

        for k, v in anno_dict.items():
            boxes = torch.tensor(v, dtype=torch.float)
            colors = 'red'
            if k == 3:
                colors = 'blue'
            image = draw_bounding_boxes(image, boxes, colors=colors, width=2)

        img = F.to_pil_image(image)  # torch.tensor 에서 pil 이미지로 변환
        plt.figure(figsize=figsize)
        plt.imshow(np.asarray(img))  # numpy 배열로 변경후, 가로로 이미지를 나열
        plt.axis('off')
        plt.savefig(f'{self.folder_path}/{self.file_name}_labeled.png', bbox_inches="tight", pad_inches=0)
        plt.plot()
        plt.close()


if __name__ == '__main__':
    folder_name_list = [
        # 'spec_16000_5/',
        # 'spec_16000_7/',
        # 'spec_16000_10/',
        'spec_20000_3/'
    ]
    for f_n in folder_name_list:
        max_second = int(f_n.replace('/', '').split('_')[-1])
        folder_path = f'{f_n}/'
        bbox = Bbox(f_n, m_s=max_second)
        break
        file_list = os.listdir(folder_path)
        for fdx, f in enumerate(file_list):
            if '.png' in f:
                f_name = f.split('.')[0]
                try:
                    r = bbox.bbox(f_name)
                except ValueError:
                    continue
                # bbox.anno_save_file(r, max_second)
                print(fdx)
                bbox.labeled_save_image(r)

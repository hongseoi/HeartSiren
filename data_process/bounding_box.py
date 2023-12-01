import torchvision.io as io
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as trans
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

        self.total_min_y = None

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

        # 검은 부분 잘라낼때 필요함
        min_y_list = list()

        anno_list = list()
        for idx, anno in enumerate(anno_data):
            min_x, max_x, l = map(lambda a: float(a), anno)

            min_x = int(min_x * ratio)
            max_x = int(max_x * ratio)
            l = int(l)

            y_list = list()
            for x in range(min_x, max_x + 1):
                min_height = 0

                for y in range(height):
                    if self.image[2, y, x] <= 50:
                        if y > min_height:
                            min_height = y

                y_list.append(min_height)

            min_y = min(y_list)

            if idx == len(anno_data) - 1:
                avg_y = sum(min_y_list) / len(min_y_list)
                if min_y < int(avg_y * 0.8):
                    min_y = min(min_y_list)

            # 라벨: 해당 박스 리스트
            # anno_dict[l].append([min_x, min(y_list), max_x, height])
            min_y_list.append(min_y)
            anno_list.append([min_x, min_y, max_x, height, l])

        self.total_min_y = min(min_y_list)

        return anno_list

    def anno_save_file(self, anno_data, sub_name=None):
        file_name = self.file_name
        if sub_name:
            file_name += '_' + sub_name

        save_folder = f'{self.folder_path}/'
        with open(save_folder + file_name + '.json', 'w') as f:
            json.dump(anno_data, f, ensure_ascii=False, indent=4)

    def cut_black(self, anno_data, figsize):
        image = F.to_pil_image(self.image)
        # image = Image.open(f'{self.output_folder}/{self.file_name}.png')
        w, h = image.size

        for idx in range(len(anno_data)):
            anno_data[idx][1] = anno_data[idx][1] - self.total_min_y
            anno_data[idx][3] = anno_data[idx][3] - self.total_min_y

        self.anno_save_file(anno_data, 'cropped')
        try:
            # crop을 통해 이미지 자르기 (left,up, rigth, down)
            crop_image = image.crop((0, 0 + self.total_min_y, w, h))

            crop_image_path = self.save_image(crop_image, figsize, 'cropped')
            crop_image = io.read_image(crop_image_path)[:3]
        except SystemError as e:
            print(self.file_name, self.total_min_y)

        return anno_data, crop_image

    def labeled_save_image(self, anno_data, figsize, image=None, sub_name=None):
        labeled_name = 'labeled'
        if sub_name:
            labeled_name = labeled_name + '_' + sub_name

        if image == None:
            image = self.image[:3]

        anno_dict = {1: list(), 3: list()}
        for x_min, y_min, x_max, y_max, label in anno_data:
            box = [x_min, y_min, x_max, y_max]
            anno_dict[label].append(box)

        for k, v in anno_dict.items():
            boxes = torch.tensor(v, dtype=torch.float)
            colors = 'red'
            if k == 0:
                colors = 'white'
            if k == 2:
                colors = 'yellow'
            if k == 4:
                colors = 'pink'
            if k == 3:
                colors = 'blue'
            image = draw_bounding_boxes(image, boxes, colors=colors, width=2)

        image = F.to_pil_image(image)  # torch.tensor 에서 pil 이미지로 변환

        self.save_image(image, figsize, labeled_name)

    def save_image(self, image, figsize, sub_name):
        plt.figure(figsize=figsize)
        plt.imshow(np.asarray(image))  # numpy 배열로 변경후, 가로로 이미지를 나열
        plt.axis('off')
        plt.savefig(f'{self.folder_path}/{self.file_name}_{sub_name}.png', bbox_inches="tight", pad_inches=0)
        plt.plot()
        plt.close()
        return f'{self.folder_path}/{self.file_name}_{sub_name}.png'


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

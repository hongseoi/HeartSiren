from PIL import Image
import torchvision.io as io
import pandas as pd


class Bbox:
    def __init__(self, f_path):
        self.folder_path = f_path
        self.image_path = None
        self.tsv_path = None

        self.output_path = ''

        self.tsv_df = None

    def tsv_load(self, image_name):
        self.image_path = image_name + '.png'
        self.tsv_path = self.folder_path + image_name + '.tsv'
        df = pd.read_csv(self.tsv_path, delimiter='\t', names=['x_min', 'x_max', 'label'])
        self.tsv_df = df[(df['label'] == 1) | (df['label'] == 3)]

    def bbox(self, image_path=None):
        if image_path:
            self.image_path = image_path
        image = io.read_image(self.folder_path + self.image_path)
        channel, height, width = image.size()
        print(height, width)


folder_path = '../../spectogram/'
bbox = Bbox(folder_path)

bbox.tsv_load('2530_AV')
bbox.bbox()


# # 이미지를 불러옵니다.
# image_path = '../PAST/img_data4/14998_AV.png'  # 이미지 파일 경로를 적절히 수정하세요.
# # image_path = 'rb.png'
# image = Image.open(image_path)
# image = io.read_image(image_path)
#
# # 이미지의 크기를 가져와서 출력합니다.
# #width, height = image.size
# #print("Image Size - Width:", width, "Height:", height)
# print(image[:-1])
#
# # tsv 파일 불러와서 label_x ={'1': [[0, 15]], '3':[[25, 35], [50, 80], [150, 200]]} 형태로 바꾸는 코드
#
# import pandas as pd
# import numpy as np
#
# tsv_path = "../processed_label/45843_PV.tsv"
# df = pd.read_csv(tsv_path, delimiter='\t', names=['x_min', 'x_max', 'label'])
# label_x = dict()
#
# for row in df.iterrows():
#     if row[1]['label'] in [1.0, 3.0]:
#         if row[1]['label'] not in label_x:
#             label_x[row[1]['label']] = []
#         x_min = np.ceil(row[1]['x_min']*155)
#         x_max = np.ceil(row[1]['x_max']*155)
#         label_x[row[1]['label']].append([x_min, x_max])
#
# print(label_x)
#
# import torch
# from torchvision.utils import draw_bounding_boxes
# import torchvision.transforms as transforms
# import torchvision.io as io
# import matplotlib.pyplot as plt
# import torchvision.transforms.functional as F
# import numpy as np
# import math
#
# # 이미지 주소를 지정합니다.
# image_path = "../processed_img/14998_AV.png"
# # image_path = "./rb.png"
# # 이미지를 불러옵니다.
# image = io.read_image(image_path)
#
# # 이미지의 높이와 너비를 가져옵니다.
# height = image.size()[-2]
# print('height', height)
# width = image.size()[-1]
#
#
# def show(imgs):
#     if not isinstance(imgs, list):  # 하나의 이미지일때
#         imgs = [imgs]
#     fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)  # 총 사진의 개수만큼 plot
#     for i, img in enumerate(imgs):
#         img = img.detach()  # 학습 그래프에서 제외
#         img = F.to_pil_image(img)  # torch.tensor 에서 pil 이미지로 변환
#         axs[0, i].imshow(np.asarray(img))  # numpy 배열로 변경후, 가로로 이미지를 나열
#         axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
#
#
# y_list = list()
# for x in range(width):
#
#     max_height = 0
#     for y in range(height):
#         # 픽셀이 검정색이 아닌 경우
#         if image[2, y, x] <= 50 and y >= int(height / 2):
#             # 픽셀의 높이를 저장합니다.
#             if y > max_height:
#                 max_height = y
#     y_list.append(max_height)
#
# print(y_list)
#
# '''
# 초로 이미지 비율에 맞췄을 경우 10초 = 780
# ex) -
# 0, 10 :1,
# 25, 35 :3
# '''
# # label_x ={'blue': [[0, 15]], 'green':[[25, 35], [50, 80], [150, 200]]}
# print(len(y_list))
# label = list()
# height = 385
# for key, value in label_x.items():
#     for v in value:
#         if v[0] == 0:
#             v[0] += 1
#         if v[1] == width:
#             v[1] -= 1
#         print(v)
#
#         start = math.floor(v[0])
#         end = math.ceil(v[1])
#         min_y = max(y_list[start:end])
#         label_data = [v[0], min_y, v[1], height, key]
#         label.append(label_data)
#
# # 가장 높은 높이를 출력합니다.
# image = image[:-1]
# for l in label:
#     boxes = torch.tensor([l[:-1]], dtype=torch.float)
#     colors = [l[:-1]]
#     image = draw_bounding_boxes(image, boxes, colors=['green'], width=2)
#
# show(image)
# print(label)
#
# import csv
# # TSV 파일로 저장하는 함수
# def save_to_tsv(data, filename):
#     with open(filename, 'w', newline='') as tsvfile:
#         writer = csv.writer(tsvfile, delimiter='\t')
#         # 헤더 작성
#         #writer.writerow(['col1', 'col2', 'col3', 'col4', 'col5'])
#         # 데이터 작성
#         writer.writerows(data)
#
# # TSV 파일로 저장
# save_to_tsv(label, 'output.tsv')
#
# # 저장한 TSV 파일의 헤더 출력
# df = pd.read_csv('output.tsv', delimiter='\t', header=None)
# df.head()
#
#
# def save_bbox(image_path, tsv_path, output_path):
#     # 이미지 불러오기
#     image = io.read_image(image_path)
#
#     # 이미지의 높이와 너비를 가져옵니다.
#     height = image.size()[-2]
#     print('height', height)
#     width = image.size()[-1]
#
#     # tsv 파일에 대하여 x_min, x_max
#     df = pd.read_csv(tsv_path, delimiter='\t', names=['x_min', 'x_max', 'label'])
#     label_x = dict()
#
#     for row in df.iterrows():
#         if row[1]['label'] in [1.0, 3.0]:
#             if row[1]['label'] not in label_x:
#                 label_x[row[1]['label']] = []
#             x_min = np.ceil(row[1]['x_min'] * 155)
#             x_max = np.ceil(row[1]['x_max'] * 155)
#             label_x[row[1]['label']].append([x_min, x_max])
#
#     print(label_x)
#
#     # y_min 계산
#     y_list = list()
#     for x in range(width):
#
#         max_height = 0
#         for y in range(height):
#             # 픽셀이 검정색이 아닌 경우
#             if image[2, y, x] <= 50 and y >= int(height / 2):
#                 # 픽셀의 높이를 저장합니다.
#                 if y > max_height:
#                     max_height = y
#         y_list.append(max_height)
#
#     print(y_list)
#
#     # label_x ={'blue': [[0, 15]], 'green':[[25, 35], [50, 80], [150, 200]]}
#     print(len(y_list))
#     label = list()
#     height = 385
#     for key, value in label_x.items():
#         for v in value:
#             if v[0] == 0:
#                 v[0] += 1
#             if v[1] == width:
#                 v[1] -= 1
#             print(v)
#
#             start = math.floor(v[0])
#             end = math.ceil(v[1])
#
#             if not y_list[start:end]:
#                 with open('empty_sequences.txt', 'a') as file:
#                     file.write(tsv_path + '\n')
#             else:
#                 min_y = max(y_list[start:end])
#                 label_data = [v[0], min_y, v[1], height, key]
#                 label.append(label_data)
#
#     print(label)
#     # 저장
#     with open(output_path, 'w', newline='') as tsvfile:
#         writer = csv.writer(tsvfile, delimiter='\t')
#         writer.writerows(label)
#
# tsv_path = "./processed_label/14998_AV.tsv"
# image_path = "./processed_img/14998_AV.png"
#
# output_path = "./eval/" + tsv_path
# save_bbox(image_path, tsv_path, output_path)
#
#
# image_path = "../PAST/img_data4/14998_AV.png"
# image = io.read_image(image_path)
#
#
# label = pd.read_csv(output_path, delimiter='\t', header=None)
# label = label.values.tolist()
# print(label)
#
# height = image.size()[-2]
# width = image.size()[-1]
# image = image[:-1]
#
# for l in label:
#     boxes = torch.tensor([l[:-1]], dtype=torch.float)
#     colors = [l[:-1]]
#     image = draw_bounding_boxes(image, boxes, colors=['green'], width=2)
#
# show(image)
#
#
# import os
#
# img_folder_path = "./processed_img/"
# tsv_folder_path = "./processed_label/"
# output_folder = './output/'
#
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# tsv_files = [f for f in os.listdir(tsv_folder_path) if f.endswith(".tsv")]
# image_files = [f for f in os.listdir(img_folder_path) if f.endswith(".png")]
#
# idx = 0
#
# for i in tsv_files:
#     for j in image_files:
#         if i[:-3] == j[:-3]:
#             print("<<<<<<<<<<<<<<<<",i,">>>>>>>>>>>>>>>>>")
#             idx += 1
#             img = img_folder_path + j
#             tsv = tsv_folder_path + i
#             output_path = output_folder + i
#             save_bbox(img, tsv, output_path)
#             print(idx/len(tsv_files),"% 저장완료")
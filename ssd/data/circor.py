from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import csv


CIRCOR_CLASSES = (  # always index 0
    'S1', 'S2')

# note: if you used our download scripts, this should be right
CIRCOR_ROOT = osp.join(HOME, "data/CirCor/")


class CirCorAnnotationTransform(object):
    """Transforms a CirCor annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: ['S1', 'S2'])
    """

    def __init__(self, class_to_ind=None):
        self.class_to_ind = class_to_ind or dict(
            zip(CIRCOR_CLASSES, range(len(CIRCOR_CLASSES))))
        
    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        time_range = 0.0
        xmin_list = []
        xmax_list = []
        label_list = []
        with open(target, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for start, end, annotation in reader:
                time_range = end
                if annotation not in ['1', '3']:
                    continue
                annotation = 0.0 if annotation == '1' else 1.0
                xmin_list.append(float(start))
                xmax_list.append(float(end))
                label_list.append(annotation)
        
        time_range = float(time_range)
        for i in range(len(xmin_list)):
            xmin_list[i] /= time_range
            xmax_list[i] /= time_range

        res = []
        for xmin, xmax, label in zip(xmin_list, xmax_list, label_list):
            res.append([xmin, 0.0, xmax, 1.0, label])

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class CirCorDetection(data.Dataset):
    """CirCor Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to CirCor folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_set='RECORDS',
                 transform=None, target_transform=CirCorAnnotationTransform(),
                 dataset_name='CirCor'):
        self.root = root
        self.image_set = osp.join(CIRCOR_ROOT, image_set)
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s.tsv')
        self._imgpath = osp.join('%s.png')
        self.ids = list()
        with open(self.image_set, 'r') as f:
            self.ids = f.readlines()


    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index][:-1]
        target = osp.join(CIRCOR_ROOT, self._annopath % img_id)
        img = cv2.imread(osp.join(CIRCOR_ROOT, self._imgpath % img_id))
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

import pandas as pd
import numpy as np
import os
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from .tranforms import augment3
from PIL import Image


class ImageClassDataset(data.Dataset):
    """Single image dataset for image multi-label classification.

    Read image and its label(score) pairs.

    There is 1 mode:
    single images with a individual name + a csv file with all images` label.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            image_folder (str): the folder containing all the images.
            csv_path (str): the csv file consists of all image names and their class.
            class (int/float): the classification label of the image.
            image_size (tuple): Resize the image into a fin size (should be square).

            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(ImageClassDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        # self.io_backend_opt = opt['io_backend']  # only disk type is prepared for this dataset type.
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.resize = opt['resize'] if 'reszie' in opt else True
        # if self.mean is not None or self.std is not None:
        #   print('Normlizing Active')
        self.augment_ratio=opt['augment_ratio'] if 'augment_ratio' in opt else None

        # read csv and build a data list
        self.dt_folder = opt['image_folder']
        raw_data = pd.read_csv(opt['csv_path'])
        pd_data = pd.DataFrame(raw_data)
        self.image_names = pd_data['identifier'].tolist()
        levels_raw = pd_data['CDImm'].tolist()
        self.levels=levels_raw
        

        # make augment
        # directly increase the lenth of list, will effect the validation time and epochs counting
        if self.augment_ratio is not None:
          new_image_list=[]
          new_level_list=self.levels
          for times in range(0,self.augment_ratio-1):
            new_image_list.extend(self.image_names)
            # print(self.levels.shape)
            # print(new_level_list.shape)
            new_level_list=np.concatenate([new_level_list,self.levels],axis=0)
          new_image_list.extend(self.image_names)
          self.image_names=new_image_list
          self.levels=new_level_list
        


    def __getitem__(self, index):

        img_path = os.path.join(self.dt_folder, self.image_names[index]+"_ori.png")
        img_data = Image.open(img_path)
        level = self.levels[index]
        

        # augment
        # auto reshape and crop into size
        img_data=augment3(img_data.convert('RGB'),resize=self.resize, flip=self.opt['flip'],patch_size=self.opt['image_size'])
        # normalize (not recommanded)
        if self.mean is not None or self.std is not None:
            normalize(img_data, self.mean, self.std, inplace=True)
        # print(img_data.size())

        return {'image': img_data, 'class': level, 'img_path': img_path}

    def __len__(self):
        return len(self.image_names)

class ImageMaskDataset(data.Dataset):
    """Single image dataset for image multi-label classification.

    Read image and its label(score) pairs.

    There is 1 mode:
    single images with a individual name + a csv file with all images` label.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            image_folder (str): the folder containing all the images.
            csv_path (str): the csv file consists of all image names and their class.
            class (int/float): the classification label of the image.
            image_size (tuple): Resize the image into a fin size (should be square).

            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(ImageMaskDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        # self.io_backend_opt = opt['io_backend']  # only disk type is prepared for this dataset type.
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.resize = opt['resize'] if 'reszie' in opt else True
        # if self.mean is not None or self.std is not None:
        #   print('Normlizing Active')
        self.augment_ratio=opt['augment_ratio'] if 'augment_ratio' in opt else None

        # read csv and build a data list
        self.dt_folder = opt['image_folder']
        raw_data = pd.read_csv(opt['csv_path'])
        pd_data = pd.DataFrame(raw_data)
        self.image_names = pd_data['identifier'].tolist()
        levels_raw = pd_data['CDImm'].tolist()
        self.levels=levels_raw
        

        # make augment
        # directly increase the lenth of list, will effect the validation time and epochs counting
        if self.augment_ratio is not None:
          new_image_list=[]
          new_level_list=self.levels
          for times in range(0,self.augment_ratio-1):
            new_image_list.extend(self.image_names)
            # print(self.levels.shape)
            # print(new_level_list.shape)
            new_level_list=np.concatenate([new_level_list,self.levels],axis=0)
          new_image_list.extend(self.image_names)
          self.image_names=new_image_list
          self.levels=new_level_list
        


    def __getitem__(self, index):

        img_path = os.path.join(self.dt_folder, self.image_names[index]+"_mask.png")
        img_data = Image.open(img_path)
        level = self.levels[index]
        

        # augment
        # auto reshape and crop into size
        img_data=augment3(img_data.convert('RGB'),resize=self.resize, flip=self.opt['flip'],patch_size=self.opt['image_size'])
        # normalize (not recommanded)
        if self.mean is not None or self.std is not None:
            normalize(img_data, self.mean, self.std, inplace=True)
        # print(img_data.size())

        return {'image': img_data, 'class': level, 'img_path': img_path}

    def __len__(self):
        return len(self.image_names)


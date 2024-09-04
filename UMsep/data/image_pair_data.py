import pandas as pd
import numpy as np
import os,cv2,random
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from .tranforms import paired_augment,tensor2img
from PIL import Image


class ImagePairDataset(data.Dataset):
    """Paired image dataset for image Segmentation.

    Read image and its image pairs.

    There is 1 mode:
    single images with a individual name + image pair.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            image_folder (str): the folder containing all the images.
            csv_path (str): the csv file consists of all image names and their class.
            class (int/float): the classification label of the image.
            image_size (tuple): Resize the image into a fin size (should be square).

            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(ImagePairDataset, self).__init__()
        self.opt = opt
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.resize = opt['resize'] if 'reszie' in opt else True
        self.crop = opt['crop'] if 'crop' in opt else False
        self.score_scaler = opt['score_scaler'] if 'score_scaler' in opt else 1

        if self.mean is not None or self.std is not None:
          print('Normlizing Active')
        self.augment_ratio=opt['augment_ratio'] if 'augment_ratio' in opt else None

        # read csv and build a data list
        self.dt_folder = opt['image_folder']
        raw_data = pd.read_csv(opt['csv_path'])

        pd_data = pd.DataFrame(raw_data)
        image_list = pd_data['identifier'].tolist()
        levels_raw = [pd_data['age'].tolist(),pd_data['gender'].tolist(),pd_data['IOPod'].tolist(),pd_data['IOPos'].tolist()]
        label_raw=[sub_mm/self.score_scaler for sub_mm in pd_data['CDImm'].tolist()]

        
        

        # make augment
        # directly increase the lenth of list, will effect the validation time and epochs counting
        if self.augment_ratio is not None:
          new_img_list=[]
          new_level_list=[]
          new_lable_list=[]
          for times in range(0,self.augment_ratio):
            new_img_list.extend(image_list)
            new_level_list.extend(levels_raw)
            new_lable_list.extend(label_raw)
          
          image_list = new_img_list
          levels_raw = new_level_list
          label_raw=new_lable_list
        self.image_list=image_list
        self.label_list=label_raw



    def __getitem__(self, index):

        full_name,suf=os.path.splitext(self.image_list[index])
        ori_path = os.path.join(self.dt_folder, full_name+"_ori.png")
        
        mask_path = os.path.join(self.dt_folder, full_name+"_mask.png")
        ori_data = Image.open(ori_path)
        mask_data = Image.open(mask_path)
        sub_label = self.label_list[index]
        # augment
        # auto reshape and crop into size
        ori_data,mask_data=paired_augment(ori_data.convert('RGB'),
                                       mask_data.convert('RGB'),
                                       flip=self.opt['flip'], 
                                       fine_size=self.opt['fine_size'],
                                       crop=self.crop,
                                       crop_size=self.opt['image_size'],
                                       resize=self.opt['resize'])

        # normalize (not recommanded)
        if self.mean is not None or self.std is not None:
            normalize(ori_data, self.mean, self.std, inplace=True)
            normalize(mask_data, self.mean, self.std, inplace=True)
     
        
        if mask_data.shape[0]!=1:
           mask_data=mask_data[0,:,:].unsqueeze(dim=0)

        return {'hq': mask_data,'lq': ori_data, 'label': sub_label,'lq_path': mask_path}

    def __len__(self):
        return len(self.image_list)



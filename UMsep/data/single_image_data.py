import pandas as pd
import os
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from .tranforms import single_augment,ratio_resize
from PIL import Image

class SingleImageDataset(data.Dataset):
    """Paired image dataset for image generation.

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
        super(SingleImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        # self.io_backend_opt = opt['io_backend']  # only disk type is prepared for this dataset type.
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.resize = opt['resize'] if 'reszie' in opt else True
        self.crop = opt['crop'] if 'crop' in opt else False
        if 'min_multiplier' in opt:
          self.resize = False
          self.min_multiplier=opt['min_multiplier'] 
        else: 
           self.min_multiplier=1

        if self.mean is not None or self.std is not None:
          print('Normlizing Active')
        self.augment_ratio=opt['augment_ratio'] if 'augment_ratio' in opt else None

        # read csv and build a data list
        self.dt_folder = opt['image_folder']
        raw_data = pd.read_csv(opt['csv_path'])

        pd_data = pd.DataFrame(raw_data)
        self.image_lq = pd_data['file_name'].tolist()
        

        # make augment
        # directly increase the lenth of list, will effect the validation time and epochs counting
        if self.augment_ratio is not None:
          new_lq_list=[]
          for times in range(0,self.augment_ratio):
            new_lq_list.extend(self.image_lq)
          self.image_lq = new_lq_list

        
        # # make the levels int
        # self.lvs= np.array(self.lvs).astype(int)


    def __getitem__(self, index):

        # hq_name,suf=os.path.splitext(self.image_hq[index])
        # hq_path = os.path.join(self.dt_folder, hq_name+"-hq.PNG")
        
        img_path = os.path.join(self.dt_folder, self.image_lq[index])
        img_data = Image.open(img_path)
        

        # augment
        # auto reshape and crop into size
        if self.min_multiplier!=1:
           img_data=ratio_resize(img_data,min_multiplier=self.min_multiplier,long_edge=self.opt['fine_size'])
        img_data=single_augment(img_data.convert('RGB'),
                                       flip=self.opt['flip'], 
                                       fine_size=self.opt['fine_size'],
                                       crop=self.crop,
                                       crop_size=self.opt['image_size'],
                                       resize=self.opt['resize'])

        # normalize (not recommanded)
        if self.mean is not None or self.std is not None:
            normalize(img_data, self.mean, self.std, inplace=True)
        

        # saving=cv2.imwrite(os.path.join('/data/huden/CATINTELL/test_image',str(random.randint(0,20)))+'.PNG',tensor2img(hq_data))
        return {'hq': img_data,'lq': img_data, 'lq_path': img_path}

    def __len__(self):
        return len(self.image_lq)



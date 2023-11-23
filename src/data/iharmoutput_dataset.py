import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from PIL import Image
import torch
import random
import numpy as np
import torch.nn.functional as F
import glob


######'dir_A': shadowimage
######'dir_B': shadowmask
######'dir_C': shadowfree
######'dir_param':illumination parameter
######'dir_light': light direction
######'dir_instance':object mask


class IHarmOutputDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.is_train = self.opt.isTrain
        self.root = opt.dataroot
        # self.dir_A =  opt.shadowimg_path # not needed
        # self.dir_C = opt.shadowfree_path 
        self.dir_param = opt.param_path
        # self.dir_bg_instance = opt.bg_instance_path # not needed
        self.dir_bg_shadow = opt.bg_shadow_path
        self.dir_new_mask = opt.new_mask_path

        # get the harmonised images from the dataroot folder
        harmed_images = glob.glob(opt.dataroot+'/*_harmonized.jpg')

        # We only have; harmonised image, foreground mask, ground truth
        # NO; shadow mask, param information

        # print('total images number', len(self.imname))
        self.birdy_deshadoweds = []
        self.birdy_shadoweds = []
        self.birdy_fg_instances = []
        self.birdy_fg_shadows = []
        self.birdy_bg_instances = []
        self.birdy_bg_shadows = []
        self.birdy_edges = []
        self.birdy_shadow_params = []
        self.birdy_shadow_object_ratio = []
        self.birdy_instance_boxes = []
        self.birdy_shadow_boxes= []
        self.birdy_instance_box_areas=[]
        self.birdy_shadow_box_areas=[]
        self.birdy_imlists=[]
        for image in harmed_images:
            # adjust file naming so that the 'a' image is ground truth (as output by Iharm)
            A_img = Image.open(image.replace('harmonized', 'real')).convert('RGB').resize((self.opt.loadSize, self.opt.loadSize),Image.NEAREST)

            # 'c' image is the harmonised output of iharmony
            C_img = Image.open(image).convert('RGB').resize((self.opt.loadSize, self.opt.loadSize),Image.NEAREST)

            # foreground mask
            instance = Image.open(image.replace('harmonized', 'mask')).convert('L').resize((self.opt.loadSize, self.opt.loadSize),Image.NEAREST)
            
            A_img_array = np.array(A_img)
            C_img_array = np.array(C_img)
            instance_array = np.array(instance)

            self.birdy_deshadoweds.append(Image.fromarray(np.uint8(C_img_array), mode='RGB'))
            self.birdy_shadoweds.append(Image.fromarray(np.uint8(A_img_array), mode='RGB'))
            self.birdy_fg_instances.append(Image.fromarray(np.uint8(instance_array), mode='L'))
            self.birdy_fg_shadows.append(Image.fromarray(np.uint8(np.ones(A_img_array.shape)), mode='RGB'))
            self.birdy_imlists.append(image)
            self.birdy_bg_instances.append(Image.fromarray(np.uint8(np.zeros(instance_array.shape)), mode='L'))
            self.birdy_bg_shadows.append(Image.fromarray(np.uint8(np.zeros(instance_array.shape)), mode='L'))


        print("Done loading images")
        self.data_size = len(self.birdy_deshadoweds)
        print('datasize', self.data_size)

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=opt.norm_mean,
                                               std = opt.norm_std)]

        self.transformA = transforms.Compose(transform_list)
        self.transformB = transforms.Compose([transforms.ToTensor()])

        self.transformAugmentation = transforms.Compose([
            transforms.Resize(int(self.opt.loadSize * 1.12), Image.BICUBIC),
            transforms.RandomCrop(self.opt.loadSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

    def __getitem__(self,index):
        birdy = {}
        birdy['A'] = self.birdy_shadoweds[index]
        birdy['C'] = self.birdy_deshadoweds[index]
        birdy['instancemask'] = self.birdy_fg_instances[index]
        birdy['B'] = self.birdy_fg_shadows[index]
        birdy['bg_shadow'] = self.birdy_bg_shadows[index]
        birdy['bg_instance'] = self.birdy_bg_instances[index]
        birdy['im_list']= self.birdy_imlists[index]


        ow = birdy['A'].size[0]
        oh = birdy['A'].size[1]
        loadSize = self.opt.loadSize
        if self.opt.randomSize:
            loadSize = np.random.randint(loadSize + 1,loadSize * 1.3 ,1)[0]
        if self.opt.keep_ratio:
            if w>h:
                ratio = np.float(loadSize)/np.float(h)
                neww = np.int(w*ratio)
                newh = loadSize
            else:
                ratio = np.float(loadSize)/np.float(w)
                neww = loadSize
                newh = np.int(h*ratio)
        else:
            neww = loadSize
            newh = loadSize


        if not self.is_train:
            for k,im in birdy.items():
                if k=='im_list':
                    continue
                birdy[k] = im.resize((neww, newh),Image.NEAREST)

        if self.opt.no_flip and self.opt.no_crop and self.opt.no_rotate:
            for k,im in birdy.items():
                if k=='im_list':
                    continue
                birdy[k] = im.resize((neww, newh),Image.NEAREST)

        #### flip
        if not self.opt.no_flip:
            for i in ['A', 'B', 'C',  'instancemask', 'bg_shadow', 'bg_instance', 'edge','fg_instance_box', 'fg_shadow_box', 'fg_instance_box_area', 'fg_shadow_box_area']:
                birdy[i] = birdy[i].transpose(Image.FLIP_LEFT_RIGHT)


        for k,im in birdy.items():
            if k=='im_list':
                continue
            birdy[k] = self.transformB(im)


        for i in ['A','C','B','instancemask', 'bg_instance', 'bg_shadow', 'edge','fg_instance_box', 'fg_shadow_box', 'fg_instance_box_area', 'fg_shadow_box_area']:
            if i in birdy:
                birdy[i] = (birdy[i] - 0.5)*2


        h = birdy['A'].size()[1]
        w = birdy['A'].size()[2]
        if not self.opt.no_crop:
            w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))
            for k, im in birdy.items():
                birdy[k] = im[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
                birdy[k] = im.type(torch.FloatTensor)


        for k,im in birdy.items():
            if k=='im_list':
                continue
            im = F.interpolate(im.unsqueeze(0), size = self.opt.loadSize)
            birdy[k] = im.squeeze(0)


        birdy['w'] = ow 
        birdy['h'] = oh

        shadow_param=[0,1,0,1,0,1]

        birdy['param'] = torch.FloatTensor(np.array(shadow_param))

        # return all the necessary information information for a single image
        return birdy
    

    def __len__(self):
        return self.data_size

    def name(self):
        return 'IHarmOutputDataset'

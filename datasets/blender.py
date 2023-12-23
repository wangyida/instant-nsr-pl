import os
import json
import math
import numpy as np
from PIL import Image
from termcolor import colored

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions
from utils.misc import get_rank


class BlenderDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.apply_mask = self.config.apply_mask
        self.apply_depth = self.config.apply_depth

        with open(os.path.join(self.config.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)

        if 'w' in meta and 'h' in meta:
            W, H = int(meta['w']), int(meta['h'])
        else:
            W, H = 800, 800

        if 'img_wh' in self.config:
            w, h = self.config.img_wh
            assert round(W / w * h) == H
        elif 'img_downscale' in self.config:
            w, h = W // self.config.img_downscale, H // self.config.img_downscale
        else:
            raise KeyError("Either img_wh or img_downscale should be specified.")
        
        self.w, self.h = w, h
        self.img_wh = (self.w, self.h)
        factor = w / W

        self.near, self.far = self.config.near_plane, self.config.far_plane

        try:
            self.focal_x = meta['fl_x'] * factor
            self.focal_y = meta['fl_y'] * factor
            self.cx = meta['cx'] * factor
            self.cy = meta['cy'] * factor
        except:
            self.focal_x = 0.5 * w / math.tan(0.5 * meta['camera_angle_x']) * factor # scaled focal length
            self.focal_y = self.focal_x
            self.cx = self.w//2 * factor
            self.cy = self.h//2 * factor

        try:
            self.k1 = meta['k1']
            self.k2 = meta['k2']
            self.k3 = meta['k3']
            self.k4 = meta['k4']
            with_distort = True
        except:
            self.k1 = 0.0
            self.k2 = 0.0
            self.k3 = 0.0
            self.k4 = 0.0
            with_distort = False

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(self.w, self.h, self.focal_x, self.focal_y, self.cx, self.cy, k1=self.k1, k2=self.k2, k3=self.k3, k4=self.k4).to(self.rank) if self.config.load_data_on_gpu else get_ray_directions(self.w, self.h, self.focal_x, self.focal_y, self.cx, self.cy, k1=self.k1, k2=self.k2, k3=self.k3, k4=self.k4).cpu() # (h, w, 3)

        self.all_c2w, self.all_images, self.all_fg_masks, self.all_depths, self.all_depths_mask = [], [], [], [], []

        for i, frame in enumerate(meta['frames']):
            c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4])
            self.all_c2w.append(c2w)

            img_path = os.path.join(self.config.root_dir, f"{frame['file_path']}.png")
            img = Image.open(img_path)
            img = img.resize(self.img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
            img = img.to(self.rank) if self.config.load_data_on_gpu else img.cpu()

            if self.apply_mask:
                if with_distort:
                    # NOTE: c3vd data supplement mask from depth images
                    depth_path = img_path.replace("images", "depths").replace("color.png", "depth.tiff")
                    depth = Image.open(depth_path)
                    depth = depth.resize(self.img_wh, Image.BICUBIC)
                    depth = TF.to_tensor(depth).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
                    mask = torch.ones_like(img[...,0], device=img.device)
                    mask[depth[...,0] == 0] = 0.0
                elif img.shape[-1] != 4:
                    mask_path = img_path.replace("images", "masks") # .replace("color.png", "depth.tiff")
                    mask = Image.open(mask_path).convert('L')
                    mask = mask.resize(self.img_wh, Image.BICUBIC)
                    mask = TF.to_tensor(mask).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
                elif img.shape[-1] == 4:
                    mask = img[..., -1]
                mask = mask.to(self.rank) if self.config.load_data_on_gpu else mask.cpu()
                self.all_fg_masks.append(mask) # (h, w)
            else:
                self.all_fg_masks.append(torch.ones_like(img[...,0], device=img.device)) # (h, w)
            self.all_images.append(img[...,:3])

            if self.apply_depth:
                # load estimated or recorded depth map
                depth_path = os.path.join(self.config.root_dir, f"{frame['depth_path']}")
                if os.path.isfile(depth_path):
                    try:
                        depth = torch.load(depth_path)[...,3]
                    except:
                        depth = Image.open(depth_path)
                        depth = depth.resize(self.img_wh, Image.BICUBIC)
                        depth = TF.to_tensor(depth).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
                    self.all_depths.append(depth)
                else:
                    print(colored('skip, ', depth_path + 'does not exist', 'red'))
                    self.all_depths.append(torch.zeros_like(img[...,0], device=img.device))
                # load the mask which is used to determine rays with confident depth
                try:
                    depth_mask_path = os.path.join(self.config.root_dir, f"{frame['depth_mask_path']}")
                    if os.path.isfile(depth_mask_path):
                        depth_mask = (torch.load(depth_mask_path)[...] < 0.35).to(bool)
                        self.all_depths_mask.append(depth_mask)
                    else:
                        print(colored('skip, ', depth_mask_path + 'does not exist', 'red'))
                        self.all_depths_mask.append(torch.zeros_like(img[...,0], device=img.device))
                except:
                    self.all_depths_mask.append(torch.ones_like(img[...,0], device=img.device))
            else:
                self.all_depths.append(torch.zeros_like(img[...,0], device=img.device))
                self.all_depths_mask.append(torch.zeros_like(img[...,0], device=img.device))

        self.all_c2w, self.all_images, self.all_fg_masks, self.all_depths, self.all_depths_mask = \
            torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
            torch.stack(self.all_images, dim=0).float(), \
            torch.stack(self.all_fg_masks, dim=0).float(), \
            torch.stack(self.all_depths, dim=0).float(), \
            torch.stack(self.all_depths_mask, dim=0).float()

        # translate
        # self.all_c2w[...,3] -= self.all_c2w[...,3].mean(0)

        # rescale
        if 'cam_downscale' not in self.config:
            scale = 1.0
        elif self.config.cam_downscale:
            scale = self.config.cam_downscale
        else:
            # auto-scale with camera positions
            scale = self.all_c2w[...,3].norm(p=2, dim=-1).min()
            print('Auto-scaled by: ', scale)
        if self.split != 'val':
            self.all_c2w[...,3] /= scale
        

class BlenderDataset(Dataset, BlenderDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class BlenderIterableDataset(IterableDataset, BlenderDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('blender')
class BlenderDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = BlenderIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = BlenderDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = BlenderDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = BlenderDataset(self.config, self.config.train_split)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       

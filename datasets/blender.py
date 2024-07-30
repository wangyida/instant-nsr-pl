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
from utils.rast import rasterize


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
        self.factor = w / W

        self.near, self.far = self.config.near_plane, self.config.far_plane

        try:
            self.fx = meta['fl_x'] * self.factor
            self.fy = meta['fl_y'] * self.factor
            self.cx = meta['cx'] * self.factor
            self.cy = meta['cy'] * self.factor
        except:
            self.fx = 0.5 * w / math.tan(0.5 * meta['camera_angle_x']) * self.factor # scaled focal length
            self.fy = self.fx
            self.cx = self.w//2 * self.factor
            self.cy = self.h//2 * self.factor

        intrinsic = np.array([[self.fx, 0, self.cx],[0, self.fy, self.cy],[0, 0, 1]])

        try:
            self.k1 = meta['k1']
            self.k2 = meta['k2']
            self.p1 = meta['p1']
            self.p2 = meta['p2']
            self.k3 = meta['k3']
            self.k4 = meta['k4']
        except:
            self.k1 = 0.0
            self.k2 = 0.0
            self.p1 = 0.0
            self.p2 = 0.0
            self.k3 = 0.0
            self.k4 = 0.0

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(self.w, self.h, self.fx, self.fy, self.cx, self.cy, k1=self.k1, k2=self.k2, k3=self.k3, k4=self.k4).to(self.rank)
        if not self.config.load_data_on_gpu: 
            self.directions = self.directions.cpu() # (h, w, 3)

        self.all_c2w, self.all_images, self.all_fg_masks, self.all_depths, self.all_depth_masks, self.all_vis_masks = [], [], [], [], [], []

        import open3d as o3d
        pts_clt = o3d.geometry.PointCloud()
        self.num_imgs = len(meta['frames'])
        for i, frame in enumerate(meta['frames']):
            c2w_npy = np.array(frame['transform_matrix'])
            # NOTE: only specific dataset, e.g. Baoru's medical images needs to convert
            # c2w_npy[:3, 1:3] *= -1. # COLMAP => OpenGL
            c2w = torch.from_numpy(c2w_npy[:3, :4])
            self.all_c2w.append(c2w)

            img_path = os.path.join(self.config.root_dir, f"{frame['file_path']}.png")
            img = Image.open(img_path)
            img = img.resize(self.img_wh, Image.BICUBIC)
            img = TF.pil_to_tensor(img).permute(1, 2, 0) / 255.0 # (4, h, w) => (h, w, 4 ) and normalize it
            img = img.to(self.rank) if self.config.load_data_on_gpu else img.cpu()

            # load the estimated or recorded depth map
            if "depth_path" in frame:
                mesh_dir = os.path.join(self.config.root_dir, 'meshes')
                os.makedirs(mesh_dir, exist_ok=True)
                depth_path = os.path.join(self.config.root_dir, f"{frame['depth_path']}")
                if depth_path.split('.')[-1] == 'tiff':
                    depth = Image.open(depth_path).convert('I')
                    depth = depth.resize(self.img_wh, Image.BICUBIC)
                    depth = Image.fromarray(np.array(depth).astype("uint16") / (2**16-1) * 100)
                    depth = TF.pil_to_tensor(depth).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
                    depth = depth[..., -1] # / 27.222561594202872 # like cam_downscale
                elif depth_path.split('.')[-1] == 'pth':
                    depth = torch.load(depth_path)[...,3]
                depth /= self.config.cam_downscale

                # saving point cloud form depth
                dep_max = 5.0 # lidar limit
                import open3d as o3d
                import cv2
                if depth.max() != 0.0:
                    # depth_o3d = o3d.geometry.Image(depth.numpy() * self.config.cam_downscale)
                    distort = np.array([self.k1, self.k2, self.p1, self.p2, self.k3, self.k4, 0, 0])
                    depth_o3d = o3d.geometry.Image(cv2.undistort(depth.numpy(), intrinsic, distort))
                    img_o3d = o3d.geometry.Image(cv2.undistort(img.numpy(), intrinsic, distort))
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        img_o3d,
                        depth_o3d,
                        depth_trunc=dep_max-0.01,
                        depth_scale=1,
                        convert_rgb_to_intensity=False)
                    intrin_o3d = o3d.camera.PinholeCameraIntrinsic(
                        w, h, self.fx, self.fy, self.cx, self.cy)
                    # Open3D uses world-to-camera extrinsics
                    pts_frm = o3d.geometry.PointCloud.create_from_depth_image(
                        depth_o3d, intrinsic=intrin_o3d, extrinsic=np.linalg.inv(c2w_npy), depth_scale=1)
                    # o3d.io.write_point_cloud(os.path.join(mesh_dir, f"./layout_depth_frame_{str(i + 1)}.ply"), pts_frm)
                    pts_clt.points = (o3d.utility.Vector3dVector(
                        np.concatenate((np.array(pts_clt.points), np.array(pts_frm.points)),
                                       axis=0)))
            else:
                depth = None

            if self.apply_mask:
                if "mask_path" in frame:
                    mask_path = os.path.join(self.config.root_dir, f"{frame['mask_path']}")
                    mask = Image.open(mask_path).convert('I')
                    mask = mask.resize(self.img_wh, Image.BICUBIC)
                    # mask is either 0 or 1
                    mask = TF.to_tensor(mask).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
                elif img.shape[-1] == 4:
                    mask = img[..., -1]
                elif depth is not None:
                    mask = torch.ones_like(img[...,0], device=img.device)
                    # mask = (depth > 0.0).to(bool)
                mask = mask.to(self.rank) if self.config.load_data_on_gpu else mask.cpu()
                self.all_fg_masks.append(mask) # (h, w)
            else:
                self.all_fg_masks.append(torch.ones_like(img[...,0], device=img.device)) # (h, w)
            self.all_images.append(img[...,:3])

            if self.apply_depth and 'depth_path' in frame:
                # Using the provided depth images
                depth_path = os.path.join(self.config.root_dir, f"{frame['depth_path']}")
                if os.path.isfile(depth_path):
                    self.all_depths.append(depth)
                else:
                    print(colored('skip, ', depth_path + 'does not exist', 'red'))
                    self.all_depths.append(torch.zeros_like(img[...,0], device=img.device))
                # load the mask which is used to determine rays with confident depth
                if "depth_mask_path" in frame:
                    depth_mask_path = os.path.join(self.config.root_dir, f"{frame['depth_mask_path']}")
                    if os.path.isfile(depth_mask_path):
                        depth_mask = (torch.load(depth_mask_path)[...] < 0.35).to(bool)
                        self.all_depth_masks.append(depth_mask)
                    else:
                        print(colored('skip, ', depth_mask_path + 'does not exist', 'red'))
                        self.all_depth_masks.append(torch.zeros_like(img[...,0], device=img.device))
                else:
                    depth_mask = (depth > 0.0).to(bool)
                    self.all_depth_masks.append(depth_mask)
            elif self.apply_depth and 'depth_path' not in frame:
                # Rasterizing depth and norms from a mesh
                pcd = o3d.geometry.PointCloud()
                mesh_init_path = os.path.join(self.config.root_dir, 'points3D_mesh.ply')
                if os.path.exists(mesh_init_path):
                    print(colored(
                            'GT surface mesh is directly loaded',
                            'blue'))
                    mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(mesh_init_path))
                depth_rast_path = os.path.join(
                        self.config.root_dir, f'depths_{self.split}',
                        f"rasted_{self.config.img_downscale}")
                norm_rast_path = os.path.join(
                        self.config.root_dir, f'normals_{self.split}',
                        f"rasted_{self.config.img_downscale}")
                c2w_col = c2w
                c2w_col[:3, 1:3] *= -1. # OpenGL => COLMAP
                depth_rast, _ = rasterize(frame['file_path'], mesh_o3d, intrinsic, c2w_col, w, h, depth_rast_path, norm_rast_path)
                depth_rast = TF.pil_to_tensor(depth_rast).permute(
                    1, 2, 0) / self.config.cam_downscale
                inf_mask = (depth_rast == float("Inf"))
                depth_rast[inf_mask] = 0
                depth_rast = depth_rast.to(
                    self.rank
                ) if self.config.load_data_on_gpu else depth_rast.cpu()
                depth_rast_mask = (depth_rast > 0.0).to(bool) # trim points outside the contraction box off
                self.all_depths.append(depth_rast)
                self.all_depth_masks.append(depth_rast_mask)
            else:
                self.all_depths.append(torch.zeros_like(img[...,0], device=img.device))
                self.all_depth_masks.append(torch.zeros_like(img[...,0], device=img.device))
            
            # NOTE: Masks labeling valid training rays,
            # may comes from semantics or depth sensors
            if depth is not None:
                vis_mask = (depth > 0.0).to(bool)
            else:
                vis_mask = torch.ones_like(img[...,0], device=img.device)
            self.all_vis_masks.append(vis_mask)

        if self.apply_depth and "depth_path" in frame:
            pts_clt = pts_clt.voxel_down_sample(voxel_size=0.5)
            pts_clt.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=20))
            o3d.io.write_point_cloud(os.path.join(mesh_dir, 'layout_depth_clt.ply'), pts_clt)
            # Poisson surface on top of given GT points
            mesh_poisson_path = os.path.join(mesh_dir, 'layout_mesh_ps.ply')
            if not os.path.exists(mesh_poisson_path):
                print(colored(
                        'Extracting Poisson surface on top of given GT points',
                        'blue'))
                with o3d.utility.VerbosityContextManager(
                        o3d.utility.VerbosityLevel.Debug) as cm:
                    # Poisson
                    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pts_clt, depth=10, linear_fit=True)
                    densities = np.asarray(densities)
                    vertices_to_remove = densities < np.quantile(densities, 0.02)
                    mesh.remove_vertices_by_mask(vertices_to_remove)
                o3d.io.write_triangle_mesh(mesh_poisson_path, mesh)

        self.all_c2w, self.all_images, self.all_fg_masks, self.all_depths, self.all_depth_masks, self.all_vis_masks = \
            torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
            torch.stack(self.all_images, dim=0).float(), \
            torch.stack(self.all_fg_masks, dim=0).float(), \
            torch.stack(self.all_depths, dim=0).float(), \
            torch.stack(self.all_depth_masks, dim=0).float(), \
            torch.stack(self.all_vis_masks, dim=0).float()


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

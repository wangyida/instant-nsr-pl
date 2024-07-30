import os
import math
import numpy as np
from PIL import Image
from termcolor import colored

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from datasets.colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
from models.ray_utils import get_ray_directions
from utils.misc import get_rank
from utils.pose_utils import get_center, normalize_poses, create_spheric_poses
from utils.rast import rasterize

class ColmapDatasetBase():
    # the data only has to be processed once
    initialized = False
    properties = {}

    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        if not ColmapDatasetBase.initialized:
            camdata = read_cameras_binary(os.path.join(self.config.root_dir, 'sparse/0/cameras.bin'))

            H = int(camdata[1].height)
            W = int(camdata[1].width)

            if 'img_wh' in self.config:
                w, h = self.config.img_wh
                assert round(W / w * h) == H
            elif 'img_downscale' in self.config:
                w, h = int(W / self.config.img_downscale + 0.5), int(H / self.config.img_downscale + 0.5)
            else:
                raise KeyError("Either img_wh or img_downscale should be specified.")

            img_wh = (w, h)
            factor = w / W

            if camdata[1].model == 'SIMPLE_RADIAL':
                fx = fy = camdata[1].params[0] * factor
                cx = camdata[1].params[1] * factor
                cy = camdata[1].params[2] * factor
            elif camdata[1].model in ['PINHOLE', 'OPENCV']:
                fx = camdata[1].params[0] * factor
                fy = camdata[1].params[1] * factor
                cx = camdata[1].params[2] * factor
                cy = camdata[1].params[3] * factor
            else:
                raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
            intrinsic = {
                'width': w,
                'height': h,
                'f': fx,
                'cx': cx,
                'cy': cy,
                'b1': 0,
                'b2': 0,
                'k1': 0,
                'k2': 0,
                'k3': 0,
                'k4': 0,
                'p1': 0,
                'p2': 0,
                'p3': 0,
                'p4': 0,
            }
            intrinsic = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
            
            directions = get_ray_directions(w, h, fx, fy, cx, cy).to(self.rank) if self.config.load_data_on_gpu else get_ray_directions(w, h, fx, fy, cx, cy).cpu()

            imdata = read_images_binary(os.path.join(self.config.root_dir, 'sparse/0/images.bin'))

            mask_dir = os.path.join(self.config.root_dir, 'masks', 'sam')
            # masks labling invalid regions
            vis_mask_dir = os.path.join(self.config.root_dir, f'vis_mask')
            has_mask = os.path.exists(mask_dir) # TODO: support partial masks
            self.apply_mask = has_mask and self.config.apply_mask
            self.config.apply_depth = self.config.apply_depth
            
            all_c2w, all_images, all_fg_masks, all_depths, all_depth_masks, all_vis_masks = [], [], [], [], [], []

            pts3d = read_points3d_binary(os.path.join(self.config.root_dir, 'sparse/0/points3D.bin'))
            pts3d_rgb = np.array([pts3d[k].rgb for k in pts3d])
            pts3d = np.array([pts3d[k].xyz for k in pts3d])

            for i, d in enumerate(imdata.values()):
                R = d.qvec2rotmat()
                t = d.tvec.reshape(3, 1)
                c2w = torch.from_numpy(np.concatenate([R.T, -R.T@t], axis=1)).float()
                all_c2w.append(c2w)
                if self.split in ['train', 'val']:
                    img_path = os.path.join(self.config.root_dir, f"images_{self.config.img_downscale}", d.name)
                    if not os.path.exists(img_path):
                        os.makedirs(os.path.join(self.config.root_dir, f"images_{self.config.img_downscale}"), exist_ok=True)
                        img_path = os.path.join(self.config.root_dir, 'images', d.name)
                    img = Image.open(img_path.replace('JPG','jpg'))
                    if img.size[0] != w or img.size[1] != h:
                        img = img.resize(img_wh, Image.BICUBIC)
                        img.save(os.path.join(self.config.root_dir, f"images_{self.config.img_downscale}", d.name))
                    img = TF.to_tensor(img).permute(1, 2, 0)[...,:3]
                    img = img.to(self.rank) if self.config.load_data_on_gpu else img.cpu()
                    if self.apply_mask:
                        mask_paths = [os.path.join(mask_dir, d.name), os.path.join(mask_dir, d.name[3:])]
                        mask_paths = list(filter(os.path.exists, mask_paths))
                        assert len(mask_paths) == 1
                        mask = Image.open(mask_paths[0]).convert('L') # (H, W, 1)
                        mask = mask.resize(img_wh, Image.BICUBIC)
                        mask = TF.to_tensor(mask)[0]
                    else:
                        mask = torch.ones_like(img[...,0], device=img.device)

                    # NOTE: Visual masks    
                    self.apply_vis_mask = False # True
                    if self.apply_vis_mask:
                        vis_mask_paths = [os.path.join(vis_mask_dir, d.name), os.path.join(vis_mask_dir, d.name[3:])]
                        vis_mask_paths = list(filter(os.path.exists, vis_mask_paths))
                        assert len(vis_mask_paths) == 1
                        vis_mask = Image.open(vis_mask_paths[0]).convert('L') # (H, W, 1)
                        vis_mask = vis_mask.resize(img_wh, Image.BICUBIC)
                        vis_mask = TF.to_tensor(vis_mask)[0]
                    else:
                        vis_mask = torch.ones_like(img[...,0], device=img.device)

                    all_fg_masks.append(mask) # (h, w)
                    all_vis_masks.append(vis_mask) # (h, w)
                    all_images.append(img)
                    if self.config.apply_depth:
                        import open3d as o3d
                        # NOTE: save the point cloud using point cloud
                        pcd = o3d.geometry.PointCloud()
                        mesh_init_path = os.path.join(self.config.root_dir, 'sparse/0/points3D_mesh.ply')
                        if os.path.exists(mesh_init_path):
                            print(colored(
                                    'GT surface mesh is directly loaded',
                                    'blue'))
                            mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(mesh_init_path))
                        else:
                            print(colored(
                                    'GT surface mesh is not provided',
                                    'cyan'))
                            print(colored(
                                    'Processing with Poisson surface on top of given GT points',
                                    'blue'))
                            mesh_dir = os.path.join(self.config.root_dir, 'meshes')
                            os.makedirs(mesh_dir, exist_ok=True)
                            if not os.path.exists(os.path.join(mesh_dir, 'layout_pcd_gt.ply')):
                                pcd.points = o3d.utility.Vector3dVector(pts3d)
                                pcd.colors = o3d.utility.Vector3dVector(pts3d_rgb)
                            else:
                                pcd = o3d.io.read_point_cloud(os.path.join(mesh_dir, 'layout_pcd_gt.ply'))
                            # pts3d = torch.from_numpy(pts3d).float()
                            if not os.path.exists(os.path.join(mesh_dir, 'layout_pcd_gt.ply')):
                                # pcd = pcd.voxel_down_sample(voxel_size=0.002)
                                pcd.estimate_normals(
                                            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
                                o3d.io.write_point_cloud(os.path.join(mesh_dir, 'layout_pcd_gt.ply'), pcd)
                            else:
                                pcd.normals = o3d.io.read_point_cloud(os.path.join(mesh_dir, 'layout_pcd_gt.ply')).normals

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
                                        pcd, depth=10, linear_fit=True)
                                o3d.io.write_triangle_mesh(mesh_poisson_path, mesh)
                            else:
                                print(colored(
                                        'Poisson surface is directly loaded',
                                        'blue'))
                            mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(mesh_poisson_path))

                        depth_rast_path = os.path.join(
                                self.config.root_dir, 'depths',
                                f"rasted_{self.config.img_downscale}")
                        norm_rast_path = os.path.join(
                                self.config.root_dir, 'normals',
                                f"rasted_{self.config.img_downscale}")
                        depth_rast, _ = rasterize(d.name, mesh_o3d, intrinsic, c2w, w, h, depth_rast_path, norm_rast_path)
                        depth_rast = TF.pil_to_tensor(depth_rast).permute(
                            1, 2, 0) / self.config.cam_downscale
                        inf_mask = (depth_rast == float("Inf"))
                        depth_rast[inf_mask] = 0
                        depth_rast = depth_rast.to(
                            self.rank
                        ) if self.config.load_data_on_gpu else depth_rast.cpu()
                        depth_rast_mask = (depth_rast > 0.0).to(bool) # trim points outside the contraction box off
                        all_depths.append(depth_rast)
                        all_depth_masks.append(depth_rast_mask)
                    else:
                        all_depths.append(torch.zeros_like(img[...,0], device=img.device))
                        all_depth_masks.append(torch.zeros_like(img[...,0], device=img.device))

            if self.config.apply_depth and self.config.preprocess_only:
                print(colored('Finish preprocessing.', 'green'))
                exit()
            
            all_c2w, all_images, all_fg_masks, all_depths, all_depth_masks, all_vis_masks = \
                torch.stack(all_c2w, dim=0).float(), \
                torch.stack(all_images, dim=0).float(), \
                torch.stack(all_fg_masks, dim=0).float(), \
                torch.stack(all_depths, dim=0).float(), \
                torch.stack(all_depth_masks, dim=0).float(), \
                torch.stack(all_vis_masks, dim=0).float()

            pts3d = torch.from_numpy(pts3d).float()
            all_c2w[:,:,1:3] *= -1. # COLMAP => OpenGL
            if self.config.repose:
                all_c2w, pts3d, R, t, cam_downscale = normalize_poses(
                    all_c2w,
                    pts3d,
                    up_est_method=self.config.up_est_method,
                    center_est_method=self.config.center_est_method,
                    cam_downscale=self.config.cam_downscale)
            else:
                poses_min, poses_max = all_c2w[..., 3].min(0)[0], all_c2w[
                    ..., 3].max(0)[0]
                pts_fg = pts3d[(poses_min[0] < pts3d[:, 0])
                               & (pts3d[:, 0] < poses_max[0]) &
                               (poses_min[1] < pts3d[:, 1]) &
                               (pts3d[:, 1] < poses_max[1])]
                print(colored(get_center(pts3d), 'blue'))
                all_c2w[:, :, 3] -= get_center(pts3d)
                pts3d -= get_center(pts3d)
                R = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
                t = -get_center(pts3d)
                cam_downscale = self.config.cam_downscale

            ColmapDatasetBase.properties = {
                'w': w,
                'h': h,
                'img_wh': img_wh,
                'factor': factor,
                'has_mask': has_mask,
                'apply_mask': self.apply_mask,
                'apply_depth': self.config.apply_depth,
                'directions': directions,
                'pts3d': pts3d,
                'all_c2w': all_c2w,
                'all_images': all_images,
                'all_fg_masks': all_fg_masks,
                'all_depths': all_depths,
                'all_depth_masks': all_depth_masks,
                'all_vis_masks': all_vis_masks,
                'repose_R': R,
                'repose_t': t,
                'repose_s': cam_downscale
            }

            ColmapDatasetBase.initialized = True
        
        for k, v in ColmapDatasetBase.properties.items():
            setattr(self, k, v)

        if self.split == 'test':
            self.all_c2w = create_spheric_poses(self.all_c2w[:,:,3], n_steps=self.config.n_test_traj_steps)
            self.all_images = torch.zeros((self.config.n_test_traj_steps, self.h, self.w, 3), dtype=torch.float32)
            self.all_fg_masks = torch.zeros((self.config.n_test_traj_steps, self.h, self.w), dtype=torch.float32)
            self.all_vis_masks = torch.ones((self.config.n_test_traj_steps, self.h, self.w), dtype=torch.float32)

        """
        # for debug use
        from models.ray_utils import get_rays
        rays_o, rays_d = get_rays(self.directions.cpu(), self.all_c2w, keepdim=True)
        pts_out = []
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 1.0 0.0 0.0' for l in rays_o[:,0,0].reshape(-1, 3).tolist()]))

        t_vals = torch.linspace(0, 1, 8)
        z_vals = 0.05 * (1 - t_vals) + 0.5 * t_vals

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,0,0][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 1.0 0.0' for l in ray_pts.view(-1, 3).tolist()]))

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,self.h-1,0][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 0.0 1.0' for l in ray_pts.view(-1, 3).tolist()]))

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,0,self.w-1][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 1.0 1.0' for l in ray_pts.view(-1, 3).tolist()]))

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,self.h-1,self.w-1][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 1.0 1.0 1.0' for l in ray_pts.view(-1, 3).tolist()]))
        
        open('cameras.txt', 'w').write('\n'.join(pts_out))
        open('scene.txt', 'w').write('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 0.0 0.0' for l in self.pts3d.view(-1, 3).tolist()]))

        exit(1)
        """

        self.all_c2w = self.all_c2w.float().to(self.rank)
        if self.config.load_data_on_gpu:
            self.all_images = self.all_images.to(self.rank) 
            self.all_fg_masks = self.all_fg_masks.to(self.rank)
            self.all_vis_masks = self.all_vis_masks.to(self.rank)
        

class ColmapDataset(Dataset, ColmapDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class ColmapIterableDataset(IterableDataset, ColmapDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('colmap')
class ColmapDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = ColmapIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = ColmapDataset(self.config, self.config.get('val_split', 'train'))
        if stage in [None, 'test']:
            self.test_dataset = ColmapDataset(self.config, self.config.get('test_split', 'test'))
        if stage in [None, 'predict']:
            self.predict_dataset = ColmapDataset(self.config, 'train')         

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

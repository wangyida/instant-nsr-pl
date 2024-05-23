import os
import math
import numpy as np
from PIL import Image
from termcolor import colored
import json

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from datasets.colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
from models.ray_utils import get_ray_directions
from utils.misc import get_rank
from utils.pose_utils import align_global, get_center, normalize_poses, create_spheric_poses 

import os
import copy
import logging
import numpy as np
import argparse


def bin2camera(work_space, bin_file):
    """convert r3d bin data to camera ex/intrinsics"""
    try:
        cam_intrinsics, cam_rotations, cam_centers, resolutions, fns = R3DParser.LoadR3DBinDataset(work_space, bin_file)
    except:
        cam_intrinsics, cam_rotations, cam_centers, resolutions, fns = R3DUtil.LoadR3DBinDataset(work_space, bin_file)
    cam_intrinsics = cam_intrinsics.reshape(-1, 3, 3)
    resolutions = resolutions.reshape(-1, 2)
    cam_rotations = cam_rotations.reshape(-1, 3, 3)
    cam_centers = cam_centers.reshape(-1, 3)
    extrinsics = np.zeros((len(fns), 4, 4)).astype("float32")
    extrinsics[:, :3, :3] = cam_rotations
    extrinsics[:, :3, 3] = cam_centers
    extrinsics[:, 3, 3] = 1

    intrinsics = []
    for i in range(len(fns)):
        intrinsic = {
            'width': 0,
            'height': 0,
            'f': 0,
            'cx': 0,
            'cy': 0,
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
        cam_intrinsic = cam_intrinsics[i]
        intrinsic["width"] = resolutions[i][0]
        intrinsic["height"] = resolutions[i][1]
        intrinsic["cx"] = cam_intrinsic[0, 2]  # - resolutions[i][0] / 2
        intrinsic["cy"] = cam_intrinsic[1, 2]  # - resolutions[i][1] / 2
        intrinsic["f"] = cam_intrinsic[0, 0]
        intrinsics.append(copy.deepcopy(intrinsic))
    return extrinsics, intrinsics, fns


class ColmapDatasetBase():
    # the data only has to be processed once
    initialized = False
    properties = {}

    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        if not ColmapDatasetBase.initialized:
            try:
                try:
                    import R3DParser
                except:
                    import R3DUtil
                assert os.path.isfile(self.config.pose_path)
                all_c2w, intrinsics, fns = bin2camera(self.config.root_dir,
                                                      self.config.pose_path)
            except:
                all_c2w, intrinsics = np.load(os.path.join(self.config.root_dir,
                                    'extrinsics.npy'), allow_pickle=True), np.load(os.path.join(self.config.root_dir,
                                    'intrinsics.npy'), allow_pickle=True)
                try:
                    try:
                        all_c2w = np.concatenate((all_c2w[()]['FRONT'], all_c2w[()]['FRONT_LEFT'], all_c2w[()]['FRONT_RIGHT']), axis=0)
                        intrinsics = np.concatenate((intrinsics[()]['FRONT'], intrinsics[()]['FRONT_LEFT'], intrinsics[()]['FRONT_RIGHT']), axis=0)
                        print('Using', colored('front 3', 'blue'), 'cameras')
                    except:
                        all_c2w = all_c2w[()]['FRONT']
                        intrinsics = intrinsics[()]['FRONT']
                        print('Using', colored('front', 'blue'), 'camera alone')
                except:
                    print('Using', colored('all', 'blue'), 'available cameras')
                fns = sorted(os.listdir(os.path.join(self.config.root_dir,
                                    'images'))) 
                fns = [os.path.join(self.config.root_dir, 'images', fn) for fn in fns]
            mask_ori_dir = os.path.join(self.config.root_dir, 'sky_mask')
            mask_dir = os.path.join(self.config.root_dir,
                                    f'sky_mask_{self.config.img_downscale}')
            # masks labling invalid regions
            vis_mask_ori_dir = os.path.join(self.config.root_dir, f'vis_mask')
            vis_mask_dir = os.path.join(self.config.root_dir,
                                    f'vis_mask_{self.config.img_downscale}')
            # masks labling dynamic objects
            dynamic_mask_ori_dir = os.path.join(self.config.root_dir, f'moving_vehicle_bound')
            dynamic_mask_dir = os.path.join(self.config.root_dir,
                                    f'moving_vehicle_bound_{self.config.img_downscale}')
            # mesh folder
            mesh_dir = os.path.join(self.config.root_dir, 'meshes')
            os.makedirs(mesh_dir, exist_ok=True)

            all_images, all_vis_masks, all_fg_masks, all_depths, all_depth_masks, directions = [], [], [], [], [], []

            import open3d as o3d
            pts_clt = o3d.geometry.PointCloud()
            self.num_imgs = 0
            self.num_cams = 1
            for i, d in enumerate(all_c2w):
                self.num_imgs += 1
                if self.split in ['train', 'val']:
                    idx_tmp = fns[i].replace(self.config.root_dir,
                                             '')[1:].rfind('/')
                    img_folder = fns[i].replace(self.config.root_dir,
                                                '')[0:idx_tmp + 1]
                    img_path = fns[i].replace(
                        f"{img_folder}",
                        f"{img_folder}_{self.config.img_downscale}")
                    if not os.path.exists(img_path):
                        os.makedirs(os.path.dirname(img_path), exist_ok=True)
                        img_path = fns[i]
                    img = Image.open(img_path)
                    W = Image.open(fns[i]).size[0]
                    H = Image.open(fns[i]).size[1]

                    # Get camera parameters
                    if isinstance(intrinsics[i], np.ndarray):
                        if 'img_wh' in self.config:
                            w, h = self.config.img_wh
                            self.factor = w / W
                        else:
                            self.factor = 1.0 / self.config.img_downscale
                            w, h = int(W * self.factor), int(H * self.factor)
                        img_wh = (w, h)
                        intrinsic = intrinsics[i]
                        intrinsic[:2,:] *= self.factor
                        fx = intrinsic[0, 0] # camdata[1].params[0] * self.factor
                        fy = intrinsic[1, 1] # camdata[1].params[0] * self.factor
                        cx = intrinsic[0, 2]
                        cy = intrinsic[1, 2]
                    else:
                        intrinsic = intrinsics[i]
                        H = int(intrinsic["height"])
                        W = int(intrinsic["width"])

                        if 'img_wh' in self.config:
                            w, h = self.config.img_wh
                        elif 'img_downscale' in self.config:
                            w, h = int(W / self.config.img_downscale +
                                       0.5), int(H / self.config.img_downscale + 0.5)
                        else:
                            raise KeyError(
                                "Either img_wh or img_downscale should be specified.")

                        img_wh = (w, h)
                        self.factor = w / W
                        fx = fy = intrinsic["f"] * self.factor
                        cx = intrinsic["cx"] * self.factor
                        cy = intrinsic["cy"] * self.factor

                    direction = get_ray_directions(w, h, fx, fy, cx, cy).to(
                        self.rank
                    ) if self.config.load_data_on_gpu else get_ray_directions(
                        w, h, fx, fy, cx, cy).cpu()
                    directions.append(direction)

                    img_path = fns[i].replace(
                        img_folder,
                        f"{img_folder}_{self.config.img_downscale}")
                    if img.size[0] != w or img.size[1] != h:
                        img = img.resize(img_wh, Image.BICUBIC)
                        img.save(img_path)
                    # notice that to_tensor rescale the input in range [0, 1]
                    # img = TF.to_tensor(img).permute(1, 2, 0)[..., :3]
                    img = TF.pil_to_tensor(img).permute(1, 2, 0)[..., :3] / 255.0 # (4, h, w) => (h, w, 4 ) and normalize it
                    img = img.to(
                        self.rank
                    ) if self.config.load_data_on_gpu else img.cpu()
                    # NOTE: Visual masks    
                    vis_mask_path = os.path.join(
                        vis_mask_dir, 'mask' + fns[i].split("/")[-1][6:-3] + 'png')
                    if not os.path.exists(vis_mask_path) and os.path.exists(vis_mask_path.replace(f"{vis_mask_dir}",f"{vis_mask_ori_dir}")):
                        os.makedirs(os.path.dirname(vis_mask_path), exist_ok=True)
                        vis_mask_path = vis_mask_path.replace(
                            f"{vis_mask_dir}",
                            f"{vis_mask_ori_dir}")
                    if os.path.exists(vis_mask_path):
                        self.num_cams = np.max([self.num_cams, int(fns[i].split("/")[-1][7:-4])+1])
                        vis_mask = Image.open(vis_mask_path)
                        if vis_mask.size[0] != w or vis_mask.size[1] != h:
                            vis_mask = vis_mask.resize(img_wh, Image.NEAREST)
                            vis_mask_path = vis_mask_path.replace(
                                f"{vis_mask_ori_dir}",
                                f"{vis_mask_dir}")
                            vis_mask.save(vis_mask_path)
                        vis_mask = TF.to_tensor(vis_mask)[0]
                    else:
                        vis_mask = torch.ones_like(img[..., 0], device=img.device)

                    # NOTE: dynamic object mask
                    dynamic_mask_path = os.path.join(
                        dynamic_mask_dir, fns[i].split("/")[-1][:-3] + 'png')
                    if not os.path.exists(dynamic_mask_path):
                        os.makedirs(os.path.dirname(dynamic_mask_path), exist_ok=True)
                        dynamic_mask_path = dynamic_mask_path.replace(
                            f"{dynamic_mask_dir}",
                            f"{dynamic_mask_ori_dir}")
                    if os.path.exists(dynamic_mask_path):
                        dynamic_mask = Image.open(dynamic_mask_path)
                        if dynamic_mask.size[0] != w or dynamic_mask.size[1] != h:
                            dynamic_mask = dynamic_mask.resize(img_wh, Image.NEAREST)
                            dynamic_mask_path = dynamic_mask_path.replace(
                                f"{dynamic_mask_ori_dir}",
                                f"{dynamic_mask_dir}")
                            dynamic_mask.save(dynamic_mask_path)
                        vis_mask *= (1 - TF.to_tensor(dynamic_mask)[0])

                    depth_folder = 'lidar_depth'
                    if self.config.apply_depth and os.path.exists(fns[i].replace(f"{img_folder}", f"/{depth_folder}")):
                        depth_format = '.npy' # 'tif' 'npy' 'pth' 'png'
                        depth_path = fns[i].replace(
                            f"{img_folder}",
                            f"/{depth_folder}_{self.config.img_downscale}"
                        ).replace(".png", depth_format)
                        if not os.path.exists(depth_path):
                            os.makedirs(os.path.dirname(depth_path),
                                        exist_ok=True)
                            depth_path = fns[i].replace(
                                f"{img_folder}",
                                f"/{depth_folder}").replace(".png", depth_format)
                        # loading depth
                        if depth_format == '.tiff' or depth_format == '.png':
                            depth = Image.open(depth_path)
                        elif depth_format == '.npy':
                            depth = np.load(depth_path)
                            depth = Image.fromarray(depth)
                        elif depth_path == '.pth':
                            depth = torch.load(depth_path)[...,3]
                            depth = Image.fromarray(depth.numpy())
                        depth_path = fns[i].replace(
                            f"{img_folder}",
                            f"/{depth_folder}_{self.config.img_downscale}"
                        ).replace(".png", depth_format)
                        if depth.size[0] != w or depth.size[1] != h:
                            depth = depth.resize(img_wh, Image.NEAREST)
                            # NOTE Problem encountered
                            if depth_format == '.tif' or depth_format == '.png':
                                depth.save(depth_path)
                            elif depth_format == '.npy':
                                np.save(depth_path, depth)
                        depth = TF.pil_to_tensor(depth).permute(
                            1, 2, 0) # / self.config.cam_downscale
                        depth = depth.to(
                            self.rank
                        ) if self.config.load_data_on_gpu else depth.cpu()
                    elif self.config.apply_depth:
                        print(colored(fns[i].replace(f"{img_folder}", f"/{depth_folder}") + ' does not exist', 'red'))
                        depth = torch.zeros_like(img[..., 0], device=img.device) # (h, w)
                    else:
                        depth = torch.zeros_like(img[..., 0],
                                                 device=img.device)  # (h, w)
                    depth_mask = (depth > 0.0).to(bool)

                    # saving point cloud form depth
                    dep_max = 300.0 # lidar limit
                    import open3d as o3d
                    if depth.max() != 0.0:
                        # NOTE: removing dynamic objects
                        inf_mask = (depth == float("Inf"))
                        depth[inf_mask] = 0
                        depth_o3d = o3d.geometry.Image(depth.numpy() * (1-(np.array(dynamic_mask) / 255.0).astype(np.float32)[..., np.newaxis]))
                        img_o3d = o3d.geometry.Image(img.numpy())
                        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                            img_o3d,
                            depth_o3d,
                            depth_trunc=dep_max-0.01,
                            depth_scale=1,
                            convert_rgb_to_intensity=False)
                        intrin_o3d = o3d.camera.PinholeCameraIntrinsic(
                            w, h, fx, fy, cx, cy)
                        # Open3D uses world-to-camera extrinsics
                        pts_frm = o3d.geometry.PointCloud.create_from_depth_image(
                            depth_o3d, intrinsic=intrin_o3d, extrinsic=np.linalg.inv(all_c2w[i]), depth_scale=1)
                        o3d.io.write_point_cloud(os.path.join(mesh_dir, f"layout_depth_frame_{str(i + 1)}.ply"), pts_frm)
                        pts_clt.points = (o3d.utility.Vector3dVector(
                            np.concatenate((np.array(pts_clt.points), np.array(pts_frm.points)),
                                           axis=0)))

                    all_vis_masks.append(vis_mask)
                    all_images.append(img)

            print('Training with', colored(self.num_imgs, 'blue'), 'shot by', colored(self.num_cams, 'blue'), 'cameras')
            # pts_clt = pts_clt.voxel_down_sample(voxel_size=0.01)
            o3d.io.write_point_cloud(os.path.join(mesh_dir, 'layout_depth_clt.ply'), pts_clt)

            # ICP registration on purpose of coordinates alignment
            path_pts_target = os.path.join(mesh_dir, 'dlo_map.ply')
            if os.path.exists(path_pts_target):
                print(colored('Aligning to a given coordinates', 'yellow'))
                if not os.path.exists(os.path.join(mesh_dir, 'pose_crt.npy')):
                    reg_p2p_trans = align_global(mesh_dir, path_pts_target)
                else:
                    reg_p2p_trans = np.load(os.path.join(mesh_dir, 'pose_crt.npy'))
                all_c2w = np.matmul(reg_p2p_trans, all_c2w)
            else:
                print(colored('Skip aligning to a given coordinates, continue', 'green'))

            # Load point cloud which might be scanned
            if self.config.apply_depth:
                print(colored('point cloud fused by per-frame depth', 'green'))
                pts3d = np.array(pts_clt.points)
            elif self.config.pcd_path:
                assert os.path.isfile(self.config.pcd_path)
                print(colored('point cloud loaded with a given fused cloud', 'green'))
                if self.config.pcd_path.endswith('.npz'):
                    pts3d = []
                    pts3d_frames = np.load(self.config.pcd_path, allow_pickle=True)['pointcloud'].item()
                    for id_pts, pts_frame in pts3d_frames.items():
                        pts3d.append(np.array(pts_frame))
                    pts3d = np.vstack(pts3d)[:, :3]
                elif self.config.pcd_path.endswith('.ply'):
                    pts3d = np.asarray(
                        o3d.io.read_point_cloud(self.config.pcd_path).points)
            else:
                print(colored('point cloud supervision not given', 'red'))
                pts3d = None # []
                """
                pts3d = np.asarray(
                    o3d.io.read_point_cloud(self.config.pcd_path).points)
                """

            if pts3d is not None:
                # NOTE: save the point cloud using point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts3d)
                pts3d = torch.from_numpy(pts3d).float()
                """
                if not os.path.exists(os.path.join(mesh_dir, 'layout_pcd_gt.ply')):
                    # pcd = pcd.voxel_down_sample(voxel_size=0.002)
                    pcd.estimate_normals(
                                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=20))
                    o3d.io.write_point_cloud(os.path.join(mesh_dir, 'layout_pcd_gt.ply'), pcd)
                else:
                    pcd.normals = o3d.io.read_point_cloud(os.path.join(mesh_dir, 'layout_pcd_gt.ply')).normals
                """

                # Poisson surface on top of given GT points
                if os.path.exists(path_pts_target):
                    print(colored('Rasterizing on mesh presented in target coordinates', 'green'))
                    if os.path.exists(os.path.join(mesh_dir, 'layout_depth_clt_transform.ply')):
                        pcd = o3d.io.read_point_cloud(os.path.join(mesh_dir, 'layout_depth_clt_transform.ply'))
                        mesh_poisson_path = os.path.join(mesh_dir, 'layout_mesh_ps_transform.ply')
                    else:
                        pcd = o3d.io.read_point_cloud(path_pts_target)
                        mesh_poisson_path = os.path.join(mesh_dir, 'layout_mesh_ps_gt.ply')
                else:
                    mesh_poisson_path = os.path.join(mesh_dir, 'layout_mesh_ps.ply')
                    # mesh_poisson_path = os.path.join(mesh_dir, 'layout_mesh_nksr.ply')
                    print(colored('Rasterizing on mesh presented in original coordinates', 'green'))
                    pcd = o3d.io.read_point_cloud(os.path.join(mesh_dir, 'layout_depth_clt.ply'))

                if not os.path.exists(mesh_poisson_path):
                    pcd.estimate_normals(
                            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=20))
                    # pcd.orient_normals_towards_camera_location(camera_location=np.mean(all_c2w[:,:3,3] + np.array([0,0,100]), axis=0))
                    print('Extracting', colored(
                            'Poisson surface',
                            'blue'), 'on top of given GT points')
                    with o3d.utility.VerbosityContextManager(
                            o3d.utility.VerbosityLevel.Debug) as cm:
                        # Poisson surface
                        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                            pcd, depth=11, linear_fit=True) # safe with 10
                        densities = np.asarray(densities)
                        vertices_to_remove = densities < np.quantile(densities, 0.02)
                        mesh.remove_vertices_by_mask(vertices_to_remove)
                    """
                    print('Extracting', colored(
                            'NKSR surface',
                            'blue'), 'on top of given GT points')
                    import nksr
                    device = torch.device("cuda:0")
                    reconstructor = nksr.Reconstructor(device)
                    reconstructor.chunk_tmp_device = torch.device("cpu")

                    input_xyz = torch.from_numpy(np.asarray(pcd.points)).float().to(device)
                    input_sensor = torch.from_numpy(np.asarray(pcd.normals)).float().to(device)

                    field = reconstructor.reconstruct(
                        input_xyz, sensor=input_sensor, detail_level=None,
                        # Minor configs for better efficiency (not necessary)
                        approx_kernel_grad=True, solver_tol=1e-4, fused_mode=True, 
                        # Chunked reconstruction (if OOM)
                        # chunk_size=51.2,
                        preprocess_fn=nksr.get_estimate_normal_preprocess_fn(64, 85.0)
                    )
                    
                    # (Optional) Convert to CPU for mesh extraction
                    # field.to_("cpu")
                    # reconstructor.network.to("cpu")

                    mesh_nksr = field.extract_dual_mesh(mise_iter=1)
                    mesh = o3d.geometry.TriangleMesh(mesh_nksr.v, mesh_nksr.f)
                    """
                else:
                    mesh = o3d.io.read_triangle_mesh(mesh_poisson_path)

                o3d.io.write_triangle_mesh(mesh_poisson_path, mesh)
                mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

                # Create scene and add the mesh
                scene = o3d.t.geometry.RaycastingScene()
                scene.add_triangles(mesh_o3d)

            # Rasterizing dpeth from mesh
            for i, d in enumerate(all_c2w):
                if self.split in ['train', 'val']:
                    if self.config.apply_depth:
                        pts_clt = o3d.t.geometry.PointCloud.from_legacy(o3d.io.read_point_cloud(os.path.join(mesh_dir, 'layout_depth_clt.ply')))
                        depth_reproj = pts_clt.project_to_depth_image(w,
                                                                  h,
                                                                  intrinsics[i],
                                                                  np.linalg.inv(all_c2w[i]),
                                                                  depth_scale=1.0,
                                                                  depth_max=dep_max)
                        depth_reproj = Image.fromarray(np.asarray(depth_reproj.to_legacy()))

                        # save reprojected depth
                        depth_reproj_path = os.path.join(
                                self.config.root_dir,
                                f"depth_reproj_{self.config.img_downscale}")
                        os.makedirs(depth_reproj_path, exist_ok=True)

                        # visualize the hit distance (depth)
                        depth_reproj.save(
                            os.path.join(depth_reproj_path,
                                fns[i].split("/")[-1][:-3] + 'tiff'))

                        depth_reproj = TF.pil_to_tensor(depth_reproj).permute(
                            1, 2, 0)
                        inf_mask = (depth_reproj == float("Inf"))
                        depth_reproj[inf_mask] = 0
                        depth_reproj_mask = (depth_reproj > 0.0).to(bool)

                        # Rays are 6D vectors with origin and ray direction.
                        # Here we use a helper function to create rays
                        rays_mesh = scene.create_rays_pinhole(intrinsic_matrix=intrinsic, extrinsic_matrix=np.linalg.inv(all_c2w[i]), width_px=w, height_px=h)

                        # Compute the ray intersections.
                        rays_rast = scene.cast_rays(rays_mesh)
                        depth_rast = rays_rast['t_hit'].numpy()
                        norm_rast = rays_rast['primitive_normals'].numpy()

                        # add more confident depth values from the scaners
                        lidar_mask = (depth_reproj_mask != 0)[...,0]
                        # depth_rast = np.where(lidar_mask, depth_reproj[...,0], depth_rast)


                    # NOTE: foreground masks    
                    if self.config.apply_mask:
                        mask_path = os.path.join(
                            mask_dir, fns[i].split("/")[-1][:-3] + 'png')
                        # mask_paths = list(filter(os.path.exists, mask_paths))
                        # assert len(mask_paths) == 1
                        if not os.path.exists(mask_path):
                            os.makedirs(os.path.join(
                                self.config.root_dir,
                                f"sky_mask_{self.config.img_downscale}"),
                                        exist_ok=True)
                            mask_path = os.path.join(
                                mask_ori_dir,
                                fns[i].split("/")[-1][:-3] + 'png')
                        if os.path.exists(mask_path):
                            mask = Image.open(mask_path) # (H, W, 1)
                            if mask.size[0] != w or mask.size[1] != h:
                                mask = mask.resize(img_wh, Image.NEAREST)
                                mask.save(
                                    os.path.join(
                                        self.config.root_dir,
                                        f"sky_mask_{self.config.img_downscale}",
                                        fns[i].split("/")[-1][:-3] + 'png'))
                            mask = TF.to_tensor(mask)[0] # TF.pil_to_tensor does not rescale the input PIL mask
                            mask_fg = torch.ones_like(mask, device=img.device)
                            mask_fg[mask == 1] = 0 # check whether we use 1 or 255
                            mask = mask_fg
                        elif inf_mask is not None:
                            mask = (1 - inf_mask.to(int)).to(bool)
                        else:
                            print(colored('Foreground mask not available, need to disable apply_mask', 'red'))
                    else:
                        mask = torch.ones_like(img[..., 0], device=img.device)

                    if self.config.apply_depth:
                        # filter out invalid pixels in the rasted depth maps
                        inf_mask = (depth_rast == float("Inf"))
                        depth_rast[inf_mask] = 0
                        depth_rast *= mask.numpy()
                        norm_rast *= mask[..., None].numpy()

                        # save rasterized depth
                        depth_rast_path = os.path.join(
                                self.config.root_dir,
                                f"depth_rast_{self.config.img_downscale}")
                        os.makedirs(depth_rast_path, exist_ok=True)
                        np.save(os.path.join(depth_rast_path, fns[i].split("/")[-1][:-3] + 'npy'), depth_rast)

                        # save rasterized norm
                        norm_rast_path = os.path.join(
                                self.config.root_dir,
                                f"norm_rast_{self.config.img_downscale}")
                        os.makedirs(norm_rast_path, exist_ok=True)
                        np.save(os.path.join(norm_rast_path, fns[i].split("/")[-1][:-3] + 'npy'), norm_rast)

                        save_norm_dep_vis = True
                        depth_rast = Image.fromarray(depth_rast)
                        if save_norm_dep_vis:
                            # visualize the hit distance (depth)
                            depth_rast.save(
                                os.path.join(depth_rast_path,
                                    fns[i].split("/")[-1][:-3] + 'tiff'))

                            # save rasterized norm
                            norm_rast = Image.fromarray(((norm_rast + 1) * 128).astype(np.uint8))
                            norm_rast_path = os.path.join(
                                    self.config.root_dir,
                                    f"norm_rast_{self.config.img_downscale}")
                            os.makedirs(norm_rast_path, exist_ok=True)
                            norm_rast.save(
                                os.path.join(
                                    norm_rast_path,
                                    fns[i].split("/")[-1][:-3] + 'png'))

                        # convert depth into tensor
                        depth_rast = TF.pil_to_tensor(depth_rast).permute(
                            1, 2, 0) # / self.config.cam_downscale

                        depth_rast = depth_rast.to(
                            self.rank
                        ) if self.config.load_data_on_gpu else depth_rast.cpu()
                        depth_rast_mask = (depth_rast > 0.0).to(bool) # trim points outside the contraction box off
                        depth_rast_mask *= mask[..., None].to(bool)

                        all_depths.append(depth_rast) # (h, w)
                        all_depth_masks.append(depth_rast_mask)
                    all_fg_masks.append(mask) # (h, w)

            directions = torch.stack(directions, dim=0)
            all_c2w = torch.tensor(all_c2w)[:, :3].float()
            all_c2w[:, :, 1:3] *= -1.  # COLMAP => OpenGL

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

            all_depths = [dep/cam_downscale for dep in all_depths]

            ColmapDatasetBase.properties = {
                'num_imgs': self.num_imgs,
                'num_cams': self.num_cams,
                'w': w,
                'h': h,
                'img_wh': img_wh,
                'factor': self.factor,
                'has_mask': self.config.apply_mask,
                'apply_mask': self.config.apply_mask,
                'apply_depth': self.config.apply_depth,
                'directions': directions,
                'pts3d': pts3d,
                'all_c2w': all_c2w,
                'all_images': all_images,
                'all_vis_masks': all_vis_masks,
                'all_fg_masks': all_fg_masks,
                'all_depths': all_depths,
                'all_depth_masks': all_depth_masks,
                'repose_R': R,
                'repose_t': t,
                'repose_s': cam_downscale
            }

            ColmapDatasetBase.initialized = True

        for k, v in ColmapDatasetBase.properties.items():
            setattr(self, k, v)

        if self.split == 'test':
            """
            self.all_c2w = create_spheric_poses(
                self.all_c2w[:, :, 3], n_steps=self.config.n_test_traj_steps)
            """
            n_steps = torch.arange(self.num_imgs // self.num_cams)
            self.all_c2w = self.all_c2w[n_steps*self.num_cams] # front alone when there is 6 cameras
            self.config.n_test_traj_steps = int(self.num_imgs // self.num_cams)
            self.all_images = torch.zeros(
                (self.config.n_test_traj_steps, self.h, self.w, 3),
                dtype=torch.float32)
            self.all_vis_masks = torch.ones(
                (self.config.n_test_traj_steps, self.h, self.w),
                dtype=torch.float32)
            self.all_fg_masks = torch.zeros(
                (self.config.n_test_traj_steps, self.h, self.w),
                dtype=torch.float32)
            self.all_depths = torch.zeros(
                (self.config.n_test_traj_steps, self.h, self.w),
                dtype=torch.float32)
            self.all_depth_masks = torch.zeros(
                (self.config.n_test_traj_steps, self.h, self.w),
                dtype=torch.bool)
        else:
            self.all_images, self.all_vis_masks, self.all_fg_masks, self.all_depths, self.all_depth_masks = torch.stack(
                self.all_images, dim=0).float(), torch.stack(
                    self.all_vis_masks, dim=0).float(), torch.stack(
                    self.all_fg_masks, dim=0).float(), torch.stack(
                        self.all_depths,
                        dim=0).float(), torch.stack(self.all_depth_masks,
                                                    dim=0).bool()
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

        self.all_c2w = self.all_c2w.to(self.rank)
        if self.config.load_data_on_gpu:
            self.all_images = self.all_images.to(self.rank)
            self.all_fg_masks = self.all_fg_masks.to(self.rank)


class ColmapDataset(Dataset, ColmapDatasetBase):

    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        return {'index': index}


class ColmapIterableDataset(IterableDataset, ColmapDatasetBase):

    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('hmvs')
class ColmapDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = ColmapIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = ColmapDataset(
                self.config, self.config.get('val_split', 'train'))
        if stage in [None, 'test']:
            self.test_dataset = ColmapDataset(
                self.config, self.config.get('test_split', 'test'))
        if stage in [None, 'predict']:
            self.predict_dataset = ColmapDataset(self.config, 'train')

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(dataset,
                          num_workers=os.cpu_count(),
                          batch_size=batch_size,
                          pin_memory=True,
                          sampler=sampler)

    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)

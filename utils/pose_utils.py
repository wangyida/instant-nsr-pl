import open3d as o3d
import numpy as np
from termcolor import colored
import os
import copy
import pyransac3d as pyrsc
import torch
import torch.nn.functional as F

# get center
def get_center(pts):
    center = pts.mean(0)
    """
    dis = (pts - center[None,:]).norm(p=2, dim=-1)
    mean, std = dis.mean(), dis.std()
    q25, q75 = torch.quantile(dis, 0.25), torch.quantile(dis, 0.75)
    valid = (dis > mean - 1.5 * std) & (dis < mean + 1.5 * std) & (dis > mean - (q75 - q25) * 1.5) & (dis < mean + (q75 - q25) * 1.5)
    center = pts[valid].mean(0)
    """
    return center


# get rotation
def get_rot(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2 + 1e-10))


def normalize_poses(poses,
                    pts,
                    up_est_method,
                    center_est_method,
                    cam_downscale=None):
    if center_est_method == 'camera':
        # estimation scene center as the average of all camera positions
        center = poses[..., 3].mean(0)
    elif center_est_method == 'lookat':
        # estimation scene center as the average of the intersection of selected pairs of camera rays
        cams_ori = poses[..., 3]
        cams_dir = poses[:, :3, :3] @ torch.as_tensor([0., 0., -1.])
        cams_dir = F.normalize(cams_dir, dim=-1)
        A = torch.stack([cams_dir, -cams_dir.roll(1, 0)], dim=-1)
        b = -cams_ori + cams_ori.roll(1, 0)
        t = torch.linalg.lstsq(A, b).solution
        center = (torch.stack([cams_dir, cams_dir.roll(1, 0)], dim=-1) *
                  t[:, None, :] +
                  torch.stack([cams_ori, cams_ori.roll(1, 0)], dim=-1)).mean(
                      (0, 2))
    elif center_est_method == 'point':
        # first estimation scene center as the average of all camera positions
        # later we'll use the center of all points bounded by the cameras as the final scene center
        center = get_center(pts)
    else:
        raise NotImplementedError(
            f'Unknown center estimation method: {center_est_method}')

    if up_est_method == 'ground':
        # estimate up direction as the normal of the estimated ground plane
        # use RANSAC to estimate the ground plane in the point cloud
        ground = pyrsc.Plane()
        plane_eq, inliers = ground.fit(
            pts.numpy(),
            thresh=0.01)  # TODO: determine thresh based on scene scale
        plane_eq = torch.as_tensor(
            plane_eq)  # A, B, C, D in Ax + By + Cz + D = 0
        z = F.normalize(plane_eq[:3], dim=-1)  # plane normal as up direction
        signed_distance = (
            torch.cat([pts, torch.ones_like(pts[..., 0:1])], dim=-1) *
            plane_eq).sum(-1)
        if signed_distance.mean() < 0:
            z = -z  # flip the direction if points lie under the plane
    elif up_est_method == 'camera':
        # estimate up direction as the average of all camera up directions
        z = F.normalize((poses[..., 3] - center).mean(0), dim=0)
    elif up_est_method == 'z-axis':
        # center pose
        # NOTE: only need to do this when you use Nerf2Mesh to texture the mesh
        # poses[:, :3, 1:3] *= -1.  # OpenGL => COLMAP
        # full 4x4 poses
        onehot = torch.tile(torch.tensor([0., 0., 0., 1.0]),
                            (poses.size()[0], 1, 1))
        poses = torch.cat((poses, onehot), axis=1).cpu().numpy()
        # normalization
        z = poses[:, :3, 1].mean(0) / (
            np.linalg.norm(poses[:, :3, 1].mean(0)) + 1e-10)
        # rotate averaged camera up direction to [0,0,1]
        R_z = get_rot(z, [0, 0, 1])
        R_z = torch.tensor(np.pad(R_z, [0, 1])).float()
        R_z[-1, -1] = 1
        poses = torch.as_tensor(poses[:, :3]).float()
    elif up_est_method == 'no-change':
        R_z = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    else:
        raise NotImplementedError(
            f'Unknown up estimation method: {up_est_method}')

    if not up_est_method == 'no-change' and not up_est_method == 'z-axis':
        # new axis
        y_ = torch.as_tensor([z[1], -z[0], 0.])
        x = F.normalize(y_.cross(z), dim=0)
        y = z.cross(x)

    if center_est_method == 'point':
        # rotation
        if up_est_method == 'z-axis':
            Rc = R_z[:3, :3].T  
        elif up_est_method == 'no-change':
            Rc = R_z
        else:
            Rc = torch.stack([x, y, z], dim=1)
        R = Rc.T
        poses_homo = torch.cat([
            poses,
            torch.as_tensor([[[0., 0., 0., 1.]]]).expand(
                poses.shape[0], -1, -1)
        ],
                               dim=1)
        inv_trans = torch.cat([
            torch.cat([R, torch.as_tensor([[0., 0., 0.]]).T], dim=1),
            torch.as_tensor([[0., 0., 0., 1.]])
        ],
                              dim=0)
        poses_norm = (inv_trans @ poses_homo)[:, :3]
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:, 0:1])],
                                     dim=-1)[..., None])[:, :3, 0]

        # translation and scaling
        poses_min, poses_max = poses_norm[...,
                                          3].min(0)[0], poses_norm[...,
                                                                   3].max(0)[0]
        pts_fg = pts[(poses_min[0] < pts[:, 0]) & (pts[:, 0] < poses_max[0]) &
                     (poses_min[1] < pts[:, 1]) & (pts[:, 1] < poses_max[1])]
        center = get_center(pts_fg)
        tc = center.reshape(3, 1)
        t = -tc
        poses_homo = torch.cat([
            poses_norm,
            torch.as_tensor([[[0., 0., 0., 1.]]]).expand(
                poses_norm.shape[0], -1, -1)
        ],
                               dim=1)
        inv_trans = torch.cat([
            torch.cat([torch.eye(3), t], dim=1),
            torch.as_tensor([[0., 0., 0., 1.]])
        ],
                              dim=0)
        poses_norm = (inv_trans @ poses_homo)[:, :3]

        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:, 0:1])],
                                     dim=-1)[..., None])[:, :3, 0]
        if up_est_method == 'z-axis':
            # rectify convention
            # poses_norm[:, :3, 1:3] *= -1  # COLMAP => OpenGL
            poses_norm = poses_norm[:, [1, 0, 2], :]
            poses_norm[:, 2] *= -1
            pts = pts[:, [1, 0, 2]]
            pts[:, 2] *= -1

        # rescaling
        if not cam_downscale:
            # auto-scale with point cloud supervision
            cam_downscale = pts.norm(p=2, dim=-1).max()
            # auto-scale with camera positions
            # cam_downscale = poses_norm[..., 3].norm(p=2, dim=-1).min()
        poses_norm[..., 3] /= cam_downscale
        pts = pts / cam_downscale

    else:
        # rotation and translation
        if up_est_method == 'z-axis':
            Rc = R_z[:3, :3].T  
        elif up_est_method == 'no-change':
            Rc = R_z
        else:
            Rc = torch.stack([x, y, z], dim=1)
        tc = center.reshape(3, 1)
        R, t = Rc.T, -Rc.T @ tc
        poses_homo = torch.cat([
            poses,
            torch.as_tensor([[[0., 0., 0., 1.]]]).expand(
                poses.shape[0], -1, -1)
        ],
                               dim=1)
        inv_trans = torch.cat(
            [torch.cat([R, t], dim=1),
             torch.as_tensor([[0., 0., 0., 1.]])],
            dim=0)
        poses_norm = (inv_trans @ poses_homo)[:, :3]  # (N_images, 3, 4)

        # apply the transformation to the point cloud
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:, 0:1])],
                                     dim=-1)[..., None])[:, :3, 0]
        if up_est_method == 'z-axis':
            # rectify convention
            # poses_norm[:, :3, 1:3] *= -1  # COLMAP => OpenGL
            poses_norm = poses_norm[:, [1, 0, 2], :]
            poses_norm[:, 2] *= -1
            pts = pts[:, [1, 0, 2]]
            pts[:, 2] *= -1

        # rescaling
        if not cam_downscale:
            # auto-scale with point cloud supervision
            cam_downscale = pts.norm(p=2, dim=-1).max()
            # auto-scale with camera positions
            # cam_downscale = poses_norm[..., 3].norm(p=2, dim=-1).min()
        poses_norm[..., 3] /= cam_downscale
        pts = pts / cam_downscale

    return poses_norm, pts, R, t, cam_downscale

def create_spheric_poses(cameras, n_steps=120):
    center = torch.as_tensor([0., 0., 0.],
                             dtype=cameras.dtype,
                             device=cameras.device)
    mean_d = (cameras - center[None, :]).norm(p=2, dim=-1).mean()
    mean_h = cameras[:, 2].mean()
    r = (mean_d**2 - mean_h**2).sqrt()
    up = torch.as_tensor([0., 0., 1.],
                         dtype=center.dtype,
                         device=center.device)

    all_c2w = []
    for theta in torch.linspace(0, 2 * math.pi, n_steps):
        cam_pos = torch.stack([r * theta.cos(), r * theta.sin(), mean_h])
        l = F.normalize(center - cam_pos, p=2, dim=0)
        s = F.normalize(l.cross(up), p=2, dim=0)
        u = F.normalize(s.cross(l), p=2, dim=0)
        c2w = torch.cat([torch.stack([s, u, -l], dim=1), cam_pos[:, None]],
                        axis=1)
        all_c2w.append(c2w)

    all_c2w = torch.stack(all_c2w, dim=0)

    return all_c2w


def align_global(mesh_dir, path_pts_target):
    # ICP registration on purpose of coordinates alignment
    pts_clt = o3d.io.read_point_cloud(os.path.join(mesh_dir, 'layout_depth_clt.ply'))
    
    print(colored('Fused point cloud from depth images LOADED', 'blue'))
    pts_clt_transform = o3d.geometry.PointCloud()
    # GT lidar pcd
    pts_gt = o3d.io.read_point_cloud(path_pts_target)
    print(colored('Target point cloud LOADED', 'blue'))
    # global pts alignment
    threshold = 0.1
    trans_init = np.asarray([[1., 0., 0., 0.],
                             [0., 1., 0., 0.],
                             [0., 0., 1., 0.], 
                             [0., 0., 0., 1.]])
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pts_clt, pts_gt, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    print(colored('ICP registration done', 'blue'))
    pts_clt_transform = copy.deepcopy(pts_clt).transform(reg_p2p.transformation)
    np.save(os.path.join(mesh_dir, 'pose_crt.npy'), np.asarray(reg_p2p.transformation))
    o3d.io.write_point_cloud(os.path.join(mesh_dir, 'layout_depth_clt_transform.ply'), pts_clt_transform)
    return np.asarray(reg_p2p.transformation)

if __name__ == "__main__":
    root_dir = '/lpai/volumes/perception/sunhaiyang/world_model_dataset/liauto/easy_truth_240305/0501697614537981_linre_0318_6views_optim'
    mesh_dir = os.path.join(root_dir, 'meshes')
    path_pts_target = os.path.join(mesh_dir, 'dlo_map.ply')
    align_global(mesh_dir, path_pts_target)

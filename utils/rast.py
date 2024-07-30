import os
import numpy as np
from PIL import Image
import open3d as o3d

def rasterize(img_name, mesh_o3d, intrinsic, c2w, w, h, depth_rast_path, norm_rast_path):
    # Create scene and add the mesh
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_o3d)

    # Rays are 6D vectors with origin and ray direction.
    # Here we use a helper function to create rays
    rays_mesh = scene.create_rays_pinhole(intrinsic_matrix=intrinsic, extrinsic_matrix=np.linalg.inv(np.concatenate((c2w.numpy(), np.array([[0, 0, 0, 1.]])))), width_px=w, height_px=h)

    # Compute the ray intersections.
    rays_rast = scene.cast_rays(rays_mesh)

    # visualize the hit distance (depth)
    # save rasterized depth
    os.makedirs(depth_rast_path, exist_ok=True)
    if img_name.lower().endswith(('.png', '.jpg')):
        img_name = img_name[:-4]
    elif img_name.lower().endswith(('.jpeg')):
        img_name = img_name[:-5]
    np.save(os.path.join(depth_rast_path, img_name.split("/")[-1] + '.npy'), rays_rast['t_hit'].numpy())

    # save rasterized norm
    os.makedirs(norm_rast_path, exist_ok=True)
    # rays_rast['primitive_normals'].numpy()[:,:,1:3] *= -1 # OpenGL => COLMAP
    rays_rast['primitive_normals'].numpy()[:,:,:] *= -1
    np.save(os.path.join(norm_rast_path, img_name.split("/")[-1] + '.npy'), rays_rast['primitive_normals'].numpy())
    depth_rast = Image.fromarray(rays_rast['t_hit'].numpy())

    # visualize the hit distance (depth)
    norm_rast = Image.fromarray(((rays_rast['primitive_normals'].numpy() + 1) * 128).astype(np.uint8))
    depth_rast.save(
        os.path.join(depth_rast_path,
            img_name.split("/")[-1] + '.tiff'))
    norm_rast.save(
        os.path.join(
            norm_rast_path,
            img_name.split("/")[-1] + '.png'))

    return depth_rast, norm_rast

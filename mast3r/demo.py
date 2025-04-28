#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# sparse gradio demo functions
# --------------------------------------------------------
import math
import gradio
import os
import numpy as np
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation
from scipy.ndimage import zoom
from scipy import ndimage
import tempfile
import shutil
import torch
import open3d as o3d
import gc

from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from mast3r.image_pairs import make_pairs
from mast3r.retrieval.processor import Retriever

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.demo import get_args_parser as dust3r_get_args_parser

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dust3r.utils.video import process_video_for_reconstruction

import matplotlib.pyplot as pl


class SparseGAState:
    def __init__(self, sparse_ga, should_delete=False, cache_dir=None, outfile_name=None):
        self.sparse_ga = sparse_ga
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete

    def __del__(self):
        if not self.should_delete:
            return
        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None


def get_args_parser():
    parser = dust3r_get_args_parser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--gradio_delete_cache', default=None, type=int,
                        help='age/frequency at which gradio removes the file. If >0, matching cache is purged')
    parser.add_argument('--retrieval_model', default=None, type=str, help="retrieval_model to be loaded")

    actions = parser._actions
    for action in actions:
        if action.dest == 'model_name':
            action.choices = ["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"]
    # change defaults
    parser.prog = 'mast3r demo'
    return parser


def _convert_scene_output_to_glb(outfile, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            pts3d_i = pts3d[i].reshape(imgs[i].shape)
            msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile

def get_3D_model_from_scene(silent, scene_state, min_conf_thr=2, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, TSDF_thresh=0, do_sr=False):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene_state is None:
        return None
    outfile = scene_state.outfile_name
    if outfile is None:
        return None

    # get optimized values from scene
    scene = scene_state.sparse_ga
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()

    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    msk = to_numpy([c > min_conf_thr for c in confs])
    
    if do_sr:
        # Create a temporary file to store surface reconstruction results
        outpath = outfile.replace('.glb', '_surface.glb')
        
        # Merge all viewpoint point clouds and remove background and noise
        filtered_pts = []
        filtered_cols = []
        for i, (p, m, c) in enumerate(zip(pts3d, msk, rgbimg)):
            print(f"Processing view {i}")
            # Flatten the mask for indexing point clouds
            flat_mask = m.ravel()
            # Obtain valid points
            valid_pts = p[flat_mask]
    
            # Create corresponding masks for color images
            # The mask has the same shape as the image
            valid_cols = c[m]  # No need for ravel, directly use 2D mask to index 3D image
    
            # Filter invalid points
            finite_mask = np.isfinite(valid_pts.sum(axis=1))
            valid_pts = valid_pts[finite_mask]
            valid_cols = valid_cols[finite_mask]  # Ensure the color matches the dots
    
            print(f"  View {i} valid points: {len(valid_pts)}")
            filtered_pts.append(valid_pts)
            filtered_cols.append(valid_cols)

        if filtered_pts:
            pts = np.concatenate(filtered_pts, axis=0)
            cols = np.concatenate(filtered_cols, axis=0)
            print(f"Merged point cloud size: {pts.shape}, color size: {cols.shape}")

            # Final check for invalid values
            valid_msk = np.isfinite(pts.sum(axis=1))
            pts = pts[valid_msk]
            cols = cols[valid_msk]
            print(f"Point cloud size after filtering: {pts.shape}")

            if len(pts) > 1000000:
                idx = np.random.choice(len(pts), 1000000, replace=False)
                pts = pts[idx]
                cols = cols[idx]
                print(f"Point cloud size after downsampling: {pts.shape}")

            pts = pts.astype(np.float64)
            cols = np.clip(cols, 0, 1).astype(np.float64)

            pc = o3d.geometry.PointCloud()
            print("change to 3d vectors...")
            try:
                print("Starting batch processing of point clouds...")
                # Add points in batches to prevent processing too much data at once
                batch_size = 50000
                for i in range(0, len(pts), batch_size):
                    end = min(i + batch_size, len(pts))
                    # print(f"Processing points {i} to {end}")

                    if i == 0:
                        pc.points = o3d.utility.Vector3dVector(pts[:end])
                        pc.colors = o3d.utility.Vector3dVector(cols[:end])
                    else:
                        batch_pc = o3d.geometry.PointCloud()
                        batch_pc.points = o3d.utility.Vector3dVector(pts[i:end])
                        batch_pc.colors = o3d.utility.Vector3dVector(cols[i:end])
                        pc += batch_pc
                    gc.collect()
                    
                print("Point cloud processing completed, starting filtering...")
                gc.collect()
            except Exception as e:
                print(f"Point cloud creation error: {e}")
                pc = o3d.geometry.PointCloud()
                sample_pts = np.array([[0,0,0]], dtype=np.float64)
                sample_cols = np.array([[1,0,0]], dtype=np.float64)
                pc.points = o3d.utility.Vector3dVector(sample_pts)
                pc.colors = o3d.utility.Vector3dVector(sample_cols)
                print("Created placeholder point cloud, continuing execution...")
        else:
            print("Warning: No valid points!")
            # Create an empty point cloud
            pc = o3d.geometry.PointCloud()


        try:
            max_points = 200000
            if len(pc.points) > max_points:
                print(f"Forced downsampling of point cloud from {len(pc.points)} to {max_points} points")
                pc = pc.uniform_down_sample(int(len(pc.points) / max_points))

            # Use voxel downsampling as the first step to reduce density
            print("Performing voxel downsampling...")
            pc = pc.voxel_down_sample(voxel_size=0.02)
            gc.collect()
    
            print(f"Point cloud size after downsampling: {len(pc.points)}")

            # Enforce point limits within the set range
            max_safe_points = 40000
            if len(pc.points) > max_safe_points:
                print(f"Point cloud is still too large ({len(pc.points)}), forcibly limited to {max_safe_points} points.")

                # Calculate the proportion of points to skip
                skip_ratio = len(pc.points) / max_safe_points
                if skip_ratio <= 5:  # If it is not particularly large, use uniform downsampling
                    pc = pc.uniform_down_sample(int(skip_ratio))
                else:  # If there are too many points, use random downsampling
                    # Create a new point cloud
                    indices = np.random.choice(len(pc.points), max_safe_points, replace=False)
                    random_pc = o3d.geometry.PointCloud()
                    random_pc.points = o3d.utility.Vector3dVector(np.asarray(pc.points)[indices])
                    random_pc.colors = o3d.utility.Vector3dVector(np.asarray(pc.colors)[indices])
                    pc = random_pc
        
                print(f"Point cloud size after forced downsampling: {len(pc.points)}")
    
            # Use radius filtering instead of statistical filtering for less memory usage
            print("Performing radius filtering...")

            try:
                pc, ind = pc.remove_radius_outlier(nb_points=16, radius=0.05)
                if len(pc.points) > max_safe_points:
                    print(f"The point cloud is still too large after radius filtering ({len(pc.points)}), downsampling again.")
                    skip_ratio = len(pc.points) / max_safe_points
                    pc = pc.uniform_down_sample(int(skip_ratio))
            except Exception as e:
                print(f"Radius filtering failed: {e}")

            print("Performing statistical filtering...")
            try:
                pc, ind = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                pc = pc.select_by_index(ind)
                if len(pc.points) > max_safe_points:
                    print(f"The point cloud is still too large after statistical filtering ({len(pc.points)}), downsampling again.")
                    skip_ratio = len(pc.points) / max_safe_points
                    pc = pc.uniform_down_sample(int(skip_ratio))
            except Exception as e:
                print(f"Statistical filtering failed, continuing execution: {e}")
    
            # Final voxel downsampling
            print("Performing final voxel downsampling...")
            pc = pc.voxel_down_sample(voxel_size=0.01)
            if len(pc.points) > max_safe_points:
                print(f"The final point cloud is still too large ({len(pc.points)}), forcibly limited to {max_safe_points} points.")
                skip_ratio = len(pc.points) / max_safe_points
                pc = pc.uniform_down_sample(int(skip_ratio))
            
        except Exception as e:
            print(f"Point cloud filtering error: {e}")
            print("Skipping point cloud filtering, continuing execution...")

            # If the entire processing workflow fails, create a simplified version of the point cloud
            if len(pc.points) > max_safe_points:
                indices = np.random.choice(len(pc.points), max_safe_points, replace=False)
                simple_pc = o3d.geometry.PointCloud()
                simple_pc.points = o3d.utility.Vector3dVector(np.asarray(pc.points)[indices])
                simple_pc.colors = o3d.utility.Vector3dVector(np.asarray(pc.colors)[indices])
                pc = simple_pc
                print(f"Created a simplified point cloud with {len(pc.points)} points")

        gc.collect()
        
        
        # TSDF Voxel Fusion Hole Filling
        # Constructing a Scalable TSDF Voxel Fusion System
        print("Constructing a scalable TSDF voxel fusion system...")
        w,h = rgbimg[0].shape[1], rgbimg[0].shape[0]
        # Dynamically adjust TSDF parameters based on image size
        img_pixels = w * h
        points_count = len(pc.points) if pc is not None else 0
        
        # Comprehensive consideration of point cloud size and image resolution
        if points_count > 100000 or img_pixels > 250000:
            voxel_size = 0.012
            sdf_trunc = 0.03
            print(f"Large scene ({points_count} points, {w}x{h} pixels), using larger voxel size: {voxel_size}")
        elif points_count > 50000 or img_pixels > 120000:
            voxel_size = 0.008
            sdf_trunc = 0.02
            print(f"Medium scene ({points_count} points, {w}x{h} pixels), using medium voxel size: {voxel_size}")
        else:
            voxel_size = 0.004
            sdf_trunc = 0.01
            print(f"Small scene ({points_count} points, {w}x{h} pixels), using smaller voxel size: {voxel_size}")

        intr = o3d.camera.PinholeCameraIntrinsic(w, h,
                  focals[0].item(), focals[0].item(), w/2, h/2)
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        # Fuse the depth and color images for each frame
        depth_maps = to_numpy(scene.get_depthmaps())
        tsdf_success = False  # track success
        max_depth_pixels = 1000000
        
        for idx in range(len(rgbimg)):
            try:
                # Ensure RGB is the correct shape and C-contiguous
                rgb_array = np.ascontiguousarray((rgbimg[idx]*255).astype(np.uint8))
                # Reshape the depth map into a 2D image
                depth_flat = depth_maps[idx]
                print(f"Original shape of depth map: {depth_flat.shape}")
        
                # Improved Shape Inference
                if len(depth_flat) == w * h:  # If the total number of pixels matches
                    depth_reshaped = depth_flat.reshape(h, w)
                else:
                    print(f"Depth map size ({len(depth_flat)}) does not match the expected dimensions ({h}x{w}={h*w}), attempting to infer...")
            
                    # Match according to common depth map sizes
                    if len(depth_flat) == 3072:  # 64×48
                        depth_reshaped = depth_flat.reshape(64, 48)
                    elif len(depth_flat) == 2304:  # 48×48
                        depth_reshaped = depth_flat.reshape(48, 48)
                    elif len(depth_flat) == 4096:  # 64×64
                        depth_reshaped = depth_flat.reshape(64, 64)
                    else:
                        # Try to find the closest reasonable shape
                        side = int(math.sqrt(len(depth_flat)))
                        found_shape = False
                
                        # Try square
                        if abs(side*side - len(depth_flat)) < side:
                            depth_reshaped = depth_flat.reshape(side, side)
                            found_shape = True
                            print(f"Inferred depth map is square: {side}×{side}")
                
                        # Try common aspect ratios
                        if not found_shape:
                            aspect_ratios = [(4, 3), (16, 9), (3, 4), (9, 16)]
                            for ar in aspect_ratios:
                                ratio = ar[0] / ar[1]
                                test_h = int(math.sqrt(len(depth_flat) / ratio))
                                test_w = int(test_h * ratio)
                                if abs(test_h * test_w - len(depth_flat)) < test_h:
                                    try:
                                        depth_reshaped = depth_flat.reshape(test_h, test_w)
                                        found_shape = True
                                        print(f"Inferred depth map shape: {test_h}×{test_w}")
                                        break
                                    except:
                                        continue
                
                        # Final attempt to divide evenly
                        if not found_shape:
                            for h_test in range(max(1, side-10), side+10):
                                if len(depth_flat) % h_test == 0:
                                    w_test = len(depth_flat) // h_test
                                    depth_reshaped = depth_flat.reshape(h_test, w_test)
                                    found_shape = True
                                    print(f"Inferred depth map shape: {h_test}×{w_test}")
                                    break
                
                        if not found_shape:
                            raise ValueError(f"Unable to determine the shape of the depth map: {depth_flat.shape}")
            
                    # Check if depth map resolution needs to be limited
                    target_h, target_w = h, w
                    if h * w > max_depth_pixels:
                        scale_factor = math.sqrt(max_depth_pixels / (h * w))
                        target_h, target_w = int(h * scale_factor), int(w * scale_factor)
                        print(f"Limit the depth map resolution from {h}×{w} to {target_h}×{target_w}")
            
                    # Calculate scaling factor
                    zoom_factor = (target_h/depth_reshaped.shape[0], target_w/depth_reshaped.shape[1])
                    print(f"Apply zoom factor: {zoom_factor}")
            
                    # If the zoom level is very large, use tiled scaling to reduce memory usage
                    if depth_reshaped.shape[0] * depth_reshaped.shape[1] * target_h * target_w > 100000000:
                        print("Depth map scaling is too large, using block processing...")
                        # First, reduce the depth map to a medium size
                        interim_factor = (min(depth_reshaped.shape[0]*2, target_h)/depth_reshaped.shape[0], 
                                         min(depth_reshaped.shape[1]*2, target_w)/depth_reshaped.shape[1])
                        depth_reshaped = zoom(depth_reshaped, interim_factor, order=1)
                        # Rescale to target size
                        if depth_reshaped.shape != (target_h, target_w):
                            final_factor = (target_h/depth_reshaped.shape[0], 
                                           target_w/depth_reshaped.shape[1])
                            depth_reshaped = zoom(depth_reshaped, final_factor, order=1)
                        gc.collect()
                    else:
                        depth_reshaped = zoom(depth_reshaped, zoom_factor, order=1)
        
                # Preprocess depth map - enhance details
                try:
                    # Create depth map backup
                    depth_orig = depth_reshaped.copy()
            
                    # Use median filtering to reduce noise
                    depth_denoised = ndimage.median_filter(depth_reshaped, size=2)
            
                    # Retain original depth values
                    valid_mask = depth_orig > 0
                    depth_reshaped = np.where(valid_mask, depth_denoised, 0)
            
                    print("Depth map enhancement applied")
                except Exception as e:
                    print(f"Depth map enhancement failed: {e}")
        
                depth_array = np.ascontiguousarray((depth_reshaped*1000).astype(np.uint16))
        
                # Check if the array shape meets expectations
                print(f"RGB shape: {rgb_array.shape}, depth shape: {depth_array.shape}")
        
                # Handling cases where depth map and RGB sizes do not match
                if depth_array.shape[0] != rgb_array.shape[0] or depth_array.shape[1] != rgb_array.shape[1]:
                    print(f"Depth map shape {depth_array.shape} does not match RGB shape {rgb_array.shape}, adjusting depth map.")
                    # Create an empty depth map matching the size of RGB
                    matched_depth = np.zeros((rgb_array.shape[0], rgb_array.shape[1]), dtype=np.uint16)
                    # Calculate the size of the copied area
                    copy_h = min(depth_array.shape[0], matched_depth.shape[0])
                    copy_w = min(depth_array.shape[1], matched_depth.shape[1])
                    # Copy valid area
                    matched_depth[:copy_h, :copy_w] = depth_array[:copy_h, :copy_w]
                    depth_array = matched_depth
                    gc.collect()

                # Create Open3D image
                color_o3d = o3d.geometry.Image(rgb_array)
                depth_o3d = o3d.geometry.Image(depth_array)

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_o3d, depth_o3d,
                    depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
                pose = np.linalg.inv(to_numpy(cams2world)[idx])
                volume.integrate(rgbd, intr, pose)
                tsdf_success = True  # At least one frame processed successfully
                gc.collect()  # Clear memory after each frame
            except Exception as e:
                print(f"Error processing frame {idx}: {e}")
                continue

        gc.collect()
        
        print("Extracting triangular mesh...")

        use_point_cloud_mesh = False
        try:
            if tsdf_success:
                mesh_o3d = volume.extract_triangle_mesh()
                # Check if the grid is empty
                if len(mesh_o3d.vertices) == 0:
                    print("TSDF generated an empty mesh, the mesh will be generated directly using the point cloud...")
                    use_point_cloud_mesh = True
                else:
                    print(f"Successfully extracted mesh from TSDF, number of vertices: {len(mesh_o3d.vertices)}")

                    # Simplify large meshes
                    if len(mesh_o3d.vertices) > 100000:
                        print(f"Too many mesh vertices ({len(mesh_o3d.vertices)}), simplifying...")
                        try:
                            target_ratio = 0.25  # Retain 25% of the vertices
                            mesh_o3d = mesh_o3d.simplify_quadric_decimation(
                                int(len(mesh_o3d.vertices) * target_ratio))
                            print(f"Number of vertices in the simplified mesh: {len(mesh_o3d.vertices)}")
                            # Recalculate normals to ensure quality
                            mesh_o3d.compute_vertex_normals()
                            gc.collect()
                        except Exception as e:
                            print(f"Grid simplification failed: {e}")
                    
                    mesh_o3d.compute_vertex_normals()
            else:
                print("TSDF processing failed, point cloud will be used directly to generate the mesh...")
                use_point_cloud_mesh = True
        except Exception as e:
            print(f"Error extracting mesh: {e}, will generate mesh directly using point cloud...")
            use_point_cloud_mesh = True

        # If TSDF fails, directly generate the mesh from the point cloud
        if use_point_cloud_mesh:
            print("Generating mesh from point cloud...")
            try:
                # BPA algorithm creates a mesh
                radii = [0.05, 0.1, 0.2]
                mesh_o3d = None
            
                for radius in radii:
                    print(f"Trying radius {radius}...")
                    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                        pc, o3d.utility.DoubleVector([radius, radius * 2]))

                    # Check if a valid mesh has been generated
                    if len(bpa_mesh.triangles) > 10:
                        mesh_o3d = bpa_mesh
                        print(f"Successfully generated mesh with radius {radius}, number of triangles: {len(mesh_o3d.triangles)}")
                        break

                # If BPA fails, try Poisson reconstruction
                if mesh_o3d is None or len(mesh_o3d.triangles) < 10:
                    print("Attempting Poisson surface reconstruction...")
                    pc.estimate_normals()
                    mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=8)[0]

                # If Poisson also fails, use the alpha shape
                if mesh_o3d is None or len(mesh_o3d.triangles) < 10:
                    print("Attempting alpha shape reconstruction...")
                    mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pc, 0.1)
            
                # Final option: simple alpha shape
                if mesh_o3d is None or len(mesh_o3d.triangles) < 10:
                    print("Using a simple alpha shape...")
                    mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pc, 1.0)
            
                mesh_o3d.compute_vertex_normals()

            except Exception as e:
                print(f"Failed to generate mesh from point cloud: {e}")
                # Create a minimalist grid
                mesh_o3d = o3d.geometry.TriangleMesh()
                mesh_o3d.vertices = pc.points
                mesh_o3d.vertex_colors = pc.colors
        
        gc.collect()

        # Assign colors to the vertices of the triangular mesh
        # Check the color of the nearest neighbor for each vertex
        verts = np.asarray(mesh_o3d.vertices)
        pcd_pts = np.asarray(pc.points)
        pcd_cols = np.asarray(pc.colors)

        batch_size = 10000
        vertex_colors = np.zeros((len(verts), 3))

        try:
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=1).fit(pcd_pts)
    
            # Batch processing vertex color assignment
            for i in range(0, len(verts), batch_size):
                end = min(i + batch_size, len(verts))
                # print(f"Processing vertex colors {i} to {end}")
                batch_verts = verts[i:end]
                _, idxs = nbrs.kneighbors(batch_verts)
                vertex_colors[i:end] = pcd_cols[idxs[:,0]]
                gc.collect()

            print(f"Processing vertex colors successfully.")
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        except Exception as e:
            print(f"Vertex color assignment failed: {e}")
            # Revert to simple color assignment
            try:
                # Simply use the average color of the point cloud
                avg_color = np.mean(pcd_cols, axis=0)
                uniform_colors = np.tile(avg_color, (len(verts), 1))
                mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(uniform_colors)
            except:
                pass

        # Export colored glb
        tri = trimesh.Trimesh(
            vertices=np.asarray(mesh_o3d.vertices),
            faces=np.asarray(mesh_o3d.triangles),
            vertex_colors=np.asarray(mesh_o3d.vertex_colors)
        )
        tri.export(outpath)
        if not silent:
            print(f'(export surface mesh to {outpath})')
        outfile = outpath
    else:
        # Generate model using the original method
        outfile = _convert_scene_output_to_glb(outfile, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)
    return outfile

def set_scenegraph_options(inputfiles, win_cyclic, refid, scenegraph_type):
    num_files = len(inputfiles) if inputfiles is not None else 1
    max_winsize, min_winsize = 1, 1

    winsize = gradio.Slider(visible=False)
    win_cyclic = gradio.Checkbox(visible=False)
    graph_opt = gradio.Column(visible=False)
    refid = gradio.Slider(visible=False)

    if scenegraph_type in ["swin", "logwin"]:
        if scenegraph_type == "swin":
            if win_cyclic:
                max_winsize = max(1, math.ceil((num_files - 1) / 2))
            else:
                max_winsize = num_files - 1
        else:
            if win_cyclic:
                half_size = math.ceil((num_files - 1) / 2)
                max_winsize = max(1, math.ceil(math.log(half_size, 2)))
            else:
                max_winsize = max(1, math.ceil(math.log(num_files, 2)))

        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=min_winsize, maximum=max_winsize, step=1, visible=True)
        win_cyclic = gradio.Checkbox(value=win_cyclic, label="Cyclic sequence", visible=True)
        graph_opt = gradio.Column(visible=True)
        refid = gradio.Slider(visible=False)

    elif scenegraph_type == "retrieval":
        graph_opt = gradio.Column(visible=True)
        winsize = gradio.Slider(label="Retrieval: Num. key images", value=min(20, num_files),
                                minimum=0, maximum=num_files, step=1, visible=True)
        win_cyclic = gradio.Checkbox(visible=False)
        refid = gradio.Slider(label="Retrieval: Num neighbors", value=min(num_files - 1, 10), minimum=1,
                              maximum=num_files - 1, step=1, visible=True)

    elif scenegraph_type == "oneref":
        graph_opt = gradio.Column(visible=True)
        winsize = gradio.Slider(visible=False)
        win_cyclic = gradio.Checkbox(visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files - 1, step=1, visible=True)

    return graph_opt, winsize, win_cyclic, refid

def get_reconstructed_scene(outdir, gradio_delete_cache, model, retrieval_model, device, silent, image_size,
                            current_scene_state, filelist, optim_level, lr1, niter1, lr2, niter2, min_conf_thr,
                            matching_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics, do_sr=False, **kw):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    # Add robustness checks for filelist
    if filelist is None:
        if not silent:
            print("error: No input image or video frame")
        return current_scene_state, None
    
    # Ensure filelist is a list
    if not isinstance(filelist, list):
        filelist = [filelist]
    
    # Handle empty list cases
    if len(filelist) == 0:
        if not silent:
            print("error: File list is empty")
        return current_scene_state, None
    
    # Handling the output format of the Gradio File component
    if isinstance(filelist[0], tuple) and len(filelist[0]) >= 1:
        # Extract file paths from tuples
        filelist = [f[0] for f in filelist if isinstance(f, tuple) and len(f) >= 1]
        if not silent:
            print(f"extracted {len(filelist)} file paths from the tuple")
    
    # Ensure all paths exist
    valid_files = [f for f in filelist if os.path.exists(f)]
    if len(valid_files) != len(filelist):
        if not silent:
            print(f"warning: {len(filelist) - len(valid_files)} file paths are invalid")
        
    if len(valid_files) == 0:
        if not silent:
            print("error: No valid file")
        return current_scene_state, None
    
    filelist = valid_files
    
    # The original code continues to execute
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    elif scenegraph_type == "retrieval":
        scene_graph_params.append(str(winsize))  # Na
        scene_graph_params.append(str(refid))  # k

    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)

    sim_matrix = None
    if 'retrieval' in scenegraph_type:
        assert retrieval_model is not None
        retriever = Retriever(retrieval_model, backbone=model, device=device)
        with torch.no_grad():
            sim_matrix = retriever(filelist)

        # Cleanup
        del retriever
        torch.cuda.empty_cache()

    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=sim_matrix)
    if optim_level == 'coarse':
        niter2 = 0
    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.cache_dir is not None:
        cache_dir = current_scene_state.cache_dir
    elif gradio_delete_cache:
        cache_dir = tempfile.mkdtemp(suffix='_cache', dir=outdir)
    else:
        cache_dir = os.path.join(outdir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr, **kw)
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.outfile_name is not None:
        outfile_name = current_scene_state.outfile_name
    else:
        outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=outdir)

    scene_state = SparseGAState(scene, gradio_delete_cache, cache_dir, outfile_name)
    outfile = get_3D_model_from_scene(silent, scene_state, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size, TSDF_thresh, do_sr)
    return scene_state, outfile

def main_demo(tmpdirname, model, retrieval_model, device, image_size, server_name, server_port, silent=False,
              share=False, gradio_delete_cache=False):
    if not silent:
        print('Outputing stuff in', tmpdirname)

    recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, gradio_delete_cache, model,
                                  retrieval_model, device, silent, image_size)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, silent)

    available_scenegraph_type = [("complete: all possible image pairs", "complete"),
                                 ("swin: sliding window", "swin"),
                                 ("logwin: sliding window with long range", "logwin"),
                                 ("oneref: match one image with all", "oneref")]
    if retrieval_model is not None:
        available_scenegraph_type.insert(1, ("retrieval: connect views based on similarity", "retrieval"))

    def get_context(delete_cache):
        css = """.gradio-container {margin: 0 !important; min-width: 100%};"""
        title = "MASt3R Demo"
        if delete_cache:
            return gradio.Blocks(css=css, title=title, delete_cache=bool(delete_cache))
        else:
            return gradio.Blocks(css=css, title="MASt3R Demo")  # for compatibility with older versions

    with get_context(gradio_delete_cache) as demo:
        # scene state is saved so that you can change conf_thr, cam_size... without rerunning the inference
        scene = gradio.State(None)
        gradio.HTML('<h2 style="text-align: center;">MASt3R Demo</h2>')
        
        # Added: Video Processing UI Section
        with gradio.Column():
            # Create tabs to differentiate image and video inputs
            with gradio.Tabs() as input_tabs:
                with gradio.TabItem("Images"):
                    inputfiles = gradio.File(file_count="multiple", label="Upload Images")
                
                with gradio.TabItem("Video"):
                    inputvideo = gradio.Video(label="Upload Video")
                    with gradio.Row():
                        video_frames = gradio.Slider(
                            label="Max Frames", 
                            value=10, 
                            minimum=4, 
                            maximum=20, 
                            step=2,
                            info="Maximum number of frames to extract"
                        )
                        video_method = gradio.Dropdown(
                            ["uniform", "interval", "keyframe"], 
                            value="uniform", 
                            label="Frame Extract Method",
                            info="uniform: evenly spaced, interval: fixed interval, keyframe: scene detection"
                        )
                    
                    with gradio.Row():
                        video_interval = gradio.Slider(
                            label="Frame Interval", 
                            value=15, 
                            minimum=1, 
                            maximum=60, 
                            step=1, 
                            visible=False,
                            info="Extract every N frames (only when interval method is selected)"
                        )
                        
                        video_preview = gradio.Gallery(
                            label="Extracted Frames Preview",
                            visible=False,
                            columns=5,
                            rows=2,
                            height="200px"
                        )
                    
                    video_process_btn = gradio.Button("Extract Frames from Video")
            # End of video processing UI section
            
            # The original interface controls remain unchanged
            with gradio.Row():
                with gradio.Column():
                    with gradio.Row():
                        lr1 = gradio.Slider(label="Coarse LR", value=0.07, minimum=0.01, maximum=0.2, step=0.01)
                        niter1 = gradio.Slider(value=300, minimum=0, maximum=1000, step=1,
                                               label="Iterations", info="For coarse alignment")
                        lr2 = gradio.Slider(label="Fine LR", value=0.01, minimum=0.005, maximum=0.05, step=0.001)
                        niter2 = gradio.Slider(value=300, minimum=0, maximum=1000, step=1,
                                               label="Iterations", info="For refinement")
                        optim_level = gradio.Dropdown(["coarse", "refine", "refine+depth"],
                                                      value='refine+depth', label="OptLevel",
                                                      info="Optimization level")
                    with gradio.Row():
                        matching_conf_thr = gradio.Slider(label="Matching Confidence Thr", value=0.,
                                                          minimum=0., maximum=30., step=0.1,
                                                          info="Before Fallback to Regr3D!")
                        shared_intrinsics = gradio.Checkbox(value=False, label="Shared intrinsics",
                                                            info="Only optimize one set of intrinsics for all views")
                        scenegraph_type = gradio.Dropdown(available_scenegraph_type,
                                                          value='complete', label="Scenegraph",
                                                          info="Define how to make pairs",
                                                          interactive=True)
                        with gradio.Column(visible=False) as graph_opt:
                            winsize = gradio.Slider(label="Scene Graph: Window Size", value=1,
                                                    minimum=1, maximum=1, step=1)
                            win_cyclic = gradio.Checkbox(value=False, label="Cyclic sequence")
                            refid = gradio.Slider(label="Scene Graph: Id", value=0,
                                                  minimum=0, maximum=0, step=1, visible=False)

            run_btn = gradio.Button("Run")

            with gradio.Row():
                # adjust the confidence threshold
                min_conf_thr = gradio.Slider(label="min_conf_thr", value=1.5, minimum=0.0, maximum=10, step=0.1)
                # adjust the camera size in the output pointcloud
                cam_size = gradio.Slider(label="cam_size", value=0.2, minimum=0.001, maximum=1.0, step=0.001)
                TSDF_thresh = gradio.Slider(label="TSDF Threshold", value=0., minimum=0., maximum=1., step=0.01)
            with gradio.Row():
                as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
                # two post process implemented
                mask_sky = gradio.Checkbox(value=False, label="Mask sky")
                clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
                transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")
                # Add surface reconstruction checkbox
                do_sr = gradio.Checkbox(value=False, label="Surface Reconstruction (Open3D)")

            outmodel = gradio.Model3D()

            # Added: Video processing feature
            # Video method change event
            def update_interval_visibility(method):
                return {"visible": method == "interval"}
            
            # Video processing function
            def process_video(video_path, max_frames, method, interval=15):
                if video_path is None or not os.path.exists(video_path):
                    print("error: invalid path")
                    return [], []
    
                try:
                    # Create a temporary directory to store the extracted frames
                    output_dir = tempfile.mkdtemp(prefix='mast3r_video_', dir=tmpdirname)
        
                    if not silent:
                        print(f"video: {video_path}")
                        print(f"method: {method}, max frames: {max_frames}")
        
                    # Extract video frames
                    frame_paths = process_video_for_reconstruction(
                        video_path,
                        max_frames=max_frames,
                        method=method,
                        interval=interval if method == "interval" else 15,
                        output_dir=output_dir,
                        verbose=not silent
                    )
        
                    # Format required for preparing file components [(path, name), ...]
                    file_components = []
                    for path in frame_paths:
                        file_name = os.path.basename(path)
                        file_components.append((path, file_name))
                    print("loading...")
            
                    if not silent:
                        print(f"ready to pass {len(file_components)} files to the process")
                        # Print several example paths
                        if len(file_components) > 0:
                            print(f"sample file: {file_components[0]}")
        
                    # Select a few frames for preview
                    preview_paths = []
                    if len(frame_paths) > 0:
                        # Select up to 10 frames for preview
                        indices = np.linspace(0, len(frame_paths) - 1, min(10, len(frame_paths)), dtype=int)
                        preview_paths = [frame_paths[i] for i in indices]
        
                    return frame_paths, preview_paths
        
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"error generating file: {e}")
                    return [], []
            
            # Video method change event
            video_method.change(
                update_interval_visibility,
                inputs=[video_method],
                outputs=[video_interval]
            )

            # Video extraction button event
            video_process_btn.click(
                process_video,
                inputs=[inputvideo, video_frames, video_method, video_interval],
                outputs=[inputfiles, video_preview]
            )

            # Add logging functionality to help track file transfers
            def log_inputfiles(files):
                """Record input file information for debugging purposes"""
                try:
                    if files is None:
                        print("warning: input None")
                        return files
            
                    if not isinstance(files, list):
                        print(f"invalid input file: {type(files)}")
                        return files
            
                    print(f"recieve {len(files)} files")
                    if len(files) > 0:
                        print(f"file type of first file: {type(files[0])}")
                        if isinstance(files[0], tuple) and len(files[0]) >= 1:
                            print(f"sample path: {files[0][0]}")
                except Exception as e:
                    print(f"Error when loading file: {e}")
                return files

            # Add file input monitoring
            inputfiles.change(
                log_inputfiles,
                inputs=[inputfiles],
                outputs=[inputfiles]  # Using the same component as output is just to keep the process going
            )
            # End of video processing function

            scenegraph_type.change(set_scenegraph_options,
                                   inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                                   outputs=[graph_opt, winsize, win_cyclic, refid])
            inputfiles.change(set_scenegraph_options,
                              inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                              outputs=[graph_opt, winsize, win_cyclic, refid])
            win_cyclic.change(set_scenegraph_options,
                              inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                              outputs=[graph_opt, winsize, win_cyclic, refid])
            run_btn.click(fn=recon_fun,
                          inputs=[scene, inputfiles, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                                  as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                                  scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics, do_sr],
                          outputs=[scene, outmodel])
            min_conf_thr.release(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, TSDF_thresh, do_sr],
                                 outputs=outmodel)
            cam_size.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, TSDF_thresh, do_sr],
                            outputs=outmodel)
            TSDF_thresh.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size, TSDF_thresh, do_sr],
                               outputs=outmodel)
            as_pointcloud.change(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, TSDF_thresh, do_sr],
                                 outputs=outmodel)
            mask_sky.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, TSDF_thresh, do_sr],
                            outputs=outmodel)
            clean_depth.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size, TSDF_thresh, do_sr],
                               outputs=outmodel)
            transparent_cams.change(model_from_scene_fun,
                                    inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                            clean_depth, transparent_cams, cam_size, TSDF_thresh, do_sr],
                                    outputs=outmodel)
            # Add do_sr change event
            do_sr.change(fn=model_from_scene_fun,
                         inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                 clean_depth, transparent_cams, cam_size, TSDF_thresh, do_sr],
                         outputs=outmodel)
    demo.launch(share=share, server_name=server_name, server_port=server_port)
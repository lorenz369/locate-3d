#!/usr/bin/env python3
"""
Wrapper to preprocess MASt3R-SLAM output using locate3d preprocessing pipeline.

This script adapts the locate3d preprocessing pipeline to work with MASt3R-SLAM output,
which includes keyframes (RGB images), depth maps, and camera poses with intrinsics.

Expected MASt3R-SLAM output structure:
- keyframes/: RGB images as PNG files (000XXX.png)
- depth_maps/: Depth maps as NPY files (000XXX.npy)
- poses_file.txt: Camera poses and intrinsics (frame_id tx ty tz qx qy qz qw width height fx fy cx cy ...)
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
import logging

# Add the parent directory to Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from pointcloud_featurizer import FeatureLifter3D

logger = logging.getLogger(__name__)


def validate_input_data(data_dir, poses_file):
    """Validate that input data exists and has the expected structure."""
    data_path = Path(data_dir)
    keyframes_dir = data_path / "keyframes"
    depth_maps_dir = data_path / "depth_maps"
    
    errors = []
    
    if not data_path.exists():
        errors.append(f"Data directory does not exist: {data_dir}")
        return errors
    
    if not keyframes_dir.exists():
        errors.append(f"Keyframes directory not found: {keyframes_dir}")
    
    if not depth_maps_dir.exists():
        errors.append(f"Depth maps directory not found: {depth_maps_dir}")
    
    if not Path(poses_file).exists():
        errors.append(f"Poses file not found: {poses_file}")
    
    if errors:
        return errors
    
    # Check if we have matching keyframes and depth maps
    keyframe_ids = set()
    depth_ids = set()
    
    for kf_file in keyframes_dir.glob("*.png"):
        try:
            frame_id = int(kf_file.stem)
            keyframe_ids.add(frame_id)
        except ValueError:
            print(f"Warning: Invalid keyframe filename: {kf_file.name}")
    
    for depth_file in depth_maps_dir.glob("*.npy"):
        try:
            frame_id = int(depth_file.stem)
            depth_ids.add(frame_id)
        except ValueError:
            print(f"Warning: Invalid depth filename: {depth_file.name}")
    
    matching_ids = keyframe_ids & depth_ids
    if not matching_ids:
        errors.append("No matching keyframe and depth map pairs found")
    else:
        print(f"Found {len(matching_ids)} frames with both keyframe and depth data")
        print(f"Frame ID range: {min(matching_ids)} - {max(matching_ids)}")
    
    return errors


def parse_mast3r_poses(poses_file):
    """
    Parse MASt3R-SLAM poses file.
    
    Format: frame_id tx ty tz qx qy qz qw width height fx fy cx cy ...
    Returns dict with frame_ids as keys and pose/intrinsics as values
    """
    poses_data = {}
    
    print(f"Parsing poses file: {poses_file}")
    
    with open(poses_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 12:
                if parts:  # Skip empty lines silently
                    print(f"Warning: Line {line_num} has insufficient data ({len(parts)} fields), skipping")
                continue
                
            try:
                frame_id = int(parts[0])
                
                # Translation (tx, ty, tz)
                translation = [float(parts[1]), float(parts[2]), float(parts[3])]
                
                # Quaternion (qx, qy, qz, qw)
                quaternion = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
                
                # Camera intrinsics
                width = int(parts[8])
                height = int(parts[9])
                fx = float(parts[10])
                fy = float(parts[11])
                cx = float(parts[12]) if len(parts) > 12 else width / 2.0
                cy = float(parts[13]) if len(parts) > 13 else height / 2.0
                
                poses_data[frame_id] = {
                    'translation': translation,
                    'quaternion': quaternion,
                    'width': width,
                    'height': height,
                    'fx': fx,
                    'fy': fy,
                    'cx': cx,
                    'cy': cy
                }
            except (ValueError, IndexError) as e:
                print(f"Warning: Error parsing line {line_num}: {e}")
                continue
    
    print(f"Successfully parsed {len(poses_data)} camera poses")
    return poses_data


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    qx, qy, qz, qw = q
    
    # Normalize quaternion
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    
    return R


def create_camera_to_world_matrix(translation, quaternion):
    """Create 4x4 camera-to-world transformation matrix."""
    R = quaternion_to_rotation_matrix(quaternion)
    t = np.array(translation).reshape(3, 1)
    
    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    
    return T


def create_intrinsic_matrix(fx, fy, cx, cy):
    """Create 3x3 camera intrinsic matrix."""
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K


def load_mast3r_data(data_dir, poses_file, max_frames=None):
    """
    Load MASt3R-SLAM data and convert to locate3d format.
    
    Returns:
        camera_views: dict with keys 'rgb', 'depth_zbuffer', 'cam_to_world', 'cam_K'
    """
    # Validate input data first
    validation_errors = validate_input_data(data_dir, poses_file)
    if validation_errors:
        print("Validation errors found:")
        for error in validation_errors:
            print(f"  - {error}")
        raise ValueError("Input data validation failed")
    
    keyframes_dir = Path(data_dir) / "keyframes"
    depth_maps_dir = Path(data_dir) / "depth_maps"
    
    # Parse poses
    poses_data = parse_mast3r_poses(poses_file)
    
    # Find frames that have both keyframe, depth, and pose data
    available_frame_ids = set(poses_data.keys())
    
    keyframe_ids = set()
    for kf_file in keyframes_dir.glob("*.png"):
        try:
            frame_id = int(kf_file.stem)
            keyframe_ids.add(frame_id)
        except ValueError:
            continue
    
    depth_ids = set()
    for depth_file in depth_maps_dir.glob("*.npy"):
        try:
            frame_id = int(depth_file.stem)
            depth_ids.add(frame_id)
        except ValueError:
            continue
    
    # Get intersection of all three sets
    valid_frame_ids = available_frame_ids & keyframe_ids & depth_ids
    valid_frame_ids = sorted(valid_frame_ids)
    
    if not valid_frame_ids:
        raise ValueError("No frames found with complete data (keyframe + depth + pose)")
    
    if max_frames:
        valid_frame_ids = valid_frame_ids[:max_frames]
    
    print(f"Loading {len(valid_frame_ids)} frames with complete data...")
    if len(valid_frame_ids) < len(available_frame_ids):
        missing_count = len(available_frame_ids) - len(valid_frame_ids)
        print(f"Note: {missing_count} frames skipped due to missing keyframe or depth data")
    
    rgb_images = []
    depth_maps = []
    cam_to_world_matrices = []
    cam_K_matrices = []
    loaded_frame_ids = []
    
    for frame_id in valid_frame_ids:
        try:
            # Load RGB image
            rgb_path = keyframes_dir / f"{frame_id:06d}.png"
            rgb_image = Image.open(rgb_path).convert('RGB')
            rgb_array = np.array(rgb_image)
            
            # Load depth map
            depth_path = depth_maps_dir / f"{frame_id:06d}.npy"
            depth_map = np.load(depth_path)
            
            # Validate depth map
            if depth_map.size == 0:
                print(f"Warning: Empty depth map for frame {frame_id}, skipping")
                continue
            
            # Get pose and intrinsics
            pose_data = poses_data[frame_id]
            
            # Create camera-to-world matrix
            cam_to_world = create_camera_to_world_matrix(
                pose_data['translation'], 
                pose_data['quaternion']
            )
            
            # Create intrinsic matrix  
            cam_K = create_intrinsic_matrix(
                pose_data['fx'], pose_data['fy'], 
                pose_data['cx'], pose_data['cy']
            )
            
            # Append to lists
            rgb_images.append(rgb_array)
            depth_maps.append(depth_map)
            cam_to_world_matrices.append(cam_to_world)
            cam_K_matrices.append(cam_K)
            loaded_frame_ids.append(frame_id)
            
        except Exception as e:
            print(f"Warning: Failed to load frame {frame_id}: {e}")
            continue
    
    if not rgb_images:
        raise ValueError("No frames could be loaded successfully")
    
    print(f"Successfully loaded {len(rgb_images)} frames")
    print(f"Loaded frame IDs: {loaded_frame_ids[:5]}{'...' if len(loaded_frame_ids) > 5 else ''}")
    
    # Convert to tensors
    camera_views = {
        'rgb': torch.tensor(np.stack(rgb_images), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0,  # Convert to (N, C, H, W) and normalize to [0,1]
        'depth_zbuffer': torch.tensor(np.stack(depth_maps), dtype=torch.float32),
        'cam_to_world': torch.tensor(np.stack(cam_to_world_matrices), dtype=torch.float32),
        'cam_K': torch.tensor(np.stack(cam_K_matrices), dtype=torch.float32)
    }
    
    return camera_views


def main():
    parser = argparse.ArgumentParser(description="Process MASt3R-SLAM data with locate3d preprocessing")
    parser.add_argument(
        "--mast3r_data_dir", 
        type=str, 
        required=True,
        help="Path to MASt3R-SLAM output directory (containing keyframes/ and depth_maps/)"
    )
    parser.add_argument(
        "--poses_file", 
        type=str, 
        required=True,
        help="Path to MASt3R-SLAM poses file (with intrinsics)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Output directory to save preprocessed point clouds"
    )
    parser.add_argument(
        "--scene_name", 
        type=str, 
        default="mast3r_scene",
        help="Name for the output scene"
    )
    parser.add_argument(
        "--max_frames", 
        type=int, 
        default=None,
        help="Maximum number of frames to process (for testing)"
    )
    parser.add_argument(
        "--config_type", 
        choices=["clip", "dino", "both"], 
        default="both",
        help="Which feature extraction to use"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load MASt3R-SLAM data
    camera_views = load_mast3r_data(
        args.mast3r_data_dir, 
        args.poses_file,
        max_frames=args.max_frames
    )
    
    # Initialize feature extractors
    feature_extractors = {}
    
    if args.config_type in ["clip", "both"]:
        clip_cfg = OmegaConf.load(os.path.join(current_dir, "config/clip.yaml"))
        feature_extractors["clip"] = FeatureLifter3D(clip_cfg)
        
    if args.config_type in ["dino", "both"]:
        dino_cfg = OmegaConf.load(os.path.join(current_dir, "config/dino.yaml"))
        feature_extractors["dino"] = FeatureLifter3D(dino_cfg)
    
    # Process with each feature extractor
    for feat_type, extractor in feature_extractors.items():
        print(f"Processing with {feat_type.upper()} features...")
        
        # Extract features
        torch.manual_seed(0)  # For reproducibility
        extracted_features = extractor.lift_frames(camera_views)
        
        # Prepare output dict
        if feat_type == "clip":
            output_dict = {
                "points": extracted_features["points_reduced"],
                "rgb": extracted_features["rgb_reduced"],
                "features_clip": extracted_features["features_reduced"],
                "features_dino": None,  # Will be filled if both are used
            }
        else:  # dino
            output_dict = {
                "points": extracted_features["points_reduced"],
                "rgb": extracted_features["rgb_reduced"],
                "features_clip": None,  # Will be filled if both are used
                "features_dino": extracted_features["features_reduced"],
            }
        
        # Save the result
        output_file = os.path.join(args.output_dir, f"{args.scene_name}_{feat_type}.pt")
        torch.save(output_dict, output_file)
        print(f"Saved {feat_type} features to: {output_file}")
        
        # Print statistics
        print(f"Point cloud statistics for {feat_type}:")
        print(f"  - Number of points: {len(output_dict['points'])}")
        print(f"  - Point cloud bounds:")
        points = output_dict['points']
        print(f"    X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"    Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"    Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        if output_dict[f'features_{feat_type}'] is not None:
            print(f"  - Feature dimension: {output_dict[f'features_{feat_type}'].shape[-1]}")
    
    # If both features were extracted, create a combined version
    if args.config_type == "both" and len(feature_extractors) == 2:
        print("Creating combined CLIP+DINO features...")
        
        # Re-extract features to ensure consistency
        torch.manual_seed(0)
        clip_features = feature_extractors["clip"].lift_frames(camera_views)
        torch.manual_seed(0)
        dino_features = feature_extractors["dino"].lift_frames(camera_views)
        
        combined_output = {
            "points": dino_features["points_reduced"],  # Use DINO points as reference
            "rgb": dino_features["rgb_reduced"],
            "features_clip": clip_features["features_reduced"],
            "features_dino": dino_features["features_reduced"],
        }
        
        combined_file = os.path.join(args.output_dir, f"{args.scene_name}_combined.pt")
        torch.save(combined_output, combined_file)
        print(f"Saved combined features to: {combined_file}")
    
    print("Processing completed!")


if __name__ == "__main__":
    main() 
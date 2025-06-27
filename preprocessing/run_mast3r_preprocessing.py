#!/usr/bin/env python3
"""
Example script to run MASt3R-SLAM preprocessing using the locate3d pipeline.

This script demonstrates how to use the mast3r_slam_wrapper.py to process
MASt3R-SLAM output data and create featurized point clouds for 3D search.
"""

import os
import sys
from pathlib import Path
import subprocess

# Add the parent directory to Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

def validate_scene_directory(scene_dir):
    """
    Validate that a directory contains the required MASt3R-SLAM data.
    
    Returns:
        dict: Scene info with paths, or None if invalid
    """
    scene_path = Path(scene_dir)
    
    if not scene_path.exists():
        print(f"Error: Scene directory not found: {scene_dir}")
        return None
    
    keyframes_dir = scene_path / "keyframes"
    depth_maps_dir = scene_path / "depth_maps"
    
    if not (keyframes_dir.exists() and depth_maps_dir.exists()):
        print("Error: Missing keyframes/ or depth_maps/ directory")
        return None
    
    # Look for poses file
    poses_file = None
    for pattern in ["*_with_intrinsics.txt", "*intrinsics*.txt", "*.txt"]:
        poses_files = list(scene_path.glob(pattern))
        if poses_files:
            poses_file = poses_files[0]  # Take the first match
            break
    
    if poses_file is None:
        print("Error: No poses file found")
        return None
    
    # Count available data
    keyframe_count = len(list(keyframes_dir.glob("*.png")))
    depth_count = len(list(depth_maps_dir.glob("*.npy")))
    
    if keyframe_count == 0 or depth_count == 0:
        print("Error: No keyframes or depth maps found")
        return None
    
    return {
        'scene_dir': scene_path,
        'keyframes_dir': keyframes_dir,
        'depth_maps_dir': depth_maps_dir,
        'poses_file': poses_file,
        'keyframe_count': keyframe_count,
        'depth_count': depth_count
    }

def main():
    # Configuration - update these paths for your setup
    default_scene_dir = "../../MASt3R-SLAM/logs/test_video1_all_kfs"
    output_base_dir = "./output_pointclouds"
    
    print("MASt3R-SLAM to Locate3D Preprocessing")
    print("=" * 50)
    
    # Get scene directory
    scene_dir = input(f"Enter scene directory [{default_scene_dir}]: ").strip() or default_scene_dir
    
    # Validate scene directory
    scene_info = validate_scene_directory(scene_dir)
    if not scene_info:
        print("\nExpected directory structure:")
        print("  scene_directory/")
        print("  ├── keyframes/")
        print("  │   └── *.png")
        print("  ├── depth_maps/")
        print("  │   └── *.npy")
        print("  └── *_with_intrinsics.txt")
        return
    
    # Display scene info
    print(f"\nScene information:")
    print(f"  Keyframes: {scene_info['keyframe_count']}")
    print(f"  Depth maps: {scene_info['depth_count']}")
    print(f"  Poses file: {scene_info['poses_file'].name}")
    
    # Get processing options
    print("\nProcessing Options:")
    config_type = input("Feature extraction type (clip/dino/both) [both]: ").strip() or "both"
    max_frames_input = input("Max frames to process (leave empty for all): ").strip()
    max_frames = int(max_frames_input) if max_frames_input else None
    
    # Check if SAM weights exist
    sam_weights_path = "./weights/sam_vit_h_4b8939.pth"
    if not os.path.exists(sam_weights_path):
        print("\nSAM weights not found. Downloading...")
        os.makedirs("./weights", exist_ok=True)
        try:
            import wget
            wget.download(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                sam_weights_path
            )
            print("\nSAM weights downloaded successfully.")
        except ImportError:
            print("wget not installed. Please download SAM weights manually:")
            print(f"wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O {sam_weights_path}")
            return
        except Exception as e:
            print(f"Failed to download SAM weights: {e}")
            return
    
    # Set up output directory and scene name
    scene_name = scene_info['scene_dir'].name
    output_dir = f"{output_base_dir}/{scene_name}"
    
    print(f"\nProcessing scene: {scene_name}")
    print(f"Output directory: {output_dir}")
    
    # Create the command
    cmd = [
        "python", "mast3r_slam_wrapper.py",
        "--mast3r_data_dir", str(scene_info['scene_dir']),
        "--poses_file", str(scene_info['poses_file']),
        "--output_dir", output_dir,
        "--scene_name", scene_name,
        "--config_type", config_type,
    ]
    
    if max_frames:
        cmd.extend(["--max_frames", str(max_frames)])
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run the command
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print(f"\n✓ Processing completed successfully!")
        print(f"Output files in: {output_dir}")
    else:
        print(f"\n✗ Processing failed (return code: {result.returncode})")
    
if __name__ == "__main__":
    main() 
# MASt3R-SLAM to Locate3D Preprocessing Wrapper

This wrapper adapts the locate3d preprocessing pipeline to work with MASt3R-SLAM output data, enabling you to create featurized point clouds for 3D search using RGB keyframes, depth maps, and camera poses from MASt3R-SLAM.

## Overview

The wrapper performs the following steps:
1. **Loads MASt3R-SLAM data**: RGB keyframes, depth maps, and camera poses with intrinsics
2. **Converts data format**: Transforms MASt3R-SLAM format to locate3d's expected format
3. **Feature extraction**: Uses 2D foundation models (CLIP and/or DINOv2) to extract features
4. **3D lifting**: Projects 2D features into 3D space using depth and camera parameters
5. **Voxelization**: Creates voxelized point clouds with weighted feature averaging
6. **Saves output**: Stores featurized point clouds for use in 3D search

## MASt3R-SLAM Data Format

Expected input structure:
```
scene_directory/              # Scene directory (e.g., test_video1_all_kfs)
├── keyframes/               # RGB images
│   ├── 000000.png
│   ├── 000062.png
│   └── ...
├── depth_maps/             # Depth maps as numpy arrays
│   ├── 000000.npy
│   ├── 000062.npy
│   └── ...
└── *_with_intrinsics.txt   # Camera poses and intrinsics
```

### Poses File Format
Each line contains: `frame_id tx ty tz qx qy qz qw width height fx fy cx cy`
- `frame_id`: Frame identifier (matches keyframe/depth filenames)
- `tx, ty, tz`: Camera translation 
- `qx, qy, qz, qw`: Camera rotation as quaternion
- `width, height`: Image dimensions
- `fx, fy`: Focal lengths
- `cx, cy`: Principal point coordinates

## Setup

### 1. Install Dependencies
```bash
# Install required packages
pip install torch torchvision opencv-python pillow numpy omegaconf segment-anything
```

### 2. Download SAM Weights
```bash
cd locate-3d/preprocessing
mkdir -p weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O weights/sam_vit_h_4b8939.pth
```

## Usage

### Interactive Processing

Process a scene interactively:

```bash
cd locate-3d/preprocessing
python run_mast3r_preprocessing.py
```

This script will:
- Let you select a scene to process
- Ask for processing options (feature type, max frames)
- Handle the entire processing pipeline

### Direct Wrapper Usage

Use the wrapper directly:

```bash
cd locate-3d/preprocessing

python mast3r_slam_wrapper.py \
    --mast3r_data_dir ../../MASt3R-SLAM/logs/test_video1_all_kfs \
    --poses_file ../../MASt3R-SLAM/logs/test_video1_all_kfs/AFM_Video_Marco_1_with_intrinsics.txt \
    --output_dir ./output_pointclouds \
    --scene_name AFM_Video_Marco_1 \
    --config_type both
```

## Command Line Arguments

### run_mast3r_preprocessing.py
- Interactive script - no command line arguments needed
- Edit the script to change default paths

### mast3r_slam_wrapper.py
- `--mast3r_data_dir`: Path to single scene directory
- `--poses_file`: Path to camera poses file with intrinsics  
- `--output_dir`: Output directory for processed point clouds
- `--scene_name`: Name prefix for output files
- `--config_type`: Feature extraction type - "clip", "dino", or "both"
- `--max_frames`: Maximum number of frames to process

## Output Files

The wrapper generates the following files in the output directory:

```
output_pointclouds/
└── scene_name/
    ├── scene_name_clip.pt    # If using CLIP features
    ├── scene_name_dino.pt    # If using DINO features
    └── scene_name_combined.pt # If using both feature types
```

Each `.pt` file contains a dictionary with:
```python
{
    "points": torch.Tensor,         # [N, 3] 3D coordinates
    "rgb": torch.Tensor,           # [N, 3] RGB colors
    "features_clip": torch.Tensor, # [N, D_clip] CLIP features (if available)
    "features_dino": torch.Tensor, # [N, D_dino] DINO features (if available)
}
```

## Quick Start Example

```bash
# 1. Navigate to preprocessing directory
cd locate-3d/preprocessing

# 2. Run interactive processing
python run_mast3r_preprocessing.py

# The script will show:
# MASt3R-SLAM to Locate3D Preprocessing
# ==================================================
# Found scene:
#   - test_video1_all_kfs:
#     Keyframes: 113
#     Depth maps: 113
#     Poses file: AFM_Video_Marco_1_with_intrinsics.txt
# 
# Processing Options:
# Feature extraction type (clip/dino/both) [both]: both
# Max frames to process (leave empty for all): 10
# 
# Processing scene: test_video1_all_kfs
# ...
```

## Using the Output for 3D Search

The generated point clouds are compatible with the locate3d 3D search pipeline:

```python
import torch

# Load processed point cloud
data = torch.load("output_pointclouds/test_video1_all_kfs/test_video1_all_kfs_combined.pt")

points = data["points"]              # [N, 3] 3D coordinates
rgb = data["rgb"]                    # [N, 3] RGB colors
features_clip = data["features_clip"] # [N, 768] CLIP features
features_dino = data["features_dino"] # [N, 384] DINO features

print(f"Point cloud: {points.shape[0]} points")
print(f"CLIP features: {features_clip.shape}")
print(f"DINO features: {features_dino.shape}")
```

## Configuration

The wrapper uses the same configuration files as the original locate3d preprocessing:
- `config/clip.yaml`: CLIP feature extraction settings
- `config/dino.yaml`: DINOv2 feature extraction settings

Key settings you can modify:
- **Voxel size**: `voxel_size: 0.05` (default: 5cm voxels)
- **Feature pooling**: `feature_pool_method: "mean"` (mean/max/sum)
- **SAM parameters**: Points per side, IoU threshold, etc.
- **Device settings**: CUDA/CPU selection

## Troubleshooting

### Common Issues

1. **Missing depth maps**: Ensure depth map filenames match keyframe IDs (000XXX.npy)
2. **Pose parsing errors**: Check poses file format matches expected structure
3. **SAM weights missing**: Run the download command or check network connection
4. **CUDA out of memory**: Reduce `--max_frames` or increase voxel size in config files

### Performance Tips

- **Testing**: Use `--max_frames 10` for quick testing
- **Memory optimization**: Adjust `unproject_frame_batch_size` in config files  
- **Speed**: Use `--config_type dino` for faster processing if CLIP features aren't needed

## Integration with Locate3D Pipeline

The output point clouds can be directly used with:
- **Locate3D's PointTransformer-v3 encoder**
- **3D-JEPA self-supervised pre-training**  
- **Language-conditioned 3D object localization**
- **Other 3D scene understanding tasks**

This wrapper effectively bridges MASt3R-SLAM's output with locate3d's preprocessing requirements, enabling end-to-end 3D scene understanding from RGB-D SLAM data. 
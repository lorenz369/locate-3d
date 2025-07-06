# Downloading and preprocessing datasets for Locate 3D

This codebase currently supports preprocessing and inference on pointclouds extracted from ScanNet, ScanNet++, and ARKitScenes datasets.

The script `preprocessing/run_preprocessing.py` is the main entrypoint. However, before running this script, ensure that your datasets and directories are set up as follows. Note that you need to have _at least one_ of these datasets downloaded and processed first.

## Preparing ScanNet dataset

1. Obtain the ScanNet dataset by navigating to [the ScanNet homepage](http://www.scan-net.org/ScanNet/) and filling out the terms of use agreement
2. Extract the frames (rgb, depth, poses) by following the [official ScanNet toolkit instructions](https://github.com/ScanNet/ScanNet)
3. Follow the [dataset preparation instructions from MMDetection3D](https://mmdetection3d.readthedocs.io/en/v0.15.0/datasets/scannet_det.html) to extract additional information such as segmentation files, axis-aligned matrices, bounding boxes 

## Preparing for ScanNet++ dataset

1. Obtain the ScanNet++ dataset by navigating to [the ScanNet++ homepage](https://kaldir.vc.in.tum.de/scannetpp/) and filling out the terms of use agreement
2. Extract the frames (rgb, depth, poses) by following the [official ScanNet++ toolkit instructions](https://github.com/scannetpp/scannetpp)

## Preparing the ARKitScenes dataset

For ARKitScenes, be sure to follow [instructions to download the 3dod data split](https://github.com/apple/ARKitScenes/blob/main/DATA.md).


## Input Data Format Specifications

The preprocessing system expects input data in specific formats for each dataset. All datasets must provide RGB images, depth maps, camera poses, and intrinsic parameters with proper correspondence and alignment.

### Core Data Structure

The preprocessing pipeline expects a `camera_views` dictionary with the following structure:
```python
camera_views = {
    "cam_to_world": torch.Tensor,    # Camera-to-world transformation matrices (N, 4, 4)
    "cam_K": torch.Tensor,           # Camera intrinsic matrices (N, 3, 3)
    "rgb": torch.Tensor,             # RGB images (N, C, H, W)
    "depth_zbuffer": torch.Tensor,   # Depth maps (N, H, W)
}
```

### 1. RGB Images (`rgb` key)
- **Supported formats**: `.jpg`, `.png`
- **Tensor shape**: `(N, 3, H, W)` where N=number of frames
- **Data type**: `torch.float32`
- **Value range**: `[0.0, 1.0]` (normalized from 0-255)
- **Processing**: Images are loaded via PIL, resized to target resolution, and normalized

### 2. Depth Maps (`depth_zbuffer` key)
- **Supported formats**: 
  - 16-bit PNG files (`.png`) - most common
  - 32-bit depth images
  - NumPy arrays (`.npy`)
- **Tensor shape**: `(N, H, W)` where N=number of frames
- **Data type**: `torch.float32`
- **Units**: **Meters** (after scaling)
- **Scale factor**: `0.001` (converts from millimeters to meters)
- **Processing**: Loaded with OpenCV `IMREAD_ANYDEPTH`, scaled, and resized with `INTER_NEAREST`

### 3. Camera Intrinsics (`cam_K` key)
- **Format**: 3x3 intrinsic matrix per frame
- **Tensor shape**: `(N, 3, 3)` where N=number of frames
- **Data type**: `torch.float32`
- **Matrix structure**:
  ```
  K = [[fx,  0, cx],
       [ 0, fy, cy],
       [ 0,  0,  1]]
  ```
- **Parameters**:
  - `fx`, `fy`: Focal lengths in pixels
  - `cx`, `cy`: Principal point coordinates in pixels
- **Note**: Intrinsics are automatically scaled to match resized image dimensions

### 4. Camera Poses (`cam_to_world` key)
- **Format**: 4x4 transformation matrices
- **Tensor shape**: `(N, 4, 4)` where N=number of frames
- **Data type**: `torch.float32`
- **Convention**: **Camera-to-world** transformation matrices
- **Matrix structure**:
  ```
  cam_to_world = [[R11, R12, R13, tx],
                  [R21, R22, R23, ty],
                  [R31, R32, R33, tz],
                  [ 0,   0,   0,  1]]
  ```

### Dataset-Specific File Organization

#### ScanNet Dataset
- **RGB**: `.jpg` files in `posed_images/{scene_name}/`
- **Depth**: `.png` files (16-bit) in `posed_images/{scene_name}/`
- **Poses**: Individual `.txt` files per frame containing 4x4 matrices
- **Intrinsics**: Single `intrinsic.txt` file (4x4 matrix, shared across frames)
- **Default resolution**: 640x480 (resized from 1296x968)
- **Frame sampling**: Every 30th frame

#### ScanNet++ Dataset
- **RGB**: `.jpg` files in `data/{scene_name}/iphone/rgb/`
- **Depth**: `.png` files in `data/{scene_name}/iphone/depth/`
- **Poses & Intrinsics**: Combined in `pose_intrinsic_imu.json`
- **Default resolution**: 480x640 (resized from 1440x1920 with scale factor 1/3)
- **Frame sampling**: Every 30th frame

#### ARKitScenes Dataset
- **RGB**: `.png` files in `lowres_wide/`
- **Depth**: `.png` files in `lowres_depth/`
- **Intrinsics**: `.pincam` files in `lowres_wide_intrinsics/` (6D format: [width, height, fx, fy, cx, cy])
- **Poses**: `lowres_wide.traj` file with 6DOF poses (axis-angle + translation)
- **Default resolution**: 256x192
- **Frame sampling**: Every 30th frame

### Data Quality Requirements
- **Temporal Alignment**: RGB, depth, pose, and intrinsics must correspond to the same timestamp/frame
- **Naming Convention**: Files typically named with frame indices (e.g., `000001.jpg`, `000001.png`)
- **Depth Validity**: Depth values must be within reasonable bounds (filtered during processing)
- **Pose Validity**: No NaN values allowed in transformation matrices
- **Correspondence**: Exact 1:1 correspondence required between all data modalities

### Coordinate System Conventions
- **World Coordinate System**: Typically aligned with scene geometry
- **Camera Coordinate System**: OpenCV convention (Z-forward, Y-down, X-right)
- **Pose Convention**: Camera-to-world transformations
- **Depth Units**: Meters (after preprocessing scaling)


## Download a checkpoint for SAM-H

We use the SAM-H model to extract masks and compute per-mask CLIP features. Download this model by running the following snippet.
```
cd preprocessing 
mkdir weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```


## Running the preprocessing script

With at least one of the above datasets downloaded and prepared, the preprocessing script can be run using
```
python preprocessing/run_preprocessing.py
```

The supported commandline arguments are
```
REQUIRED
--l3dd_annotations_fpath    {locate3d_data/dataset/all.json,locate3d_data/dataset/train_scannet.json,locate3d_data/dataset/val_arkitscenes.json,locate3d_data/dataset/train.json,locate3d_data/dataset/train_scannetpp.json,locate3d_data/dataset/val_scannet.json,locate3d_data/dataset/train_arkitscenes.json,locate3d_data/dataset/val.json,locate3d_data/dataset/val_scannetpp.json} (Locate 3D annotations)

--cache_path CACHE_PATH (directory to store the cached pointclouds in)

OPTIONAL

[--scannet_data_dir SCANNET_DATA_DIR]  (path to scannet directory containing posed images)
[--scannetpp_data_dir SCANNETPP_DATA_DIR] (path to scannet++ directory containing posed images)
[--arkitscenes_data_dir ARKITSCENES_DATA_DIR] (path to arkitscenes directory containing posed images)

[--start START] (index of first scene to be processed; 0 by default)
[--end END] (index of last scene to be processed; -1 by default)
```

## Parallelizing the preprocessing script

For users interested in parallelizing the preprocessing using SLURM array jobs, look at `run_preprocessing_slurm_array.py`

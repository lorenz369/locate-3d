# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union
from torchvision.transforms import functional as F
from pathlib import Path
import torch
import cv2
import numpy as np
import imageio.v2 as imageio

from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def get_image_from_path(
    image_path: Union[str, Path],
    height: Optional[int] = None,
    width: Optional[int] = None,
    keep_alpha: bool = False,
    resample=Image.BILINEAR,
) -> torch.Tensor:

    pil_image = Image.open(image_path)

    # Explicitly convert RGBA images to RGB to remove the alpha channel.
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')

    assert (height is None) == (width is None)  # Neither or both
    if height is not None and pil_image.size != (width, height):
        # Pass resample filter positionally for wider compatibility
        pil_image = pil_image.resize((width, height), resample)

    # Convert the final PIL image to a PyTorch tensor
    image = F.to_tensor(pil_image)

    return image


def get_depth_image_from_path(
    fpath: str, height: int, width: int, scale_factor: float
) -> np.ndarray:
    """
    Reads a depth image from a path, handles different formats, and resizes.
    """
    # Convert the fpath object (which may be a Path object) to a string
    fpath = str(fpath)

    if fpath.endswith(".png") or fpath.endswith(".jpg"):
        image = imageio.imread(fpath)
    elif fpath.endswith(".npz"):
        try:
            with np.load(fpath) as data:
                if 'depth' in data:
                    image = data['depth']
                else:
                    image = data['arr_0']
        except Exception as e:
            print(f"Error loading NPZ file {fpath}: {e}")
            return None
    else:
        image = None

    if image is None:
        return None

    current_height, current_width = image.shape[:2]
    if current_height != height or current_width != width:
        image = np.array(Image.fromarray(image).resize((width, height), Image.NEAREST))

    image = image.astype(np.float64) * scale_factor
    return image


def intrinsic_array_to_matrix(intrinsics: np.ndarray):
    """
    Converts the loaded 6D array to a 4x4 transformation matrix
    Loaded 6D array has the format: [width height focal_length_x focal_length_y principal_point_x principal_point_y]
    """
    fx = intrinsics[2]
    fy = intrinsics[3]
    cx = intrinsics[4]
    cy = intrinsics[5]

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K


def six_dim_pose_to_transform(pose):
    """convert traj_str into translation and rotation matrices
    Adapted from: https://github.com/apple/ARKitScenes/blob/9ec0b99c3cd55e29fc0724e1229e2e6c2909ab45/threedod/benchmark_scripts/utils/tenFpsDataLoader.py
    Args:
        pose: A list representing a camera position at a particular timestamp.
        The list has seven elements:
        * Elements 1-3: rotation (axis-angle representation in radians)
        * Elements 4-6: translation (usually in meters)

    Returns:
        Rt: cam to world transformation matrix
    """

    assert len(pose) == 6
    # Rotation in angle axis
    angle_axis = [float(pose[0]), float(pose[1]), float(pose[2])]
    r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))
    # Translation
    t_w_to_p = np.asarray([float(pose[3]), float(pose[4]), float(pose[5])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    Rt = np.linalg.inv(extrinsics)
    return Rt


def convert_angle_axis_to_matrix3(angle_axis):
    """Return a Matrix3 for the angle axis.
    Arguments:
        angle_axis {Point3} -- a rotation in angle axis form.
    """
    matrix, _ = cv2.Rodrigues(angle_axis)
    return matrix


def interpolate_camera_poses(times_subset, poses_subset, times_interpolate):
    """
    Example:
    times = np.array(list(range(len(pose))))
    subset = list(range(len(pose)))[::5]
    if len(pose) - 1 not in subset:
        subset = subset + [len(pose) - 1]
    poses_out = interpolate_camera_poses(times[subset], pose[subset], times)
    """
    # Extract rotations and translations from the poses
    rotations = R.from_matrix(poses_subset[:, :3, :3])  # for pose in poses_subset]
    translations = [pose[:3, 3] for pose in poses_subset]

    # Initialize Slerp for rotations
    slerp = Slerp(times_subset, rotations)

    # Initialize CubicSpline for translations
    cubic_spline = CubicSpline(times_subset, translations, axis=0)

    # Interpolate rotations and translations at the desired times
    interpolated_rotations = slerp(times_interpolate)
    interpolated_translations = cubic_spline(times_interpolate)

    # Combine rotations and translations into 4x4 poses
    interpolated_poses = np.array(
        [
            np.hstack([rot.as_matrix(), trans.reshape(3, 1)])
            for rot, trans in zip(interpolated_rotations, interpolated_translations)
        ]
    )
    interpolated_poses = np.array(
        [np.vstack([pose, [0, 0, 0, 1]]) for pose in interpolated_poses]
    )

    return interpolated_poses


def get_rotation_matrix_z(k: int):
    """
    Generate a rotation matrix for k*90 degrees rotation around the Z-axis.

    Args:
        k (int): The number of 90-degree increments to rotate.

    Returns:
        torch.Tensor: A 4x4 rotation matrix.
    """
    theta = np.radians(90 * k)  # Convert degrees to radians
    cos = np.cos(theta)
    sin = np.sin(theta)

    return torch.tensor(
        [[cos, -sin, 0, 0], [sin, cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
    )


def infer_sky_direction_from_poses(poses_cam_to_world):
    """
    Infer the sky direction from a list of camera poses by calculating the average "up" and "right" relative to the camera, then identifying which 90 degree rotation would make the "up" vector closest to the world "up".

    Logic copied from DUSt3R codebase: https://github.com/naver/dust3r/blob/69192aa322d279438390c109b8b85d5b859b5cdd/datasets_preprocess/preprocess_arkitscenes.py#L308

    Args:
        poses_cam_to_world (List[np.array]): A list of camera poses.

    Returns:
        str: The inferred sky direction. One of "UP", "DOWN", "LEFT", or "RIGHT".
    """

    def get_up_vectors(pose_device_to_world):
        return np.matmul(pose_device_to_world, np.array([[0.0], [-1.0], [0.0], [0.0]]))

    def get_right_vectors(pose_device_to_world):
        return np.matmul(pose_device_to_world, np.array([[1.0], [0.0], [0.0], [0.0]]))

    if len(poses_cam_to_world) > 0:
        up_vector = sum(get_up_vectors(p) for p in poses_cam_to_world) / len(
            poses_cam_to_world
        )
        right_vector = sum(get_right_vectors(p) for p in poses_cam_to_world) / len(
            poses_cam_to_world
        )
        up_world = np.array([[0.0], [0.0], [1.0], [0.0]])
    else:
        up_vector = np.array([[0.0], [-1.0], [0.0], [0.0]])
        right_vector = np.array([[1.0], [0.0], [0.0], [0.0]])
        up_world = np.array([[0.0], [0.0], [1.0], [0.0]])

    # value between 0, 180
    device_up_to_world_up_angle = (
        np.arccos(np.clip(np.dot(np.transpose(up_world), up_vector), -1.0, 1.0)).item()
        * 180.0
        / np.pi
    )
    device_right_to_world_up_angle = (
        np.arccos(
            np.clip(np.dot(np.transpose(up_world), right_vector), -1.0, 1.0)
        ).item()
        * 180.0
        / np.pi
    )

    up_closest_to_90 = abs(device_up_to_world_up_angle - 90.0) < abs(
        device_right_to_world_up_angle - 90.0
    )
    if up_closest_to_90:
        assert abs(device_up_to_world_up_angle - 90.0) < 45.0
        if device_right_to_world_up_angle > 90.0:
            # LEFT
            sky_direction_scene = "LEFT"
        else:
            # RIGHT
            sky_direction_scene = "RIGHT"
    else:
        # right is close to 90
        assert abs(device_right_to_world_up_angle - 90.0) < 45.0
        if device_up_to_world_up_angle > 90.0:
            sky_direction_scene = "DOWN"
        else:
            sky_direction_scene = "UP"
    return sky_direction_scene


def rotate_intrinsics_90_degrees_clockwise_about_camera_z(cam_K, W, H, k):
    """
    Adjust the intrinsic matrix for camera rotation.

    Parameters:
    cam_K : torch.Tensor
        The original intrinsic matrix [batch_size, 3, 3].
    W : int
        The width of the image.
    H : int
        The height of the image.
    k : int
        Number of 90-degree clockwise rotations.

    Returns:
    cam_K : torch.Tensor
        The rotated intrinsic matrix [batch_size, 3, 3].
    """

    k = k % 4

    if k == 0:
        return cam_K

    f_x = cam_K[:, 0, 0].clone()
    f_y = cam_K[:, 1, 1].clone()
    c_x = cam_K[:, 0, 2].clone()
    c_y = cam_K[:, 1, 2].clone()

    assert (cam_K[:, 0, 1] == 0).all()
    assert (cam_K[:, 1, 0] == 0).all()
    assert (cam_K[:, 2, :2] == 0).all()
    assert (cam_K[:, 2, 2] == 1).all()

    if k == 1:  # 90-degree clockwise rotation
        cam_K[:, 0, 0] = f_y
        cam_K[:, 0, 2] = c_y
        cam_K[:, 1, 1] = f_x
        cam_K[:, 1, 2] = W - c_x

    elif k == 2:  # 180-degree rotation
        cam_K[:, 0, 2] = W - c_x
        cam_K[:, 1, 2] = H - c_y

    elif k == 3:  # 270-degree clockwise rotation
        cam_K[:, 0, 0] = f_y
        cam_K[:, 0, 2] = H - c_y
        cam_K[:, 1, 1] = f_x
        cam_K[:, 1, 2] = c_x

    else:
        raise ValueError(
            "k must be 1, 2, or 3 for 90, 180, or 270-degree rotation respectively."
        )

    return cam_K


def rotate_frames_90_degrees_clockwise_about_camera_z(
    rgb, depth_zbuffer, cam_to_world, cam_K, orig_W: int, orig_H: int, k: int = 1
):
    """
    Rotates the frame history 90 degrees clockwise about the camera Z-axis.

    Args:
        orig_W (int): The original width of the frames.
        orig_H (int): The original height of the frames
        k (int): Number of times to rotate the observations 90 degrees clockwise. Default is 1.

    """
    k = k % 4
    if k == 0:
        # if k == 0, do nothing
        return rgb, depth_zbuffer, cam_to_world, cam_K
    # Rotate RGB and depth zbuffer
    rgb = torch.rot90(rgb, k=k, dims=(2, 3))  # Rotate along H-W plane
    depth_zbuffer = torch.rot90(depth_zbuffer, k=k, dims=(1, 2))

    # Define the rotation matrix for 90 degrees clockwise k times about the z-axis
    rotation_matrix = get_rotation_matrix_z(k)
    # Rotate camera-to-world pose matrix
    cam_to_world = torch.matmul(cam_to_world, rotation_matrix)

    cam_K = rotate_intrinsics_90_degrees_clockwise_about_camera_z(
        cam_K, orig_W, orig_H, k
    )

    return rgb, depth_zbuffer, cam_to_world, cam_K

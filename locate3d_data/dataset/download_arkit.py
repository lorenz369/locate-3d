#!/usr/bin/env python3
"""
Download a small subset of ARKitScenes scenes for testing.
This script downloads the first few scenes from the annotation file.
"""

import os
import subprocess
import json
from pathlib import Path

def get_scene_list(annotation_file, num_scenes=5):
    """Extract scene IDs from annotation file."""
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Get unique scene IDs
    scene_ids = set()
    for annotation in annotations:
        scene_ids.add(annotation['scene_id'])
    
    # Return first N scenes
    return list(scene_ids)[:num_scenes]

def download_arkitscenes_scene(scene_id, output_dir):
    """Download a single ARKitScenes scene."""
    # This is a placeholder - you'll need to implement the actual download logic
    # based on the ARKitScenes download instructions
    print(f"Would download scene {scene_id} to {output_dir}")
    print(f"Please follow the ARKitScenes download instructions for scene {scene_id}")
    print(f"Download URL: https://github.com/apple/ARKitScenes/blob/main/DATA.md")

def main():
    # Configuration
    annotation_file = "locate3d_data/dataset/train_arkitscenes.json"
    output_dir = "locate3d_data/dataset/arkitscenes/raw"
    num_scenes = 3  # Start with just 3 scenes for testing
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get scene list
    scene_ids = get_scene_list(annotation_file, num_scenes)
    print(f"Will download {len(scene_ids)} scenes: {scene_ids}")
    
    # Download each scene
    for scene_id in scene_ids:
        scene_dir = os.path.join(output_dir, "Training", scene_id)
        Path(scene_dir).mkdir(parents=True, exist_ok=True)
        download_arkitscenes_scene(scene_id, scene_dir)

if __name__ == "__main__":
    main()
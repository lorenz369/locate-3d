#!/usr/bin/env python3
"""
Filter annotation file to only include scenes that are actually available in the dataset.
"""

import json
import os
from pathlib import Path

def filter_annotations(annotation_file, available_scenes, output_file):
    """
    Filter annotation file to only include scenes that are available.
    
    Args:
        annotation_file: Path to the original annotation file
        available_scenes: List of scene IDs that are available
        output_file: Path to save the filtered annotation file
    """
    
    # Read the original annotation file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Filter annotations to only include available scenes
    filtered_annotations = []
    available_scenes_set = set(available_scenes)
    
    for annotation in annotations:
        scene_id = annotation.get('scene_id')
        if scene_id in available_scenes_set:
            filtered_annotations.append(annotation)
    
    # Save the filtered annotations
    with open(output_file, 'w') as f:
        json.dump(filtered_annotations, f, indent=2)
    
    print(f"Original annotations: {len(annotations)}")
    print(f"Filtered annotations: {len(filtered_annotations)}")
    print(f"Available scenes: {available_scenes}")
    print(f"Filtered file saved to: {output_file}")

if __name__ == "__main__":
    # Available scenes in your dataset
    available_scenes = ["42444821", "42447230", "45261495"]
    
    # Input and output files
    input_file = "locate3d_data/dataset/train_arkitscenes.json"
    output_file = "locate3d_data/dataset/train_arkitscenes_subset.json"
    
    # Create the filtered annotation file
    filter_annotations(input_file, available_scenes, output_file)
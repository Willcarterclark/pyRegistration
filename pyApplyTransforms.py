#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Apply Transform Script
--
by Will Clark 
POLARIS - University of Sheffield
--
This script applies existing transforms (single or composite) to a set of images.
Can handle both standard registrations (affine + warp) and composite transforms (no affine & "composite" in name).
"""

import os
import argparse
import subprocess
import SimpleITK as sitk
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

def find_transform_files(transform_dir):
    """
    Find and identify transform files in a directory.
    
    Args:
        transform_dir: Directory containing transform files
    
    Returns:
        dict: Dictionary with transform type and file paths
    """
    transforms = {
        'type': None,
        'files': []
    }
    
    # Check for composite transform (single file)
    composite_files = glob.glob(os.path.join(transform_dir, "*composite_1Warp.nii.gz"))
    
    # Check for standard transform pair
    warp_files = glob.glob(os.path.join(transform_dir, "*1Warp.nii.gz"))
    affine_files = glob.glob(os.path.join(transform_dir, "*0GenericAffine.mat"))
    
    # Exclude composite files from warp files
    warp_files = [w for w in warp_files if "Composite" not in w]
    
    if composite_files:
        transforms['type'] = 'composite'
        transforms['files'] = composite_files
        print(f"Found composite transform: {composite_files[0]}")
    elif warp_files and affine_files:
        transforms['type'] = 'standard'
        transforms['files'] = [warp_files[0], affine_files[0]]
        print(f"Found standard transforms:")
        print(f"  Warp: {warp_files[0]}")
        print(f"  Affine: {affine_files[0]}")
    else:
        print("Warning: Could not identify transform type")
        # Return whatever files we found
        transforms['type'] = 'unknown'
        transforms['files'] = warp_files + affine_files
    
    return transforms

def load_dir_structure(patient_dir, img_folder, subfolder=None):
    """
    Load directory structure to find images to transform.
    Simplified version of the load_dir() function from the registration script.
    """
    Table_out = pd.DataFrame()
    
    # Walk through the patient directory
    for root, dirs, files in os.walk(patient_dir):
        # Check if we're in the right folder
        if img_folder in root:
            # If subfolder specified, check for it
            if subfolder and subfolder not in root:
                continue
            
            # Get all image files
            img_files = [f for f in files if f.endswith(('.nii', '.nii.gz', '.mha'))]
            
            if img_files:
                data_to_bind = {
                    'Images': [np.array(img_files)],
                    'Directory': [root]
                }
                output_scans = pd.DataFrame(data_to_bind)
                Table_out = pd.concat([Table_out, output_scans], ignore_index=True)
    
    return Table_out

def apply_transform_to_image(image_path, output_path, transform_info, reference_image, 
                            ants_path=None, dimensions="3", interpolation="Linear"):
    """
    Apply transform to a single image.
    
    Args:
        image_path: Path to input image
        output_path: Path to output transformed image
        transform_info: Dictionary with transform type and file paths
        reference_image: Reference image for output space
        ants_path: Path to ANTs binaries
        dimensions: Image dimensions (2 or 3)
        interpolation: Interpolation method (Linear, NearestNeighbor, etc.)
    """
    
    # Set ANTs tool path
    ants_apply = os.path.join(ants_path, 'antsApplyTransforms') if ants_path else 'antsApplyTransforms'
    
    # Build the command
    apply_command = [
        ants_apply,
        "-d", dimensions,
        "-v", "1",
        "-i", image_path,
        "-r", reference_image,
        "-n", interpolation,
        "-o", output_path
    ]
    
    # Add transforms based on type
    if transform_info['type'] == 'composite':
        # Single composite transform
        for transform_file in transform_info['files']:
            apply_command.extend(["-t", transform_file])
    
    elif transform_info['type'] == 'standard':
        # Standard registration: apply warp then affine
        # ANTs applies transforms in reverse order (last to first)
        # So we add warp first, then affine
        warp_file = [f for f in transform_info['files'] if "1Warp" in f]
        affine_file = [f for f in transform_info['files'] if "0GenericAffine" in f]
        
        if warp_file:
            apply_command.extend(["-t", warp_file[0]])
        if affine_file:
            apply_command.extend(["-t", affine_file[0]])
    
    else:
        # Unknown type - just add whatever transforms we found
        for transform_file in transform_info['files']:
            apply_command.extend(["-t", transform_file])
    
    print(f"Applying transform to: {os.path.basename(image_path)}")
    print(f"Command: {' '.join(apply_command)}")
    
    try:
        subprocess.run(apply_command, check=True)
        print(f"  Success! Output: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Failed to apply transform: {e}")
        return False

def process_patient(patient_dir, transform_dir, img_folder, reference_img, 
                   ants_path=None, output_dir=None, subfolder=None,
                   img_identifiers=None, exclude_identifiers=None,
                   is_composite=None, dimensions="3"):
    """
    Process all images for a patient using specified transforms.
    
    Args:
        patient_dir: Patient directory containing images
        transform_dir: Directory containing transform files
        img_folder: Folder name containing images to transform
        reference_img: Reference image identifier or path
        ants_path: Path to ANTs binaries
        output_dir: Output directory for transformed images
        subfolder: Optional subfolder within img_folder
        img_identifiers: List of strings to identify specific images to transform
        exclude_identifiers: List of strings to exclude certain images
        is_composite: Force interpretation as composite (True) or standard (False) transform
        dimensions: Image dimensions
    """
    
    print(f"Processing patient: {patient_dir}")
    print(f"Using transforms from: {transform_dir}")
    print("-" * 50)
    
    # Find transform files
    transform_info = find_transform_files(transform_dir)
    
    # Override type if specified
    if is_composite is not None:
        transform_info['type'] = 'composite' if is_composite else 'standard'
        print(f"Forcing transform type: {transform_info['type']}")
    
    if not transform_info['files']:
        print("Error: No transform files found!")
        return
    
    # Load images to transform
    image_df = load_dir_structure(patient_dir, img_folder, subfolder)
    
    if image_df.empty:
        print(f"No images found in {img_folder}")
        return
    
    # Find or set reference image
    reference_path = None
    
    # Check if reference_img is a full path
    if os.path.exists(reference_img):
        reference_path = reference_img
    else:
        # Search for reference image in the dataframe
        for index, row in image_df.iterrows():
            imgs = row['Images']
            img_dir = row['Directory']
            
            for img in imgs:
                if reference_img in img:
                    reference_path = os.path.join(img_dir, img)
                    break
            
            if reference_path:
                break
    
    if not reference_path:
        print(f"Error: Could not find reference image: {reference_img}")
        return
    
    print(f"Using reference image: {reference_path}")
    
    # Determine output directory
    if output_dir:
        # Create nested structure in output directory
        rel_path = os.path.relpath(patient_dir, os.path.dirname(patient_dir))
        final_output_dir = os.path.join(output_dir, rel_path, "transformed_" + os.path.basename(transform_dir))
    else:
        # Save in transform directory
        final_output_dir = os.path.join(transform_dir, "transformed_images")
    
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Output directory: {final_output_dir}")
    print("-" * 50)
    
    # Process each image
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for index, row in image_df.iterrows():
        imgs = row['Images']
        img_dir = row['Directory']
        
        for img in imgs:
            # Skip reference image
            if reference_img in img:
                print(f"Skipping reference image: {img}")
                skip_count += 1
                continue
            
            # Check if image should be included
            if img_identifiers:
                if not any(identifier in img for identifier in img_identifiers):
                    print(f"Skipping (not in include list): {img}")
                    skip_count += 1
                    continue
            
            # Check if image should be excluded
            if exclude_identifiers:
                if any(identifier in img for identifier in exclude_identifiers):
                    print(f"Skipping (in exclude list): {img}")
                    skip_count += 1
                    continue
            
            # Apply transform
            img_path = os.path.join(img_dir, img)
            
            # Create output filename
            output_name = img.replace('.nii.gz', '_transformed.nii.gz')
            if '.nii.gz' not in output_name and '.nii' in output_name:
                output_name = output_name.replace('.nii', '_transformed.nii.gz')
            
            output_path = os.path.join(final_output_dir, output_name)
            
            # Determine interpolation based on image type
            # Use NearestNeighbor for masks/segmentations
            interpolation = "NearestNeighbor" if any(x in img.lower() for x in ['mask', 'seg', 'label']) else "Linear"
            
            # Apply transform
            if apply_transform_to_image(img_path, output_path, transform_info, 
                                      reference_path, ants_path, dimensions, interpolation):
                success_count += 1
            else:
                fail_count += 1
    
    print("-" * 50)
    print(f"Processing complete!")
    print(f"  Successful: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Failed: {fail_count}")
    
    # Create summary file
    summary_file = os.path.join(final_output_dir, "transform_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Transform Application Summary\n")
        f.write("="*50 + "\n")
        f.write(f"Patient: {patient_dir}\n")
        f.write(f"Transform directory: {transform_dir}\n")
        f.write(f"Transform type: {transform_info['type']}\n")
        f.write(f"Reference image: {reference_path}\n")
        f.write(f"Output directory: {final_output_dir}\n")
        f.write("-"*50 + "\n")
        f.write(f"Results:\n")
        f.write(f"  Successful: {success_count}\n")
        f.write(f"  Skipped: {skip_count}\n")
        f.write(f"  Failed: {fail_count}\n")
        f.write("-"*50 + "\n")
        f.write(f"Transform files used:\n")
        for tf in transform_info['files']:
            f.write(f"  - {tf}\n")
    
    print(f"Summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Apply transforms to a set of images")
    
    parser.add_argument('-pat_dir', '--patient_dir', type=Path, required=True,
                       help="Patient directory containing images")
    parser.add_argument('-trans_dir', '--transform_dir', type=Path, required=True,
                       help="Directory containing transform files")
    parser.add_argument('-img_folder', '--image_folder', type=str, required=True,
                       help="Folder name containing images to transform")
    parser.add_argument('-ref', '--reference', type=str, required=True,
                       help="Reference image identifier or full path")
    parser.add_argument('-ants_path', '--ants_dir', type=Path,
                       help="Path to ANTs binaries")
    parser.add_argument('-out_dir', '--output_dir', type=Path,
                       help="Output directory for transformed images")
    parser.add_argument('-sub_folder', '--sub_folder', type=str,
                       help="Optional subfolder within image folder")
    parser.add_argument('-include', '--include_images', type=str, nargs='+',
                       help="List of identifiers for images to include")
    parser.add_argument('-exclude', '--exclude_images', type=str, nargs='+',
                       help="List of identifiers for images to exclude")
    parser.add_argument('-composite', '--is_composite', action='store_true',
                       help="Force interpretation as composite transform")
    parser.add_argument('-standard', '--is_standard', action='store_true',
                       help="Force interpretation as standard transform")
    parser.add_argument('-dim', '--dimensions', type=str, default="3",
                       help="Image dimensions (2 or 3)")
    
    args = parser.parse_args()
    
    # Determine transform type
    is_composite = None
    if args.is_composite:
        is_composite = True
    elif args.is_standard:
        is_composite = False
    
    # Process the patient
    process_patient(
        patient_dir=str(args.patient_dir),
        transform_dir=str(args.transform_dir),
        img_folder=str(args.image_folder),
        reference_img=str(args.reference),
        ants_path=str(args.ants_dir) if args.ants_dir else None,
        output_dir=str(args.output_dir) if args.output_dir else None,
        subfolder=str(args.sub_folder) if args.sub_folder else None,
        img_identifiers=args.include_images,
        exclude_identifiers=args.exclude_images,
        is_composite=is_composite,
        dimensions=str(args.dimensions)
    )

if __name__ == "__main__":
    main()

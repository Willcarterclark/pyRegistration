#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Apply Transform Script - Fixed Version v2
--
by Will Clark 
POLARIS - University of Sheffield
--
Supports three input modes:
1. DIRECT: Apply transforms to a single image file
2. TREE: Apply transforms to images in a nested tree structure (Patient/visit/folder/images)
3. VENT: Apply transforms to ventilation images within registration folders 
         (Patient/visit/Reg_folder/Vent_folder/images)

Output Structure:
    TREE mode: {output_dir}/{patient}/{visit}/{regfolder}/{img_folder}/(subfolder)/
    VENT mode: {output_dir}/{patient}/{visit}/{regfolder}/{vent_folder}/
    DIRECT mode: {output_dir}/{patient_id}/
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
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

#%% MANUAL MODE PARAMETERS
manual = False  # Set to True when running in IDE

# Manual parameters for testing - edit these when running in IDE
manual_params = {
    # Mode selection: 'direct', 'tree', or 'vent'
    'mode': 'tree',
    
    # Common parameters
    'patient_dir': r'',
    'ants_path': r'',
    'dimensions': '3',
    
    # Transform location parameters
    'transform_folder': 'Reg__TLC_2__RV',  # Name of registration folder within patient tree
    'timepoint': None,  # Specific timepoint, or None to auto-detect
    
    # For DIRECT mode
    'direct_image': None,  # Full path to single image to transform
    'direct_transform_dir': None,  # Full path to directory containing transforms
    'direct_reference': None,  # Full path to reference image
    'direct_output': None,  # Full path to output file
    'patient_id': None,  # Patient identifier for direct mode output structure
    
    # For TREE mode
    'img_folder': 'img',  # Folder name containing images
    'sub_folder': None,  # Optional subfolder within img_folder
    'reference_identifier': 'RV',  # String to identify reference image in tree
    
    # For VENT mode (images within registration folder)
    'vent_dirs': ['Vent_Int', 'Vent_Trans', 'Vent_Hyb3'],  # Ventilation type folders
    'vent_strings': {'Vent_Int': 'sVent', 'Vent_Trans': 'JacVent', 'Vent_Hyb3': 'HYCID'},
    'vent_img_filters': ['_medfilt_3.nii.gz'],  # Filters for vent images
    
    # Output settings
    'output_dir': None,  # Base output directory (None = save in transform folder)
    
    # Image filtering
    'include_images': None,  # List of identifiers to include
    'exclude_images': ['mask', 'seg'],  # List of identifiers to exclude
}

# Default ventilation directories and strings (backward compatibility)
DEFAULT_VENT_DIRS = ["Vent_Int", "Vent_Trans", "Vent_Hyb", "Vent_Hyb2", "Vent_Hyb3"]
DEFAULT_VENT_STRINGS = {
    'Vent_Int': 'sVent', 
    'Vent_Trans': 'JacVent', 
    'Vent_Hyb': 'SIJacVent', 
    'Vent_Hyb2': 'SqSIJacVent', 
    'Vent_Hyb3': 'HYCID'
}

#%% Directory Loading Functions

def load_img_dir(load_dir):
    """Load image directory contents"""
    if not os.path.exists(load_dir):
        logger.warning(f"Directory does not exist: {load_dir}")
        return []
    img_list = os.listdir(load_dir)
    logger.debug(f"Found files: {img_list}")
    return img_list

def load_folder(fdir, dirs_to_find=False):
    """
    Load folder's subdirectories.
    
    Args:
        fdir: Directory to search in
        dirs_to_find: Specific directories to look for (string, list, or False for all)
    
    Returns:
        numpy array of directory names found
    """
    single_dir_requested = False

    if dirs_to_find:
        if isinstance(dirs_to_find, (list, np.ndarray)):
            dirs_to_find_list = np.array(dirs_to_find)
            if len(dirs_to_find_list) == 1:
                single_dir_requested = True
        else:
            dirs_to_find_list = np.array([dirs_to_find])
            single_dir_requested = True

        if not os.path.exists(fdir):
            logger.error(f"Directory '{fdir}' does not exist.")
            return np.array([])
    else:
        if not os.path.exists(fdir):
            logger.error(f"Directory '{fdir}' does not exist.")
            return np.array([])

        dirs_to_find_list = np.array(sorted([
            d for d in os.listdir(fdir)
            if os.path.isdir(os.path.join(fdir, d))
        ]))

    existing_dirs = []
    for directory in dirs_to_find_list:
        if os.path.isdir(os.path.join(fdir, directory)):
            existing_dirs.append(directory)
        else:
            logger.debug(f"Subdirectory '{directory}' not found in '{fdir}'")

    if single_dir_requested and len(existing_dirs) == 0:
        return None

    return np.array(existing_dirs)

def load_folder_imgs(fdir, files_to_find=False, extensions=('.nii', '.nii.gz', '.mha')):
    """
    Find image files in a directory matching specified substring patterns.
    
    Args:
        fdir: Directory to search
        files_to_find: Specific file patterns to match (string, list, or False for all)
        extensions: Valid file extensions
    
    Returns:
        numpy array of matching filenames
    """
    if not os.path.exists(fdir):
        logger.error(f"Directory '{fdir}' does not exist.")
        return np.array([])

    single_pattern_requested = False

    if files_to_find:
        if isinstance(files_to_find, (list, np.ndarray)):
            patterns_list = np.array(files_to_find)
            if len(patterns_list) == 1:
                single_pattern_requested = True
        else:
            patterns_list = np.array([files_to_find])
            single_pattern_requested = True

        matching_files = []
        for pattern in patterns_list:
            pattern_matches = glob.glob(os.path.join(fdir, f"*{pattern}*"))
            pattern_matches = [os.path.basename(f) for f in pattern_matches 
                             if f.endswith(extensions)]
            matching_files.extend(pattern_matches)

        matching_files = sorted(list(set(matching_files)))
    else:
        # Get all image files
        matching_files = sorted([
            os.path.basename(f) for f in glob.glob(os.path.join(fdir, "*"))
            if f.endswith(extensions)
        ])

    if single_pattern_requested and len(matching_files) == 0:
        logger.debug(f"No matching files found for pattern in {fdir}")
        return None

    logger.debug(f"Files found in {fdir}: {matching_files}")
    return np.array(matching_files) if matching_files else np.array([])


def load_tree_structure(source_dir, folder, subfolder=None, timepoint=None, filenames=None):
    """
    Load images from nested patient directory structure.
    Matches the pattern: source_dir / timepoint / folder / (subfolder) / images
    
    Args:
        source_dir: Patient directory (e.g., /data/Patient01)
        folder: Folder name to load from each timepoint (e.g., 'img', 'Images')
        subfolder: Optional subfolder within the specified folder
        timepoint: Specific timepoint(s) to load (None = auto-detect all)
        filenames: Specific filename patterns to search for
    
    Returns:
        DataFrame with columns: Images, Directory, Timepoint, Folder, Subfolder
    """
    Table_out = pd.DataFrame()
    
    logger.info("Loading Directory Tree...")
    logger.info(f"  Source directory: {source_dir}")
    logger.info(f"  Folder: {folder}")
    logger.info(f"  Subfolder: {subfolder}")
    logger.info(f"  Timepoint(s): {timepoint if timepoint else 'auto-detect'}")
    
    # Get timepoints
    timepoints = load_folder(fdir=source_dir, dirs_to_find=timepoint)
    
    if timepoints is None or len(timepoints) == 0:
        logger.warning(f"No timepoints found in {source_dir}")
        return Table_out
    
    for tp in timepoints:
        logger.debug(f"Processing timepoint: {tp}")
        time_point_path = os.path.join(source_dir, tp)
        
        # Look for the specified folder
        folders = load_folder(fdir=time_point_path, dirs_to_find=folder)
        
        if folders is None or len(folders) == 0:
            logger.debug(f"Folder '{folder}' not found in timepoint {tp}")
            continue
        
        for fld in folders:
            folder_path = os.path.join(time_point_path, fld)
            
            if subfolder:
                # Look for subfolder
                subfolders = load_folder(fdir=folder_path, dirs_to_find=subfolder)
                
                if subfolders is None or len(subfolders) == 0:
                    logger.debug(f"Subfolder '{subfolder}' not found in {folder_path}")
                    continue
                
                for sf in subfolders:
                    data_folder = os.path.join(folder_path, sf)
                    imgs = load_folder_imgs(fdir=data_folder, files_to_find=filenames)
                    
                    if imgs is None or len(imgs) == 0:
                        logger.debug(f"No images in {data_folder}")
                        imgs = np.array(["None"])
                    
                    data_to_bind = {
                        'Images': [imgs],
                        'Directory': [data_folder],
                        'Timepoint': [tp],
                        'Folder': [fld],
                        'Subfolder': [sf]
                    }
                    Table_out = pd.concat([Table_out, pd.DataFrame(data_to_bind)], ignore_index=True)
            else:
                # No subfolder, load directly from folder
                data_folder = folder_path
                imgs = load_folder_imgs(fdir=data_folder, files_to_find=filenames)
                
                if imgs is None or len(imgs) == 0:
                    logger.debug(f"No images in {data_folder}")
                    imgs = np.array(["None"])
                
                data_to_bind = {
                    'Images': [imgs],
                    'Directory': [data_folder],
                    'Timepoint': [tp],
                    'Folder': [fld],
                    'Subfolder': [None]
                }
                Table_out = pd.concat([Table_out, pd.DataFrame(data_to_bind)], ignore_index=True)
    
    logger.info(f"  Found {len(Table_out)} directories with images")
    return Table_out


def load_vent_structure(source_dir, reg_folder, timepoint=None, vent_dirs=None, 
                       vent_strings=None, vent_filters=None):
    """
    Load ventilation images from within registration folder structure.
    Pattern: source_dir / timepoint / reg_folder / vent_dir / images
    
    Args:
        source_dir: Patient directory (e.g., /data/Patient01)
        reg_folder: Registration folder name (e.g., 'Reg__TLC_2__RV')
        timepoint: Specific timepoint(s) to load (None = auto-detect all)
        vent_dirs: List of ventilation type folders (e.g., ['Vent_Int', 'Vent_Trans'])
        vent_strings: Dict mapping vent folder to image prefix (e.g., {'Vent_Int': 'sVent'})
        vent_filters: Additional filename filters (e.g., ['_medfilt_3.nii.gz'])
    
    Returns:
        DataFrame with columns: Images, Directory, Timepoint, VentType, RegFolder
    """
    if vent_dirs is None:
        vent_dirs = DEFAULT_VENT_DIRS
    if vent_strings is None:
        vent_strings = DEFAULT_VENT_STRINGS
    
    Table_out = pd.DataFrame()
    
    logger.info("Loading Ventilation Directory Structure...")
    logger.info(f"  Source directory: {source_dir}")
    logger.info(f"  Registration folder: {reg_folder}")
    logger.info(f"  Vent directories: {vent_dirs}")
    logger.info(f"  Vent strings: {vent_strings}")
    logger.info(f"  Vent filters: {vent_filters}")
    
    # Get timepoints
    timepoints = load_folder(fdir=source_dir, dirs_to_find=timepoint)
    
    if timepoints is None or len(timepoints) == 0:
        logger.warning(f"No timepoints found in {source_dir}")
        return Table_out
    
    for tp in timepoints:
        logger.debug(f"Processing timepoint: {tp}")
        time_point_path = os.path.join(source_dir, tp)
        
        # Find registration folder
        reg_path = os.path.join(time_point_path, reg_folder)
        actual_reg_folder = reg_folder  # Track actual folder name used
        
        if not os.path.exists(reg_path):
            # Try finding it with pattern matching
            matching_dirs = glob.glob(os.path.join(time_point_path, f"*{reg_folder}*"))
            if matching_dirs:
                reg_path = matching_dirs[0]
                actual_reg_folder = os.path.basename(reg_path)
                logger.info(f"Found registration folder: {reg_path}")
            else:
                logger.debug(f"Registration folder not found in {time_point_path}")
                continue
        
        # Look for ventilation directories within registration folder
        for vent_dir in vent_dirs:
            vent_path = os.path.join(reg_path, vent_dir)
            
            if not os.path.exists(vent_path):
                logger.debug(f"Vent directory '{vent_dir}' not found in {reg_path}")
                continue
            
            # Get the expected image prefix
            vent_prefix = vent_strings.get(vent_dir, None)
            
            # Build search pattern
            search_patterns = []
            if vent_prefix:
                if vent_filters:
                    for filt in vent_filters:
                        search_patterns.append(f"{vent_prefix}{filt}")
                else:
                    search_patterns.append(vent_prefix)
            elif vent_filters:
                search_patterns = vent_filters
            
            # Find images
            if search_patterns:
                imgs = load_folder_imgs(fdir=vent_path, files_to_find=search_patterns)
            else:
                imgs = load_folder_imgs(fdir=vent_path)
            
            if imgs is None or len(imgs) == 0:
                logger.debug(f"No matching images in {vent_path}")
                imgs = np.array(["None"])
            
            data_to_bind = {
                'Images': [imgs],
                'Directory': [vent_path],
                'Timepoint': [tp],
                'VentType': [vent_dir],
                'RegFolder': [actual_reg_folder]
            }
            Table_out = pd.concat([Table_out, pd.DataFrame(data_to_bind)], ignore_index=True)
    
    logger.info(f"  Found {len(Table_out)} ventilation directories with images")
    return Table_out


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
        'files': [],
        'inverse_files': []
    }
    
    if not os.path.exists(transform_dir):
        logger.error(f"Transform directory does not exist: {transform_dir}")
        return transforms
    
    # Check for composite transform (single file)
    composite_files = glob.glob(os.path.join(transform_dir, "*composite*1Warp.nii.gz"))
    
    # Check for standard transform pair
    warp_files = glob.glob(os.path.join(transform_dir, "*1Warp.nii.gz"))
    inv_warp_files = glob.glob(os.path.join(transform_dir, "*1InverseWarp.nii.gz"))
    affine_files = glob.glob(os.path.join(transform_dir, "*0GenericAffine.mat"))
    
    # Exclude composite files from warp files
    warp_files = [w for w in warp_files if "composite" not in w.lower() and "Inverse" not in w]
    
    if composite_files:
        transforms['type'] = 'composite'
        transforms['files'] = [composite_files[0]]
        logger.info(f"Found composite transform: {composite_files[0]}")
    elif warp_files and affine_files:
        transforms['type'] = 'standard'
        transforms['files'] = [warp_files[0], affine_files[0]]
        if inv_warp_files:
            transforms['inverse_files'] = [inv_warp_files[0], affine_files[0]]
        logger.info(f"Found standard transforms:")
        logger.info(f"  Warp: {warp_files[0]}")
        logger.info(f"  Affine: {affine_files[0]}")
        if inv_warp_files:
            logger.info(f"  Inverse Warp: {inv_warp_files[0]}")
    else:
        logger.warning(f"Could not identify transform type in {transform_dir}")
        logger.info(f"  Directory contents: {os.listdir(transform_dir)}")
        transforms['type'] = 'unknown'
        transforms['files'] = warp_files + affine_files
    
    return transforms


def find_transform_in_tree(patient_dir, reg_folder, timepoint=None):
    """
    Find transform files within a tree structure.
    Pattern: patient_dir / timepoint / reg_folder / transforms
    
    Args:
        patient_dir: Patient directory
        reg_folder: Registration folder name or pattern
        timepoint: Specific timepoint (None = auto-detect first)
    
    Returns:
        tuple: (transform_dir, transform_info, actual_reg_folder_name, timepoint_used)
    """
    logger.info("Searching for transforms in tree structure...")
    logger.info(f"  Patient dir: {patient_dir}")
    logger.info(f"  Reg folder pattern: {reg_folder}")
    
    # Get timepoints
    timepoints = load_folder(fdir=patient_dir, dirs_to_find=timepoint)
    
    if timepoints is None or len(timepoints) == 0:
        logger.error(f"No timepoints found in {patient_dir}")
        return None, None, None, None
    
    for tp in timepoints:
        tp_path = os.path.join(patient_dir, tp)
        
        # Look for registration folder
        reg_path = os.path.join(tp_path, reg_folder)
        actual_reg_folder = reg_folder
        
        if not os.path.exists(reg_path):
            # Try pattern matching
            matching_dirs = glob.glob(os.path.join(tp_path, f"*{reg_folder}*"))
            if matching_dirs:
                reg_path = matching_dirs[0]
                actual_reg_folder = os.path.basename(reg_path)
            else:
                continue
        
        # Check for transform files
        transform_info = find_transform_files(reg_path)
        
        if transform_info['files']:
            logger.info(f"Found transforms in: {reg_path}")
            return reg_path, transform_info, actual_reg_folder, tp
    
    logger.error(f"No transform files found in tree structure")
    return None, None, None, None


def find_reference_image(reference_identifier, search_dirs, img_folder=None):
    """
    Find reference image with flexible search strategy.
    
    Args:
        reference_identifier: Reference image identifier string or full path
        search_dirs: List of directories to search
        img_folder: Optional folder name to restrict search
    
    Returns:
        str: Full path to reference image, or None if not found
    """
    logger.info(f"Searching for reference image: {reference_identifier}")
    
    # Check if it's already a full path
    if os.path.exists(reference_identifier):
        logger.info(f"  Found (full path): {reference_identifier}")
        return reference_identifier
    
    # Search in provided directories
    for search_dir in search_dirs:
        if not search_dir or not os.path.exists(search_dir):
            continue
        
        logger.debug(f"  Searching in: {search_dir}")
        
        # Walk through directory tree
        for root, dirs, files in os.walk(search_dir):
            # If img_folder specified, only search in matching folders
            if img_folder and img_folder not in root:
                continue
            
            for f in files:
                if reference_identifier in f and f.endswith(('.nii', '.nii.gz', '.mha')):
                    ref_path = os.path.join(root, f)
                    logger.info(f"  Found: {ref_path}")
                    return ref_path
    
    logger.error(f"Reference image not found: {reference_identifier}")
    return None


def apply_transform_to_image(image_path, output_path, transform_info, reference_image, 
                            ants_path=None, dimensions="3", interpolation="Linear"):
    """
    Apply transform to a single image using ANTs.
    
    Args:
        image_path: Path to input image
        output_path: Path to output transformed image
        transform_info: Dictionary with transform type and file paths
        reference_image: Reference image for output space
        ants_path: Path to ANTs binaries
        dimensions: Image dimensions (2 or 3)
        interpolation: Interpolation method (Linear, NearestNeighbor, etc.)
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Set ANTs tool path
    if ants_path:
        ants_apply = os.path.join(ants_path, 'antsApplyTransforms')
    else:
        ants_apply = 'antsApplyTransforms'
    
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
    
    logger.info(f"Applying transform to: {os.path.basename(image_path)}")
    logger.debug(f"Command: {' '.join(apply_command)}")
    
    try:
        result = subprocess.run(apply_command, check=True, capture_output=True, text=True)
        logger.info(f"  Success! Output: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"  Failed to apply transform: {e}")
        logger.error(f"  stderr: {e.stderr}")
        return False


def filter_images(images, include_list=None, exclude_list=None):
    """
    Filter image list based on include/exclude patterns.
    
    Args:
        images: numpy array or list of image filenames
        include_list: List of patterns to include (None = include all)
        exclude_list: List of patterns to exclude (None = exclude none)
    
    Returns:
        Filtered list of images
    """
    if not isinstance(images, (list, np.ndarray)):
        return images
    
    filtered = []
    for img in images:
        if img == "None":
            continue
        
        # Check include list
        if include_list:
            if not any(pattern in img for pattern in include_list):
                continue
        
        # Check exclude list
        if exclude_list:
            if any(pattern in img for pattern in exclude_list):
                continue
        
        filtered.append(img)
    
    return filtered


def build_output_path_tree(output_dir, patient_name, timepoint, reg_folder, img_folder, subfolder=None):
    """
    Build output path for TREE mode preserving directory structure.
    
    Structure: {output_dir}/{patient}/{timepoint}/{reg_folder}/{img_folder}/(subfolder)/
    
    Args:
        output_dir: Base output directory
        patient_name: Patient identifier
        timepoint: Visit/timepoint name
        reg_folder: Registration folder name
        img_folder: Image folder name
        subfolder: Optional subfolder name
    
    Returns:
        str: Full output directory path
    """
    if subfolder:
        return os.path.join(output_dir, patient_name, timepoint, reg_folder, img_folder, subfolder)
    else:
        return os.path.join(output_dir, patient_name, timepoint, reg_folder, img_folder)


def build_output_path_vent(output_dir, patient_name, timepoint, reg_folder, vent_folder):
    """
    Build output path for VENT mode preserving directory structure.
    
    Structure: {output_dir}/{patient}/{timepoint}/{reg_folder}/{vent_folder}/
    
    Args:
        output_dir: Base output directory
        patient_name: Patient identifier
        timepoint: Visit/timepoint name
        reg_folder: Registration folder name
        vent_folder: Ventilation type folder name
    
    Returns:
        str: Full output directory path
    """
    return os.path.join(output_dir, patient_name, timepoint, reg_folder, vent_folder)


def build_output_path_direct(output_dir, patient_id):
    """
    Build output path for DIRECT mode.
    
    Structure: {output_dir}/{patient_id}/
    
    Args:
        output_dir: Base output directory
        patient_id: Patient identifier (provided as parameter)
    
    Returns:
        str: Full output directory path
    """
    if patient_id:
        return os.path.join(output_dir, patient_id)
    else:
        return output_dir


def process_direct_mode(image_path, transform_dir, reference_path, output_path,
                       ants_path=None, dimensions="3", patient_id=None, output_dir=None):
    """
    Process a single image in direct mode.
    
    Args:
        image_path: Full path to image to transform
        transform_dir: Directory containing transform files
        reference_path: Full path to reference image
        output_path: Full path to output file (if None, uses output_dir + patient_id)
        ants_path: Path to ANTs binaries
        dimensions: Image dimensions
        patient_id: Patient identifier for output structure
        output_dir: Base output directory (used with patient_id if output_path is None)
    
    Returns:
        bool: True if successful
    """
    logger.info("=" * 70)
    logger.info("DIRECT MODE - Single Image Transform")
    logger.info("=" * 70)
    logger.info(f"Image: {image_path}")
    logger.info(f"Transform dir: {transform_dir}")
    logger.info(f"Reference: {reference_path}")
    
    # Find transform files
    transform_info = find_transform_files(transform_dir)
    
    if not transform_info['files']:
        logger.error("No transform files found!")
        return False
    
    # Determine output path
    if output_path:
        final_output_path = output_path
    elif output_dir:
        final_output_dir = build_output_path_direct(output_dir, patient_id)
        os.makedirs(final_output_dir, exist_ok=True)
        
        # Create output filename
        img_name = os.path.basename(image_path)
        output_name = img_name.replace('.nii.gz', '_transformed.nii.gz')
        if '.nii.gz' not in output_name and '.nii' in output_name:
            output_name = output_name.replace('.nii', '_transformed.nii.gz')
        
        final_output_path = os.path.join(final_output_dir, output_name)
    else:
        logger.error("Either output_path or output_dir must be specified!")
        return False
    
    logger.info(f"Output: {final_output_path}")
    logger.info("=" * 70)
    
    # Determine interpolation
    img_name = os.path.basename(image_path).lower()
    interpolation = "NearestNeighbor" if any(x in img_name for x in ['mask', 'seg', 'label']) else "Linear"
    
    # Create output directory
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    
    # Apply transform
    success = apply_transform_to_image(
        image_path, final_output_path, transform_info, reference_path,
        ants_path, dimensions, interpolation
    )
    
    return success


def process_tree_mode(patient_dir, img_folder, transform_folder, reference_identifier,
                     ants_path=None, output_dir=None, subfolder=None, timepoint=None,
                     include_images=None, exclude_images=None, dimensions="3"):
    """
    Process images in tree structure mode.
    
    Output structure: {output_dir}/{patient}/{timepoint}/{reg_folder}/{img_folder}/(subfolder)/
    
    Args:
        patient_dir: Patient directory
        img_folder: Folder containing images
        transform_folder: Registration folder name
        reference_identifier: String to identify reference image
        ants_path: Path to ANTs binaries
        output_dir: Output directory (None = save in transform folder)
        subfolder: Optional subfolder within img_folder
        timepoint: Specific timepoint(s)
        include_images: Patterns to include
        exclude_images: Patterns to exclude
        dimensions: Image dimensions
    
    Returns:
        dict: Summary of processing results
    """
    logger.info("=" * 70)
    logger.info("TREE MODE - Process Images from Tree Structure")
    logger.info("=" * 70)
    logger.info(f"Patient directory: {patient_dir}")
    logger.info(f"Image folder: {img_folder}")
    logger.info(f"Transform folder: {transform_folder}")
    logger.info(f"Reference identifier: {reference_identifier}")
    logger.info("=" * 70)
    
    results = {'success': 0, 'skipped': 0, 'failed': 0}
    
    # Extract patient name from path
    patient_name = os.path.basename(patient_dir.rstrip(os.sep))
    logger.info(f"Patient name: {patient_name}")
    
    # Load images from tree structure
    image_df = load_tree_structure(patient_dir, img_folder, subfolder, timepoint)
    
    if image_df.empty:
        logger.error("No images found in tree structure!")
        return results
    
    # Find transforms in tree structure
    transform_dir, transform_info, actual_reg_folder, tp_used = find_transform_in_tree(
        patient_dir, transform_folder, timepoint
    )
    
    if not transform_info or not transform_info['files']:
        logger.error("No transform files found!")
        return results
    
    # Find reference image
    search_dirs = [patient_dir, transform_dir]
    reference_path = find_reference_image(reference_identifier, search_dirs, img_folder)
    
    if not reference_path:
        logger.error(f"Reference image not found: {reference_identifier}")
        return results
    
    logger.info(f"Using reference: {reference_path}")
    
    # Process each image
    for idx, row in image_df.iterrows():
        imgs = row['Images']
        img_dir = row['Directory']
        row_timepoint = row.get('Timepoint', tp_used)
        row_folder = row.get('Folder', img_folder)
        row_subfolder = row.get('Subfolder', None)
        
        if not isinstance(imgs, (list, np.ndarray)):
            continue
        
        # Filter images
        filtered_imgs = filter_images(imgs, include_images, exclude_images)
        
        # Determine output directory for this row
        if output_dir:
            final_output_dir = build_output_path_tree(
                output_dir, patient_name, row_timepoint, actual_reg_folder, 
                row_folder, row_subfolder
            )
        else:
            # Save in transform folder, preserving img_folder structure
            if row_subfolder:
                final_output_dir = os.path.join(transform_dir, row_folder, row_subfolder)
            else:
                final_output_dir = os.path.join(transform_dir, row_folder)
        
        os.makedirs(final_output_dir, exist_ok=True)
        logger.info(f"Output directory: {final_output_dir}")
        
        for img in filtered_imgs:
            if img == "None" or reference_identifier in img:
                results['skipped'] += 1
                continue
            
            img_path = os.path.join(img_dir, img)
            
            # Determine interpolation
            interpolation = "NearestNeighbor" if any(x in img.lower() for x in ['mask', 'seg', 'label']) else "Linear"
            
            # Create output filename
            output_name = img.replace('.nii.gz', '_transformed.nii.gz')
            if '.nii.gz' not in output_name and '.nii' in output_name:
                output_name = output_name.replace('.nii', '_transformed.nii.gz')
            
            output_path = os.path.join(final_output_dir, output_name)
            
            # Apply transform
            if apply_transform_to_image(img_path, output_path, transform_info, 
                                       reference_path, ants_path, dimensions, interpolation):
                results['success'] += 1
            else:
                results['failed'] += 1
    
    logger.info("=" * 70)
    logger.info(f"PROCESSING COMPLETE: {results['success']} success, {results['skipped']} skipped, {results['failed']} failed")
    logger.info("=" * 70)
    
    return results


def process_vent_mode(patient_dir, reg_folder, reference_identifier,
                     ants_path=None, output_dir=None, timepoint=None,
                     vent_dirs=None, vent_strings=None, vent_filters=None,
                     include_images=None, exclude_images=None, dimensions="3"):
    """
    Process ventilation images within registration folder structure.
    
    Output structure: {output_dir}/{patient}/{timepoint}/{reg_folder}/{vent_folder}/
    
    Args:
        patient_dir: Patient directory
        reg_folder: Registration folder name
        reference_identifier: String to identify reference image
        ants_path: Path to ANTs binaries
        output_dir: Output directory (None = save in transform folder)
        timepoint: Specific timepoint(s)
        vent_dirs: List of ventilation type folders
        vent_strings: Dict mapping vent folder to image prefix
        vent_filters: Additional filename filters
        include_images: Patterns to include
        exclude_images: Patterns to exclude
        dimensions: Image dimensions
    
    Returns:
        dict: Summary of processing results
    """
    logger.info("=" * 70)
    logger.info("VENT MODE - Process Ventilation Images")
    logger.info("=" * 70)
    logger.info(f"Patient directory: {patient_dir}")
    logger.info(f"Registration folder: {reg_folder}")
    logger.info(f"Reference identifier: {reference_identifier}")
    logger.info(f"Vent directories: {vent_dirs}")
    logger.info("=" * 70)
    
    if vent_dirs is None:
        vent_dirs = DEFAULT_VENT_DIRS
    if vent_strings is None:
        vent_strings = DEFAULT_VENT_STRINGS
    
    results = {'success': 0, 'skipped': 0, 'failed': 0}
    
    # Extract patient name from path
    patient_name = os.path.basename(patient_dir.rstrip(os.sep))
    logger.info(f"Patient name: {patient_name}")
    
    # Load ventilation images
    image_df = load_vent_structure(patient_dir, reg_folder, timepoint, 
                                   vent_dirs, vent_strings, vent_filters)
    
    if image_df.empty:
        logger.error("No ventilation images found!")
        return results
    
    # Find transforms - they should be in the same reg_folder as the vent images
    transform_dir, transform_info, actual_reg_folder, tp_used = find_transform_in_tree(
        patient_dir, reg_folder, timepoint
    )
    
    if not transform_info or not transform_info['files']:
        logger.error("No transform files found!")
        return results
    
    # Find reference image (usually in the registration folder or nearby)
    search_dirs = [patient_dir, transform_dir]
    reference_path = find_reference_image(reference_identifier, search_dirs)
    
    if not reference_path:
        logger.error(f"Reference image not found: {reference_identifier}")
        return results
    
    logger.info(f"Using reference: {reference_path}")
    
    # Process each ventilation image
    for idx, row in image_df.iterrows():
        imgs = row['Images']
        img_dir = row['Directory']
        row_timepoint = row.get('Timepoint', tp_used)
        vent_type = row.get('VentType', 'unknown')
        row_reg_folder = row.get('RegFolder', actual_reg_folder)
        
        if not isinstance(imgs, (list, np.ndarray)):
            continue
        
        # Filter images
        filtered_imgs = filter_images(imgs, include_images, exclude_images)
        
        # Determine output directory for this vent type
        if output_dir:
            final_output_dir = build_output_path_vent(
                output_dir, patient_name, row_timepoint, row_reg_folder, vent_type
            )
        else:
            # Save in same location (vent folder within reg folder)
            final_output_dir = img_dir
        
        os.makedirs(final_output_dir, exist_ok=True)
        logger.info(f"Output directory: {final_output_dir}")
        
        for img in filtered_imgs:
            if img == "None" or reference_identifier in img:
                results['skipped'] += 1
                continue
            
            img_path = os.path.join(img_dir, img)
            
            # Use linear interpolation for ventilation images
            interpolation = "Linear"
            
            # Create output filename
            output_name = img.replace('.nii.gz', '_transformed.nii.gz')
            if '.nii.gz' not in output_name and '.nii' in output_name:
                output_name = output_name.replace('.nii', '_transformed.nii.gz')
            
            output_path = os.path.join(final_output_dir, output_name)
            
            # Apply transform
            if apply_transform_to_image(img_path, output_path, transform_info, 
                                       reference_path, ants_path, dimensions, interpolation):
                results['success'] += 1
            else:
                results['failed'] += 1
    
    logger.info("=" * 70)
    logger.info(f"PROCESSING COMPLETE: {results['success']} success, {results['skipped']} skipped, {results['failed']} failed")
    logger.info("=" * 70)
    
    return results


#%% Argument Parsing

def create_parser():
    parser = argparse.ArgumentParser(
        description="Apply transforms to images - supports direct, tree, and ventilation modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    
    # DIRECT mode - single image:
    python script.py --mode direct \\
        --direct_image /data/patient/image.nii.gz \\
        --direct_transform_dir /data/patient/Reg_folder \\
        --direct_reference /data/patient/reference.nii.gz \\
        --direct_output /output/transformed.nii.gz

    # TREE mode - images in tree structure:
    python script.py --mode tree \\
        --patient_dir /data/Patient01 \\
        --img_folder Images \\
        --transform_folder Reg_TLC_2_RV \\
        --reference RV \\
        --exclude mask seg

    # VENT mode - ventilation images within registration folder:
    python script.py --mode vent \\
        --patient_dir /data/Patient01 \\
        --transform_folder Reg_TLC_2_RV \\
        --reference RV \\
        --vent_dirs Vent_Int Vent_Trans Vent_Hyb3 \\
        --vent_strings Vent_Int:sVent Vent_Trans:JacVent Vent_Hyb3:HYCID
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', choices=['direct', 'tree', 'vent'], required=True,
                       help='Processing mode')
    
    # Common parameters
    parser.add_argument('-pat_dir', '--patient_dir', type=Path,
                       help="Patient directory")
    parser.add_argument('-ants_path', '--ants_path', type=Path,
                       help="Path to ANTs binaries")
    parser.add_argument('-dim', '--dimensions', type=str, default="3",
                       help="Image dimensions (2 or 3)")
    parser.add_argument('-out_dir', '--output_dir', type=Path,
                       help="Output base directory")
    parser.add_argument('-timepoint', '--timepoint', type=str, nargs='*',
                       help="Specific timepoint(s) to process")
    
    # DIRECT mode arguments
    parser.add_argument('--direct_image', type=Path,
                       help="[DIRECT] Full path to image to transform")
    parser.add_argument('--direct_transform_dir', type=Path,
                       help="[DIRECT] Directory containing transform files")
    parser.add_argument('--direct_reference', type=Path,
                       help="[DIRECT] Full path to reference image")
    parser.add_argument('--direct_output', type=Path,
                       help="[DIRECT] Full path to output file")
    parser.add_argument('-patient_id', '--patient_id', type=str,
                       help="[DIRECT] Patient identifier for output structure")
    
    # TREE mode arguments
    parser.add_argument('-img_folder', '--img_folder', type=str,
                       help="[TREE] Folder name containing images")
    parser.add_argument('-sub_folder', '--sub_folder', type=str,
                       help="[TREE] Optional subfolder within img_folder")
    parser.add_argument('-ref', '--reference', type=str,
                       help="[TREE/VENT] Reference image identifier")
    parser.add_argument('-trans_folder', '--transform_folder', type=str,
                       help="[TREE/VENT] Registration folder name")
    
    # VENT mode arguments
    parser.add_argument('-vent_dirs', '--vent_dirs', type=str, nargs='+',
                       help="[VENT] Ventilation type folders")
    parser.add_argument('-vent_strings', '--vent_strings', type=str, nargs='+',
                       help="[VENT] Vent folder to prefix mapping (format: folder:prefix)")
    parser.add_argument('-vent_filters', '--vent_filters', type=str, nargs='+',
                       help="[VENT] Additional filename filters")
    
    # Filtering
    parser.add_argument('-include', '--include_images', type=str, nargs='+',
                       help="List of identifiers for images to include")
    parser.add_argument('-exclude', '--exclude_images', type=str, nargs='+',
                       help="List of identifiers for images to exclude")
    
    return parser


def parse_vent_strings(vent_strings_list):
    """Parse vent_strings from command line format (folder:prefix) to dict"""
    if not vent_strings_list:
        return None
    
    result = {}
    for item in vent_strings_list:
        if ':' in item:
            folder, prefix = item.split(':', 1)
            result[folder] = prefix
    
    return result if result else None


#%% Main

def main():
    if manual:
        # Use manual parameters
        logger.info("=" * 50)
        logger.info("Running in MANUAL MODE with hardcoded parameters")
        logger.info("=" * 50)
        
        mode = manual_params['mode']
        
        if mode == 'direct':
            process_direct_mode(
                image_path=manual_params['direct_image'],
                transform_dir=manual_params['direct_transform_dir'],
                reference_path=manual_params['direct_reference'],
                output_path=manual_params['direct_output'],
                ants_path=manual_params['ants_path'],
                dimensions=manual_params['dimensions'],
                patient_id=manual_params.get('patient_id'),
                output_dir=manual_params.get('output_dir')
            )
        
        elif mode == 'tree':
            process_tree_mode(
                patient_dir=manual_params['patient_dir'],
                img_folder=manual_params['img_folder'],
                transform_folder=manual_params['transform_folder'],
                reference_identifier=manual_params['reference_identifier'],
                ants_path=manual_params['ants_path'],
                output_dir=manual_params.get('output_dir'),
                subfolder=manual_params.get('sub_folder'),
                timepoint=manual_params.get('timepoint'),
                include_images=manual_params.get('include_images'),
                exclude_images=manual_params.get('exclude_images'),
                dimensions=manual_params['dimensions']
            )
        
        elif mode == 'vent':
            process_vent_mode(
                patient_dir=manual_params['patient_dir'],
                reg_folder=manual_params['transform_folder'],
                reference_identifier=manual_params['reference_identifier'],
                ants_path=manual_params['ants_path'],
                output_dir=manual_params.get('output_dir'),
                timepoint=manual_params.get('timepoint'),
                vent_dirs=manual_params.get('vent_dirs'),
                vent_strings=manual_params.get('vent_strings'),
                vent_filters=manual_params.get('vent_img_filters'),
                include_images=manual_params.get('include_images'),
                exclude_images=manual_params.get('exclude_images'),
                dimensions=manual_params['dimensions']
            )
    
    else:
        # Parse command line arguments
        parser = create_parser()
        args = parser.parse_args()
        
        mode = args.mode
        
        if mode == 'direct':
            # Validate required arguments
            if not args.direct_image or not args.direct_transform_dir or not args.direct_reference:
                parser.error("DIRECT mode requires: --direct_image, --direct_transform_dir, --direct_reference")
            
            if not args.direct_output and not args.output_dir:
                parser.error("DIRECT mode requires either --direct_output or --output_dir (with --patient_id)")
            
            process_direct_mode(
                image_path=str(args.direct_image),
                transform_dir=str(args.direct_transform_dir),
                reference_path=str(args.direct_reference),
                output_path=str(args.direct_output) if args.direct_output else None,
                ants_path=str(args.ants_path) if args.ants_path else None,
                dimensions=args.dimensions,
                patient_id=args.patient_id,
                output_dir=str(args.output_dir) if args.output_dir else None
            )
        
        elif mode == 'tree':
            # Validate required arguments
            if not all([args.patient_dir, args.img_folder, args.transform_folder, args.reference]):
                parser.error("TREE mode requires: --patient_dir, --img_folder, "
                           "--transform_folder, --reference")
            
            process_tree_mode(
                patient_dir=str(args.patient_dir),
                img_folder=args.img_folder,
                transform_folder=args.transform_folder,
                reference_identifier=args.reference,
                ants_path=str(args.ants_path) if args.ants_path else None,
                output_dir=str(args.output_dir) if args.output_dir else None,
                subfolder=args.sub_folder,
                timepoint=args.timepoint,
                include_images=args.include_images,
                exclude_images=args.exclude_images,
                dimensions=args.dimensions
            )
        
        elif mode == 'vent':
            # Validate required arguments
            if not all([args.patient_dir, args.transform_folder, args.reference]):
                parser.error("VENT mode requires: --patient_dir, --transform_folder, --reference")
            
            # Parse vent_strings
            vent_strings = parse_vent_strings(args.vent_strings)
            
            process_vent_mode(
                patient_dir=str(args.patient_dir),
                reg_folder=args.transform_folder,
                reference_identifier=args.reference,
                ants_path=str(args.ants_path) if args.ants_path else None,
                output_dir=str(args.output_dir) if args.output_dir else None,
                timepoint=args.timepoint,
                vent_dirs=args.vent_dirs,
                vent_strings=vent_strings,
                vent_filters=args.vent_filters,
                include_images=args.include_images,
                exclude_images=args.exclude_images,
                dimensions=args.dimensions
            )


if __name__ == "__main__":
    main()

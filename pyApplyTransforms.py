#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Apply Transform Script - Version 3.2 (Fixed Inverse Transform Order)
--
by Will Clark 
POLARIS - University of Sheffield
--
Supports three input modes:
1. DIRECT: Apply transforms to a single image file
2. TREE: Apply transforms to images in a nested tree structure (Patient/visit/folder/images)
3. VENT: Apply transforms to ventilation images within registration folders 
         (Patient/visit/Reg_folder/Vent_folder/images)

NEW in v3.1: Flexible transform mode control
    - forward: Standard affine + warp (default, backward compatible)
    - inverse: Affine (inverted) + InverseWarp 
    - warp_only: Single warp file only (for composite transforms)
    - inverse_warp_only: Single inverse warp only

NEW in v3.2: 
    - Fixed inverse transform order: Now correctly applies InverseWarp first, then inverted Affine
      (Command line: -t [Affine.mat,1] -t InverseWarp.nii.gz)
    - Added inverse_use_ants flag: Force using forward warp with ANTs inversion flag instead
      of pre-generated inverse warp file

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
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


#%% Transform Mode Definitions

class TransformMode(Enum):
    """
    Transform application modes.
    
    FORWARD: Apply affine + forward warp (standard registration output)
    INVERSE: Apply inverse affine + inverse warp (reverse the registration)
    WARP_ONLY: Apply only the warp transform (no affine)
    INVERSE_WARP_ONLY: Apply only the inverse warp transform (no affine)
    """
    FORWARD = "forward"
    INVERSE = "inverse"
    WARP_ONLY = "warp_only"
    INVERSE_WARP_ONLY = "inverse_warp_only"


@dataclass
class TransformFiles:
    """
    Container for transform file paths with explicit control over which to use.
    
    Attributes:
        affine: Path to affine transform file (.mat)
        warp: Path to forward warp file (1Warp.nii.gz)
        inverse_warp: Path to inverse warp file (1InverseWarp.nii.gz)
        composite: Path to composite transform (if single-file format)
        transform_type: 'standard', 'composite', 'explicit', or 'unknown'
        user_specified: Whether files were explicitly specified by user
        inverse_use_ants: If True, use forward warp with ANTs inversion flag instead of inverse warp file
    """
    affine: Optional[str] = None
    warp: Optional[str] = None
    inverse_warp: Optional[str] = None
    composite: Optional[str] = None
    transform_type: str = "unknown"
    user_specified: bool = False
    inverse_use_ants: bool = False
    
    @classmethod
    def from_explicit_files(cls, transform_dir: str, 
                           affine_file: Optional[str] = None,
                           warp_file: Optional[str] = None,
                           inverse_warp_file: Optional[str] = None) -> 'TransformFiles':
        """
        Create TransformFiles from explicitly specified filenames.
        
        Args:
            transform_dir: Base directory for relative paths
            affine_file: Affine transform filename (relative or absolute)
            warp_file: Warp transform filename (relative or absolute)
            inverse_warp_file: Inverse warp transform filename (relative or absolute)
        
        Returns:
            TransformFiles instance with specified files
        """
        instance = cls(user_specified=True)
        
        def resolve_path(filename: Optional[str]) -> Optional[str]:
            """Resolve filename to full path, checking existence."""
            if filename is None:
                return None
            # Check if absolute path
            if os.path.isabs(filename):
                if os.path.exists(filename):
                    return filename
                else:
                    logger.warning(f"Specified file not found: {filename}")
                    return None
            # Relative path - resolve against transform_dir
            full_path = os.path.join(transform_dir, filename)
            if os.path.exists(full_path):
                return full_path
            else:
                logger.warning(f"Specified file not found: {full_path}")
                return None
        
        instance.affine = resolve_path(affine_file)
        instance.warp = resolve_path(warp_file)
        instance.inverse_warp = resolve_path(inverse_warp_file)
        
        # Determine transform type based on what was provided
        if instance.warp and instance.affine:
            instance.transform_type = 'explicit'
            logger.info(f"Using explicitly specified transforms:")
            logger.info(f"  Affine: {instance.affine}")
            logger.info(f"  Warp: {instance.warp}")
            if instance.inverse_warp:
                logger.info(f"  Inverse Warp: {instance.inverse_warp}")
        elif instance.warp:
            # Single warp file - could be composite or warp-only
            instance.transform_type = 'explicit'
            logger.info(f"Using explicitly specified warp (no affine):")
            logger.info(f"  Warp: {instance.warp}")
            if instance.inverse_warp:
                logger.info(f"  Inverse Warp: {instance.inverse_warp}")
        elif instance.affine:
            instance.transform_type = 'explicit'
            logger.info(f"Using explicitly specified affine only:")
            logger.info(f"  Affine: {instance.affine}")
        else:
            instance.transform_type = 'unknown'
            logger.warning("No valid transform files found from explicit specification")
        
        return instance
    
    def has_forward_transforms(self) -> bool:
        """Check if forward transforms are available."""
        if self.transform_type == 'composite':
            return self.composite is not None
        return self.warp is not None
    
    def has_inverse_transforms(self) -> bool:
        """Check if inverse transforms are available."""
        if self.transform_type == 'composite':
            # Composite transforms can be inverted via ANTs flag
            return self.composite is not None
        return self.inverse_warp is not None
    
    def get_transforms_for_mode(self, mode: TransformMode) -> List[Dict[str, str]]:
        """
        Get the transform specifications for a given mode.
        
        Returns list of dicts with 'path' and 'invert' keys.
        ANTs applies transforms in reverse order (last specified is applied first).
        For forward: warp is applied first, then affine (so specify: warp, affine)
        For inverse: inverse_affine first, then inverse_warp (so specify: inverse_warp, affine[inverted])
        
        Returns:
            List of transform specifications: [{'path': str, 'invert': bool}, ...]
        """
        transforms = []
        
        if mode == TransformMode.FORWARD:
            if self.transform_type == 'composite':
                if self.composite:
                    transforms.append({'path': self.composite, 'invert': False})
            else:
                # Standard/explicit: warp then affine (ANTs applies in reverse order)
                if self.warp:
                    transforms.append({'path': self.warp, 'invert': False})
                if self.affine:
                    transforms.append({'path': self.affine, 'invert': False})
                    
        elif mode == TransformMode.INVERSE:
            if self.transform_type == 'composite':
                if self.composite:
                    transforms.append({'path': self.composite, 'invert': True})
            else:
                # Inverse of T = Warp ∘ Affine is T^{-1} = Affine^{-1} ∘ InverseWarp
                # Execution order: InverseWarp first, then Affine^{-1} second
                # ANTs applies transforms in REVERSE order (last specified = first applied)
                # So we specify: affine[inverted] first, then inverse_warp second
                # This produces: -t [Affine.mat,1] -t InverseWarp.nii.gz
                # ANTs will apply InverseWarp first, then inverted Affine second
                if self.affine:
                    transforms.append({'path': self.affine, 'invert': True})
                
                # Handle warp inversion - either use inverse warp file or ANTs inversion flag
                if self.inverse_use_ants:
                    # Force using forward warp with ANTs inversion flag
                    if self.warp:
                        logger.info("Using forward warp with ANTs inversion flag (inverse_use_ants=True)")
                        transforms.append({'path': self.warp, 'invert': True})
                    else:
                        logger.warning("inverse_use_ants=True but no forward warp found")
                elif self.inverse_warp:
                    transforms.append({'path': self.inverse_warp, 'invert': False})
                elif self.warp:
                    # Fallback: invert the forward warp if no inverse available
                    logger.warning("No inverse warp found, using forward warp with inversion flag")
                    transforms.append({'path': self.warp, 'invert': True})
                    
        elif mode == TransformMode.WARP_ONLY:
            if self.transform_type == 'composite':
                if self.composite:
                    transforms.append({'path': self.composite, 'invert': False})
            elif self.warp:
                transforms.append({'path': self.warp, 'invert': False})
                
        elif mode == TransformMode.INVERSE_WARP_ONLY:
            if self.transform_type == 'composite':
                if self.composite:
                    transforms.append({'path': self.composite, 'invert': True})
            elif self.inverse_warp:
                transforms.append({'path': self.inverse_warp, 'invert': False})
            elif self.warp:
                logger.warning("No inverse warp found, using forward warp with inversion flag")
                transforms.append({'path': self.warp, 'invert': True})
        
        return transforms
    
    def validate_mode(self, mode: TransformMode) -> tuple[bool, str]:
        """
        Validate that the required transforms exist for the given mode.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if mode == TransformMode.FORWARD:
            if self.transform_type == 'composite':
                if not self.composite:
                    return False, "Composite transform file not found"
            else:
                if not self.warp:
                    return False, "Forward warp file not found"
            return True, ""
            
        elif mode == TransformMode.INVERSE:
            if self.transform_type == 'composite':
                if not self.composite:
                    return False, "Composite transform file not found for inversion"
            else:
                if not self.inverse_warp and not self.warp:
                    return False, "Neither inverse warp nor forward warp found for inverse mode"
            return True, ""
            
        elif mode == TransformMode.WARP_ONLY:
            if self.transform_type == 'composite':
                if not self.composite:
                    return False, "Composite transform file not found"
            elif not self.warp:
                return False, "Warp file not found for warp_only mode"
            return True, ""
            
        elif mode == TransformMode.INVERSE_WARP_ONLY:
            if self.transform_type == 'composite':
                if not self.composite:
                    return False, "Composite transform file not found for inversion"
            elif not self.inverse_warp and not self.warp:
                return False, "Neither inverse warp nor forward warp found for inverse_warp_only mode"
            return True, ""
        
        return False, f"Unknown transform mode: {mode}"
    
    def __str__(self) -> str:
        """String representation showing available transforms."""
        parts = [f"TransformFiles(type={self.transform_type}"]
        if self.affine:
            parts.append(f"affine={os.path.basename(self.affine)}")
        if self.warp:
            parts.append(f"warp={os.path.basename(self.warp)}")
        if self.inverse_warp:
            parts.append(f"inv_warp={os.path.basename(self.inverse_warp)}")
        if self.composite:
            parts.append(f"composite={os.path.basename(self.composite)}")
        if self.user_specified:
            parts.append("user_specified=True")
        return ", ".join(parts) + ")"


#%% MANUAL MODE PARAMETERS
manual = False  # Set to True when running in IDE

# Manual parameters for testing - edit these when running in IDE
manual_params = {
    # Mode selection: 'direct', 'tree', or 'vent'
    'mode': 'tree',
    
    # NEW: Transform mode selection
    'transform_mode': 'forward',  # 'forward', 'inverse', 'warp_only', 'inverse_warp_only'
    
    # NEW: Explicit transform file specification (optional - overrides auto-detection)
    # If None, auto-detection is used. Paths can be relative to transform folder or absolute.
    'affine_file': None,         # e.g., 'Reg__TLC_2__RV_0GenericAffine.mat'
    'warp_file': None,           # e.g., 'Reg__TLC_2__RV_1Warp.nii.gz'
    'inverse_warp_file': None,   # e.g., 'Reg__TLC_2__RV_1InverseWarp.nii.gz'
    'inverse_use_ants': False,   # If True, use forward warp with ANTs inversion flag instead of inverse warp file
    
    # Common parameters
    'patient_dir': r'/path',
    'ants_path': r'/path', #Use None if you want to use antspy
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
    'vent_dir': None,  # Ventilation patient directory (if different from patient_dir)
    'vent_transform_folder': None,  # Ventilation registration folder
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


#%% Transform Finding Functions (Updated for TransformFiles dataclass)

def find_transform_files(transform_dir, affine_file=None, warp_file=None, 
                        inverse_warp_file=None, inverse_use_ants=False) -> TransformFiles:
    """
    Find and identify transform files in a directory.
    
    If explicit filenames are provided, uses those instead of auto-detection.
    
    Args:
        transform_dir: Directory containing transform files
        affine_file: Optional explicit affine filename (relative or absolute)
        warp_file: Optional explicit warp filename (relative or absolute)
        inverse_warp_file: Optional explicit inverse warp filename (relative or absolute)
        inverse_use_ants: If True, use forward warp with ANTs inversion flag instead of inverse warp file
    
    Returns:
        TransformFiles: Dataclass with all found transform file paths
    """
    # Check if any explicit files were specified
    has_explicit = any([affine_file, warp_file, inverse_warp_file])
    
    if has_explicit:
        logger.info("Using explicitly specified transform files")
        result = TransformFiles.from_explicit_files(
            transform_dir, 
            affine_file=affine_file,
            warp_file=warp_file,
            inverse_warp_file=inverse_warp_file
        )
        result.inverse_use_ants = inverse_use_ants
        return result
    
    # Auto-detection mode
    transforms = TransformFiles()
    transforms.inverse_use_ants = inverse_use_ants
    
    if not os.path.exists(transform_dir):
        logger.error(f"Transform directory does not exist: {transform_dir}")
        return transforms
    
    # Check for composite transform (single file)
    composite_files = glob.glob(os.path.join(transform_dir, "*composite*1Warp.nii.gz"))
    # Also check for Composite.h5 format
    composite_h5_files = glob.glob(os.path.join(transform_dir, "*Composite.h5"))
    
    # Check for standard transform files
    warp_files = glob.glob(os.path.join(transform_dir, "*1Warp.nii.gz"))
    inv_warp_files = glob.glob(os.path.join(transform_dir, "*1InverseWarp.nii.gz"))
    affine_files = glob.glob(os.path.join(transform_dir, "*0GenericAffine.mat"))
    
    # Exclude composite files from warp files
    warp_files = [w for w in warp_files if "composite" not in w.lower() and "Inverse" not in w]
    
    if composite_files or composite_h5_files:
        transforms.transform_type = 'composite'
        transforms.composite = composite_files[0] if composite_files else composite_h5_files[0]
        logger.info(f"Found composite transform: {transforms.composite}")
        
    elif warp_files:
        transforms.transform_type = 'standard'
        transforms.warp = warp_files[0]
        logger.info(f"Found forward warp: {transforms.warp}")
        
        if affine_files:
            transforms.affine = affine_files[0]
            logger.info(f"Found affine: {transforms.affine}")
            
        if inv_warp_files:
            transforms.inverse_warp = inv_warp_files[0]
            logger.info(f"Found inverse warp: {transforms.inverse_warp}")
    else:
        transforms.transform_type = 'unknown'
        logger.warning(f"Could not identify transform type in {transform_dir}")
        logger.info(f"  Directory contents: {os.listdir(transform_dir)}")
        # Still try to capture any transforms found
        if affine_files:
            transforms.affine = affine_files[0]
    
    if inverse_use_ants:
        logger.info("inverse_use_ants=True: Will use forward warp with ANTs inversion flag for inverse transforms")
    
    return transforms


def find_transform_in_tree(patient_dir, reg_folder, timepoint=None,
                          affine_file=None, warp_file=None, inverse_warp_file=None,
                          inverse_use_ants=False):
    """
    Find transform files within a tree structure.
    Pattern: patient_dir / timepoint / reg_folder / transforms
    
    Args:
        patient_dir: Patient directory
        reg_folder: Registration folder name or pattern
        timepoint: Specific timepoint (None = auto-detect first)
        affine_file: Optional explicit affine filename
        warp_file: Optional explicit warp filename
        inverse_warp_file: Optional explicit inverse warp filename
        inverse_use_ants: If True, use forward warp with ANTs inversion flag instead of inverse warp file
    
    Returns:
        tuple: (transform_dir, TransformFiles, actual_reg_folder_name, timepoint_used)
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
        
        # Check for transform files (with optional explicit specification)
        transform_files = find_transform_files(
            reg_path, 
            affine_file=affine_file,
            warp_file=warp_file,
            inverse_warp_file=inverse_warp_file,
            inverse_use_ants=inverse_use_ants
        )
        
        if transform_files.has_forward_transforms() or transform_files.has_inverse_transforms():
            logger.info(f"Found transforms in: {reg_path}")
            return reg_path, transform_files, actual_reg_folder, tp
    
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


#%% Transform Application Function (Updated for TransformMode)

def apply_transform_to_image(image_path, output_path, transform_files: TransformFiles, 
                            reference_image, transform_mode: TransformMode = TransformMode.FORWARD,
                            ants_path=None, dimensions="3", interpolation="Linear"):
    """
    Apply transform to a single image using ANTs.
    
    Args:
        image_path: Path to input image
        output_path: Path to output transformed image
        transform_files: TransformFiles dataclass with transform file paths
        reference_image: Reference image for output space
        transform_mode: TransformMode enum specifying which transforms to apply
        ants_path: Path to ANTs binaries
        dimensions: Image dimensions (2 or 3)
        interpolation: Interpolation method (Linear, NearestNeighbor, etc.)
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Validate mode
    is_valid, error_msg = transform_files.validate_mode(transform_mode)
    if not is_valid:
        logger.error(f"Transform mode validation failed: {error_msg}")
        return False
    
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
    
    # Get transforms for the specified mode
    transforms = transform_files.get_transforms_for_mode(transform_mode)
    
    if not transforms:
        logger.error(f"No transforms available for mode: {transform_mode.value}")
        return False
    
    # Add transforms to command
    # ANTs syntax: -t [transform,invert_flag] where invert_flag is 0 or 1
    for t in transforms:
        if t['invert']:
            apply_command.extend(["-t", f"[{t['path']},1]"])
        else:
            apply_command.extend(["-t", t['path']])
    
    logger.info(f"Applying transform ({transform_mode.value}) to: {os.path.basename(image_path)}")
    logger.debug(f"Command: {' '.join(apply_command)}")
    
    
    #Output formatted Transform command to .txt file (similar to the main registration script!)
    
    
    try:
        result = subprocess.run(apply_command, check=True, capture_output=True, text=True)
        logger.info(f"  Success! Output: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"  Failed to apply transform: {e}")
        logger.error(f"  stderr: {e.stderr}")
        return False


#%% Filtering and Path Building Functions

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
    """
    if subfolder:
        return os.path.join(output_dir, patient_name, timepoint, reg_folder, img_folder, subfolder)
    else:
        return os.path.join(output_dir, patient_name, timepoint, reg_folder, img_folder)


def build_output_path_vent(output_dir, patient_name, timepoint, reg_folder, vent_folder):
    """
    Build output path for VENT mode preserving directory structure.
    
    Structure: {output_dir}/{patient}/{timepoint}/{reg_folder}/{vent_folder}/
    """
    return os.path.join(output_dir, patient_name, timepoint, reg_folder, vent_folder)


def build_output_path_direct(output_dir, patient_id):
    """
    Build output path for DIRECT mode.
    
    Structure: {output_dir}/{patient_id}/
    """
    return os.path.join(output_dir, patient_id) if patient_id else output_dir


#%% Processing Functions

def process_direct_mode(image_path, transform_dir, reference_path, output_path,
                       transform_mode=TransformMode.FORWARD,
                       affine_file=None, warp_file=None, inverse_warp_file=None,
                       inverse_use_ants=False,
                       ants_path=None, dimensions="3", patient_id=None, output_dir=None):
    """
    Process a single image directly.
    
    Args:
        image_path: Path to image to transform
        transform_dir: Directory containing transforms
        reference_path: Path to reference image
        output_path: Path for output (optional if output_dir provided)
        transform_mode: TransformMode enum for which transforms to apply
        affine_file: Optional explicit affine filename
        warp_file: Optional explicit warp filename
        inverse_warp_file: Optional explicit inverse warp filename
        inverse_use_ants: If True, use forward warp with ANTs inversion flag instead of inverse warp file
        ants_path: Path to ANTs binaries
        dimensions: Image dimensions
        patient_id: Patient ID for output structure
        output_dir: Base output directory
    
    Returns:
        dict: Processing results
    """
    logger.info("=" * 70)
    logger.info("DIRECT MODE - Single Image Transform")
    logger.info("=" * 70)
    logger.info(f"Image: {image_path}")
    logger.info(f"Transform directory: {transform_dir}")
    logger.info(f"Reference: {reference_path}")
    logger.info(f"Transform mode: {transform_mode.value}")
    if any([affine_file, warp_file, inverse_warp_file]):
        logger.info(f"Explicit files: affine={affine_file}, warp={warp_file}, inv_warp={inverse_warp_file}")
    if inverse_use_ants:
        logger.info(f"inverse_use_ants: {inverse_use_ants}")
    logger.info("=" * 70)
    
    results = {'success': 0, 'skipped': 0, 'failed': 0}
    
    # Find transform files (with optional explicit specification)
    transform_files = find_transform_files(
        transform_dir,
        affine_file=affine_file,
        warp_file=warp_file,
        inverse_warp_file=inverse_warp_file,
        inverse_use_ants=inverse_use_ants
    )
    
    if transform_files.transform_type == 'unknown':
        logger.error("Could not identify transforms!")
        results['failed'] += 1
        return results
    
    # Validate transform mode
    is_valid, error_msg = transform_files.validate_mode(transform_mode)
    if not is_valid:
        logger.error(f"Cannot use transform mode '{transform_mode.value}': {error_msg}")
        results['failed'] += 1
        return results
    
    # Determine interpolation
    img_name = os.path.basename(image_path).lower()
    interpolation = "NearestNeighbor" if any(x in img_name for x in ['mask', 'seg', 'label']) else "Linear"
    
    # Determine output path
    if not output_path:
        if output_dir and patient_id:
            out_dir = build_output_path_direct(output_dir, patient_id)
            os.makedirs(out_dir, exist_ok=True)
            base_name = os.path.basename(image_path)
            output_path = os.path.join(out_dir, base_name.replace('.nii.gz', '_transformed.nii.gz'))
        else:
            logger.error("No output path specified!")
            results['failed'] += 1
            return results
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Apply transform
    if apply_transform_to_image(image_path, output_path, transform_files, 
                               reference_path, transform_mode, ants_path, 
                               dimensions, interpolation):
        results['success'] += 1
    else:
        results['failed'] += 1
    
    logger.info("=" * 70)
    logger.info(f"PROCESSING COMPLETE: {results['success']} success, {results['failed']} failed")
    logger.info("=" * 70)
    
    return results


def process_tree_mode(patient_dir, img_folder, transform_folder, reference_identifier,
                     transform_mode=TransformMode.FORWARD,
                     affine_file=None, warp_file=None, inverse_warp_file=None,
                     inverse_use_ants=False,
                     ants_path=None, output_dir=None, subfolder=None, timepoint=None,
                     include_images=None, exclude_images=None, dimensions="3"):
    """
    Process images from a tree directory structure.
    
    Output structure: {output_dir}/{patient}/{timepoint}/{reg_folder}/{img_folder}/(subfolder)/
    
    Args:
        patient_dir: Patient directory
        img_folder: Folder containing images
        transform_folder: Registration folder name
        reference_identifier: String to identify reference image
        transform_mode: TransformMode enum for which transforms to apply
        affine_file: Optional explicit affine filename
        warp_file: Optional explicit warp filename
        inverse_warp_file: Optional explicit inverse warp filename
        inverse_use_ants: If True, use forward warp with ANTs inversion flag instead of inverse warp file
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
    logger.info(f"Transform mode: {transform_mode.value}")
    if any([affine_file, warp_file, inverse_warp_file]):
        logger.info(f"Explicit files: affine={affine_file}, warp={warp_file}, inv_warp={inverse_warp_file}")
    if inverse_use_ants:
        logger.info(f"inverse_use_ants: {inverse_use_ants}")
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
    
    # Find transforms in tree structure (with optional explicit specification)
    transform_dir, transform_files, actual_reg_folder, tp_used = find_transform_in_tree(
        patient_dir, transform_folder, timepoint,
        affine_file=affine_file,
        warp_file=warp_file,
        inverse_warp_file=inverse_warp_file,
        inverse_use_ants=inverse_use_ants
    )
    
    if transform_files is None:
        logger.error("No transform files found!")
        return results
    
    # Validate transform mode
    is_valid, error_msg = transform_files.validate_mode(transform_mode)
    if not is_valid:
        logger.error(f"Cannot use transform mode '{transform_mode.value}': {error_msg}")
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
            
            # Create output filename with mode suffix for clarity
            mode_suffix = f"_{transform_mode.value}" if transform_mode != TransformMode.FORWARD else ""
            output_name = img.replace('.nii.gz', f'{mode_suffix}_transformed.nii.gz')
            if '.nii.gz' not in output_name and '.nii' in output_name:
                output_name = output_name.replace('.nii', f'{mode_suffix}_transformed.nii.gz')
            
            output_path = os.path.join(final_output_dir, output_name)
            
            # Apply transform
            if apply_transform_to_image(img_path, output_path, transform_files, 
                                       reference_path, transform_mode, ants_path, 
                                       dimensions, interpolation):
                results['success'] += 1
            else:
                results['failed'] += 1
    
    logger.info("=" * 70)
    logger.info(f"PROCESSING COMPLETE: {results['success']} success, {results['skipped']} skipped, {results['failed']} failed")
    logger.info("=" * 70)
    
    return results


def process_vent_mode(patient_dir, ventilation_patient_dir, reg_folder, vent_reg_folder, 
                     reference_identifier, transform_mode=TransformMode.FORWARD,
                     affine_file=None, warp_file=None, inverse_warp_file=None,
                     inverse_use_ants=False,
                     ants_path=None, output_dir=None, timepoint=None,
                     vent_dirs=None, vent_strings=None, vent_filters=None,
                     include_images=None, exclude_images=None, dimensions="3"):
    """
    Process ventilation images within registration folder structure.
    
    Output structure: {output_dir}/{patient}/{timepoint}/{reg_folder}/{vent_folder}/
    
    Args:
        patient_dir: Patient directory
        ventilation_patient_dir: Ventilation patient directory (if different)
        reg_folder: Registration folder name
        vent_reg_folder: Ventilation registration folder
        reference_identifier: String to identify reference image
        transform_mode: TransformMode enum for which transforms to apply
        affine_file: Optional explicit affine filename
        warp_file: Optional explicit warp filename
        inverse_warp_file: Optional explicit inverse warp filename
        inverse_use_ants: If True, use forward warp with ANTs inversion flag instead of inverse warp file
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
    logger.info(f"Transform mode: {transform_mode.value}")
    logger.info(f"Vent directories: {vent_dirs}")
    if any([affine_file, warp_file, inverse_warp_file]):
        logger.info(f"Explicit files: affine={affine_file}, warp={warp_file}, inv_warp={inverse_warp_file}")
    if inverse_use_ants:
        logger.info(f"inverse_use_ants: {inverse_use_ants}")
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
    if not ventilation_patient_dir:
        logger.info("No ventilation image folder specified - using default patient folder...")
        if not vent_reg_folder:
            logger.info("No ventilation-registration folder specified - using default registration folder...")
            image_df = load_vent_structure(patient_dir, reg_folder, timepoint, 
                                       vent_dirs, vent_strings, vent_filters)
        else:
            image_df = load_vent_structure(patient_dir, vent_reg_folder, timepoint, 
                                       vent_dirs, vent_strings, vent_filters)
    else: 
        image_df = load_vent_structure(ventilation_patient_dir, vent_reg_folder, timepoint, 
                                   vent_dirs, vent_strings, vent_filters)
    
    if image_df.empty:
        logger.error("No ventilation images found!")
        return results
    
    # Find transforms - they should be in the original patient folder! (differs to vent folder)
    # With optional explicit specification
    transform_dir, transform_files, actual_reg_folder, tp_used = find_transform_in_tree(
        patient_dir, reg_folder, timepoint,
        affine_file=affine_file,
        warp_file=warp_file,
        inverse_warp_file=inverse_warp_file,
        inverse_use_ants=inverse_use_ants
    )
    
    if transform_files is None:
        logger.error("No transform files found!")
        return results
    
    # Validate transform mode
    is_valid, error_msg = transform_files.validate_mode(transform_mode)
    if not is_valid:
        logger.error(f"Cannot use transform mode '{transform_mode.value}': {error_msg}")
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
        
        # Determine output directory
        if output_dir:
            final_output_dir = build_output_path_vent(
                output_dir, patient_name, row_timepoint, actual_reg_folder, vent_type
            )
        else:
            # Save in same ventilation folder
            final_output_dir = img_dir
        
        os.makedirs(final_output_dir, exist_ok=True)
        logger.info(f"Output directory: {final_output_dir}")
        
        for img in filtered_imgs:
            if img == "None":
                results['skipped'] += 1
                continue
            
            img_path = os.path.join(img_dir, img)
            
            # Determine interpolation (ventilation images typically use linear)
            interpolation = "NearestNeighbor" if any(x in img.lower() for x in ['mask', 'seg', 'label']) else "Linear"
            
            # Create output filename with mode suffix for clarity
            mode_suffix = f"_{transform_mode.value}" if transform_mode != TransformMode.FORWARD else ""
            output_name = img.replace('.nii.gz', f'{mode_suffix}_transformed.nii.gz')
            if '.nii.gz' not in output_name and '.nii' in output_name:
                output_name = output_name.replace('.nii', f'{mode_suffix}_transformed.nii.gz')
            
            output_path = os.path.join(final_output_dir, output_name)
            
            # Apply transform
            if apply_transform_to_image(img_path, output_path, transform_files, 
                                       reference_path, transform_mode, ants_path, 
                                       dimensions, interpolation):
                results['success'] += 1
            else:
                results['failed'] += 1
    
    logger.info("=" * 70)
    logger.info(f"PROCESSING COMPLETE: {results['success']} success, {results['skipped']} skipped, {results['failed']} failed")
    logger.info("=" * 70)
    
    return results


#%% Argument Parsing

def create_parser():
    """Create argument parser with all supported options."""
    parser = argparse.ArgumentParser(
        description="Apply transforms to images. Supports DIRECT, TREE, and VENT modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Transform Modes:
  forward          Apply affine + warp (standard, default)
  inverse          Apply affine (inverted) + inverse warp
  warp_only        Apply only the warp transform (no affine)
  inverse_warp_only Apply only the inverse warp transform (no affine)

Examples:
  # Forward transform (default)
  python %(prog)s --mode tree -pat_dir /data/Patient01 -trans_folder Reg__TLC_2__RV ...
  
  # Inverse transform
  python %(prog)s --mode tree -pat_dir /data/Patient01 -trans_folder Reg__TLC_2__RV -transform_mode inverse ...
  
  # Warp only (for composite transforms)
  python %(prog)s --mode tree -pat_dir /data/Patient01 -trans_folder Reg__TLC_2__RV -transform_mode warp_only ...
"""
    )
    
    # Required mode argument
    parser.add_argument('--mode', type=str, required=True,
                       choices=['direct', 'tree', 'vent'],
                       help="Processing mode: direct, tree, or vent")
    
    # NEW: Transform mode argument
    parser.add_argument('-transform_mode', '--transform_mode', type=str, 
                       default='forward',
                       choices=['forward', 'inverse', 'warp_only', 'inverse_warp_only'],
                       help="Transform mode: forward (default), inverse, warp_only, inverse_warp_only")
    
    # NEW: Explicit transform file specification (optional - overrides auto-detection)
    parser.add_argument('-affine_file', '--affine_file', type=str,
                       help="Explicit affine transform filename (relative to transform folder or absolute path)")
    parser.add_argument('-warp_file', '--warp_file', type=str,
                       help="Explicit warp transform filename (relative to transform folder or absolute path)")
    parser.add_argument('-inverse_warp_file', '--inverse_warp_file', type=str,
                       help="Explicit inverse warp transform filename (relative to transform folder or absolute path)")
    parser.add_argument('-inverse_use_ants', '--inverse_use_ants', action='store_true',
                       default=False,
                       help="For inverse mode: use forward warp with ANTs inversion flag instead of pre-generated inverse warp file")
    
    # Common arguments
    parser.add_argument('-pat_dir', '--patient_dir', type=Path,
                       help="Patient directory path")
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
    parser.add_argument('-vent_dir', '--vent_dir', type=Path,
                       help="Patient directory containing ventilation images")
    parser.add_argument('-vent_dirs', '--vent_dirs', type=str, nargs='+',
                       help="[VENT] Ventilation type folders")
    parser.add_argument('-vent_strings', '--vent_strings', type=str, nargs='+',
                       help="[VENT] Vent folder to prefix mapping (format: folder:prefix)")
    parser.add_argument('-vent_filters', '--vent_filters', type=str, nargs='+',
                       help="[VENT] Additional filename filters")
    parser.add_argument('-vent_trans_folder', '--vent_transform_folder', type=str,
                       help="[TREE/VENT] Ventilation images Registration folder name")
    
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


def parse_transform_mode(mode_str: str) -> TransformMode:
    """Parse transform mode string to TransformMode enum."""
    mode_map = {
        'forward': TransformMode.FORWARD,
        'inverse': TransformMode.INVERSE,
        'warp_only': TransformMode.WARP_ONLY,
        'inverse_warp_only': TransformMode.INVERSE_WARP_ONLY,
    }
    return mode_map.get(mode_str.lower(), TransformMode.FORWARD)


#%% Main

def main():
    if manual:
        # Use manual parameters
        logger.info("=" * 50)
        logger.info("Running in MANUAL MODE with hardcoded parameters")
        logger.info("=" * 50)
        
        mode = manual_params['mode']
        transform_mode = parse_transform_mode(manual_params.get('transform_mode', 'forward'))
        
        # Get explicit file specifications (if any)
        affine_file = manual_params.get('affine_file')
        warp_file = manual_params.get('warp_file')
        inverse_warp_file = manual_params.get('inverse_warp_file')
        inverse_use_ants = manual_params.get('inverse_use_ants', False)
        
        if mode == 'direct':
            process_direct_mode(
                image_path=manual_params['direct_image'],
                transform_dir=manual_params['direct_transform_dir'],
                reference_path=manual_params['direct_reference'],
                output_path=manual_params['direct_output'],
                transform_mode=transform_mode,
                affine_file=affine_file,
                warp_file=warp_file,
                inverse_warp_file=inverse_warp_file,
                inverse_use_ants=inverse_use_ants,
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
                transform_mode=transform_mode,
                affine_file=affine_file,
                warp_file=warp_file,
                inverse_warp_file=inverse_warp_file,
                inverse_use_ants=inverse_use_ants,
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
                ventilation_patient_dir=manual_params.get('vent_dir'),
                reg_folder=manual_params['transform_folder'],
                vent_reg_folder=manual_params.get('vent_transform_folder'),
                reference_identifier=manual_params['reference_identifier'],
                transform_mode=transform_mode,
                affine_file=affine_file,
                warp_file=warp_file,
                inverse_warp_file=inverse_warp_file,
                inverse_use_ants=inverse_use_ants,
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
        transform_mode = parse_transform_mode(args.transform_mode)
        
        # Get explicit file specifications (if any)
        affine_file = args.affine_file
        warp_file = args.warp_file
        inverse_warp_file = args.inverse_warp_file
        inverse_use_ants = args.inverse_use_ants
        
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
                transform_mode=transform_mode,
                affine_file=affine_file,
                warp_file=warp_file,
                inverse_warp_file=inverse_warp_file,
                inverse_use_ants=inverse_use_ants,
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
                transform_mode=transform_mode,
                affine_file=affine_file,
                warp_file=warp_file,
                inverse_warp_file=inverse_warp_file,
                inverse_use_ants=inverse_use_ants,
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
                ventilation_patient_dir=str(args.vent_dir) if args.vent_dir else None,
                reg_folder=args.transform_folder,
                vent_reg_folder=args.vent_transform_folder,
                reference_identifier=args.reference,
                transform_mode=transform_mode,
                affine_file=affine_file,
                warp_file=warp_file,
                inverse_warp_file=inverse_warp_file,
                inverse_use_ants=inverse_use_ants,
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

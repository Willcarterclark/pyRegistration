# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 11:26:59 2025

@author: will Clark

FIXED VERSION:
- Registration uses EXPANDED masks
- Transform application uses ORIGINAL (non-masked) images
- Output masking uses ORIGINAL (non-expanded) masks
- Added file type specification for outputs
"""
#%%Packages

import numpy as np
import os
import SimpleITK as sitk
import pandas as pd
import argparse
import pathlib
import glob
import ants
import subprocess
import shlex
import csv
import ast

from pathlib import Path
from typing import Optional, Tuple
from scipy import stats
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import sys

"""
PYTHON NESTED DIRECTORY REGISTRATION SCRIPT
--
by Will Clark - wclark2@sheffield.ac.uk
POLARIS - University of Sheffield
--
Running this script requires the paired BASH .sh script, although the script can be run manually
--
To run manually:
    Run the sections:
        >"Packages"
        DO NOT RUN "BASH Input"

        Uncomment the section "Manual input", and edit the strings in the section, then run.

        >"Load Directory"
        >"Run script"
--
Registration parameters can be edited: in the run_reg() function
    Search for: <Reg Params>
    Currently set up for General MRI alignment - very good on 1H MRI
"""

#%% BASH Input

# Initialize argument parser
parser = argparse.ArgumentParser(description="Script for image registration using ANTs")

# Define the command-line arguments
parser.add_argument('-pat_dir', '--dir_pat', help="Directory of a single patient", type=Path, required=True)
parser.add_argument('-scn_dir', '--scan_folder', help="Folder to find within patient directory", type=str, required=True)
parser.add_argument('-seg_dir', '--seg_folder', help="Segmentation folder", type=str)
parser.add_argument('-ants_path', '--ants_dir', type=Path, required=True)
parser.add_argument('-f', '--img_fix', help="Fixed image identifier", type=str, required=True)
parser.add_argument('-m', '--img_mov', help="Moving image identifier", type=str, required=True)
parser.add_argument('-f_mask', '--img_fix_mask', help="Mask Fixed image identifier", type=str)
parser.add_argument('-m_mask', '--img_mov_mask', help="Mask Moving image identifier", type=str)
parser.add_argument('-sub_dir', '--sub_folder',type=str)
parser.add_argument('-dim','--dimensions', help="Registration Dimensions", type=str)
parser.add_argument('-ants_reg_params', '--ants_registration_parameters', help="ANTs registration command parameters", type=str, required=True)
parser.add_argument('-out_dir', '--output_directory', help="Optional output directory for registration results", type=Path)
parser.add_argument('-reg_exp_mask', '--reg_expand_mask', help="Expand Registration Mask - use 0 for none, integer 1-10 for voxel size filter, default is 8", type=str)
parser.add_argument('-out_type', '--output_filetype', help="Output file type (e.g., '.nii.gz', '.mha', '.nii'). Default is '.nii.gz'", type=str, default='.nii.gz')
parser.add_argument('-saveinputs', '--save_input_copies', 
                    help="Save copies of input images and masks to registration directory", 
                    action='store_true', default=False)

# Parse arguments
args = parser.parse_args()

# Extract and convert arguments
path_patient_dir = str(args.dir_pat)
scan_folder = str(args.scan_folder)
seg_folder = str(args.seg_folder) if args.seg_folder else None
path_ants = str(args.ants_dir)
img_fix = str(args.img_fix)
img_fix_mask = str(args.img_fix_mask) if args.img_fix_mask else None
img_mov_mask = str(args.img_mov_mask) if args.img_mov_mask else None
img_mov = str(args.img_mov)
sub_folder = str(args.sub_folder) if args.sub_folder else None
dimensions = str(args.dimensions) if args.dimensions else "3"
ants_reg_params_str = str(args.ants_registration_parameters)
output_directory = str(args.output_directory) if args.output_directory else None
reg_expand_mask = int(args.reg_expand_mask) if args.reg_expand_mask else 8
output_filetype = str(args.output_filetype) if args.output_filetype else '.nii.gz'
save_input_copies = args.save_input_copies

# Ensure output filetype starts with a dot
if not output_filetype.startswith('.'):
    output_filetype = '.' + output_filetype

# Output the parsed paths and arguments for verification
print(f"Patient Directory Path: {path_patient_dir}")
print(f"Scan Folder: {scan_folder}")
print(f"Segmentation Folder: {seg_folder}")
print(f"Path to ANTs binaries: {path_ants}")
print(f"Fixed Image Identifier: {img_fix}")
print(f"Moving Image Identifier: {img_mov}")
print(f"Registration Dimensions: {dimensions}")
print(f"ANTs Registration Parameters String: {ants_reg_params_str}")
print(f"Output Directory: {output_directory if output_directory else 'In-folder (default)'}")
print(f"Registration Mask Expansion: {reg_expand_mask}")
print(f"Output File Type: {output_filetype}")

#%% Load Directory functions

# Try to import the helper function, fall back to local if it fails
try:
    sys.path.append(r'\shared\polaris2\shared\will\_scripts')
    from Organisation.Directory_Functions import load_dir, load_img_dir
    print("Successfully imported helper functions from shared directory")

except (ImportError, ModuleNotFoundError) as e:
    print(f"Could not import helper functions: {e}")
    print("Falling back to local implementation")

    def load_img_dir(load_dir):
        """Load image directory"""
        img_list = os.listdir(load_dir)
        print(f"Found images: {img_list}")
        return img_list

    def load_dir(source_dir, folder, subfolder=False, timepoint=False, filenames=False, List_all=False):
        """
        Load images from nested patient directory. Operates on a single patient.

        Structure should be: Data_Folder / Patient IDs / Scan session / folder.

        Args:
            source_dir: Base directory where patient folders are located.
            folder: Folder name to load from each scan session.
            subfolder: Subfolder within the specified folder (optional).
            timepoint: Specific time points to load (default: loads all).
            filenames: Specific filename patterns to search for.
            List_all: If True, includes directories in the output; otherwise, only lists images.

        Returns:
            Table_out: DataFrame containing image filenames and their directory paths.
        """
        Table_out = pd.DataFrame()

        print("Loading Directory Tree...")
        print(f"Directory to search: {source_dir}")
        print(f"Specified Timepoint(s): {timepoint}")
        print(f"Specified Folder(s): {folder}")
        print(f"Specified Subfolder(s): {subfolder}")
        print(f"Specified filename(s): {filenames}")

        def load_folder(fdir, dirs_to_find=False):
            """Load single folder's contents"""
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
                    print(f"Error: Directory '{fdir}' does not exist.")
                    return np.array([])
            else:
                if not os.path.exists(fdir):
                    print(f"Error: Directory '{fdir}' does not exist.")
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
                    print(f"Warning: Subdirectory '{directory}' not found in '{fdir}'")

            if single_dir_requested and len(existing_dirs) == 0:
                return None

            return np.array(existing_dirs)

        def load_folder_imgs(fdir, files_to_find=False):
            """Find image files in a directory matching specified substring patterns"""
            if not os.path.exists(fdir):
                print(f"Error: Directory '{fdir}' does not exist.")
                return np.array([])

            single_pattern_requested = False

            print("Loading files in directory...")
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
                    pattern_matches = [os.path.basename(f) for f in pattern_matches]
                    matching_files.extend(pattern_matches)

                matching_files = sorted(list(set(matching_files)))
            else:
                print("No specified files, listing all...")
                matching_files = sorted(glob.glob(os.path.join(fdir, "*")))
                matching_files = [os.path.basename(f) for f in matching_files if os.path.isfile(f)]

            if single_pattern_requested and len(matching_files) == 0:
                print("No matching files...")
                return None

            print(f"Files found: {matching_files}")
            return np.array(matching_files)

        # Run script
        patient_path = os.path.join(source_dir)
        print("--")

        print("Loading Patient Time Point(s)...")
        timepoints = load_folder(fdir=patient_path, dirs_to_find=timepoint)

        for timepoint in timepoints:
            print(f"Time point - {timepoint}")
            time_point_path = os.path.join(patient_path, timepoint)

            print("Loading specified Folder(s)...")
            Folders = load_folder(fdir=time_point_path, dirs_to_find=folder)

            if Folders is None:
                print("No folders located")
                return None

            for Folder in Folders:
                print(f"Folder - {Folder}")
                folder_path = os.path.join(time_point_path, Folder)

                if subfolder:
                    print("Loading specified subfolder(s)...")
                    subfolders = load_folder(fdir=folder_path, dirs_to_find=subfolder)

                    if subfolders is None:
                        print("No subfolders located")
                        return None

                    for subfolder in subfolders:
                        print(f"Sub-Folder - {subfolder}")
                        DataFolder = os.path.join(folder_path, subfolder)
                        imgs = load_folder_imgs(fdir=DataFolder, files_to_find=filenames)

                        if imgs is None or (isinstance(imgs, np.ndarray) and len(imgs) == 0):
                            print(f"{folder} Directory is empty!")
                            imgs = np.array(["None"])

                        data_to_bind = {
                            'Images': [imgs],
                            'Directory': [DataFolder]
                        }
                        output_scans = pd.DataFrame(data_to_bind)
                else:
                    DataFolder = folder_path
                    imgs = load_folder_imgs(fdir=DataFolder, files_to_find=filenames)

                    if imgs is None or (isinstance(imgs, np.ndarray) and len(imgs) == 0):
                        print(f"{folder} Directory is empty!")
                        imgs = np.array(["None"])

                    data_to_bind = {
                        'Images': [imgs],
                        'Directory': [DataFolder]
                    }
                    output_scans = pd.DataFrame(data_to_bind)

                Table_out = pd.concat([Table_out, output_scans], ignore_index=True)

        return Table_out

#%% Load directories

print("-----------")
print("Loading Directory of Images...")
List_Scans = load_dir(source_dir=path_patient_dir, folder=scan_folder, subfolder=sub_folder)

print("-----------")
print("Loading Directory of Segmentations...")

# Initialize List_Segs to None to avoid NameError
List_Segs = None

if seg_folder:
    List_Segs = load_dir(source_dir=path_patient_dir, folder=seg_folder, subfolder=sub_folder)
else:
    print("Segmentations not specified... Skipping...")

print("--")

#%% Utility functions

def ants_2_itk(image):
    """Convert ANTs image to SimpleITK image"""
    imageITK = sitk.GetImageFromArray(image.numpy().T)
    imageITK.SetOrigin(image.origin)
    imageITK.SetSpacing(image.spacing)
    imageITK.SetDirection(image.direction.reshape(9))
    return imageITK

def itk_2_ants(image):
    """Convert SimpleITK image to ANTs image"""
    image_ants = ants.from_numpy(sitk.GetArrayFromImage(image).T,
                                 origin=image.GetOrigin(),
                                 spacing=image.GetSpacing(),
                                 direction=np.array(image.GetDirection()).reshape(3, 3))
    return image_ants

def apply_mask(image, mask):
    """Apply mask to image"""
    if image.GetPixelID() == sitk.sitkFloat64:
        print("Casting image from 64-bit float to 32-bit float for masking...")
        image = sitk.Cast(image, sitk.sitkFloat32)

    if mask.GetPixelID() != sitk.sitkUInt8:
        print("Casting mask to 8-bit unsigned integer...")
        mask = sitk.Cast(mask, sitk.sitkUInt8)

    if image.GetOrigin() != mask.GetOrigin() or image.GetSpacing() != mask.GetSpacing() or image.GetDirection() != mask.GetDirection():
        print("Resampling mask to match the image...")
        mask = resample_image(mask, image)

    return sitk.Mask(image, mask)

def resample_image(image, reference_image):
    """Resample 'image' to match the physical space of 'reference_image'."""
    print("Resampling image to match the reference image...")
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(sitk.Transform())
    resampled_image = resampler.Execute(image)
    return resampled_image

def load_sitk(file_path):
    """Load SimpleITK image"""
    print(f"Loading sitk file from: {file_path}")
    return sitk.ReadImage(file_path)

def save_sitk(image, output_path):
    """Save SimpleITK image"""
    print(f"Saving sitk file to: {output_path}")
    return sitk.WriteImage(image, str(output_path))

def find_match(array, search_string):
    """Find string in array containing search_string"""
    for string in array:
        if search_string in string:
            return string
    return None

def expand_mask_radially(mask_image, radius=8):
    """
    Expand a binary mask radially by specified number of voxels.
    Ensures mask is in UInt8 format before dilation.
    """
    if mask_image.GetPixelID() != sitk.sitkUInt8:
        print(f"Converting mask from {mask_image.GetPixelIDTypeAsString()} to UInt8 for dilation...")
        mask_image = sitk.Cast(mask_image > 0, sitk.sitkUInt8)  # binarize + cast
    
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelRadius(radius)
    dilate_filter.SetKernelType(sitk.sitkBall)
    expanded_mask = dilate_filter.Execute(mask_image)
    return expanded_mask

def calculate_registration_accuracy(fixed_img_path: str,
                                   warped_moving_path: str,
                                   fixed_mask_path: Optional[str] = None,
                                   warped_moving_mask_path: Optional[str] = None,
                                   output_path: str = None) -> dict:
    """
    Calculate comprehensive registration accuracy metrics between fixed and warped moving images.
    
    Args:
        fixed_img_path: Path to the fixed image
        warped_moving_path: Path to the warped moving image
        fixed_mask_path: Optional path to the fixed mask
        warped_moving_mask_path: Optional path to the warped moving mask
        output_path: Path to save the metrics CSV file
    
    Returns:
        Dictionary containing all calculated metrics
    """
    # Load images
    fixed_img = sitk.ReadImage(fixed_img_path)
    warped_moving = sitk.ReadImage(warped_moving_path)
    
    # Convert to numpy arrays for calculations
    fixed_array = sitk.GetArrayFromImage(fixed_img).astype('float')
    warped_array = sitk.GetArrayFromImage(warped_moving).astype('float')
    
    # Initialize metrics dictionary
    metrics = {}
    
    # If masks are provided, calculate mask-based metrics
    if fixed_mask_path and warped_moving_mask_path:
        fixed_mask = sitk.ReadImage(fixed_mask_path)
        warped_moving_mask = sitk.ReadImage(warped_moving_mask_path)
        
        original_mask_array = sitk.GetArrayFromImage(fixed_mask)
        warped_mask_array = sitk.GetArrayFromImage(warped_moving_mask)
        
        # Calculate overlap metrics using masks
        fixed_binary = (original_mask_array > 0).astype(np.uint8)
        warped_binary = (warped_mask_array > 0).astype(np.uint8)
        
        # Calculate intersection and union for overlap metrics
        intersection = np.logical_and(fixed_binary, warped_binary)
        union = np.logical_or(fixed_binary, warped_binary)
        
        # Calculate Jaccard index
        if np.sum(union) > 0:
            jaccard = np.sum(intersection) / np.sum(union)
        else:
            jaccard = 0.0
        
        # Calculate Dice score
        if (np.sum(fixed_binary) + np.sum(warped_binary)) > 0:
            dice = 2 * np.sum(intersection) / (np.sum(fixed_binary) + np.sum(warped_binary))
        else:
            dice = 0.0
        
        metrics['Jaccard'] = jaccard
        metrics['Dice'] = dice
        
        print("Computing additional quality measures...")
        # Create masked versions for correlation calculations
        fixed_array_masked = np.multiply(original_mask_array, fixed_array)
        warped_array_masked = np.multiply(original_mask_array, warped_array)
        
        # Get non-zero values using the original mask
        mask_bool = original_mask_array > 0
        fixed_masked = fixed_array[mask_bool]
        warped_masked = warped_array[mask_bool]
    else:
        # No masks provided - use whole images
        print("No masks provided, computing metrics on whole images...")
        metrics['Jaccard'] = np.nan
        metrics['Dice'] = np.nan
        
        fixed_array_masked = fixed_array
        warped_array_masked = warped_array
        fixed_masked = fixed_array.flatten()
        warped_masked = warped_array.flatten()
    
    # Calculate additional quality measures
    print("Calculating SSIM...")
    metrics["SSIM"] = ssim(fixed_array_masked, warped_array_masked,
                          data_range=warped_array_masked.max() - warped_array_masked.min())
    
    print("Calculating PSNR...")
    metrics["PSNR"] = np.abs(psnr(fixed_array_masked, warped_array_masked,
                                 data_range=warped_array_masked.max() - warped_array_masked.min()))
    
    if len(fixed_masked) > 0 and len(warped_masked) > 0:
        print("Calculating Pearson correlation...")
        metrics["Pearson"] = pearsonr(fixed_masked, warped_masked)[0]
        
        print("Calculating Spearman correlation...")
        metrics["Spearman"] = spearmanr(fixed_masked, warped_masked, nan_policy='omit')[0]
    else:
        metrics["Pearson"] = np.nan
        metrics["Spearman"] = np.nan
    
    print("Calculating error metrics...")
    metrics["MSE"] = np.nanmean(np.square(fixed_array_masked - warped_array_masked))
    metrics["RMSE"] = np.sqrt(metrics["MSE"])
    metrics["MAE"] = np.nanmean(np.abs(fixed_array_masked - warped_array_masked))
    
    # Save to CSV if output path is provided
    if output_path:
        metrics_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': list(metrics.values()),
            'Fixed_Image': [fixed_img_path] * len(metrics),
            'Warped_Moving_Image': [warped_moving_path] * len(metrics)
        })
        metrics_df.to_csv(output_path, index=False)
        print(f"Registration accuracy metrics saved to CSV: {output_path}")
    
    # Print metrics to console
    print("\nRegistration accuracy metrics:")
    for metric, value in metrics.items():
        if not np.isnan(value):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: N/A (no masks provided)")
    
    return metrics

#save cmd
def format_command_string(cmd_string):
    """
    Convert Python list string to command format for .txt files.
    
    Removes outer brackets, single quotes, and commas between parameters.
    Preserves commas inside parameter brackets like MI[a,b,c,d].
    
    Args:
        cmd_string (str): String representation of command list
    
    Returns:
        str: Formatted command string
    """
    cmd_list = ast.literal_eval(cmd_string.strip())
    return ' '.join(cmd_list)

#%% Prepare Registration

def prep_reg(Input_imgs, Fixed_img, Moving_img, Input_segs=None, reg_mask=8, 
             ants_reg_params_str=None, output_base_dir=None, Fixed_mask=None, 
             Moving_mask=None, dimensions="3", output_filetype='.nii.gz'):
    
    
    global fix_img_path
    global mov_img_path
    global fix_seg_path
    global mov_seg_path
    
    """
    Prepare and run registrations on input images.
    
    Args:
        Input_imgs: DataFrame with images and directories
        Fixed_img: Fixed image identifier
        Moving_img: Moving image identifier
        Input_segs: Optional DataFrame with segmentations
        reg_mask: Radius for mask expansion (for registration constraint only)
        ants_reg_params_str: ANTs registration parameters
        output_base_dir: Optional output directory
        Fixed_mask: Optional fixed mask identifier
        Moving_mask: Optional moving mask identifier
        dimensions: Registration dimensions (default "3")
        output_filetype: Output file extension (default '.nii.gz')
    """
    print("--")
    print("Preparing registrations...")
    print("--")
    
    # Clean input DataFrames
    df1 = Input_imgs[~Input_imgs['Images'].astype(str).str.contains('None', case=False)]
    
    # Check if Input_segs is a DataFrame with data
    has_segs = Input_segs is not None and isinstance(Input_segs, pd.DataFrame) and not Input_segs.empty
    
    if has_segs:
        df2 = Input_segs[~Input_segs['Images'].astype(str).str.contains('None', case=False)]
        print("Segmentations provided, will use for masking")
    else:
        print("No segmentations specified... Running without masks")
    
    # Iterate over rows
    for index1, row1 in df1.iterrows():
        imgs = row1['Images']
        img_dir = row1['Directory']
        
        # Get segmentation data if available
        if has_segs and index1 < len(df2):
            row2 = df2.iloc[index1]
            segs = row2['Images']
            seg_dir = row2['Directory']
        else:
            segs = None
            seg_dir = None
        
        # Find the fixed and moving images
        fix_img = find_match(imgs, Fixed_img)
        mov_img = find_match(imgs, Moving_img)
        
        # Find the fixed and moving segmentations if available
        if has_segs and segs is not None:
            # Fixed Mask
            if Fixed_mask:
                fix_seg = find_match(segs, Fixed_mask)
            else:
                fix_seg = find_match(segs, Fixed_img)
            
            # Moving Mask
            if Moving_mask:
                mov_seg = find_match(segs, Moving_mask)
            else:
                mov_seg = find_match(segs, Moving_img)
        else:
            fix_seg = None
            mov_seg = None
        
        print(f"Found fixed image: {fix_img}")
        print(f"Found moving image: {mov_img}")
        
        if has_segs and segs is not None:
            print(f"Found fixed segmentation: {fix_seg}")
            print(f"Found moving segmentation: {mov_seg}")
        print("--")
        
        # Check if required images were found
        if not fix_img or not mov_img:
            print(f"Skipping row {index1} due to missing images.")
            print("ERROR: Please check image filenames or if image files exist.")
            continue
        
        if has_segs and segs is not None and (not fix_seg or not mov_seg):
            print(f"Skipping row {index1} due to missing segmentations.")
            print("ERROR: Please check segmentation filenames or if segmentation files exist.")
            continue
        
        # Load the image paths (ORIGINAL, NON-MASKED images)
        fix_img_path = os.path.join(img_dir, fix_img)
        mov_img_path = os.path.join(img_dir, mov_img)
        
        # Load seg paths if available
        if has_segs and fix_seg and mov_seg:
            fix_seg_path = os.path.join(seg_dir, fix_seg)
            mov_seg_path = os.path.join(seg_dir, mov_seg)
        else:
            fix_seg_path = None
            mov_seg_path = None
        
        # Print paths
        print(f"Fixed Image Path: {fix_img_path}")
        print(f"Moving Image Path: {mov_img_path}")
        if fix_seg_path:
            print(f"Fixed Seg Path: {fix_seg_path}")
        if mov_seg_path:
            print(f"Moving Seg Path: {mov_seg_path}")
        
        # Load images and masks using sitk
        fix_img_sitk = load_sitk(fix_img_path)
        mov_img_sitk = load_sitk(mov_img_path)
        
        # Load original masks (non-expanded) if available
        fix_seg_sitk_original = None
        mov_seg_sitk_original = None
        fix_seg_sitk_expanded = None
        mov_seg_sitk_expanded = None
        
        if has_segs and fix_seg_path and mov_seg_path:
            fix_seg_sitk_original = load_sitk(fix_seg_path)
            mov_seg_sitk_original = load_sitk(mov_seg_path)
            
            # Create expanded masks for registration constraint
            if reg_mask > 0:
                print(f"Creating expanded registration masks (radius={reg_mask} voxels)...")
                fix_seg_sitk_expanded = expand_mask_radially(mask_image=fix_seg_sitk_original, radius=reg_mask)
                mov_seg_sitk_expanded = expand_mask_radially(mask_image=mov_seg_sitk_original, radius=reg_mask)
                print("Expanded masks created for registration constraint")
            else:
                print("No mask expansion (reg_mask=0)")
                fix_seg_sitk_expanded = fix_seg_sitk_original
                mov_seg_sitk_expanded = mov_seg_sitk_original
        
        print("Images loaded!")
        print("--")
        
        # Save Registration Dir
        reg_str = f"Reg_{Moving_img}_2_{Fixed_img}"
        print(f"Registration string: {reg_str}")
        
        # OUTPUT DIRECTORY LOGIC
        if output_base_dir:
            # If output directory is specified, recreate nested structure there
            rel_path = os.path.relpath(img_dir, path_patient_dir)
            new_rel_path = rel_path.replace(scan_folder, reg_str, 1)
            
            if sub_folder and sub_folder in new_rel_path:
                new_rel_path = new_rel_path.replace(os.sep + sub_folder, "", 1)
            
            patient_id = rel_path.split(os.sep)[0]
            Reg_dir = os.path.join(output_base_dir, patient_id, new_rel_path)
        else:
            # Original in-folder behavior
            Reg_dir = img_dir.replace(scan_folder, reg_str, 1)
            
            if sub_folder:
                Reg_dir = os.path.dirname(Reg_dir)
                Reg_dir = os.path.join(Reg_dir, reg_str)
        
        print(f"Directory for registration: {Reg_dir}")
        
        # Create the registration directory if it doesn't exist
        print("Creating Registration Directory...")
        os.makedirs(Reg_dir, exist_ok=True)
        
        
        fix_seg_expanded_path = None
        mov_seg_expanded_path = None
        # Save expanded masks for ANTs registration
        if has_segs and fix_seg_sitk_expanded is not None and mov_seg_sitk_expanded is not None:
            
            #
            fix_seg_expanded_path = os.path.join(Reg_dir, "_fixed_mask_expanded.nii.gz")
            mov_seg_expanded_path = os.path.join(Reg_dir, "_moving_mask_expanded.nii.gz")
            
            save_sitk(fix_seg_sitk_expanded, fix_seg_expanded_path)
            save_sitk(mov_seg_sitk_expanded, mov_seg_expanded_path)
            print("Expanded masks saved for registration")
        
        #Save output images -saveinputs
        
        if save_input_copies:
            print("Saving input copies to registration directory...")
              
            #Save fixed and moving image to registration folder
            fix_img_path_tmp = os.path.join(Reg_dir, "_fixed_image.nii.gz")
            mov_img_path_tmp = os.path.join(Reg_dir, "_moving_image.nii.gz")
            
            save_sitk(fix_img_sitk, fix_img_path_tmp)
            save_sitk(mov_img_sitk, mov_img_path_tmp)
            print("Moving and fixed image saved to ouput directory")
            
            #Save fixed and moving maks to registration folder
            if fix_seg_sitk_original and mov_seg_sitk_original:
                fix_seg_path_tmp = os.path.join(Reg_dir, "_fixed_mask.nii.gz")
                mov_seg_path_tmp = os.path.join(Reg_dir, "_moving_mask.nii.gz")
                
                save_sitk(fix_seg_sitk_original, fix_seg_path_tmp)
                save_sitk(mov_seg_sitk_original, mov_seg_path_tmp)
                print("Moving and fixed masks saved to ouput directory")
        
        # Run Registration
        
        #Run registration just using images and masks
            
        
        print("Running Registration with unmasked images and expanded masks")
        success = run_reg(
            fixed=fix_img_path,  # ORIGINAL image, not masked
            moving=mov_img_path,  # ORIGINAL image, not masked
            fixed_mask_original=fix_seg_path,  # Original mask for output
            moving_mask_original=mov_seg_path,  # Original mask for output
            fixed_mask_expanded=fix_seg_expanded_path,  # Expanded mask for registration
            moving_mask_expanded=mov_seg_expanded_path,  # Expanded mask for registration
            output_dir=Reg_dir,
            output_prefix=reg_str + "_",
            ants_path=path_ants,
            ants_reg_params_template=ants_reg_params_str,
            use_masks=(has_segs and fix_seg_path is not None),
            dimensions=dimensions,
            output_filetype=output_filetype
        )
            
        if success:
            print("Iteration complete...")
            
        else:
            print("Registration failed for this iteration")
        print("--")
    
    return

#%% Run registration

def run_reg(fixed, moving, output_dir, output_prefix, 
            fixed_mask_original=None, moving_mask_original=None,
            fixed_mask_expanded=None, moving_mask_expanded=None,
            ants_path=None, use_masks=False, 
            ants_reg_params_template=None, dimensions="3", output_filetype='.nii.gz'):
    """
    Perform image registration using ANTs with optional masking and output the transforms.
    
    MASK HANDLING:
    - Registration uses EXPANDED masks (fixed_mask_expanded, moving_mask_expanded)
    - Transform application uses ORIGINAL images (fixed, moving - not pre-masked)
    - Output masking uses ORIGINAL masks (fixed_mask_original, moving_mask_original)
    
    Args:
        fixed: Path to ORIGINAL fixed image (not pre-masked)
        moving: Path to ORIGINAL moving image (not pre-masked)
        fixed_mask_original: Path to original (non-expanded) fixed mask
        moving_mask_original: Path to original (non-expanded) moving mask
        fixed_mask_expanded: Path to expanded fixed mask (for registration constraint)
        moving_mask_expanded: Path to expanded moving mask (for registration constraint)
        output_filetype: Output file extension (e.g., '.nii.gz', '.mha')
    
    Returns:
        Tuple or False: Returns tuple of output files if successful, False if failed
    """
    # Set the ANTs registration and apply transform paths
    ants_registration = os.path.join(ants_path, 'antsRegistration') if ants_path else 'antsRegistration'
    ants_apply = os.path.join(ants_path, 'antsApplyTransforms') if ants_path else 'antsApplyTransforms'
    
    # Construct output paths
    output_prefix_full = os.path.join(output_dir, output_prefix)
    warp_file = f"{output_prefix_full}1Warp.nii.gz"
    affine_file = f"{output_prefix_full}0GenericAffine.mat"
    inv_warp_file = f"{output_prefix_full}1InverseWarp.nii.gz"
    ants_command = f"{output_prefix_full}0ANTsCall.txt"
    
    print("=" * 80)
    print("REGISTRATION CONFIGURATION:")
    print(f"  Using ORIGINAL images for transform application: {os.path.basename(fixed)}, {os.path.basename(moving)}")
    if use_masks and fixed_mask_expanded:
        print(f"  Using EXPANDED masks for registration constraint")
    if use_masks and fixed_mask_original:
        print(f"  Using ORIGINAL masks for output warping")
    print(f"  Output file type: {output_filetype}")
    print("=" * 80)
    
    print("Defining ANTs parameters and running registration...")
    
    # Replace placeholders in the ANTs command string
    formatted_ants_reg_params = ants_reg_params_template.replace(
        "{fixed_placeholder}", fixed
    ).replace(
        "{moving_placeholder}", moving
    ).replace(
        "{output_prefix_full_placeholder}", output_prefix_full
    )
    
    # Build registration command
    registration_command = [ants_registration] + shlex.split(formatted_ants_reg_params)
    
    
    # Add the EXPANDED mask argument if provided (for registration constraint)
    if fixed_mask_expanded and moving_mask_expanded and use_masks:
        registration_command.extend(["--masks", f"[{fixed_mask_expanded},{moving_mask_expanded}]"])
        print(f"Using EXPANDED masks for registration: fixed - {fixed_mask_expanded}, moving - {moving_mask_expanded}")
    elif fixed_mask_expanded and use_masks:
        registration_command.extend(["--masks", f"[{fixed_mask_expanded}]"])
        print(f"Using EXPANDED fixed mask for registration: {fixed_mask_expanded}")
    
    #output registration command to .txt file
    ANTsCommand_output = os.path.join(output_dir, ants_command)
    #format raw ants command
    ants_command=format_command_string(str(registration_command))
    #write output
    with open(ANTsCommand_output, "w") as f:
        f.write(ants_command)
    #
    print(f"Written ANTs command to {ANTsCommand_output}")
    
    # Run the registration command
    print(f"Running ANTs registration...")
    try:
        subprocess.run(registration_command, check=True)
        print("Registration completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Registration failed: {e}")
        return False
    
    # Apply the resulting transformations to the ORIGINAL moving image
    moving_warped_output = os.path.join(output_dir, f"{output_prefix}warped{output_filetype}")
    apply_transform_command = [
        ants_apply,
        "-d", dimensions,
        "-v", "1",
        "-i", moving,  # ORIGINAL moving image
        "-r", fixed,  # ORIGINAL fixed image as reference
        "-n", "linear",
        "-t", warp_file,
        "-t", affine_file,
        "-o", moving_warped_output
    ]
    
    try:
        subprocess.run(apply_transform_command, check=True)
        print(f"Applied transform to ORIGINAL moving image. Output: {moving_warped_output}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to apply transform to moving image: {e}")
    
    # Apply inverse transforms to the ORIGINAL fixed image
    fixed_warped_output = os.path.join(output_dir, f"{output_prefix}inv_warped{output_filetype}")
    inverse_transform_command = [
        ants_apply,
        "-d", dimensions,
        "-v", "1",
        "-i", fixed,  # ORIGINAL fixed image
        "-r", moving,  # ORIGINAL moving image as reference
        "-n", "linear",
        "-t", f"[{affine_file},1]",
        "-t", inv_warp_file,
        "-o", fixed_warped_output
    ]
    
    try:
        subprocess.run(inverse_transform_command, check=True)
        print(f"Applied inverse transform to ORIGINAL fixed image. Output: {fixed_warped_output}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to apply inverse transform to fixed image: {e}")
    
    # Apply transforms to ORIGINAL masks (non-expanded) if they exist
    moving_mask_warped_output = None
    fixed_mask_warped_output = None
    
    if moving_mask_original:
        moving_mask_warped_output = os.path.join(output_dir, f"{output_prefix}mask_warped{output_filetype}")
        apply_transform_mask_command = [
            ants_apply,
            "-d", dimensions,
            "-v", "1",
            "-i", moving_mask_original,  # ORIGINAL (non-expanded) mask
            "-r", fixed,
            "-n", "NearestNeighbor",
            "-t", warp_file,
            "-t", affine_file,
            "-o", moving_mask_warped_output
        ]
        
        try:
            subprocess.run(apply_transform_mask_command, check=True)
            print(f"Applied transform to ORIGINAL moving mask. Output: {moving_mask_warped_output}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to apply transform to moving mask: {e}")
            moving_mask_warped_output = None
    
    if fixed_mask_original:
        fixed_mask_warped_output = os.path.join(output_dir, f"{output_prefix}mask_inv_warped{output_filetype}")
        inverse_transform_mask_command = [
            ants_apply,
            "-d", dimensions,
            "-v", "1",
            "-i", fixed_mask_original,  # ORIGINAL (non-expanded) mask
            "-r", moving,
            "-n", "NearestNeighbor",
            "-t", inv_warp_file,
            "-t", f"[{affine_file},1]",
            "-o", fixed_mask_warped_output
        ]
        
        try:
            subprocess.run(inverse_transform_mask_command, check=True)
            print(f"Applied inverse transform to ORIGINAL fixed mask. Output: {fixed_mask_warped_output}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to apply inverse transform to fixed mask: {e}")
            fixed_mask_warped_output = None
    
    print("All transformations applied successfully.")
    
    # Calculate registration accuracy metrics if masks are available
    if fixed_mask_original and moving_mask_original and moving_mask_warped_output:
        print("Computing Registration quality measures...")
        metrics_output = os.path.join(output_dir, f"{output_prefix}0_reg_accuracy.csv")
        
        try:
            metrics = calculate_registration_accuracy(
                fixed_img_path=fixed,
                warped_moving_path=moving_warped_output,
                fixed_mask_path=fixed_mask_original,
                warped_moving_mask_path=moving_mask_warped_output,
                output_path=metrics_output
            )
            
            if 'Jaccard' in metrics and not np.isnan(metrics['Jaccard']):
                print(f"Jaccard: {metrics['Jaccard']:.4f}")
            if 'Dice' in metrics and not np.isnan(metrics['Dice']):
                print(f"Dice: {metrics['Dice']:.4f}")
        except Exception as e:
            print(f"Failed to calculate registration metrics: {e}")
    else:
        print("Skipping registration metrics (no masks available)")
    
    return (warp_file, affine_file, moving_warped_output, fixed_warped_output, 
            moving_mask_warped_output, fixed_mask_warped_output)

#%% Run script


DFs = prep_reg(
    Input_imgs=List_Scans,
    Input_segs=List_Segs,
    Fixed_img=img_fix,
    Moving_img=img_mov,
    ants_reg_params_str=ants_reg_params_str,
    output_base_dir=output_directory,
    Fixed_mask=img_fix_mask,
    Moving_mask=img_mov_mask,
    reg_mask=reg_expand_mask,
    dimensions=dimensions,
    output_filetype=output_filetype
)

print("Script execution complete.")

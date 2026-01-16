#!/bin/bash
#$ -l h_rt=8:00:00                     # Job time - applying transforms is faster than registration
#$ -t 1-10:1                           # Job array for multiple patients - adjust range as needed
#$ -l rmem=8G                         # RAM needed (less than registration)
#$ -P insigneo-polaris                 # Project
#$ -o qsuboutput/                      # Output logs
#$ -e qsuboutput/                      # Error logs
#$ -pe smp 1                           # CPU cores
#$ -tc 2                               # Parallel jobs
#$ -m be                               # Email notifications
#$ -M @mail.com          # Email address

# Apply Transforms Script
# ------------------------
# Applies existing transforms (single or composite) to sets of images
# Can handle:
#   - Standard registrations (0GenericAffine.mat + 1Warp.nii.gz)
#   - Composite transforms (single Composite.nii.gz file)
#   - Multiple images per patient
#   - Automatic detection of transform type

# Load environment
source ~/.bashrc
module load apps/python/conda
source activate pyRegistration

# ANTs path
ap=/usr/local/community/polaris/tools/ants/2.5.1/bin/

# ==================================================================================================================
# CONFIGURATION SECTION - Please edit these parameters
# ==================================================================================================================

# Data directory containing patient folders
dir=/path/to/directory/with/patient/folders  									# <<-[EDIT]

# Transform directory pattern - where to find the transforms
# Options:
#   1. For single registration (normal mode): "Reg_TLC_2_RV"
#   2. For composite transforms: "Reg_TLC_2_RV_composite"

transform_folder="Reg_TLC_2_RV"  												# <<-[EDIT] - Name of folder containing transforms

# Image folder to process (foldername contining the image you want to transform)
image_folder="img"  															# <<-[EDIT] - Folder containing images to transform 

# Reference image identifier (what the images will be transformed to)
reference_identifier="RV"  														# <<-[EDIT] - String to identify reference image

# Optional: Specific subfolder within image folder (if you need another level to search fr)
subfolder=""  																	# <<-[EDIT] - Leave empty if not needed

# Optional: Output directory (if empty, saves in transform folder)
output_base_dir=""  															# <<-[EDIT] - Leave empty to save in transform folder

# Transform type (leave empty for auto-detection)
# Options: "--composite" for composite, "--standard" for standard, or "" for auto
transform_type=""  																# <<-[EDIT] - Usually leave empty for auto-detection

# Optional: Include only specific images (space-separated identifiers)
# Example: "TLC FRC" to only transform TLC and FRC images
include_images=""  																# <<-[EDIT] - Leave empty to include all

# Optional: Exclude specific images (space-separated identifiers)
# Example: "mask seg" to exclude masks and segmentations
exclude_images="mask seg"  														# <<-[EDIT] - Common exclusions

# Image dimensions
dimensions="3"  																# <<-[EDIT] - Change to "2" for 2D images


# Path to apply transforms script
apply_script=/apply_transforms.py  # <<-[EDIT] - Update path to where python script is located

# ==================================================================================================================
# SCRIPT EXECUTION - Don't edit below unless you know what you're doing
# ==================================================================================================================

# Get patient ID from job array
patient_id=`ls "$dir" | sed -n "$SGE_TASK_ID"p`
patient_dir=$dir/$patient_id

echo "=========================================="
echo "Processing patient: $patient_id"
echo "Patient directory: $patient_dir"
echo "Transform folder: $transform_folder"
echo "=========================================="

# Find the transform directory
# Look for the transform folder within the patient directory
transform_dir=""
for d in $(find "$patient_dir" -type d -name "*$transform_folder*" 2>/dev/null); do
    if [ -d "$d" ]; then
        transform_dir="$d"
        echo "Found transform directory: $transform_dir"
        break
    fi
done

# Check if transform directory was found
if [ -z "$transform_dir" ]; then
    echo "ERROR: Could not find transform directory matching pattern: $transform_folder"
    echo "Searched in: $patient_dir"
    exit 1
fi

# Build the Python command from inputs
cmd="python $apply_script"
cmd="$cmd -pat_dir $patient_dir"
cmd="$cmd -trans_dir $transform_dir"
cmd="$cmd -img_folder $image_folder"
cmd="$cmd -ref $reference_identifier"
cmd="$cmd -ants_path $ap"
cmd="$cmd -dim $dimensions"

# Add optional parameters if specified
if [ ! -z "$subfolder" ]; then
    cmd="$cmd -sub_folder $subfolder"
fi

if [ ! -z "$output_base_dir" ]; then
    cmd="$cmd -out_dir $output_base_dir"
fi

if [ ! -z "$transform_type" ]; then
    cmd="$cmd $transform_type"
fi

if [ ! -z "$include_images" ]; then
    cmd="$cmd -include $include_images"
fi

if [ ! -z "$exclude_images" ]; then
    cmd="$cmd -exclude $exclude_images"
fi

echo "Running command:"
echo "$cmd"
echo "------------------------------------------"

# Run the Python script
$cmd

echo "=========================================="
echo "Completed processing for patient $patient_id"
echo "=========================================="


# ==================================================================================================================
# PARAMETER REFERENCE
# ==================================================================================================================
# --mode          : Processing mode - "direct", "tree", or "vent" (REQUIRED)
# -transform_mode   : Transform mode - "forward" (default), "inverse", "warp_only", "inverse_warp_only" (REQUIRED)
#
# Explicit transform file specification (optional - overrides auto-detection):
# -affine_file      : Explicit affine transform filename (relative to transform folder or absolute)
# -warp_file        : Explicit warp transform filename (relative to transform folder or absolute)
# -inverse_warp_file: Explicit inverse warp transform filename (relative to transform folder or absolute)
# -inverse_use_ants : For inverse mode: use forward warp with ANTs inversion flag instead of pre-generated 
#                     inverse warp file (flag, default: False)
#
# -pat_dir        : Patient directory path
# -vent_dir		  : Patient Ventilation path (usually needed if using vent mode - as ventilation images are not in main patient folder, but in parallel folder)
# -trans_folder   : Registration folder name containing transform files (*0GenericAffine.mat, *1Warp.nii.gz)
# -vent_trans_folder : Registration folder name for ventilation images containing ventilation images (usually needed if using vent mode - as ventilation images are not in main patient folder, but in parallel folder)
# -img_folder     : Folder name containing images to transform (e.g., "img", "seg")
# -ref            : Reference image identifier string (e.g., "RV", "TLC") - searches patient tree automatically
# -out_dir        : Output directory name (created within patient tree if relative path)
# -ants_path      : Path to ANTs binaries
# -dim            : Image dimensions - "2" or "3" (default: "3")
# -sub_folder     : [TREE] Optional subfolder within img_folder
# -include        : Include only images matching these patterns (space-separated)
# -exclude        : Exclude images matching these patterns (space-separated)
# -timepoint      : Process specific timepoint(s) only (e.g., "mrA", "Visit1")
# -vent_dirs      : [VENT] Ventilation type folders (e.g., "Vent_Int Vent_Trans")
# -vent_strings   : [VENT] Folder:prefix mapping (e.g., "Vent_Int:sVent Vent_Trans:JacVent")
# -vent_filters   : [VENT] Additional filename filters (e.g., "_medfilt_3.nii.gz")
#

# ==================================================================================================================
# TRANSFORM MODE REFERENCE
# ==================================================================================================================
#
# Given a registration: Reg_{moving}_2_{fixed}
# 
# Transform files produced:
#   - Reg_{moving}_2_{fixed}_0GenericAffine.mat    (affine transform)
#   - Reg_{moving}_2_{fixed}_1Warp.nii.gz          (forward warp: moving -> fixed)
#   - Reg_{moving}_2_{fixed}_1InverseWarp.nii.gz   (inverse warp: fixed -> moving)
#
# TRANSFORM MODES:
#
# 1. forward (default):
#    - Applies: Warp + Affine (ANTs command: -t Warp.nii.gz -t Affine.mat)
#    - Use case: Transform image from moving space to fixed space
#    - Example: You have TLC image, reg is Reg__TLC_2__RV, you want TLC in RV space
#
# 2. inverse:
#    - Applies: InverseWarp followed by Affine(inverted)
#    - ANTs command: -t [Affine.mat,1] -t InverseWarp.nii.gz
#    - (ANTs applies last-to-first, so InverseWarp is applied first, then inverted Affine)
#    - Use case: Transform image from fixed space to moving space  
#    - Example: You have RV image, reg is Reg__TLC_2__RV, you want RV in TLC space
#    - Note: Use -inverse_use_ants flag to force using forward warp with ANTs inversion 
#            flag instead of pre-generated inverse warp file
#
# 3. warp_only:
#    - Applies: Only the Warp (no affine)
#    - Use case: When you have composite transforms or only want deformable part
#
# 4. inverse_warp_only:
#    - Applies: Only the InverseWarp (no affine)
#    - Use case: Reverse direction without affine component
#

# ==================================================================================================================
# EXAMPLE COMMANDS
# ==================================================================================================================
# --- TREE MODE: Images in standard tree structure (Patient/visit/folder/images) ---
# python $apply_script --mode tree \
#     -pat_dir /data/Patient01 \
#     -trans_folder "Reg__TLC_2__RV" \
#     -img_folder "img" \
#     -ref "RV" \
#     -include "TLC" "FRC" \
#     -exclude "mask" "seg" \
#     -out_dir "transformed_images" \
#     -ants_path /path/to/ants/bin \
#     -dim "3"

# --- VENT MODE: Ventilation images within registration folder ---
# python $apply_script --mode vent \
#     -pat_dir /data/Patient01 \
#     -trans_folder "Reg__TLC_2__RV" \
#     -ref "RV" \
#     -vent_dirs "Vent_Int" "Vent_Trans" "Vent_Hyb3" \
#     -vent_strings "Vent_Int:sVent" "Vent_Trans:JacVent" "Vent_Hyb3:HYCID" \
#     -vent_filters "_medfilt_3.nii.gz" \
#     -ants_path /path/to/ants/bin \
#     -dim "3"

# --- DIRECT MODE: Single image transform ---
# python $apply_script --mode direct \
#     --direct_image /data/Patient01/visit1/img/TLC.nii.gz \
#     --direct_transform_dir /data/Patient01/visit1/Reg__TLC_2__RV \
#     --direct_reference /data/Patient01/visit1/img/RV.nii.gz \
#     --direct_output /data/Patient01/visit1/output/TLC_transformed.nii.gz \
#     -ants_path /path/to/ants/bin \
#     -dim "3"

# ==================================================================================================================

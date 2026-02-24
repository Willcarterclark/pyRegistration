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
# -------------------------
# REQUIRES a transform folder to already be generated - this script does not perform registration!

# Load environment
source ~/.bashrc
module load apps/python/conda
source activate pyRegistration

# ANTs path
ap=/usr/local/community/polaris/tools/ants/2.5.1/bin/

# ==================================================================================================================
# CONFIGURATION SECTION - Please edit these parameters
# ==================================================================================================================

# Data directory containing patient folders with reference and registered images.
dir=/path/to/shared/dir/of/images  # <<-[EDIT]

#Ventilation directory (designed to support nested 1H Ventilation trees) - Shouldnt need to use this
vent_dir=/path/to/shared/dir/of/ventilation/images # <<-[EDIT] (seperate directory for ventilation images stored as: vent_dir/patient#/visit#/{-vent_trans_folder}/{-vent_dir}/{-vent_strings}/{-vent_filters} - only used for compatibility with 1H Ventilation directories

# ==================================================================================================================
# SCRIPT EXECUTION - Don't edit below unless you know what you're doing
# ==================================================================================================================

# Get patient ID from job array
patient_id=`ls "$dir" | sed -n "$SGE_TASK_ID"p`

# Get Patient ID 
patient_dir=$dir/$patient_id
patient_dir_vent=$vent_dir/$patient_id # if your using the vent_dir

#Print for current patient
echo "Running Patient: "
echo $patient_id
echo "=========================="


# Path to apply transforms script
apply_script=/apply_transforms.py  # <<-[EDIT] - Update path to where python script is located

# ==================================================================================================================
# SCRIPT EXECUTION - Don't edit below unless you know what you're doing
# ==================================================================================================================

#Examples

# XeFRC images to RV, using - use "tree" mode for normal nested structures
python $apply_script --mode tree \
    -pat_dir $patient_dir \
    -trans_folder "Reg__1H-XeFRC_2__RV" \
    -img_folder "img" \
    -ref "_RV" \
	-ref_folder "img" \
    -include "_XeFRC" \
    -exclude "mask" "seg" \
    -out_dir "Vent_XeFRC_2_RV" \
    -ants_path $ap \
    -dim "3"






echo "=========================================="
echo "Completed processing for patient $patient_id"
echo "=========================================="


 ==================================================================================================================
# PARAMETER REFERENCE
# ==================================================================================================================
# --mode              : Processing mode - "direct", "tree", or "vent" (REQUIRED)
#
# --- Common Arguments ---
# -pat_dir            : Patient directory path
# -ants_path          : Path to ANTs binaries
# -dim                : Image dimensions - "2" or "3" (default: "3")
# -out_dir            : Output base directory (created if doesn't exist)
# -timepoint          : Process specific timepoint(s) only (e.g., "mrA", "Visit1")
#
# --- Transform Arguments ---
# -trans_folder       : Registration folder name containing transform files (*0GenericAffine.mat, *1Warp.nii.gz)
# -transform_mode     : Transform direction - "forward" (default), "inverse", "warp_only", "inverse_warp_only"
# -affine_file        : Explicit affine transform filename (optional override)
# -warp_file          : Explicit warp transform filename (optional override)
# -inverse_warp_file  : Explicit inverse warp transform filename (optional override)
#
# --- Reference Image Arguments ---
# -ref                : Reference image identifier string (e.g., "RV", "_TLC") OR full path to reference image
# -ref_folder         : Folder containing reference images (e.g., "img", "seg") - uses tree structure search
# -ref_subfolder      : Optional subfolder within ref_folder
#                       Search pattern: patient_dir/visit/ref_folder/(ref_subfolder)/ for -ref identifier
#                       If -ref_folder not specified, falls back to recursive search
#
# --- Output Filename Arguments ---
# -out_str            : Output filename suffix. Options:
#                         - Not specified/empty: No suffix (filename unchanged)
#                         - "True": Adds "_transformed" suffix
#                         - Custom string: Adds that string as suffix (e.g., "_warped")
# -o_filetype         : Output file type - ".nii.gz", ".nii", ".mha", ".mhd", ".nrrd" (default: preserve input type)
#
# --- Interpolation ---
# -interp             : ANTs interpolation method - "Linear", "NearestNeighbor", "Gaussian", "BSpline", "GenericLabel"
#                       Default is None (auto-detect: Linear for images, NearestNeighbor for masks/segs)
#
# --- Image Filtering ---
# -include            : Include only images matching these patterns (space-separated)
# -exclude            : Exclude images matching these patterns (space-separated)
#
# --- TREE Mode Specific ---
# -img_folder         : Folder name containing images to transform (e.g., "img", "seg")
# -sub_folder         : Optional subfolder within img_folder
#
# --- VENT Mode Specific ---
# -vent_dir           : Patient Ventilation base path (if vent images are in parallel folder structure)
# -vent_trans_folder  : Registration folder where ventilation images are stored (source folder for images)
# -vent_dirs          : Ventilation type folders (e.g., "Vent_Int" "Vent_Trans" "Vent_Hyb3")
# -vent_strings       : Folder:prefix mapping (e.g., "Vent_Int:sVent" "Vent_Trans:JacVent")
# -vent_filters       : Additional filename filters (e.g., ".nii.gz" "_medfilt_3.nii.gz")
#                       Output folder auto-generated as: {vent_trans_folder}_2_{ref_target}
#
# --- DIRECT Mode Specific ---
# --direct_image      : Full path to image to transform
# --direct_transform_dir : Directory containing transform files
# --direct_reference  : Full path to reference image
# --direct_output     : Full path to output file
# -patient_id         : Patient identifier for output structure (if using -out_dir instead of --direct_output)
#
# ==================================================================================================================
# EXAMPLE COMMANDS
# ==================================================================================================================
#
# --- TREE MODE: Images in standard tree structure (Patient/visit/folder/images) ---
# python $apply_script --mode tree \
#     -pat_dir /data/Patient01 \
#     -trans_folder "Reg__TLC_2__RV" \
#     -img_folder "img" \
#     -ref "RV" \
#     -ref_folder "img" \
#     -include "TLC" "FRC" \
#     -exclude "mask" "seg" \
#     -ants_path /path/to/ants/bin \
#     -dim "3"
#
# --- TREE MODE: With inverse transform and custom output ---
# python $apply_script --mode tree \
#     -pat_dir /data/Patient01 \
#     -trans_folder "Reg__TLC_2__RV" \
#     -transform_mode inverse \
#     -img_folder "img" \
#     -ref "_TLC" \
#     -ref_folder "img" \
#     -out_dir "/output/path" \
#     -out_str "_warped" \
#     -o_filetype ".nii.gz" \
#     -ants_path /path/to/ants/bin
#
# --- VENT MODE: Transform ventilation images to new reference space ---
# python $apply_script --mode vent \
#     -pat_dir /data/Patient01 \
#     -vent_dir /data/Ventilation/Patient01 \
#     -trans_folder "Reg__TLC_2__1H-HeTLC" \
#     -vent_trans_folder "Reg__TLC_2__RV" \
#     -vent_dirs "Vent_Int" "Vent_Trans" "Vent_Hyb3" \
#     -vent_strings "Vent_Int:sVent" "Vent_Trans:JacVent" "Vent_Hyb3:HYCID" \
#     -vent_filters ".nii.gz" \
#     -ref "_1H-HeTLC" \
#     -ref_folder "img" \
#     -ants_path /path/to/ants/bin \
#     -interp "Linear"
#     # Output folder auto-generated as: Reg__TLC_2__RV_2_1H-HeTLC
#
# --- VENT MODE: Inverse transform ventilation images ---
# python $apply_script --mode vent \
#     -pat_dir /data/Patient01 \
#     -vent_dir /data/Ventilation/Patient01 \
#     -trans_folder "Reg__TLC_2__RV" \
#     -vent_trans_folder "Reg__TLC_2__RV" \
#     -vent_dirs "Vent_Int" "Vent_Trans" \
#     -vent_strings "Vent_Int:sVent" "Vent_Trans:JacVent" \
#     -vent_filters ".nii.gz" \
#     -transform_mode inverse \
#     -ref "_TLC" \
#     -ref_folder "img" \
#     -ants_path /path/to/ants/bin
#
# --- DIRECT MODE: Single image transform ---
# python $apply_script --mode direct \
#     --direct_image /data/Patient01/visit1/img/TLC.nii.gz \
#     --direct_transform_dir /data/Patient01/visit1/Reg__TLC_2__RV \
#     --direct_reference /data/Patient01/visit1/img/RV.nii.gz \
#     --direct_output /data/Patient01/visit1/output/TLC_transformed.nii.gz \
#     -ants_path /path/to/ants/bin \
#     -dim "3"
#
# ==================================================================================================================

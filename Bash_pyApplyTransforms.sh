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
#$ -M @sheffield.ac.uk          # Email address

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

# ANTs path (on HPC System)
ap=/path/to/ANTS/ # <<-[EDIT] #/usr/local/community/polaris/tools/ants/2.5.1/bin/

# ==================================================================================================================
# CONFIGURATION SECTION - Please edit these parameters
# ==================================================================================================================

# Data directory containing patient folders
dir=/shared/director/to/dataset/  # <<-[EDIT]

# ==================================================================================================================
# SCRIPT EXECUTION - Don't edit below unless you know what you're doing
# ==================================================================================================================

# Path to apply transforms script
apply_script=/path/to/script/pyApplyTransforms.py  # <<-[EDIT] - Update path to where python script is located

# Get patient ID from job array
patient_id=`ls "$dir" | sed -n "$SGE_TASK_ID"p`
patient_dir=$dir/$patient_id

#
echo "Running Patient: "
echo $patient_id
echo "=========================="

# Run the Python script
# ------------------------------------------
# Transform images
# ------------------------------------------
# Apply XeFRC->RV registration to XeFRC images
python $apply_script --mode tree \
    -pat_dir $patient_dir \
    -trans_folder "Reg__1H-XeFRC_2__RV" \
    -img_folder "img" \
    -ref "RV" \
    -include "_XeFRC" \
    -exclude "mask" "seg" \ #excluse mask and seg images
    -out_dir "Vent_XeFRC_2_RV" \
    -transform_mode forward \
    -ants_path $ap \
    -dim "3"

# Apply HeFRC->RV registration to HeFRC images
python $apply_script --mode tree \
    -pat_dir $patient_dir \
    -trans_folder "Reg__1H-HeFRC_2__RV" \
    -img_folder "img" \
    -ref "RV" \
    -include "_HeFRC" \
    -exclude "mask" "seg" \
    -out_dir "Vent_HeFRC_2_RV" \
    -transform_mode forward \
    -ants_path $ap \
    -dim "3"

# ------------------------------------------
# Transform masks (segmentations)
# ------------------------------------------
# XeFRC masks
python $apply_script --mode tree \
    -pat_dir $patient_dir \
    -trans_folder "Reg__1H-XeFRC_2__RV" \
    -img_folder "seg" \
    -ref "RV" \
    -include "_1H-XeFRC" \
    -exclude "img" \
    -out_dir "Vent_XeFRC_2_RV" \
    -transform_mode forward \
    -ants_path $ap \
    -dim "3"

# HeFRC masks
python $apply_script --mode tree \
    -pat_dir $patient_dir \
    -trans_folder "Reg__1H-HeFRC_2__RV" \
    -img_folder "seg" \
    -ref "RV" \
    -include "_1H-HeFRC" \
    -exclude "img" \
    -out_dir "Vent_HeFRC_2_RV" \
    -ants_path $ap \
    -transform_mode forward \
    -dim "3"

# ------------------------------------------
# Inverse transforms (commented out)
# ------------------------------------------
# python $apply_script --mode tree \
#     -pat_dir $patient_dir \
#     -trans_folder "Reg__RV_2__1H-XeFRC" \
#     -img_folder "img" \
#     -ref "_XeFRC" \
#     -include "_XeFRC" \
#     -exclude "mask" "seg" \
#     -out_dir "Vent_1HVent_2_XeFRC" \
#     -ants_path $ap \
#     -dim "3"

# python $apply_script --mode tree \
#     -pat_dir $patient_dir \
#     -trans_folder "Reg__RV_2__1H-HeFRC" \
#     -img_folder "img" \
#     -ref "_HeFRC" \
#     -include "_HeFRC" \
#     -exclude "mask" "seg" \
#     -out_dir "Vent_1HVent_2_HeFRC" \
#     -ants_path $ap \
#     -dim "3"

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
#    - Applies: Warp + Affine
#    - Use case: Transform image from moving space to fixed space
#    - Example: You have TLC image, reg is Reg__TLC_2__RV, you want TLC in RV space
#
# 2. inverse:
#    - Applies: InverseWarp + Affine(inverted)
#    - Use case: Transform image from fixed space to moving space  
#    - Example: You have RV image, reg is Reg__TLC_2__RV, you want RV in TLC space
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
#     -transform_mode forward \
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
#     -transform_mode forward \
#     -dim "3"

# --- DIRECT MODE: Single image transform ---
# python $apply_script --mode direct \
#     --direct_image /data/Patient01/visit1/img/TLC.nii.gz \
#     --direct_transform_dir /data/Patient01/visit1/Reg__TLC_2__RV \
#     --direct_reference /data/Patient01/visit1/img/RV.nii.gz \
#     --direct_output /data/Patient01/visit1/output/TLC_transformed.nii.gz \
#     -ants_path /path/to/ants/bin \
#     -transform_mode forward \
#     -dim "3"

# ==================================================================================================================

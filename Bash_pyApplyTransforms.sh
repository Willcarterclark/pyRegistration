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

#!/bin/bash
#$ -l h_rt=48:00:00					# Total Job time, lower numbers will put job as a higher priority, but jobs may need longer to finish. 
#$ -t 1-20:1 						    # Job-array number <#-#:#> # Runs # to # : with # gap # Job-array number, if 1-10:1, itll run jobs for the first 1-10 patient folders, 1 at a time. 
#
#$ -l rmem=8G 						  # Amount of RAM assigned to script per run. (Note that for parallel jobs, each script will take this much memory/ core). (this is per core!)
#$ -P insigneo-polaris 			# Permissions/Identifier - no need to change. 
#$ -o qsuboutput/ 					# Folder for outputting script logs - saved as ".o" in the folder. 
#$ -e qsuboutput/ 					# Folder for outputting error logs - saved as ".e" in the folder.
#$ -pe smp 1 						    # Assigned CPU cores, more = faster job, but shouldnt need more than 1, since python doesnt support multi-threading. 
#$ -tc 2  						      # parallel jobs #, depending on script complexity to speed up completion, only works on 'job-array' tasks
#$ -m be 							      # <b>-beginning <e>-end <be>-beginning & end #Email notifications for script running 
#$ -M youremail@email.ac.uk # <ADD YOUR EMAIL HERE> #Email for reciving notifications of script

#Info 
# By Will Clark - wclark2@sheffield.ac.uk
# POLARIS - University of Sheffield 

mkdir -p qsuboutput/ #Make logging directory
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$NSLOTS #itk multi-threading

# ==================================================================================================================
# Image Registration Script
# ==================================================================================================================
# TWO MODES:
#   1. YAML config (recommended): All stable settings in a .yaml file
#   2. Full CLI (original): All settings as bash variables + command-line flags
#
# TWO RUNTIMES:
#   - ants (default): CPU-based, ANTs subprocess (antsRegistration)
#   - fireants: GPU-accelerated diffeomorphic registration (requires GPU node + FireANTs)
# ==================================================================================================================


# Directory structure should be:
#		"DataFolder" / "PatientNumber" / "Visit" / "Image" & "Masks" / *.nii.gz or *.mha
#		*Should* work with any image types, provided the masks and images are consistent.
#
# Outputs registrations to folder: "Reg_{moving image name}_2_{fixed image name}" in Patient folder
#		"DataFolder" / "PatientNumber" / "Visit" / Reg_X_2_Y / *.nii.gz
# 		Images are outputted as ".nii.gz" image types! - if there is enough demand to move to another image type I'll implement a way of specifiying the output file type. 
#
# Parts of the script you should edit have "<<-[EDIT]" as a comment. 
#
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Will's top tip! : Always register the "Bigger" image (more voxels) to the "smaller" image (less voxels), as this is more reliable, and doesnt interpolate additional "imaginary" values between the volumes! i.e: Moving: TLC or FRC, Fixed: RV. 
# This registration is symmetric, so you can always use the "inverse" if you need the smaller image registered to the larger image.  
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# ==================================================================================================================
# Script RUNNING and setup

# Python Environment setup
#-----------------------------
# If you dont have a python environment set up on Bash you will need to set one up to perform registration
#
# 1) Log onto SHARC, and load into a worker node
#	> qrshx
#
# 2) Run each line of code > :
#	> module load apps/python/conda
#	> conda create -n pyRegistration numpy
#
# 3) With the python environment "pyRegistration" set up now run this code to 'enter' the enviromnent to set up modules > :
#	> source activate pyRegistration
# 
# 4)a Install via requirements.txt -> pip install -r requirements.txt  
#
# or:
#
# 4)b Install the required python packages using "conda install" (or "pip install" if it doesnt work)
#	> conda install:
# 		SimpleITK
# 		pandas
# 		argparse
# 		pathlib
#		  antspyx
# 		subprocess
# 		scipy
# 		skimage
#     fireants (USE pip install) [NOTE: Not currently functional/Required]
#
# 5) Once installed you can type: > source deactivate    Which takes you out of the enviromnent, and > exit		to exit environment

#---------------------
# SCRIPT # EDIT FROM HERE!
#---------------------
#
source ~/.bashrc 											# source for .bashrc - This file should be setup on your home ~ directory, if not the script will not function.
#
#load python modules and environment
module load apps/python/conda								#
source activate pyRegistration 								# Python Environment activation - If you havent set one up on SHARC: Read the "Python Environment setup" section.
#
#ANTS Path
ap=/usr/local/community/polaris/tools/ants/2.5.1/bin/ 		#Path to ANTS on Sharc
itk=/usr/local/community/polaris/tools/itk 					#Path to ITK on Sharc (unused!)
#
# Set Patient directory
dir=/shared/path/to/your/data/directory/with/each/patient	# Edit this to be the path to the directory of your patients you want to register images in. 										#<<-[EDIT]
reg_script=/path/to/py_Registration.py              # <<-[EDIT]
reg_config=/path/to/registration_config.yaml                  # <<-[EDIT]
#
#
patient_id=`ls "$dir" | sed -n "$SGE_TASK_ID"p` 			# Read directory of patients, should be 1 folder per patient. Uses Job-array ($SGE_TASK_ID) number specified by -t flag above.
patient_dir=$dir/$patient_id 								# Get Patient directory from patient ID. 
#


cho "Running Patient: $patient_id"
echo "=========================="

# ==================================================================================================================
# RECOMMENDED: YAML config mode
# ==================================================================================================================
# Everything (paths, ANTs command, masks, FireANTs params) is in the YAML.
# Only patient directory changes per job.

python $reg_script \
    -yaml $reg_config \
    -pat_dir $patient_dir


# ==================================================================================================================
# YAML + per-run overrides
# ==================================================================================================================

# # Use FireANTs GPU registration (request GPU node with: #$ -l gpu=1) #NOT YET IMPLEMENTED!
# python $reg_script \
#     -yaml $reg_config \
#     -pat_dir $patient_dir \
#     -runtime fireants

# # Override image identifiers
# python $reg_script \
#     -yaml $reg_config \
#     -pat_dir $patient_dir \
#     -f "_FRC" -m "_TLC"


# ==================================================================================================================
# ALTERNATIVE: Full CLI mode (no YAML needed, backward compatible)
# ==================================================================================================================

# ap=/usr/local/community/polaris/tools/ants/2.5.1/bin/
#
# AntsRegCmd=""
# AntsRegCmd+="--dimensionality 3 "
# AntsRegCmd+="--verbose 1 "
# AntsRegCmd+="--output \"{output_prefix_full_placeholder}\" "
# AntsRegCmd+="--use-histogram-matching 1 "
# AntsRegCmd+="--initial-moving-transform \"[{fixed_placeholder},{moving_placeholder},1]\" "
# AntsRegCmd+="{nomasks} "
# AntsRegCmd+="--transform \"Rigid[0.1]\" "
# AntsRegCmd+="--metric \"MI[{fixed_placeholder},{moving_placeholder},1,32,Regular,0.25]\" "
# AntsRegCmd+="--convergence \"1000x500x250x100\" "
# AntsRegCmd+="--smoothing-sigmas \"3x2x1x0\" "
# AntsRegCmd+="--shrink-factors \"8x4x2x1\" "
# AntsRegCmd+="{nomasks} "
# AntsRegCmd+="--transform \"Affine[0.1]\" "
# AntsRegCmd+="--metric \"MI[{fixed_placeholder},{moving_placeholder},1,32,Regular,0.75]\" "
# AntsRegCmd+="--convergence \"1000x500x250x100\" "
# AntsRegCmd+="--smoothing-sigmas \"3x2x1x0\" "
# AntsRegCmd+="--shrink-factors \"8x4x2x1\" "
# AntsRegCmd+="{addmasks} "
# AntsRegCmd+="--transform \"BSplineSyN[0.2,65,0,3]\" "
# AntsRegCmd+="--metric \"CC[{fixed_placeholder},{moving_placeholder},1,2]\" "
# AntsRegCmd+="--convergence \"500x200x70x50x10\" "
# AntsRegCmd+="--smoothing-sigmas \"5x3x2x1x0\" "
# AntsRegCmd+="--shrink-factors \"10x6x4x2x1\""
#
# python $reg_script \
#     -pat_dir $patient_dir \
#     -scn_dir "img" -seg_dir "seg" \
#     -f "_RV" -m "_TLC" \
#     -f_mask "EXPIRATION" -m_mask "INSPIRATION" \
#     -ants_path $ap \
#     -ants_reg_params "$AntsRegCmd" \
#     -saveinputs -masked_inputs


echo "=========================================="
echo "Completed processing for patient $patient_id"
echo "=========================================="



################################################################################
# PYTHON SCRIPT PARAMETERS REFERENCE - for CLI
################################################################################
#
# REQUIRED:
#   -pat_dir     	 : Patient directory path
#   -scn_dir     	 : Image folder name
#   -seg_dir     	 : Mask folder name
#   -f           	 : Fixed image identifier substring
#   -m           	 : Moving image identifier substring
#   -ants_path   	 : Path to ANTs binaries
#   -ants_reg_params : ANTs registration command string
#
# OPTIONAL:
#   -f_mask      	: Fixed mask identifier substring
#   -m_mask      	: Moving mask identifier substring
#   -out_dir     	: Output directory path
#   -reg_exp_mask	: Mask expansion size (0-10, default: 8) (this is for registration only!)
#   -dim         	: Image dimensions (2 or 3, default: 3)
#   -sub_dir     	: Additional subdirectory level (if needed)
#	-out_type	 	: Image output filetype (default is .nii.gz, supports most 3d medical image types)
#	-saveinputs  	: Saves input images to registration directory (images, masks, expanded masks)
#	-masked_inputs	: Overrides input fixed and moving images with masked copies (will always save input images to directory) - Potentially fixes bug with using ANTs's mask handling.
#
################################################################################

################################################################################
# OUTPUT FILES EXPLANATION
################################################################################
#
# After successful registration, you'll find these files:
#
#   Reg_MOVING_2_FIXED_warped.nii.gz
#       - Moving image warped to fixed image space
#
#   Reg_MOVING_2_FIXED_inv_warped.nii.gz
#       - Fixed image warped to moving image space (inverse)
#
#   Reg_MOVING_2_FIXED0GenericAffine.mat
#       - Affine transformation matrix
#
#   Reg_MOVING_2_FIXED1Warp.nii.gz
#       - Forward deformation field
#
#   Reg_MOVING_2_FIXED1InverseWarp.nii.gz
#       - Inverse deformation field
#
#   Reg_MOVING_2_FIXED_mask_warped.nii.gz (if masks used)
#       - Moving mask warped to fixed space
#
#   Reg_MOVING_2_FIXED_mask_inv_warped.nii.gz (if masks used)
#       - Fixed mask warped to moving space
#
#   Reg_MOVING_2_FIXED_0_reg_accuracy.csv (if masks used)
#       - Registration quality metrics (Dice, Jaccard, etc.)
#
#---OPTIONAL OUTPUTS-----------------------------------------
#
#	_fixed_image.nii.gz (if -saveinputs)
#		- Fixed image (used in registration)
#
#	_moving_image.nii.gz (if -saveinputs)
#		- Fixed image (used in registration)
#
#	_fixed_mask.nii.gz (if -saveinputs and if masks used)
#		-Fixed image mask
#
#	_moving_mask.nii.gz (if -saveinputs and if masks used)
#		-Moving image mask
#
#	_fixed_mask_expanded.nii.gz (if -saveinputs and if masks used)
#		-Expanded Fixed image mask (used in registration)
#
#	_moving_mask_expanded.nii.gz (if -saveinputs and if masks used)
#		-Expanded Moving image mask (used in registration)
#
#
################################################################################


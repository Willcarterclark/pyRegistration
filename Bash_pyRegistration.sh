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

# Dummy BASH script for image registration. 
# -----------------------------------------
#
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
# 4) Install the required python packages using "conda install" (or "pip install" if it doesnt work)
#	> conda install:
# 		SimpleITK
# 		pandas
# 		argparse
# 		pathlib
# 		ants
#		antspyx
# 		subprocess
# 		scipy
# 		skimage
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
#
patient_id=`ls "$dir" | sed -n "$SGE_TASK_ID"p` 			# Read directory of patients, should be 1 folder per patient. Uses Job-array ($SGE_TASK_ID) number specified by -t flag above.
patient_dir=$dir/$patient_id 								# Get Patient directory from patient ID. 
#
sdir=/pyRegistration.py # Registration Script 			# Change this to where the pyRegistration.py python file is stored 															#<<--[EDIT]
#--------------------------------------
# ANTS COMMAND CALL - Currently using a generic high-resolution multi-step (Rigid, Affine, Non-linear), multi-level (downsampled resolution -> full resolution) registration.

# The placeholders {output_prefix_full_placeholder}, {fixed_placeholder}, and {moving_placeholder} will be replaced by the Python script with actual file paths

#REGISTRATION DEFINITION
AntsRegCmd=""
AntsRegCmd+="--dimensionality 3 "                                                                                          											# Change if using 2D Images
AntsRegCmd+="--verbose 1 "                                                                                                 											# Enable verbosity
AntsRegCmd+="--output \"{output_prefix_full_placeholder}\" " 																										# Handles output for raw transform files
AntsRegCmd+="--use-histogram-matching 1 "                                                                                  											# Histogram Matching
AntsRegCmd+="--initial-moving-transform \"[{fixed_placeholder},{moving_placeholder},1]\" "                                 											# Initial Transform step

# ====== First registration step - RIGID ======
AntsRegCmd+="{nomasks} "
AntsRegCmd+="--transform \"Rigid[0.1]\" "
AntsRegCmd+="--metric \"MI[{fixed_placeholder},{moving_placeholder},1,32,Regular,0.25]\" "
AntsRegCmd+="--convergence \"1000x500x250x100\" "
AntsRegCmd+="--smoothing-sigmas \"3x2x1x0\" "
AntsRegCmd+="--shrink-factors \"8x4x2x1\" "

# ====== Second registration step - AFFINE ======
AntsRegCmd+="{addmasks} "
AntsRegCmd+="--transform \"Affine[0.1]\" "
AntsRegCmd+="--metric \"MI[{fixed_placeholder},{moving_placeholder},1,32,Regular,0.25]\" "
AntsRegCmd+="--convergence \"1000x500x250x100\" "
AntsRegCmd+="--smoothing-sigmas \"3x2x1x0\" "
AntsRegCmd+="--shrink-factors \"8x4x2x1\" "

# ====== Third registration step - DEFORMABLE (BSplineSyN) ======
AntsRegCmd+="{addmasks} "
AntsRegCmd+="--transform \"BSplineSyN[0.2,65,0,3]\" "
AntsRegCmd+="--metric \"CC[{fixed_placeholder},{moving_placeholder},1,2]\" "
AntsRegCmd+="--convergence \"500x200x70x50x10\" "
AntsRegCmd+="--smoothing-sigmas \"5x3x2x1x0\" "
AntsRegCmd+="--shrink-factors \"10x6x4x2x1\""

#AntsRegCmd+="{addmasks} " #If you want to globally define masks to use for all steps, remove other lines with masks and uncomment this.

#--------------------------------------
# RUN PYTHON SCRIPT - Python call
python $sdir -pat_dir $patient_dir -scn_dir "Image_Folder" -seg_dir "Mask_Folder" -f "FixedImageName" -m "MovingImageName" -f_mask "EXPIRATION" -m_mask "INSPIRATION" -ants_path $ap -ants_reg_params "$AntsRegCmd" -saveinputs -masked_inputs																			#<<-[EDIT]
#

################################################################################
# PYTHON SCRIPT PARAMETERS REFERENCE
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


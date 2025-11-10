# pyRegistration
Deformable Image Registration for medical images using ANTs. Designed to support nested-patient folder structures. 

# Pre-requisites:
Need nested structure of patient image folders:

	"DataFolder" / "PatientNumber" / "Visit" / "Image" & "Masks" / *.nii.gz or *.mha
	
*Should* work with any image types, provided the masks and images are consistent filetypes.

# Output
Outputs registrations to folder: "Reg_{moving image name}_2_{fixed image name}" in Patient folder:

		"DataFolder" / "PatientNumber" / "Visit" / Reg_X_2_Y / *.nii.gz

Supported output image types are .nii.gz (defualt), .nii, .mha .

# Scripts:

- pyRegistration.py - All-in-one Primary python script for handling image reading and registration processing.

- Bash_pyRegistration.sh - HPC script for running patient-wise registration

- pyApplyTransforms.py - Python script for applying computed registrations transforms to image files

- Bash_pyApplyTransforms.sh - HPC script for running apply transform script

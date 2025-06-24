import numpy as np
import nibabel as nib

# Parameters
input_path = 'spect.nii.gz'  # Path to the input SPECT NIfTI file
output_path = 'fixed_threshold_mask.nii.gz'  # Path to save the binary mask
fixed_value = 100  # Fixed intensity threshold; adjust as needed

# Load the SPECT volume
img = nib.load(input_path)
data = img.get_fdata()

# Apply fixed threshold to create binary mask
mask = (data > fixed_value).astype(np.uint8)

# Save the mask as NIfTI
mask_img = nib.Nifti1Image(mask, img.affine, img.header)
mask_img.to_filename(output_path)
print(f"Fixed threshold mask saved to {output_path}")
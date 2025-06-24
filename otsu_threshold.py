import numpy as np
import nibabel as nib
from skimage.filters import threshold_otsu

# Parameters
input_path = 'spect.nii.gz'  # Path to the input SPECT NIfTI file
output_path = 'otsu_threshold_mask.nii.gz'  # Path to save the binary mask

# Load the SPECT volume
img = nib.load(input_path)
data = img.get_fdata()

# Compute global Otsu threshold
global_thresh = threshold_otsu(data)
print(f"Computed Otsu threshold: {global_thresh}")

# Apply threshold to create binary mask
mask = (data > global_thresh).astype(np.uint8)

# Save the mask as NIfTI
mask_img = nib.Nifti1Image(mask, img.affine, img.header)
mask_img.to_filename(output_path)
print(f"Otsu threshold mask saved to {output_path}")
import numpy as np
import nibabel as nib
from skimage.filters import threshold_local

# Parameters
input_path = 'spect.nii.gz'  # Path to the input SPECT NIfTI file
output_path = 'adaptive_threshold_mask.nii.gz'  # Path to save the binary mask
block_size = 51  # Size of the local neighborhood
offset = 10      # Constant subtracted from mean or weighted mean

# Load the SPECT volume
img = nib.load(input_path)
data = img.get_fdata()

# Initialize mask array
mask = np.zeros_like(data, dtype=np.uint8)

# Process slice-by-slice
for i in range(data.shape[2]):
    slice_img = data[:, :, i]
    # Compute local threshold for each slice
    local_thresh = threshold_local(slice_img, block_size=block_size, offset=offset)
    # Create binary mask for this slice
    mask[:, :, i] = (slice_img > local_thresh).astype(np.uint8)

# Save the mask as NIfTI
mask_img = nib.Nifti1Image(mask, img.affine, img.header)
mask_img.to_filename(output_path)
print(f"Adaptive threshold mask saved to {output_path}")


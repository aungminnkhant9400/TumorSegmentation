# advanced_segmentation_random_walker.py
# -----------------------------------
# Script for 3D SPECT tumor segmentation using Random Walker (seeded) and post-processing

import numpy as np
import nibabel as nib
from skimage import filters, morphology, segmentation
from skimage.restoration import denoise_nl_means, estimate_sigma

# --- Parameters ---
spect_path       = 'spect.nii.gz'
out_path         = 'random_walker_mask.nii.gz'
den_sigmasmooth = 1.0      # Gaussian smoothing sigma for noise reduction

# --- Load SPECT volume ---
img  = nib.load(spect_path)
data = img.get_fdata().astype(np.float32)

# --- (Optional) Denoise & Smooth ---
sigma_est = np.mean(estimate_sigma(data))
data_denoised = denoise_nl_means(
    data,
    h=1.15 * sigma_est,
    patch_size=5,
    patch_distance=3,
    fast_mode=True
)
data_smooth = filters.gaussian(data_denoised, sigma=den_sigmasmooth)

# --- Compute Intensity Markers ---
# Use Otsu thresholds to define background & object seeds
global_otsu = filters.threshold_otsu(data_smooth)
low_thresh   = global_otsu * 0.3  # background where intensities are very low
high_thresh  = global_otsu        # object seeds where intensities are high

markers = np.zeros(data.shape, dtype=np.int32)
markers[data_smooth < low_thresh] = 1    # background label
markers[data_smooth > high_thresh] = 2   # object label

# --- Random Walker Segmentation ---
labels = segmentation.random_walker(
    data_smooth,
    markers,
    beta=20,
    mode='bf'
)
mask = (labels == 2)

# --- Morphological Cleanup ---
mask = morphology.remove_small_objects(mask, min_size=5000)
mask = morphology.binary_closing(mask, morphology.ball(2))

# --- Save the final mask ---
mask_img = nib.Nifti1Image(mask.astype(np.uint8), img.affine, img.header)
mask_img.to_filename(out_path)
print(f"Random Walker mask saved to {out_path}")

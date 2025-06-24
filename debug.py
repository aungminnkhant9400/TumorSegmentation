import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from skimage.restoration import denoise_nl_means, estimate_sigma

# --- Parameters to adjust ---
spect_path = 'spect.nii.gz'
background_level = 0  # intensity below which is considered background
block_sizes = [31, 51, 71]
offsets = [0, 10, 25]
normalize = True
denoise = True

# --- Load SPECT volume ---
img = nib.load(spect_path)
data = img.get_fdata()
data_shape = data.shape

# Determine a representative slice index (mid-axial)
slice_index = data_shape[2] // 2

# --- Precompute background mask ---
brain_mask = data > background_level  # True for voxels to consider


# -- Utility: display slice, threshold map, and mask --
def show_debug(slice_img, local_thresh, mask, title_suffix=''):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title(f'Original Slice {title_suffix}')
    plt.imshow(slice_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f'Local Threshold Map {title_suffix}')
    plt.imshow(local_thresh, cmap='hot')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f'Binary Mask {title_suffix}')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.show()


# --- Process one slice for visualization and tuning ---
slice_img = data[:, :, slice_index].astype(np.float32)
mask_bg = brain_mask[:, :, slice_index]

# Optional denoising
if denoise:
    sigma_est = np.mean(estimate_sigma(slice_img))
    slice_img = denoise_nl_means(
        slice_img,
        h=1.15 * sigma_est,
        fast_mode=True,
        patch_size=5,
        patch_distance=3
    )

# Optional normalization
if normalize:
    min_val, max_val = slice_img.min(), slice_img.max()
    slice_img = (slice_img - min_val) / (max_val - min_val)

# --- Sweep block_size and offset ---
for bs in block_sizes:
    for off in offsets:
        # Compute local threshold only on brain region
        local_thresh = threshold_local(
            slice_img * mask_bg,
            block_size=bs,
            offset=off
        )
        mask = (slice_img > local_thresh) & mask_bg
        voxel_count = mask.sum()
        print(f'block_size={bs}, offset={off}, voxels={voxel_count}')

        # Show the maps for first parameter set
        if bs == block_sizes[0] and off == offsets[0]:
            show_debug(slice_img, local_thresh, mask, f'(bs={bs}, off={off})')

# --- Final mask with chosen parameters ---
chosen_bs, chosen_off = 51, 10  # adjust after tuning
local_thresh = threshold_local(
    slice_img * mask_bg,
    block_size=chosen_bs,
    offset=chosen_off
)
final_mask = (slice_img > local_thresh) & mask_bg

# Display final result
display_title = f'Final bs={chosen_bs}, off={chosen_off}'
show_debug(slice_img, local_thresh, final_mask, display_title)

# Save final mask back to volume space and write NIfTI
full_mask = np.zeros_like(data, dtype=np.uint8)
full_mask[:, :, slice_index] = final_mask.astype(np.uint8)
mask_img = nib.Nifti1Image(full_mask, img.affine, img.header)
mask_img.to_filename('adaptive_debug_mask.nii.gz')

print('Debugged adaptive mask saved as adaptive_debug_mask.nii.gz')
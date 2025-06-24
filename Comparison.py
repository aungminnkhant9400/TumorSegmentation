# compare_thresholds.py
# --------------------
# Script to load the original SPECT volume and the three threshold masks,
# compute simple quantitative metrics (voxel counts, volume fractions, mean intensity)
# and Dice similarity between each pair of segmentation methods.

import numpy as np
import nibabel as nib

# Function to compute Dice coefficient
def dice_coefficient(m1, m2):
    m1_bool = m1.astype(bool)
    m2_bool = m2.astype(bool)
    intersection = np.logical_and(m1_bool, m2_bool).sum()
    return 2.0 * intersection / (m1_bool.sum() + m2_bool.sum()) if (m1_bool.sum() + m2_bool.sum()) > 0 else 1.0

# Paths
data_path = 'spect.nii.gz'
masks = {
    'adaptive': 'adaptive_threshold_mask.nii.gz',
    'otsu':     'otsu_threshold_mask.nii.gz',
    'fixed':    'fixed_threshold_mask.nii.gz'
}

# Load data
img = nib.load(data_path)
volume = img.get_fdata()
voxel_volume = np.abs(np.linalg.det(img.affine[:3, :3]))  # mm^3 per voxel

# Load masks and compute metrics
mask_data = {}
metrics = {}
for name, path in masks.items():
    m = nib.load(path).get_fdata().astype(np.uint8)
    mask_data[name] = m
    voxel_count = m.sum()
    volume_mm3 = voxel_count * voxel_volume
    fraction = voxel_count / m.size
    mean_intensity = volume[m.astype(bool)].mean()
    metrics[name] = {
        'voxels': int(voxel_count),
        'volume_mm3': float(volume_mm3),
        'fraction': float(fraction),
        'mean_intensity': float(mean_intensity)
    }

# Print summary table
print("Segmentation Metrics:\n")
print(f"{'Method':<10} {'Voxels':>10} {'Volume(mm^3)':>15} {'Fraction':>10} {'Mean Intensity':>15}")
for name, vals in metrics.items():
    print(f"{name:<10} {vals['voxels']:>10} {vals['volume_mm3']:>15.2f} {vals['fraction']:>10.4f} {vals['mean_intensity']:>15.2f}")

# Compute Dice between each pair
print("\nDice Similarity Coefficients:")
names = list(metrics.keys())
for i in range(len(names)):
    for j in range(i+1, len(names)):
        a = names[i]
        b = names[j]
        d = dice_coefficient(mask_data[a], mask_data[b])
        print(f"Dice({a} vs {b}): {d:.4f}")

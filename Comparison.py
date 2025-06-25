import numpy as np
import nibabel as nib
import os

def dice_coefficient(m1, m2):
    """Compute Dice similarity between two binary masks."""
    m1_bool = m1.astype(bool)
    m2_bool = m2.astype(bool)
    inter   = np.logical_and(m1_bool, m2_bool).sum()
    denom   = m1_bool.sum() + m2_bool.sum()
    return 2.0*inter/denom if denom>0 else 1.0

# --- PATHS FOR CASE 1 ---
spect_path = 'spect.nii.gz'
gt_path    = 'tumor_1.nii.gz'

# Your three threshold masks for case 1:
masks = {
    'adaptive': 'random_walker_mask.nii.gz',
    'otsu':     'otsu_threshold_mask.nii.gz',
    'fixed':    'mask_pct99.nii.gz'
}

# --- LOAD SPECT DATA & BASIC INFO ---
img          = nib.load(spect_path)
volume       = img.get_fdata()
voxel_volume = abs(np.linalg.det(img.affine[:3,:3]))  # mm³ per voxel
total_voxels = volume.size

# --- LOAD GOLD STANDARD ---
gt_img  = nib.load(gt_path).get_fdata().astype(bool)

# --- COLLECT METRICS ---
metrics   = {}
mask_data = {}

# First, do the gold mask
metrics['gold'] = {
    'voxels':       int(gt_img.sum()),
    'volume_mm3':   gt_img.sum() * voxel_volume,
    'fraction':     gt_img.sum() / total_voxels,
    'mean_intensity': volume[gt_img].mean() if gt_img.sum()>0 else 0.0
}
mask_data['gold'] = gt_img

# Now the threshold methods
for name, path in masks.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask not found: {path}")
    m      = nib.load(path).get_fdata().astype(bool)
    mask_data[name] = m

    voxel_count    = int(m.sum())
    vol_mm3        = voxel_count * voxel_volume
    frac           = voxel_count / total_voxels
    mean_int       = volume[m].mean() if voxel_count>0 else 0.0

    metrics[name] = {
        'voxels':         voxel_count,
        'volume_mm3':     vol_mm3,
        'fraction':       frac,
        'mean_intensity': mean_int
    }

# --- PRINT METRICS TABLE ---
print("\n Segmentation Metrics (Case 1):\n")
hdr = f"{'Method':<10} {'Voxels':>10} {'Volume(mm³)':>15} {'Fraction':>10} {'Mean Inten.':>15}"
print(hdr)
print("-"*len(hdr))
for name, vals in metrics.items():
    print(f"{name:<10} {vals['voxels']:>10} {vals['volume_mm3']:>15.2f} "
          f"{vals['fraction']:>10.4f} {vals['mean_intensity']:>15.2f}")

# --- DICE SIMILARITY ---
print("\n Dice Similarity Coefficients:\n")
# Between each threshold and gold
for name in masks:
    d = dice_coefficient(mask_data[name], mask_data['gold'])
    print(f"Dice({name} vs gold): {d:.4f}")
# Among thresholds
names = list(masks.keys())
for i in range(len(names)):
    for j in range(i+1, len(names)):
        a, b = names[i], names[j]
        d = dice_coefficient(mask_data[a], mask_data[b])
        print(f"Dice({a} vs {b}): {d:.4f}")

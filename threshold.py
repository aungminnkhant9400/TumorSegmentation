import numpy as np, nibabel as nib

# Load
img  = nib.load('spect.nii.gz')
data = img.get_fdata()
nz   = data[data>0]

# Voxel volume in mm³ (or cm³)
zoom   = img.header.get_zooms()            # e.g. (2.0, 2.0, 2.0)
voxvol = np.prod(zoom)                     # mm³

for p in [90, 95, 99]:
    thresh = np.percentile(nz, p)
    mask   = (data > thresh).astype(np.uint8)

    # Compute volume in mL:
    vol_ml = mask.sum() * voxvol / 1000.0

    # Save
    out_name = f'mask_pct{p}.nii.gz'
    nib.save(nib.Nifti1Image(mask, img.affine), out_name)

    print(f"{p} percentile: thresh={thresh:.0f}, volume={vol_ml:.1f} mL, saved → {out_name}")

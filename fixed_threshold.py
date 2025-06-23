#!/usr/bin/env python3
import nibabel as nib
import numpy as np

def fixed_threshold(input_file: str, output_file: str):
    """
    1. Load the SPECT volume (.nii.gz)
    2. Compute a single global threshold (mean intensity of the central slice)
    3. Apply that threshold to the entire 3D volume
    4. Save the resulting binary mask as a NIfTI file
    """
    # 1) Load
    img  = nib.load(input_file)
    data = img.get_fdata()
    aff  = img.affine
    hdr  = img.header

    # 2) Compute threshold from middle slice
    mid_z = data.shape[2] // 2
    mid_slice = data[:, :, mid_z]
    thresh = mid_slice.mean()
    print(f"Fixed threshold (mean of slice {mid_z}): {thresh:.2f}")

    # 3) Apply globally
    mask = (data > thresh).astype(np.uint8)

    # 4) Save mask
    out = nib.Nifti1Image(mask, aff, hdr)
    nib.save(out, output_file)
    print(f"Saved fixed‐threshold mask → {output_file}")

if __name__ == "__main__":
    fixed_threshold("spect.nii.gz", "fixed_seg.nii.gz")

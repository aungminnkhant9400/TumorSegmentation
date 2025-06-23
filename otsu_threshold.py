#!/usr/bin/env python3
import nibabel as nib
import numpy as np
from skimage.filters import threshold_otsu

def otsu_segmentation(input_file: str, output_file: str):
    """
    1. Load the compressed SPECT volume (.nii.gz)
    2. Compute Otsu’s threshold on the middle slice
    3. Apply that global threshold to the entire volume
    4. Save the binary mask as a new NIfTI
    """
    # 1. Load
    img  = nib.load(input_file)
    data = img.get_fdata()
    aff  = img.affine
    hdr  = img.header

    # 2. Compute Otsu on the central slice
    mid_idx = data.shape[2] // 2
    mid_slice = data[:, :, mid_idx]
    thresh = threshold_otsu(mid_slice)
    print(f"Otsu threshold (mid‐slice): {thresh:.2f}")

    # 3. Apply globally
    mask = (data > thresh).astype(np.uint8)

    # 4. Save result
    out = nib.Nifti1Image(mask, aff, hdr)
    nib.save(out, output_file)
    print(f"Saved Otsu‐threshold mask → {output_file}")

if __name__ == "__main__":
    otsu_segmentation("spect.nii.gz", "otsu_seg.nii.gz")

#!/usr/bin/env python3
import nibabel as nib
import numpy as np

from skimage.filters import gaussian, threshold_local
from skimage.morphology import remove_small_objects, closing, ball
from skimage.measure import label

def adaptive_tumor_segmentation(
    in_nii: str,
    out_nii: str,
    sigma: float = 1.0,
    block_size: int = 81,
    offset: float = 25.0,
    min_size: int = 2000,
    closing_radius: int = 3,
    roi_slices: tuple[int,int] = None
):
    # Load
    img    = nib.load(in_nii)
    data   = img.get_fdata()
    affine = img.affine
    header = img.header

    # Smooth
    smoothed = gaussian(data, sigma=sigma, preserve_range=True)

    # Adaptive threshold (slice-wise)
    mask = np.zeros_like(smoothed, dtype=bool)
    for z in range(data.shape[2]):
        slice_img = smoothed[:, :, z]
        thr       = threshold_local(slice_img, block_size, offset=offset)
        mask[:, :, z] = slice_img > thr

    # Cleanup
    mask = remove_small_objects(mask, min_size=min_size)
    mask = closing(mask, footprint=ball(closing_radius))

    # Restrict to ROI
    if roi_slices:
        z1, z2 = roi_slices
        mask[:, :, :z1] = False
        mask[:, :, z2:] = False

    # Keep largest component
    lbls   = label(mask)
    counts = np.bincount(lbls.flat)
    if counts.size > 1:
        largest    = counts[1:].argmax() + 1
        final_mask = (lbls == largest)
    else:
        final_mask = mask

    # Save
    nib.save(nib.Nifti1Image(final_mask.astype(np.uint8), affine, header), out_nii)
    print(f"Saved adaptive mask → {out_nii}")

if __name__ == "__main__":
    adaptive_tumor_segmentation(
        in_nii="spect.nii.gz",
        out_nii="adaptive_clean.nii.gz",
        sigma=1.0,
        block_size=81,
        offset=25.0,
        min_size=2000,
        closing_radius=3,
        roi_slices=(110, 135),
    )

import nibabel as nib
import numpy as np
import os

# Load the 4 modalities
modalities = ["T1", "T1c", "T2", "FLAIR"]
imgs = [nib.load(f"Data/{m}.nii.gz") for m in modalities]
data = [img.get_fdata() for img in imgs]

# Stack along last axis (H, W, D, C)
stacked = np.stack(data, axis=-1)  # shape: (H, W, D, 4)

# Use affine from first images
affine = imgs[0].affine

# Save combined NIfTI
combined_path = "combined_4mod.nii.gz"
combined_img = nib.Nifti1Image(stacked, affine)
nib.save(combined_img, combined_path)
print("Combined NIfTI saved:", combined_path)

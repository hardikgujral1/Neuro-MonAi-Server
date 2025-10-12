import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Load segmentation
seg = nib.load("tmp9jqd369z.nii.gz").get_fdata()

# Pick a slice to visualize
slice_index = seg.shape[2] // 2  # middle slice
plt.imshow(seg[:, :, slice_index], cmap='gray')
plt.title("Segmentation Slice")
plt.show()

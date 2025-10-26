import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nibabel.affines import apply_affine

# -------------------------
# Parameters
# -------------------------
view = 'axial'  # options: 'axial', 'coronal', 'sagittal'

# -------------------------
# Load MRI and tumor mask
# -------------------------
mri_path = "combined_4mod.nii.gz"
mask_path = "tmp__62r7_2.nii.gz"

mri_nii = nib.load(mri_path)
mask_nii = nib.load(mask_path)

mri_data = mri_nii.get_fdata()
tumor_data = mask_nii.get_fdata()
affine = mri_nii.affine

if mri_data.ndim == 4:
    mri_data = mri_data[:,:,:,0]

# -------------------------
# Select plane for projection
# -------------------------
if view == 'axial':      # top-down
    brain_proj = np.max(mri_data, axis=2)
    tumor_proj = np.max(tumor_data, axis=2) > 0
    plane_axes = (0,1)
    slice_axis = 2
elif view == 'coronal':  # front view
    brain_proj = np.max(mri_data, axis=1)
    tumor_proj = np.max(tumor_data, axis=1) > 0
    plane_axes = (0,2)
    slice_axis = 1
elif view == 'sagittal': # side view
    brain_proj = np.max(mri_data, axis=0)
    tumor_proj = np.max(tumor_data, axis=0) > 0
    plane_axes = (1,2)
    slice_axis = 0
else:
    raise ValueError("View must be 'axial', 'coronal', or 'sagittal'")

# -------------------------
# Brain center (midlines)
# -------------------------
center_voxel = np.array(mri_data.shape)[list(plane_axes)] / 2
vertical_line = center_voxel[1]
horizontal_line = center_voxel[0]

# -------------------------
# Tumor extreme points
# -------------------------
tumor_voxels_3d = np.argwhere(tumor_data > 0)

# Project to selected plane
tumor_voxels = tumor_voxels_3d[:, plane_axes]

left_most_voxel   = tumor_voxels[np.argmin(tumor_voxels[:,1])]
right_most_voxel  = tumor_voxels[np.argmax(tumor_voxels[:,1])]
top_most_voxel    = tumor_voxels[np.argmin(tumor_voxels[:,0])]
bottom_most_voxel = tumor_voxels[np.argmax(tumor_voxels[:,0])]

# Convert to mm
def voxel_to_mm(voxel, slice_axis):
    coords = [0,0,0]
    coords[plane_axes[0]] = voxel[0]
    coords[plane_axes[1]] = voxel[1]
    coords[slice_axis] = mri_data.shape[slice_axis]//2
    return apply_affine(affine, coords)

left_mm   = voxel_to_mm(left_most_voxel, slice_axis)
right_mm  = voxel_to_mm(right_most_voxel, slice_axis)
top_mm    = voxel_to_mm(top_most_voxel, slice_axis)
bottom_mm = voxel_to_mm(bottom_most_voxel, slice_axis)
center_mm = voxel_to_mm(center_voxel, slice_axis)

# Distances from midlines in mm
dist_left   = np.abs(left_mm[plane_axes[1]] - center_mm[plane_axes[1]])
dist_right  = np.abs(right_mm[plane_axes[1]] - center_mm[plane_axes[1]])
dist_top    = np.abs(top_mm[plane_axes[0]] - center_mm[plane_axes[0]])
dist_bottom = np.abs(bottom_mm[plane_axes[0]] - center_mm[plane_axes[0]])

print(f"Left-most distance from midline: {dist_left:.2f} mm")
print(f"Right-most distance from midline: {dist_right:.2f} mm")
print(f"Top-most distance from midline: {dist_top:.2f} mm")
print(f"Bottom-most distance from midline: {dist_bottom:.2f} mm")

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(8,8))
plt.imshow(brain_proj.T, cmap='Greys', origin='lower', alpha=0.5)
plt.contour(tumor_proj.T, colors='red', linewidths=2)

# Draw midlines
plt.axhline(y=vertical_line, color='blue', linestyle='--', label='Vertical Midline')
plt.axvline(x=horizontal_line, color='green', linestyle='--', label='Horizontal Midline')

# Mark tumor extreme points
plt.scatter(left_most_voxel[0], left_most_voxel[1], color='cyan', s=50, label='Left-most')
plt.scatter(right_most_voxel[0], right_most_voxel[1], color='magenta', s=50, label='Right-most')
plt.scatter(top_most_voxel[0], top_most_voxel[1], color='yellow', s=50, label='Top-most')
plt.scatter(bottom_most_voxel[0], bottom_most_voxel[1], color='orange', s=50, label='Bottom-most')

# Lines to midlines
plt.plot([left_most_voxel[0], left_most_voxel[0]], [left_most_voxel[1], vertical_line], color='cyan')
plt.plot([right_most_voxel[0], right_most_voxel[0]], [right_most_voxel[1], vertical_line], color='magenta')
plt.plot([top_most_voxel[0], horizontal_line], [top_most_voxel[1], top_most_voxel[1]], color='yellow')
plt.plot([bottom_most_voxel[0], horizontal_line], [bottom_most_voxel[1], bottom_most_voxel[1]], color='orange')

plt.title(f'{view.capitalize()} View: Tumor Extreme Points and Distances')
plt.xlabel('X axis (voxels)')
plt.ylabel('Y axis (voxels)')
plt.legend()
plt.show()

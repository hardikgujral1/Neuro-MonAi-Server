import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

# === Load MRI and segmentation ===
combined_path = "combined_4mod.nii.gz"
seg_path = "tmp__62r7_2.nii.gz"

combined = nib.load(combined_path).get_fdata()
seg = nib.load(seg_path).get_fdata()

# === Pick a single modality for brain outline ===
# Usually FLAIR (index 3) or T1c (index 1)
brain_vol = combined[..., 3]

# === Extract surfaces using marching cubes ===
# Brain outline (light gray, low opacity)
verts_brain, faces_brain, _, _ = measure.marching_cubes(brain_vol, level=np.mean(brain_vol))

# Tumor surface (solid red)
verts_tumor, faces_tumor, _, _ = measure.marching_cubes(seg, level=0.5)

# === Create 3D figure ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Brain outline (wireframe-like)
ax.plot_trisurf(verts_brain[:, 0], verts_brain[:, 1], faces_brain, verts_brain[:, 2],
                color='lightgray', alpha=0.1, edgecolor='gray', linewidth=0.2)

# Tumor surface (solid)
ax.plot_trisurf(verts_tumor[:, 0], verts_tumor[:, 1], faces_tumor, verts_tumor[:, 2],
                color='red', alpha=0.9)

# === Adjust 3D view ===
ax.set_box_aspect([1, 1, 1])  # Equal scaling
ax.view_init(elev=20, azim=130)
ax.set_title("3D Brain Outline and Tumor", fontsize=14)
ax.set_xlabel("X-axis (sagittal)")
ax.set_ylabel("Y-axis (coronal)")
ax.set_zlabel("Z-axis (axial)")
plt.tight_layout()
plt.show()

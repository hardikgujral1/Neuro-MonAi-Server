import nibabel as nib
import numpy as np

# === Load segmentation ===
seg_path = "tmp__62r7_2.nii.gz"
seg_img = nib.load(seg_path)
seg_data = seg_img.get_fdata()
affine = seg_img.affine

# === Get voxel spacing (mm) ===
voxel_dims = np.abs(seg_img.header.get_zooms())  # (x, y, z)
voxel_volume = np.prod(voxel_dims)

# === Tumor voxel count and total volume ===
tumor_voxels = np.sum(seg_data > 0)
tumor_volume_mm3 = tumor_voxels * voxel_volume
tumor_volume_ml = tumor_volume_mm3 / 1000

print(f"Tumor Volume: {tumor_volume_mm3:.2f} mm³ ({tumor_volume_ml:.2f} mL)")

# === Tumor coordinates ===
coords = np.array(np.where(seg_data > 0))  # shape (3, N)

# Bounding box (voxel space)
min_coords_voxel = coords.min(axis=1)
max_coords_voxel = coords.max(axis=1)

# Convert to world (mm) space using affine
min_coords_mm = nib.affines.apply_affine(affine, min_coords_voxel)
max_coords_mm = nib.affines.apply_affine(affine, max_coords_voxel)

# === Tumor physical dimensions (in mm) ===
size_mm = np.abs(max_coords_mm - min_coords_mm)
max_diameter = np.linalg.norm(size_mm)

print(f"Tumor Dimensions (mm): X={size_mm[0]:.2f}, Y={size_mm[1]:.2f}, Z={size_mm[2]:.2f}")
print(f"Approx. Max Diameter: {max_diameter:.2f} mm")

# === Tumor centroid (voxel + mm) ===
centroid_voxel = coords.mean(axis=1)
centroid_mm = nib.affines.apply_affine(affine, centroid_voxel)

print(f"Centroid (voxel): {centroid_voxel.round(2)}")
print(f"Centroid (mm): {centroid_mm.round(2)}")

# === Image center (voxel + mm) ===
image_center_voxel = np.array(seg_data.shape) / 2
image_center_mm = nib.affines.apply_affine(affine, image_center_voxel)
offset_mm = centroid_mm - image_center_mm

# === Determine orientation axes (e.g., ('L', 'P', 'S')) ===
axes = nib.orientations.aff2axcodes(affine)
x_axis, y_axis, z_axis = axes

# === Anatomical direction based on offset ===
def get_direction(offset, axis_code):
    if axis_code in ("R", "L"):
        return "Right" if offset > 0 and axis_code == "R" else "Left" if offset > 0 else "Right"
    elif axis_code in ("A", "P"):
        return "Posterior (Back)" if offset > 0 and axis_code == "P" else "Anterior (Front)" if offset > 0 else "Posterior (Back)"
    elif axis_code in ("S", "I"):
        return "Superior (Top)" if offset > 0 and axis_code == "S" else "Inferior (Bottom)" if offset > 0 else "Superior (Top)"
    else:
        return "Unknown"

side_x = get_direction(offset_mm[0], x_axis)
side_y = get_direction(offset_mm[1], y_axis)
side_z = get_direction(offset_mm[2], z_axis)

print(f"Tumor likely located on: {side_x}, {side_y}, {side_z}")
print(f"Offset from brain center (mm): {offset_mm.round(2)}")

# === Orientation info ===
print("\nImage Orientation (affine axes):", axes)

# === Optional: Output bounding box in mm (for 3D planning) ===
print(f"\nBounding Box (mm):\n  Min: {min_coords_mm.round(2)}\n  Max: {max_coords_mm.round(2)}")

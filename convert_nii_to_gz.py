import os
import nibabel as nib

folder_path = "Data" 

for filename in os.listdir(folder_path):
    if filename.endswith(".nii") and not filename.endswith(".nii.gz"):
        nii_path = os.path.join(folder_path, filename)
        print(f"Processing: {nii_path}")

        img = nib.load(nii_path)

        gz_path = nii_path + ".gz"
        nib.save(img, gz_path)
        print(f"Saved: {gz_path}")

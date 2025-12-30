import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Load segmentation
seg = nib.load("tmp_nrjndwp.nii.gz").get_fdata()

# Initial slice
initial_slice = seg.shape[2] // 2

# Create figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)  # Leave space for the slider

# Display initial slice
img_display = ax.imshow(seg[:, :, initial_slice], cmap="inferno")
ax.set_title(f"Segmentation Slice: {initial_slice}")

# Define slider axis and create slider
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])  # [left, bottom, width, height]
slice_slider = Slider(ax_slider, 'Slice', 0, seg.shape[2] - 1, 
                      valinit=initial_slice, valstep=1)

# Update function
def update(val):
    slice_idx = int(slice_slider.val)
    img_display.set_data(seg[:, :, slice_idx])
    ax.set_title(f"Segmentation Slice: {slice_idx}")
    fig.canvas.draw_idle()

# Connect slider to update function
slice_slider.on_changed(update)

plt.show()

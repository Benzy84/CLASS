import os
import shutil
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import torch
import torch.fft as fft
from tqdm import tqdm

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_array_or_image(file_path):
    if file_path.endswith('.npy'):
        return torch.from_numpy(np.load(file_path)).to(device)
    else:
        image = Image.open(file_path)
        if image.mode == 'I;16':
            image_array = np.array(image, dtype=np.uint16)
            image_array = image_array.astype(np.int32)
        else:
            image_array = np.array(image)
        return torch.from_numpy(image_array).to(device)

# Ask the user to select the directories containing the reference fields and fields
root = tk.Tk()
root.withdraw()  # Hide the root window
ref_path = filedialog.askdirectory(title='Select References Images Folder')
fields_path = filedialog.askdirectory(title='Select Fields Folder')

# List the image files in the selected directories
ref_images_names = [f for f in os.listdir(ref_path) if f.endswith(('.tif', '.png'))]
fields_names = [f for f in os.listdir(fields_path) if f.endswith('.npy')]

# Load fields and reference images
fields = [load_array_or_image(os.path.join(fields_path, fields_name)) for fields_name in fields_names]
field_shape = fields[0].shape[0]
ref_images = [load_array_or_image(os.path.join(ref_path, img_name)) for img_name in ref_images_names]

# Convert the list of reference images to a tensor of shape (N, H, W)
ref_images_tensor = torch.stack(ref_images)
del ref_images

# Perform FFT on reference images
ref_fft = fft.fftshift(fft.fft2(ref_images_tensor))

# Crop FFT of reference images to match the field shape
start_y = (ref_fft.shape[1] - field_shape) // 2
end_y = start_y + field_shape
start_x = (ref_fft.shape[2] - field_shape) // 2
end_x = start_x + field_shape
cropped_ref_fft = ref_fft[:, start_y:end_y, start_x:end_x]

# Reconstruct the reference fields and compute their intensities
ref_fields = fft.ifft2(fft.ifftshift(cropped_ref_fft))
ref_intensities = torch.abs(ref_fields) ** 2

# Calculate the mean of the reference fields
mean_ref_intensity = torch.mean(ref_intensities, dim=0)

# Normalize each field by the mean reference field
normalized_fields = [field / torch.sqrt(mean_ref_intensity) for field in fields]

# Define directories for saving original and normalized fields
original_fields_dir = os.path.join(fields_path, 'original_fields')
normalized_fields_dir = os.path.join(fields_path, 'normalized_fields')

# Create directories if they don't exist
os.makedirs(original_fields_dir, exist_ok=True)
os.makedirs(normalized_fields_dir, exist_ok=True)

# Move original fields to the 'original_fields' directory
for field_name in fields_names:
    shutil.move(os.path.join(fields_path, field_name), os.path.join(original_fields_dir, field_name))

# Save normalized fields to the 'normalized_fields' directory with updated filenames
for field, field_name in zip(normalized_fields, fields_names):
    normalized_field_name = f"normalized_{field_name}"
    torch.save(field.cpu(), os.path.join(normalized_fields_dir, normalized_field_name))

print(f"Original fields moved to {original_fields_dir}")
print(f"Normalized fields saved to {normalized_fields_dir}")
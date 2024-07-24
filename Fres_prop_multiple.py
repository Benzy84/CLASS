import tkinter as tk
from tkinter import filedialog
import numpy as np
import os
import torch
import torch.fft
from tqdm import tqdm
import matplotlib.pyplot as plt


def fresnel_propagation(initial_field, dx, dy, z, wavelength):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convert initial field to PyTorch tensor and move to GPU if available
    # field_fft = np.fft.fft2(initial_field)
    # padding_size = 1000  # array.shape[0]//8
    # padded_field_fft = np.pad(field_fft, ((padding_size, padding_size), (padding_size, padding_size)), mode='constant')
    # padded_field = np.fft.ifft2(padded_field_fft)
    # new_dx = dx*(initial_field.shape[0]/(initial_field.shape[0] + 2 * padding_size))
    field = torch.from_numpy(initial_field).to(device).to(dtype=torch.complex64)
    [Nx, Ny] = field.shape
    k = 2 * np.pi / wavelength
    # Create coordinate grids
    fx = torch.fft.fftfreq(Nx, dx).to(device)
    fy = torch.fft.fftfreq(Ny, dx).to(device)
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')
    # Fresnel kernel
    # Ensure z is a tensor for compatibility
    z_tensor = torch.tensor(z, device=device, dtype=torch.float32)
    H = torch.exp(torch.tensor(-1j * k, device=device) * z_tensor) * torch.exp(
        torch.tensor(-1j * np.pi * wavelength, device=device) * z_tensor * (FX ** 2 + FY ** 2)
    )
    # Propagated field
    field_propagated = torch.fft.ifft2(torch.fft.fft2(field) * H)
    return field_propagated.cpu().numpy()  # Move result back to CPU and convert to numpy array

# Create a Tkinter root window and hide it
root = tk.Tk()
root.withdraw()

# Ask the user to select the folder containing the .npy files
folder_path = filedialog.askdirectory(title='Select the folder containing the .npy files')
if not os.listdir(folder_path):  # Check if folder is empty
    print("Selected folder is empty. Please select a folder with .npy files.")
    exit()

# Get a list of all .npy files in the selected folder
npy_files = [file for file in os.listdir(folder_path) if file.endswith('.npy')]

# Define the list of z values in meter
z_min = 10e-3
z_max = 30e-3
num_z = 11
z_values = np.linspace(z_min, z_max, num_z)
flag = True

# Create folders for each z value
for i, z in enumerate(z_values, start=1):
    z_mm = z * 1000
    folder_name = f'{i:02d}_z_{z_mm:.2f}mm'  # Format z value as millimeters
    folder_path_z = os.path.join(folder_path, folder_name)
    os.makedirs(folder_path_z, exist_ok=True)

# proped_path = filedialog.askopenfilename(title='Select the propagated field')
# proped_field = np.load(proped_path)

# Propagate fields for each file
total_files = len(npy_files)
for file in tqdm(npy_files, desc='Processing files', unit='file', total=total_files):
    file_path = os.path.join(folder_path, file)
    field = np.load(file_path)


    if flag:
        system_mag = 1.5
        original_shape = 2472
        blob_shape = field.shape[0]
        mag = system_mag * blob_shape / original_shape
        dx = dy = 5.5e-6 / mag  # Pixel size in x-direction (m)
        print(f'dx is :{dx} microns')
        wavelength = 632.8e-9  # Wavelength (m)
        flag = False

    for z in z_values:
        propagated_field = fresnel_propagation(field, dx, dy, z, wavelength)
        # propagated_field = propagated_field * proped_field
        z_mm = z * 1000
        folder_name = f'{z_values.tolist().index(z) + 1:02d}_z_{z_mm:.2f}mm'  # Format z value as millimeters
        folder_path_z = os.path.join(folder_path, folder_name)
        output_file = os.path.join(folder_path_z, file)
        np.save(output_file, propagated_field)

print("Field propagation completed.")



a=5
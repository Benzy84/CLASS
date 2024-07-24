import os
import re
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.ion()

# Function to load a .npy file or convert an image to a numpy array
def load_array_or_image(file_path):
    if file_path.endswith('.npy'):
        return np.load(file_path)
    else:
        # Load the image and convert to a numpy array
        image = Image.open(file_path)
        return np.array(image)

# Natural sort key function
def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

# Function to load and convert all files in a directory
def load_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith(('.npy', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    files.sort(key=natural_sort_key)
    data = []
    for file in files:
        file_path = os.path.join(directory, file)
        array = load_array_or_image(file_path)
        if array.dtype not in [np.float64, np.float32, np.float16, np.complex64, np.complex128]:
            array = array.astype(np.float32)
        data.append(array)
    return data

def create_cov_matrices(fields):
    # Normalize each field such that the sum of squared magnitudes is 1
    normalized_fields = [field / torch.sqrt(torch.sum(torch.abs(field) ** 2)) for field in fields]

    # Center the magnitudes to see their correlation
    centered_abs_fields = [torch.abs(field) - torch.mean(torch.abs(field)) for field in normalized_fields]
    # Introduce negative correlation
    # centered_abs_fields[150] = - centered_abs_fields[250]
    # Normalize the centered magnitudes
    normalized_abs_fields = [field / torch.sqrt(torch.sum(field ** 2)) for field in centered_abs_fields]

    # Flatten each normalized field to 1D and stack them into a single tensor
    flattened_fields = [field.flatten() for field in normalized_fields]
    fields_tensor = torch.stack(flattened_fields, dim=1).type(torch.cfloat)

    # Flatten each centered and normalized magnitude field to 1D and stack them into a single tensor
    flattened_abs_fields = [field.flatten() for field in normalized_abs_fields]
    abs_fields_tensor = torch.stack(flattened_abs_fields, dim=1).type(torch.float)

    # Calculate the covariance matrix using the conjugate transpose for complex fields
    cov_matrix = fields_tensor.T.conj() @ fields_tensor
    # Calculate the covariance matrix for the magnitudes
    cov_abs_matrix = abs_fields_tensor.T @ abs_fields_tensor

    # Output the covariance matrix
    cov_matrix_numpy = cov_matrix.numpy()
    cov_abs_matrix_numpy = cov_abs_matrix.numpy()

    return cov_matrix_numpy, cov_abs_matrix_numpy

def create_saturation_vec (fields):
    saturation_counts = np.array([])
    means = np.array([])
    for field in fields:
        field = field / torch.max(torch.abs(field))
        saturation_count = torch.sum(torch.abs(field) == torch.max(torch.abs(field)))
        saturation_counts = np.append(saturation_counts, saturation_count)
        mean = torch.mean(torch.abs(field))
        means = np.append(means, mean)
    return saturation_counts, means


# Ask for directories of the two sets
root = tk.Tk()
root.withdraw()
pth1_images = filedialog.askdirectory(title='Select Folder for Set 1 Images')
pth1_npy = filedialog.askdirectory(title='Select Folder for Set 1 NPY Files')
pth2_images = filedialog.askdirectory(title='Select Folder for Set 2 Images')
pth2_npy = filedialog.askdirectory(title='Select Folder for Set 2 NPY Files')


# Load files for both sets and subsets
data_set1_images = load_files(pth1_images)
data_set1_npy = load_files(pth1_npy)
data_set2_images = load_files(pth2_images)
data_set2_npy = load_files(pth2_npy)

data_set1_images = [torch.from_numpy(arr) for arr in data_set1_images]
data_set1_npy = [torch.from_numpy(arr) for arr in data_set1_npy]
data_set2_images = [torch.from_numpy(arr) for arr in data_set2_images]
data_set2_npy = [torch.from_numpy(arr) for arr in data_set2_npy]


cov_matrix_set1_images, cov_abs_matrix_set1_images = create_cov_matrices(data_set1_images)
cov_matrix_set1_npy, cov_abs_matrix_set1_npy = create_cov_matrices(data_set1_npy)
cov_matrix_set2_images, cov_abs_matrix_set2_images = create_cov_matrices(data_set2_images)
cov_matrix_set2_npy, cov_abs_matrix_set2_npy = create_cov_matrices(data_set2_npy)

saturation_counts_set1_images, means_set1_images = create_saturation_vec(data_set1_images)
saturation_counts_set1_npy, means_set1_npy = create_saturation_vec(data_set1_npy)
saturation_counts_set2_images, means_set2_images = create_saturation_vec(data_set2_images)
saturation_counts_set2_npy, means_set2_npy = create_saturation_vec(data_set2_npy)

plt.figure()
plt.subplot(2,2,1)
plt.plot(saturation_counts_set1_images)
plt.subplot(2,2,2)
plt.plot(saturation_counts_set1_npy)
plt.subplot(2,2,3)
plt.plot(saturation_counts_set2_images)
plt.subplot(2,2,4)
plt.plot(saturation_counts_set1_npy)
plt.show()


plt.figure()
plt.subplot(2,2,1)
plt.plot(means_set1_images)
plt.subplot(2,2,2)
plt.plot(means_set1_npy)
plt.subplot(2,2,3)
plt.plot(means_set2_images)
plt.subplot(2,2,4)
plt.plot(means_set1_npy)
plt.show()



# Now data_set1_images, data_set1_npy, data_set2_images, and data_set2_npy contain numpy arrays of all files
print("Data Set 1 Images Loaded:", len(data_set1_images), "files.")
print("Data Set 1 NPY Files Loaded:", len(data_set1_npy), "files.")
print("Data Set 2 Images Loaded:", len(data_set2_images), "files.")
print("Data Set 2 NPY Files Loaded:", len(data_set2_npy), "files.")

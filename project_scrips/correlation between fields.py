import os
import re
import numpy as np
import torch
from tkinter import filedialog, Tk
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.ion()

def natural_sort_key(s):
    # Function to extract numbers from the string for natural sorting
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

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


# Hide the main Tkinter window
root = Tk()
root.withdraw()

# Ask for the directory
pth = filedialog.askdirectory(title='Select Folder')

# Collect the files and sort them using natural sort
files = [f for f in os.listdir(pth) if f.endswith('.npy')]
files.sort(key=natural_sort_key)
# files = files[1::10]


# Collect the fields
fields = [torch.from_numpy(np.load(os.path.join(pth, f))) for f in files]
# fields = fields[1::2]

# random_fields = []
# for _ in files:
#     magnitude = np.random.rand(500, 500)
#     phase = np.random.uniform(-np.pi, np.pi, (500, 500))
#     complex_matrix = magnitude * np.exp(1j * phase)
#     random_fields.append(torch.from_numpy(complex_matrix))
# fields = random_fields
# fields[100] = - fields[200]

cov_matrix_numpy, cov_abs_matrix_numpy = create_cov_matrices(fields)

# Plot the correlation matrix (which is actually the covariance matrix in this case)
min_cbar = np.min(cov_abs_matrix_numpy)
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.imshow(np.abs(cov_matrix_numpy), cmap='gray', interpolation='nearest', vmin=0.5, vmax=1)
plt.colorbar()
plt.title("Correlation Matrix (Abs)")
plt.xlabel("Fields")
plt.ylabel("Fields")

plt.subplot(1, 2, 2)
plt.imshow(cov_abs_matrix_numpy, cmap='gray', interpolation='nearest', vmin=0.5, vmax=1)
plt.colorbar()
plt.title("Correlation Matrix of \n Absolute Value of Fields")
plt.xlabel("Fields")
plt.ylabel("Fields")

plt.show()

line_index = len(fields) // 2
line = cov_abs_matrix_numpy[line_index, :]
mean = np.mean(line)
std = np.std(line)
# Find indices of unusual values
unusual_indices = np.where(np.abs(line - mean) > 2 * std)[0]
unusual_indices = unusual_indices[unusual_indices != line_index]
# Remove the unusual fields from the original vector
filtered_fields = [fields[i] for i in range(len(fields)) if i not in unusual_indices]
filtered_files = [files[i] for i in range(len(fields)) if i not in unusual_indices]
# Re-analyze the filtered fields
filtered_cov_matrix_numpy, filtered_cov_abs_matrix_numpy = create_cov_matrices(filtered_fields)
# Plot the filtered correlation matrix

min_cbar = np.min(filtered_cov_abs_matrix_numpy)
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.imshow(np.abs(filtered_cov_matrix_numpy), cmap='gray', interpolation='nearest', vmin=0, vmax=1)
plt.colorbar()
plt.title("Filtered Correlation Matrix (Abs)")
plt.xlabel("Fields")
plt.ylabel("Fields")

plt.subplot(1, 2, 2)
plt.imshow(filtered_cov_abs_matrix_numpy, cmap='gray', interpolation='nearest', vmin=min_cbar, vmax=1)
plt.colorbar()
plt.title("Filtered Correlation Matrix of \n Absolute Value of Fields")
plt.xlabel("Fields")
plt.ylabel("Fields")

plt.show()

# Find the sum of absolute correlations for each field
sum_corr = np.sum(np.abs(filtered_cov_matrix_numpy), axis=1) - 1  # Subtract 1 to exclude self-correlation

# Select the indices of the fields with the lowest sum of correlations
N = 180  # Number of fields to select
selected_indices = np.argsort(sum_corr)[:N]
selected_indices = np.asarray(selected_indices, dtype=int)

# Extract the selected uncorrelated fields in the original order
selected_indices_sorted = np.sort(selected_indices)
selected_fields_tensor = filtered_fields[selected_indices_sorted]

# Save the selected fields to a new folder with a name depending on N
output_folder = os.path.join(pth, f'most_{N}_uncorrelated_fields')
os.makedirs(output_folder, exist_ok=True)

for idx in selected_indices_sorted:
    original_file = filtered_files[idx]
    field = filtered_fields[idx]
    output_file = os.path.join(output_folder, f'selected_{original_file}')
    np.save(output_file, field)

# Plot the correlation matrix of the selected fields
selected_cov_matrix = selected_fields_tensor.T.conj() @ selected_fields_tensor
selected_corr_matrix = selected_cov_matrix.numpy()

plt.figure()
plt.imshow(np.abs(selected_corr_matrix), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Correlation Matrix of Selected Fields")
plt.xlabel("Fields")
plt.ylabel("Fields")
plt.show()


a=5
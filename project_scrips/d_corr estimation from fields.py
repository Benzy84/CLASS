import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from tkinter import filedialog
import scipy.io
import re
import matplotlib.colors as mcolors


# Function to extract the frame number from a filename
def extract_frame_number(filename):
    match = re.search(r'frame_(\d+(\.\d+)?)', filename)  # This regex matches numbers after 'frame_' including decimals
    if match:
        return float(match.group(1))  # Convert the number to float to handle decimals
    return float('inf')  # Return infinity if no number is found, so these files sort last

pth = filedialog.askdirectory(title='Select Folder')


# Load .npy files, ensuring they are sorted numerically
npy_files = [f for f in os.listdir(pth) if f.endswith('.npy')]
npy_files = sorted(npy_files, key=extract_frame_number)

# npy_files = npy_files[::5]  # Take every third file
arrays = [np.load(os.path.join(pth, file)) for file in npy_files]
# cropped_arrays = [arr[90:180, 90:180] for arr in arrays]
# arrays = cropped_arrays
powers = [np.sum(np.abs(arr)**2) for arr in arrays]
max_power = max(powers)
min_power = min(powers)
max_idx = np.argmax(powers)
min_idx = np.argmin(powers)
max_field_name = npy_files[max_idx]
min_field_name = npy_files[min_idx]
max_field = np.abs(arrays[max_idx])
min_field = np.abs(arrays[min_idx])
max_field_phase = np.angle(arrays[max_idx])
min_field_phase = np.angle(arrays[min_idx])
phase_difference = np.angle(np.exp(1j * (max_field_phase - min_field_phase)))

# Visualize the phase difference
plt.figure()
plt.imshow(phase_difference)  # 'twilight' is a cyclic colormap
plt.colorbar(label='Phase difference (radians)')
plt.title('Phase Difference between Max and Min Power Fields')
plt.show()

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(max_field, cmap='inferno')
plt.colorbar()
plt.title('Max Power Field')

plt.subplot(1, 2, 2)
plt.imshow(min_field, cmap='inferno')
plt.colorbar()
plt.title('Min Power Field')

plt.show()

norm_arrays_to_max = [arr * np.sqrt(max_power / np.sum(np.abs(arr)**2)) for arr in arrays]



flattened_arrays = [arr.flatten() for arr in arrays]
flattened_norm_arrays_to_max = [arr.flatten() for arr in norm_arrays_to_max]
combined_array = np.stack(flattened_arrays, axis=1)
combined_norm_array = np.stack(flattened_norm_arrays_to_max, axis=1)



# # Create a subfolder for the normalized .npy files
# norm_subfolder = os.path.join(pth, 'Normalized fields')
# if not os.path.exists(norm_subfolder):
#     os.makedirs(norm_subfolder)
#
# # Save each normalized array as a .npy file
# for i, norm_arr in enumerate(norm_arrays_to_max):
#     npy_filename = os.path.join(norm_subfolder, f'norm_frame_{i+1}.npy')
#     np.save(npy_filename, norm_arr)
#
# # Create a subfolder for the .mat files
# subfolder = os.path.join(pth, 'MatlabFiles')
# if not os.path.exists(subfolder):
#     os.makedirs(subfolder)
# # Save each normalized array as a .mat file
# for i, arr in enumerate(arrays):
#     filename = os.path.join(subfolder, f'frame_{i+1}.mat')
#     scipy.io.savemat(filename, {'frame': arr})

# # Optional: Save the combined arrays
# scipy.io.savemat(os.path.join(subfolder, 'combined_array.mat'), {'combined_array': combined_array})
# scipy.io.savemat(os.path.join(subfolder, 'combined_norm_array.mat'), {'combined_norm_array': combined_norm_array})

ac_mat = np.conj(combined_array.T) @ combined_array
ac_mat = ac_mat / np.max(np.abs(ac_mat))
ac_norm_mat = np.conj(combined_norm_array.T) @ combined_norm_array
ac_norm_mat = ac_norm_mat / np.max(np.abs(ac_norm_mat))

plt.figure()
plt.subplot(1,2,1)
plt.imshow(np.abs(ac_mat))
plt.title('cc')
plt.colorbar()

plt.subplot(1,2,2)
plt.plot(np.round(np.diag(np.abs(ac_mat)),4), 'o', markersize=1)
plt.title('Diagonal of Matrix')
plt.show(block=False)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(np.abs(ac_norm_mat))
plt.title('normed ac')
plt.colorbar()

plt.subplot(1,2,2)
plt.plot(np.round(np.diag(np.abs(ac_norm_mat)),4), 'o', markersize=1)
plt.title('Diagonal of Normalized AC Matrix')
plt.show(block=False)

a =5
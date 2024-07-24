import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch
import torch.fft
from tqdm import tqdm
from skimage.restoration import unwrap_phase

def autocorrelation(x):
    """ Compute the normalized autocorrelation of a matrix using FFT. """
    x_fft = np.fft.fft2(x)
    x_conj_fft = np.conj(x_fft)
    ac = np.fft.ifft2(x_fft * x_conj_fft).real
    ac = np.fft.fftshift(ac)  # Shift to center the zero frequency component
    ac_normalized = (ac - np.min(ac)) / (np.max(ac) - np.min(ac))  # Normalize from 0 to 1
    return ac_normalized

def find_d_corr(ac, threshold=1/np.e):  # Adjust the threshold as needed based on empirical observations
    """ Find the correlation distance where the autocorrelation drops below a given threshold. """
    center = ac.shape[0] // 2
    center_line = ac[center, :]

    mask = center_line < threshold
    indices = np.nonzero(mask)[0]

    # Filter indices to only those greater than the center (looking rightwards)
    filtered_indices = indices[indices > center]
    if len(filtered_indices) > 0:
        d_corr = filtered_indices[0] - center
        return 2 * d_corr.item()
    else:
        return None

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


def create_field(sigma, center_x, center_y, Nx, Ny):
    """
    Create a Gaussian field with given parameters.

    Parameters:
    - sigma: Standard deviation of the Gaussian (controls the width/narrowness).
    - center_x, center_y: The center of the Gaussian in the grid.
    - Nx, Ny: The size of the grid (number of points in x and y directions).

    Returns:
    - A 2D numpy array representing the Gaussian field.
    """
    x = np.linspace(0, Nx - 1, Nx)
    y = np.linspace(0, Ny - 1, Ny)
    X, Y = np.meshgrid(x, y)

    # Calculate the 2D Gaussian
    gaussian = np.exp(-(((X - center_x) ** 2) / (2 * sigma ** 2) + ((Y - center_y) ** 2) / (2 * sigma ** 2)))

    return gaussian

Nx, Ny = 300, 300  # Size of the grid
sigma = 15  # Standard deviation of the Gaussian
center_x, center_y = Nx // 2, Ny // 2  # Center of the Gaussian
gaussian_field = create_field(sigma, center_x, center_y, Nx, Ny)
# Load the initial field from a .npy file
init_dir = 'C:/Users/Owner/PycharmProjects/CLASS_Benzy/CLASS reconstructions'
root = tk.Tk()
root.withdraw()
array_1_path = filedialog.askopenfilename(initialdir=init_dir, filetypes=[("NumPy files", "*.npy")],title='Select Field 1.')
array_2_path = filedialog.askopenfilename(initialdir=init_dir, filetypes=[("NumPy files", "*.npy")],title='Select Field 2 to prop.')


array_1 = np.load(array_1_path)
array_2 = np.load(array_2_path)
array_1 /= np.max(np.abs(array_1))
array_2 /= np.max(np.abs(array_2))
# diffuzer = array_1 * np.exp(-1j * np.angle(array_2))

# plt.figure()
# # Plotting array_1
# plt.subplot(1, 3, 1)  # 1 row, 3 columns, 1st subplot
# plt.imshow(np.abs(array_1), cmap='gray')  # Use a colormap that suits your data
# plt.colorbar()
# plt.title('Array 1 Magnitude')
#
# # Plotting array_2
# plt.subplot(1, 3, 2)  # 1 row, 3 columns, 2nd subplot
# plt.imshow(np.abs(array_2), cmap='gray')
# plt.colorbar()
# plt.title('Array 2 Magnitude')
#
# # Plotting diffuzer
# plt.subplot(1, 3, 3)  # 1 row, 3 columns, 3rd subplot
# plt.imshow(np.abs(diffuzer), cmap='gray')
# plt.colorbar()
# plt.title('Diffuzer phase')
#
# plt.show()

#
# # Set the propagation parameters
# system_mag = 300/125
# original_shape = 2469
# blob_shape = array.shape[0]
# mag = system_mag * blob_shape/original_shape
# dx = dy = 5.5e-6 / mag  # Pixel size in x-direction (m)
# wavelength = 632.8e-9  # Wavelength (m)
#
# array = fresnel_propagation(array, dx, dy, 0.1139, wavelength)
# GT_array = fresnel_propagation(GT_array, dx, dy, 0.1139, wavelength)
#
# plt.figure()
# plt.imshow(np.abs(array))
# plt.title('with diffuzer')
#
# plt.figure()
# plt.imshow(np.abs(GT_array))
# plt.title('without diffuzer')
# plt.show()
#
#
# # GT_array = gaussian_field
#
# GT_phase = np.angle(GT_array)
# e2 = array * np.exp(-1j*GT_phase)
# GT_array = e2 #[270:380,230:360]
# GT_phase = np.angle(GT_array)
#
# plt.figure()
# plt.imshow(np.abs(array))
# plt.title('initial field')
# plt.show()
#
# plt.figure()
# plt.imshow(GT_phase)
# plt.title('phase diff')
# plt.colorbar()  # Add colorbar
# plt.show()
#
# #
# plt.figure()
# diffuzer_phase = np.angle(diffuzer)
# flattened_phase_array = diffuzer_phase.flatten()
# phase_std = np.std(flattened_phase_array)
# plt.hist(flattened_phase_array, bins=101)
# plt.title(phase_std)
# plt.show()

padding_size = 100
padded_array_1 = np.pad(array_1, ((padding_size, padding_size), (padding_size, padding_size)), mode='constant')
# plt.imshow(np.abs(padded_array))
# plt.show()
padded_array_2 = np.pad(array_2, ((padding_size, padding_size), (padding_size, padding_size)), mode='constant')
# U0_1 = torch.from_numpy(padded_array_1).to(torch.complex64)
# U0_2 = torch.from_numpy(padded_array_2).to(torch.complex64)



# Set the propagation parameters
system_mag = 3.2
original_shape = 2469
blob_shape = array_1.shape[0]
mag = system_mag * blob_shape/original_shape
dx = dy = 5.5e-6 / mag  # Pixel size in x-direction (m)
wavelength = 632.8e-9  # Wavelength (m)



# Compute the field at different z-positions
z_min = -7e-2
z_max = 7e-2
num_z = 1401
z_values = np.linspace(z_min, z_max, num_z)
z_values_rounded = np.round(z_values, 5)

# z_values = np.concatenate(([0], z_values))


# Initialize lists to store the computed fields
array_1_all = []
array_2_all = []
diffuzer_all = []
for z in tqdm(z_values, desc='Computing fields'):
    field_1 = fresnel_propagation(padded_array_1, dx, dy, z, wavelength)
    field_2 = fresnel_propagation(padded_array_2, dx, dy, z, wavelength)
    field_1 /= np.max(np.abs(field_1))
    field_2 /= np.max(np.abs(field_2))
    # current_diffuzer = field_1 * np.exp(-1j * np.angle(field_2))
    array_1_all.append(field_1)
    array_2_all.append(field_2)
    # diffuzer_all.append(current_diffuzer)


# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(bottom=0.3)

# Plot the initial field in the first subplot
im_1 = ax1.imshow(np.abs(array_1), cmap='hot')
ax1.set_title('Propagated Field 1')
plt.colorbar(im_1, ax=ax1)

# Plot the ground truth in the second subplot
im_2 = ax2.imshow(np.abs(array_2), cmap='hot')
ax2.set_title('Propagated Field 2')
plt.colorbar(im_2, ax=ax2)

# Create the sliders
ax_slider1 = plt.axes([0.25, 0.2, 0.5, 0.03])
slider1 = Slider(
    ax_slider1, 'z (m) - Propagated', z_min, z_max, valinit=0, valstep=(z_max - z_min) / (num_z - 1)
)

ax_slider2 = plt.axes([0.25, 0.1, 0.5, 0.03])
slider2 = Slider(
    ax_slider2, 'z (m) - Ground Truth', z_min, z_max, valinit=0, valstep=(z_max - z_min) / (num_z - 1)
)

# Store the current zoom level for both subplots
xlim1 = ax1.get_xlim()
ylim1 = ax1.get_ylim()
xlim2 = ax2.get_xlim()
ylim2 = ax2.get_ylim()


# Update the plot based on the slider values
def update(val):
    z_index1 = int((slider1.val - z_min) / ((z_max - z_min) / (num_z - 1)))
    im_1.set_data(np.abs(array_1_all[z_index1]))
    ax1.set_title(f'Propagated Field 1 (z = {z_values[z_index1]:.2f} m)')

    z_index2 = int((slider2.val - z_min) / ((z_max - z_min) / (num_z - 1)))
    im_2.set_data(np.abs(array_2_all[z_index2]))
    ax2.set_title(f'Propagated Field 2 (z = {z_values[z_index2]:.2f} m)')

    # Restore the previous zoom levels for both subplots
    ax1.set_xlim(xlim1)
    ax1.set_ylim(ylim1)
    ax2.set_xlim(xlim2)
    ax2.set_ylim(ylim2)

    # Redraw only the updated artists
    fig.canvas.draw_idle()


# Store the current zoom levels before updating the plot
def store_zoom_levels(event):
    global xlim1, ylim1, xlim2, ylim2
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    xlim2 = ax2.get_xlim()
    ylim2 = ax2.get_ylim()


slider1.on_changed(update)
slider2.on_changed(update)
fig.canvas.mpl_connect('button_release_event', store_zoom_levels)

# Disable automatic redrawing
plt.ioff()

# Trigger an initial plot update
update(slider1.val)

# Enable blitting for efficient updates
fig.canvas.blit(fig.bbox)

plt.show(block=False)

field_d_at_diff = array_1_all[1267]
field_m_at_diff = array_2_all[1267]
difuuser_phase = np.angle(field_d_at_diff)-np.angle(field_m_at_diff)
diffuser = np.exp(1j*difuuser_phase)
diffuser_fft = np.fft.fftshift(np.fft.fft2(diffuser))

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(np.abs(diffuser_fft))
plt.title('FFT(diffuser) magnitude')
plt.colorbar()  # Optionally add a colorbar to visualize the color mapping

plt.subplot(1, 2, 2)
plt.imshow(np.angle(diffuser_fft), cmap='hsv')
plt.title('FFT(diffuser) phase')
plt.colorbar()  # Optionally add a colorbar to visualize the color mapping
plt.show()

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(np.round(np.abs(diffuser), 5))
plt.title('diffuser magnitude')
plt.colorbar()  # Optionally add a colorbar to visualize the color mapping

plt.subplot(1, 2, 2)
plt.imshow(np.angle(diffuser), cmap='hsv')
plt.title('diffuser phase')
plt.colorbar()  # Optionally add a colorbar to visualize the color mapping
plt.show()

plt.figure()
plt.imshow(np.angle(diffuser), cmap='hsv')  # Set the colormap to 'hsv'
plt.colorbar()  # Optionally add a colorbar to visualize the color mapping
plt.show()

unwrapped_phase_skimage = unwrap_phase(difuuser_phase)
plt.figure()
plt.imshow(unwrapped_phase_skimage)
plt.figure()
plt.imshow(np.angle(field_d_at_diff))
plt.show()
plt.show()




cropped_unwrapped_phase_skimage = unwrapped_phase_skimage[180:380, 180:380]
plt.figure()
plt.imshow(cropped_unwrapped_phase_skimage)
plt.show()

ac = autocorrelation(cropped_unwrapped_phase_skimage)
plt.figure()
plt.imshow(ac)
plt.show()
d_corr_estimated = find_d_corr(ac)

a=5
from matplotlib.widgets import Slider
import torch
import matplotlib.pyplot as plt
import torch.fft
from tqdm import tqdm
import matplotlib
from PIL import Image
import tifffile as tiff
from matplotlib.lines import Line2D
import imageio.v3 as iio
import tkinter as tk
from tkinter import filedialog
from scipy.ndimage import zoom
from numpy.fft import fft2 as fft
from numpy.fft import ifft2 as ifft
import colorsys
import numpy as np
from tkinter import filedialog, Tk
from PIL import Image
from numpy.fft import fftshift, ifftshift
matplotlib.use('TkAgg')
plt.ion()


def load_array_or_image(file_path):
    if file_path.endswith('.npy'):
        return np.load(file_path)
    else:
        # Load the image and keep its original mode and depth
        image = Image.open(file_path)
        return np.array(image)


def phplot(field, ax=None, log_scale=False):
    """
    Plots the phase of the input field using the HSV color space,
    where hue represents the phase and brightness represents the normalized amplitude.
    Adds a colorbar to visualize the phase mapping. Optionally uses log scale for amplitude.

    Parameters:
    field (numpy.ndarray): The complex-valued input field.
    ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to current axes.
    log_scale (bool, optional): Whether to use logarithmic scale for amplitude. Defaults to False.

    Returns:
    None
    """
    if isinstance(field, torch.Tensor):
        field = field.detach().cpu().numpy()  # Convert PyTorch tensor to NumPy array

    if field.ndim == 3 and field.shape[2] == 3:  # If the input is an RGB image
        field = np.mean(field, axis=2)  # Convert to grayscale by averaging the RGB channels
        field = field.astype(np.complex128)  # Convert to complex type for demonstration
        # Generate a complex field with some arbitrary phase and amplitude for visualization
        field = field * np.exp(1j * field)

    phase = np.angle(field)  # Phase of the field
    amplitude = np.abs(field)  # Amplitude of the field

    if log_scale:
        amplitude = np.log1p(amplitude)  # Use log scale for amplitude

    amplitude = amplitude / np.max(amplitude)  # Normalize the amplitude

    # Convert phase from radians to the range [0, 1] for HSV colormap
    phase_normalized = (phase + np.pi) / (2 * np.pi)

    # Create an HSV image where hue represents phase and value represents amplitude
    hsv_image = np.zeros((field.shape[0], field.shape[1], 3), dtype=np.float64)
    hsv_image[:, :, 0] = phase_normalized  # Hue
    hsv_image[:, :, 1] = 1.0  # Saturation (full saturation)
    hsv_image[:, :, 2] = amplitude  # Value (brightness)

    # Convert HSV image to RGB
    rgb_image = np.zeros_like(hsv_image)
    for i in range(hsv_image.shape[0]):
        for j in range(hsv_image.shape[1]):
            rgb_image[i, j] = colorsys.hsv_to_rgb(hsv_image[i, j, 0], hsv_image[i, j, 1], hsv_image[i, j, 2])

    if ax is None:
        ax = plt.gca()  # Get current axes if none is provided

    # Plot the RGB image
    im = ax.imshow(rgb_image)
    ax.axis('off')

    # Create a colorbar to represent the phase
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

    # Create a colormap for the colorbar that matches the HSV colormap
    cbar_ticks = np.linspace(-np.pi, np.pi, 9)  # Define ticks from -π to π
    cbar_labels = [f'{t:.2f}' for t in cbar_ticks]  # Format labels
    cbar.set_ticks(np.linspace(0, 1, 9))
    cbar.set_ticklabels(cbar_labels)
    cbar.set_label('Phase (radians)')


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

def create_line_field(line_length, line_width, num_lines_x, num_lines_y, Nx, Ny):
    """
    Create a field with specified number of horizontal and vertical lines in the center of the matrix.

    Parameters:
    - line_length: The length of each line.
    - line_width: The thickness of each line.
    - num_lines_x: Number of vertical lines.
    - num_lines_y: Number of horizontal lines.
    - Nx, Ny: The size of the grid (number of points in x and y directions).

    Returns:
    - A 2D numpy array representing the field with lines.
    """
    field = np.zeros((Nx, Ny))

    # Center points
    center_x, center_y = Nx // 2, Ny // 2

    # Horizontal lines
    if num_lines_y > 0:
        h_total_height = num_lines_y * (line_width * 2) - line_width
        h_start = center_x - h_total_height // 2
        for i in range(num_lines_y):
            y_position = h_start + i * 2 * line_width
            field[y_position : y_position + line_width, center_y - line_length // 2 : center_y + line_length // 2] = 1

    # Vertical lines
    if num_lines_x > 0:
        v_total_width = num_lines_x * (line_width * 2) - line_width
        v_start = center_y - v_total_width // 2
        for i in range(num_lines_x):
            x_position = v_start + i * 2 * line_width
            field[center_x - line_length // 2 : center_x + line_length // 2, x_position : x_position + line_width] = 1

    return field



def load_and_resize_image_as_npy(dx=1e-6, wanted_size=1e-3):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.tif;*.png")])
    if not file_path:
        return None
    try:
        image = iio.imread(file_path)
        if image.ndim == 3:
            # Convert to grayscale by averaging the color channels
            image = np.mean(image, axis=2)

        # Calculate the desired number of pixels
        wanted_size_pixels = int(wanted_size / dx)

        # Get the current size of the image
        current_size_pixels = image.shape

        # Calculate the zoom factors for each dimension
        zoom_factors = [wanted_size_pixels / current_size_pixels[0], wanted_size_pixels / current_size_pixels[1]]

        # Resize the image
        resized_image_array = zoom(image, zoom_factors, order=1)  # Use order=1 for bilinear interpolation
        # Threshold the image to make it binary
        threshold = resized_image_array.mean()
        binary_image_array = (resized_image_array > threshold).astype(np.uint8)

        return binary_image_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


# Example usage:
size = 62 // 3
width = 5
line_length, line_width, num_lines_x, num_lines_y, Nx, Ny = size , width , int(0.5 * size / width), 0, 800, 800
from numpy.fft import fft2, ifft2, fftshift, ifftshift
# Create the line field
line_field = create_line_field(line_length, line_width, num_lines_x, num_lines_y, Nx, Ny)
np.save('cross field', line_field)
# plt.figure()
# plt.imshow(line_field)
# plt.show()

# final_image_size = 600
# real_area = 2472*5.5e-6
# dx = real_area / final_image_size
# image_array = 1 - load_and_resize_image_as_npy(dx=dx, wanted_size=1e-3)
# padding_size = (final_image_size-image_array.shape[0]) // 2
# padded_image_array = np.pad(image_array, ((padding_size, padding_size), (padding_size, padding_size)), mode='constant')
# plt.figure()
# plt.imshow(padded_image_array)
# plt.show()
# np.save('1 mm USAF at 0', padded_image_array)



sigma = 15  # Standard deviation of the Gaussian
center_x, center_y = Nx // 2, Ny // 2  # Center of the Gaussian
gaussian_field = create_field(sigma, center_x, center_y, Nx, Ny)
# Load the initial field from a .npy file
init_dir = 'D:\Lab Images and data local'
root = tk.Tk()
root.withdraw()
# Ask for the first field
array_1_path = filedialog.askopenfilename(
    initialdir=init_dir, filetypes=[("NumPy files", "*.npy"), ("Image files", "*.jpeg;*.jpg;*.tiff;*.tif;*.png")],
    title='Select Field 1 to propagate.'
)
# Ask for the second field
array_2_path = filedialog.askopenfilename(
    initialdir=init_dir, filetypes=[("NumPy files", "*.npy"), ("Image files", "*.jpeg;*.jpg;*.tiff;*.tif;*.png")],
    title='Select Field 2 to propagate.')

root.destroy()

# Load arrays or images and convert to 2D NumPy arrays
original_array_1 = load_array_or_image(array_1_path)
original_array_2 = load_array_or_image(array_2_path)


# Normalization
original_array_1 = original_array_1.astype(np.complex64)
original_array_1 /= np.max(np.abs(original_array_1))
original_array_2 = original_array_2.astype(np.complex64)
original_array_2 /= np.max(np.abs(original_array_2))

array_1_fft = fftshift(fft(original_array_1))
array_2_fft = fftshift(fft(original_array_2))

# Create the figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the first FFT with phplot using log scale
axs[0].set_title('Array 1 FFT (Log Scale)')
phplot(array_1_fft, axs[0], log_scale=True)

# Plot the second FFT with phplot using log scale
axs[1].set_title('Array 2 FFT (Log Scale)')
phplot(array_2_fft, axs[1], log_scale=True)

# Show the plots
plt.show()

rows_to_roll_1 = 0
cols_to_roll_1 = 0
rows_to_roll_2 = 0
cols_to_roll_2 = 0

# Roll the FFT results
array_1_fft_rolled = np.roll(array_1_fft, shift=(rows_to_roll_1, cols_to_roll_1), axis=(0, 1))
array_2_fft_rolled = np.roll(array_2_fft, shift=(rows_to_roll_2, cols_to_roll_2), axis=(0, 1))

plt.figure()
plt.subplot(1,2,1)
plt.title('array 1 fft rolled')
plt.imshow(np.abs(array_1_fft_rolled))
plt.subplot(1,2,2)
plt.title('array 2 fft rolled')
plt.imshow(np.abs(array_2_fft_rolled))
plt.show()

array_1 = ifft(ifftshift(array_1_fft_rolled))
array_2 = ifft(ifftshift(array_2_fft_rolled))


# hann = np.outer(*(2*[np.hanning(array_1.shape[-1])]))
# hanned_1 = array_1 * hann
# Fourier_1 = np.fft.fft2(hanned_1)
# Fourier_1 *= hann
# hanned_1 = np.fft.ifft2(Fourier_1)

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

padding_size = 500
padded_array_1 = np.pad(array_1, ((padding_size, padding_size), (padding_size, padding_size)), mode='constant')
# plt.imshow(np.abs(padded_array))
# plt.show()
padded_array_2 = np.pad(array_2, ((padding_size, padding_size), (padding_size, padding_size)), mode='constant')
# U0_1 = torch.from_numpy(padded_array_1).to(torch.complex64)
# U0_2 = torch.from_numpy(padded_array_2).to(torch.complex64)



# Set the propagation parameters
system_mag_1 = 200/125
system_mag_2 = 200/125
original_shape = 2472
blob_shape_1 = array_1.shape[0]
blob_shape_2 = array_2.shape[0]
mag_1 = system_mag_1 * blob_shape_1/original_shape
mag_2 = system_mag_2 * blob_shape_2/original_shape
dx_1 = dy_1 = 5.5e-6 / mag_1  # Pixel size in x-direction (m)
dx_2 = dy_2 = 5.5e-6 / mag_2  # Pixel size in x-direction (m)
wavelength = 632.8e-9  # Wavelength (m)



# Compute the field at different z-positions
z_min_mm = -300
z_max_mm = 300
z_min, z_max = z_min_mm * 1e-3 ,z_max_mm * 1e-3
step_in_mm = 3
num_z = int((z_max_mm - z_min_mm) // step_in_mm + 1)
z_values = np.linspace(z_min, z_max, num_z)
z_values_rounded = np.round(z_values, 5)

# z_values = np.concatenate(([0], z_values))

original_height_1, original_width_1 = array_1.shape
original_height_2, original_width_2 = array_2.shape

# Calculate the start and end indices for slicing
start_row_1 = padding_size
end_row_1 = start_row_1 + original_height_1
start_col_1 = padding_size
end_col_1 = start_col_1 + original_width_1

start_row_2 = padding_size
end_row_2 = start_row_2 + original_height_2
start_col_2 = padding_size
end_col_2 = start_col_2 + original_width_2



# Initialize lists to store the computed fields
array_1_all = []
array_2_all = []
diffuzer_all = []
for z in tqdm(z_values, desc='Computing fields'):
    field_1 = fresnel_propagation(padded_array_1, dx_1, dy_1, z, wavelength)
    field_2 = fresnel_propagation(padded_array_2, dx_2, dy_2, z, wavelength)
    # Normalization
    field_1 /= np.max(np.abs(field_1))
    field_2 /= np.max(np.abs(field_2))

    # Slice the padded array to retrieve the original field
    original_field_1 = field_1[start_row_1:end_row_1, start_col_1:end_col_1]
    original_field_2 = field_2[start_row_2:end_row_2, start_col_2:end_col_2]

    # current_diffuzer = field_1 * np.exp(-1j * np.angle(field_2))
    array_1_all.append(original_field_1)
    array_2_all.append(original_field_2)
    # diffuzer_all.append(current_diffuzer)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(bottom=0.3)

extent_1 = [0, array_1.shape[1]*dx_1*1e3, 0, array_1.shape[0]*dy_1*1e3]
extent_2 = [0, array_2.shape[1]*dx_2*1e3, 0, array_2.shape[0]*dy_2*1e3]

im_1 = ax1.imshow(np.abs(array_1), cmap='hot', extent=extent_1)
ax1.set_title('Propagated Field 1')
plt.colorbar(im_1, ax=ax1)

im_2 = ax2.imshow(np.abs(array_2), cmap='hot', extent=extent_2)
ax2.set_title('Propagated Field 2')
plt.colorbar(im_2, ax=ax2)


# Initialize the PSF lines and text labels in axes coordinates
psf_line_1 = Line2D([0, 0], [0, 0], color='white', linewidth=2, transform=ax1.transAxes)
psf_line_2 = Line2D([0, 0], [0, 0], color='white', linewidth=2, transform=ax2.transAxes)
ax1.add_line(psf_line_1)
ax2.add_line(psf_line_2)
text_label_1 = ax1.text(0.05, 0.95, 'PSF size', color='white', fontsize=12, ha='left', va='top', transform=ax1.transAxes)
text_label_2 = ax2.text(0.05, 0.95, 'PSF size', color='white', fontsize=12, ha='left', va='top', transform=ax2.transAxes)

ax_slider1 = plt.axes([0.25, 0.2, 0.5, 0.03])
slider1 = Slider(
    ax_slider1, 'z (m) - Propagated', z_min, z_max, valinit=0, valstep=(z_max - z_min) / (num_z - 1)
)

ax_slider2 = plt.axes([0.25, 0.1, 0.5, 0.03])
slider2 = Slider(
    ax_slider2, 'z (m) - Ground Truth', z_min, z_max, valinit=0, valstep=(z_max - z_min) / (num_z - 1)
)

xlim1 = ax1.get_xlim()
ylim1 = ax1.get_ylim()
xlim2 = ax2.get_xlim()
ylim2 = ax2.get_ylim()

theta = np.deg2rad(0.5)

# Define the update function
# Define the update function
def update(val):
    z_index1 = int((slider1.val - z_min) / ((z_max - z_min) / (num_z - 1)))
    im_1.set_data(np.abs(array_1_all[z_index1]))
    ax1.set_title(f'Propagated Field 1 (z = {z_values[z_index1]:.2f} m)')
    psf_size_1 = theta * z_values[z_index1] * 1e3  # in mm
    normalized_psf_size_1 = np.abs(psf_size_1 / (extent_1[1] - extent_1[0]))  # Normalize relative to the plot's width
    psf_line_1.set_data([0.05, 0.05 + normalized_psf_size_1], [0.9, 0.9])  # Horizontal line from (0.05, 0.9) to (0.05 + normalized_psf_size_1, 0.9)

    z_index2 = int((slider2.val - z_min) / ((z_max - z_min) / (num_z - 1)))
    im_2.set_data(np.abs(array_2_all[z_index2]))
    ax2.set_title(f'Propagated Field 2 (z = {z_values[z_index2]:.2f} m)')
    psf_size_2 = theta * z_values[z_index2] * 1e3  # in mm
    normalized_psf_size_2 = np.abs(psf_size_2 / (extent_2[1] - extent_2[0]))  # Normalize relative to the plot's width
    psf_line_2.set_data([0.05, 0.05 + normalized_psf_size_2], [0.9, 0.9])  # Horizontal line from (0.05, 0.9) to (0.05 + normalized_psf_size_2, 0.9)

    ax1.set_xlim(xlim1)
    ax1.set_ylim(ylim1)
    ax2.set_xlim(xlim2)
    ax2.set_ylim(ylim2)
    fig.canvas.draw_idle()
def store_zoom_levels(event):
    global xlim1, ylim1, xlim2, ylim2
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    xlim2 = ax2.get_xlim()
    ylim2 = ax2.get_ylim()

# Add the sliders and connect the update function
slider1.on_changed(update)
slider2.on_changed(update)


fig.canvas.mpl_connect('button_release_event', store_zoom_levels)

plt.ioff()

# Trigger an initial plot update for both sliders
update(slider1.val)
update(slider2.val)
plt.show()
a=5
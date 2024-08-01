import torch
import torch.nn.functional as F
from torch.fft import fftshift, ifftshift, fft2 as fft, ifft2 as ifft
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, Button
import imageio.v3 as iio
import tkinter as tk
from scipy.ndimage import zoom
import colorsys
import numpy as np
from tkinter import filedialog, Tk
from PIL import Image
import time

matplotlib.use('TkAgg')
plt.ion()

def update(val):
    z_index1 = int((slider1.val - z_min) / ((z_max - z_min) / (num_z - 1)))
    im_1.set_data(np.abs(array_1_all_numpy[z_index1]))
    ax1.set_title(f'Propagated Field 1 (z = {z_values[z_index1]:.2f} m)')
    psf_size_1 = theta * z_values[z_index1] * 1e3  # in mm
    normalized_psf_size_1 = np.abs(psf_size_1 / (extent_1[1] - extent_1[0]))  # Normalize relative to the plot's width
    psf_line_1.set_data([0.05, 0.05 + normalized_psf_size_1], [0.9, 0.9])  # Horizontal line from (0.05, 0.9) to (0.05 + normalized_psf_size_1, 0.9)

    z_index2 = int((slider2.val - z_min) / ((z_max - z_min) / (num_z - 1)))
    im_2.set_data(np.abs(array_2_all_numpy[z_index2]))
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

def roll_fft(event):
    global array_1_fft_rolled, array_2_fft_rolled
    ax = event.inaxes
    if ax == ax1:
        y, x = int(event.ydata), int(event.xdata)
        rows_to_roll = array_1_fft_rolled.shape[0] // 2 - y
        cols_to_roll = array_1_fft_rolled.shape[1] // 2 - x
        array_1_fft_rolled = torch.roll(array_1_fft_rolled, shifts=(rows_to_roll, cols_to_roll), dims=(0, 1))
        update_plot(ax1, array_1_fft_rolled)
    elif ax == ax2:
        y, x = int(event.ydata), int(event.xdata)
        rows_to_roll = array_2_fft_rolled.shape[0] // 2 - y
        cols_to_roll = array_2_fft_rolled.shape[1] // 2 - x
        array_2_fft_rolled = torch.roll(array_2_fft_rolled, shifts=(rows_to_roll, cols_to_roll), dims=(0, 1))
        update_plot(ax2, array_2_fft_rolled)
    plt.draw()

def update_plot(ax, field):
    rgb_image = create_rgb_image(field)
    ax.get_images()[0].set_data(rgb_image)


def create_rgb_image(field):
    if isinstance(field, torch.Tensor):
        field = field.detach().cpu().numpy()

    phase = np.angle(field)
    amplitude = np.abs(field)
    amplitude = np.log1p(amplitude)  # Use log scale for amplitude
    amplitude = amplitude / np.max(amplitude)

    phase_normalized = (phase + np.pi) / (2 * np.pi)
    hsv_image = np.zeros((field.shape[0], field.shape[1], 3), dtype=np.float64)
    hsv_image[:, :, 0] = phase_normalized
    hsv_image[:, :, 1] = 1.0
    hsv_image[:, :, 2] = amplitude

    rgb_image = np.zeros_like(hsv_image)
    for i in range(hsv_image.shape[0]):
        for j in range(hsv_image.shape[1]):
            rgb_image[i, j] = colorsys.hsv_to_rgb(hsv_image[i, j, 0], hsv_image[i, j, 1], hsv_image[i, j, 2])

    return rgb_image


def phplot(field, ax=None, log_scale=True):
    rgb_image = create_rgb_image(field)

    if ax is None:
        ax = plt.gca()

    im = ax.imshow(rgb_image)
    ax.axis('off')

    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar_ticks = np.linspace(-np.pi, np.pi, 9)
    cbar_labels = [f'{t:.2f}' for t in cbar_ticks]
    cbar.set_ticks(np.linspace(0, 1, 9))
    cbar.set_ticklabels(cbar_labels)
    cbar.set_label('Phase (radians)')

    return im


# Global variables for optimization and drag control
last_update_time = 0
update_interval = 0.1  # Update every 100ms
preview_scale = 0.25  # Scale factor for preview during drag
is_dragging = False


def drag_roll(event):
    global array_1_fft_rolled, array_2_fft_rolled, start_x, start_y, last_update_time, is_dragging

    if not is_dragging:
        return

    current_time = time.time()
    if current_time - last_update_time < update_interval:
        return  # Skip this update if it's too soon

    if event.inaxes == ax1 or event.inaxes == ax2:
        dx = int(start_x - event.xdata)
        dy = int(start_y - event.ydata)

        if event.inaxes == ax1:
            array_1_fft_rolled = torch.roll(array_1_fft_rolled, shifts=(-dy, -dx), dims=(0, 1))
            update_plot_preview(ax1, array_1_fft_rolled)
        else:
            array_2_fft_rolled = torch.roll(array_2_fft_rolled, shifts=(-dy, -dx), dims=(0, 1))
            update_plot_preview(ax2, array_2_fft_rolled)

        start_x, start_y = event.xdata, event.ydata
        last_update_time = current_time
        plt.draw()


def update_plot_preview(ax, field):
    preview = create_preview(field)
    ax.get_images()[0].set_data(preview)


def create_preview(field):
    # Create a lower resolution preview
    preview = field[::4, ::4]  # Take every 4th pixel
    return create_rgb_image(preview)


def on_press(event):
    global start_x, start_y, is_dragging
    if event.inaxes == ax1 or event.inaxes == ax2:
        start_x, start_y = event.xdata, event.ydata
        is_dragging = True


def on_release(event):
    global start_x, start_y, is_dragging
    if is_dragging:
        if event.inaxes == ax1:
            update_plot(ax1, array_1_fft_rolled)
        elif event.inaxes == ax2:
            update_plot(ax2, array_2_fft_rolled)
    start_x, start_y = 0, 0
    is_dragging = False
    plt.draw()




def confirm(event):
    plt.close()

def load_array_or_image(file_path):
    if file_path.endswith('.npy'):
        return np.load(file_path)
    else:
        # Load the image and keep its original mode and depth
        image = Image.open(file_path)
        return np.array(image)




def fresnel_propagation(initial_field, dx, dy, z, wavelength):
    # Convert initial field to PyTorch tensor and move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # field_fft = fft(initial_field)
    # padding_size = 1000  # array.shape[0]//8
    # padded_field_fft = np.pad(field_fft, ((padding_size, padding_size), (padding_size, padding_size)), mode='constant')
    # padded_field = ifft(padded_field_fft)
    # new_dx = dx*(initial_field.shape[0]/(initial_field.shape[0] + 2 * padding_size))
    if not isinstance(initial_field, torch.Tensor):
        field = torch.from_numpy(initial_field).to(device).to(dtype=torch.complex64)
    else:
        field = initial_field.to(device).to(dtype=torch.complex64)
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
    field_propagated = ifft(fft(field) * H)
    return field_propagated




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

# Convert to PyTorch tensors and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

original_array_1 = torch.from_numpy(original_array_1).to(device).to(torch.complex64)
original_array_2 = torch.from_numpy(original_array_2).to(device).to(torch.complex64)

# Normalization
original_array_1 /= torch.max(torch.abs(original_array_1))
original_array_2 /= torch.max(torch.abs(original_array_2))

# Compute FFTs
array_1_fft = fftshift(fft(original_array_1))
array_2_fft = fftshift(fft(original_array_2))

# Create the figure and subplots for interactive rolling
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Initial plot
array_1_fft_rolled = array_1_fft.clone()
array_2_fft_rolled = array_2_fft.clone()
phplot(array_1_fft_rolled, ax1)
ax1.set_title('Array 1 FFT')
phplot(array_2_fft_rolled, ax2)
ax2.set_title('Array 2 FFT')

# After creating the plots
start_x, start_y = 0, 0



# In the main part of your script:
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', drag_roll)

# Remove the old roll_fft connection
# fig.canvas.mpl_connect('button_press_event', roll_fft)
ax_button = plt.axes([0.81, 0.05, 0.1, 0.075])
button = Button(ax_button, 'Confirm')
button.on_clicked(confirm)

plt.show()

# After the plot is closed, you can use array_1_fft_rolled and array_2_fft_rolled
array_1 = ifft(ifftshift(array_1_fft_rolled))
array_2 = ifft(ifftshift(array_2_fft_rolled))
del ax1, ax2, ax_button, button, fig

# hann = np.outer(*(2*[np.hanning(array_1.shape[-1])]))
# hanned_1 = array_1 * hann
# Fourier_1 = fft(hanned_1)
# Fourier_1 *= hann
# hanned_1 = ifft(Fourier_1)

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
# Pad array_1
padded_array_1 = F.pad(array_1, (padding_size, padding_size, padding_size, padding_size), mode='constant', value=0)
phplot(padded_array_1, log_scale=True)

# Pad array_2
padded_array_2 = F.pad(array_2, (padding_size, padding_size, padding_size, padding_size), mode='constant', value=0)




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
z_min, z_max = z_min_mm * 1e-3, z_max_mm * 1e-3
step_in_mm = 3
num_z = int((z_max_mm - z_min_mm) // step_in_mm + 1)
z_values = np.linspace(z_min, z_max, num_z)

# Check if 0 is in the range and not already in z_values
if z_min <= 0 <= z_max and 0 not in z_values:
    # Find the index where 0 should be inserted
    insert_index = np.searchsorted(z_values, 0)
    # Insert 0 into z_values
    z_values = np.insert(z_values, insert_index, 0)

# Round the values
z_values_rounded = np.round(z_values, 5)

# Find the index of 0
zero_index = np.where(z_values_rounded == 0)[0][0]

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
    field_1 /= torch.max(torch.abs(field_1))
    field_2 /= torch.max(torch.abs(field_2))

    # Slice the padded array to retrieve the original field
    original_field_1 = field_1[start_row_1:end_row_1, start_col_1:end_col_1]
    original_field_2 = field_2[start_row_2:end_row_2, start_col_2:end_col_2]

    # current_diffuzer = field_1 * np.exp(-1j * np.angle(field_2))
    array_1_all.append(original_field_1)
    array_2_all.append(original_field_2)
    # diffuzer_all.append(current_diffuzer)


array_1_all_numpy = [tensor.cpu().numpy() for tensor in array_1_all]
array_2_all_numpy = [tensor.cpu().numpy() for tensor in array_2_all]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(bottom=0.3)

extent_1 = [0, array_1.shape[1]*dx_1*1e3, 0, array_1.shape[0]*dy_1*1e3]
extent_2 = [0, array_2.shape[1]*dx_2*1e3, 0, array_2.shape[0]*dy_2*1e3]

im_1 = ax1.imshow(np.abs(array_1_all_numpy[zero_index]), cmap='hot', extent=extent_1)
ax1.set_title('Propagated Field 1')
plt.colorbar(im_1, ax=ax1)

im_2 = ax2.imshow(np.abs(array_2_all_numpy[zero_index]), cmap='hot', extent=extent_2)
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
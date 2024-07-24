import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.fft import fft2, ifft2, fftshift, ifftshift
phs = lambda x:x/x.abs()
import tkinter as tk
from tkinter import filedialog
import os
import matplotlib
matplotlib.use('TkAgg')
plt.ion()



def autocorrelation(x):
    """ Compute the normalized autocorrelation of a matrix using FFT. """
    x_fft = torch.fft.fft2(x)
    x_conj_fft = torch.conj(x_fft)
    ac = torch.fft.ifft2(x_fft * x_conj_fft).real
    ac = torch.fft.fftshift(ac)  # Shift to center the zero frequency component
    ac_normalized = (ac - torch.min(ac)) / (torch.max(ac) - torch.min(ac))  # Normalize from 0 to 1
    return ac_normalized

def find_d_corr1(ac, threshold=1/np.e):  # Adjust the threshold as needed based on empirical observations
    """ Find the correlation distance where the autocorrelation drops below a given threshold. """
    center = ac.shape[0] // 2
    center_line = ac[center, :]

    mask = center_line < threshold
    indices = torch.nonzero(mask, as_tuple=False).squeeze()

    # Filter indices to only those greater than the center (looking rightwards)
    filtered_indices = indices[indices > center]
    if len(filtered_indices) > 0:
        d_corr = filtered_indices[0] - center
        return 2 * d_corr.item()
    else:
        return None

def find_d_corr(ac, threshold=1/np.e):  # Adjust the threshold as needed based on empirical observations
    """ Find the correlation distance where the autocorrelation drops below a given threshold. """
    center = ac.shape[0] // 2
    center_line = ac[center, :]

    mask = center_line < threshold ** 2
    indices = torch.nonzero(mask, as_tuple=False).squeeze()

    # Filter indices to only those greater than the center (looking rightwards)
    filtered_indices = indices[indices > center]
    if len(filtered_indices) > 0:
        d_corr = filtered_indices[0] - center
        return 2 * d_corr.item()
    else:
        return None

def shifted_phase(diffusers):
    phase_values = torch.angle(diffusers)
    # Shift phase from [-π, π] to [0, 2π]
    phase_values[phase_values < 0] += 2 * np.pi
    return phase_values

def gauss2D(sigma, size):
    """Generate a 2D Gaussian filter."""
    ax = np.linspace(-size // 2, size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return torch.tensor(kernel, dtype=torch.float32)


def create_diffuser(mx_shp, d_corr, speckle_size):
    """Create a diffuser by applying Gaussian filters to a random phase matrix."""
    sigma1 = mx_shp // d_corr
    sigma2 = mx_shp // speckle_size

    # Generate random phase matrix correctly spanning from -π to π
    random_phase_matrix = 2 * np.pi * torch.rand(1, mx_shp, mx_shp) - np.pi
    complex_exponential_of_phase = torch.exp(1j * random_phase_matrix)

    # Apply Gaussian filter in the frequency domain
    filtered_frequency_data = gauss2D(sigma1, mx_shp) * fftshift(fft2(complex_exponential_of_phase))
    freq_phase_values = shifted_phase(filtered_frequency_data[0])
    print(f"Min phase1: {torch.min(freq_phase_values).item()}, Max phase1: {torch.max(freq_phase_values).item()}")


    # Transform to spatial domain and extract phase
    diffusers = phs(ifft2(ifftshift(filtered_frequency_data)))
    diffusers_phase_values = shifted_phase(diffusers[0])
    print(f"Min phase2: {torch.min(diffusers_phase_values).item()}, Max phase2: {torch.max(diffusers_phase_values).item()}")

    # Adjust speckle size in the spatial domain and crop to the desired output size
    filtered_diffusers = diffusers # * gauss2D(sigma2, mx_shp)
    filtered_diffusers_phase_values = shifted_phase(filtered_diffusers[0])
    print(f"Min phase3: {torch.min(filtered_diffusers_phase_values).item()}, Max phase3: {torch.max(filtered_diffusers_phase_values).item()}")

    return filtered_diffusers

def generate_diffuser(shape, d_corr):
    random_phase_matrix = 2 * np.pi * torch.rand(1, shape, shape ) - np.pi
    complex_exponential_of_phase = torch.exp(1j * random_phase_matrix)
    ax = np.linspace(-shape // 2, shape // 2,shape)
    X, Y = np.meshgrid(ax,ax)
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
    R = torch.sqrt(X**2 + Y**2)
    F = fftshift(fft2(complex_exponential_of_phase))
    filtered_frequency_data = F * torch.exp(-(R/torch.max(R[:]) * d_corr) ** 2)
    filtered = ifft2(ifftshift(filtered_frequency_data))
    diffuser = torch.exp(1j * torch.angle(filtered))
    return diffuser



#  parameters
num_diffusers = 10
folder_path = filedialog.askdirectory(title='Select the folder containing the .npy files')
npy_files = [file for file in os.listdir(folder_path) if file.endswith('.npy')]

fields = torch.cat([torch.from_numpy(np.load(os.path.join(folder_path, f))).unsqueeze(0) for f in os.listdir(folder_path)],0)
calibration_field = fields[0]
only_miror = fields[1]
fields = fields[2:]

# system parameters
system_mag = 2
original_shape = 2469
shp = fields.shape[1:]
shape = shp[0]
mag = system_mag * shape/original_shape
dx = dy = 5.5e-6 / mag  # Pixel size in x-direction (m)


# Generate and plot diffusers
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()
for i in range(num_diffusers):
    theta = np.deg2rad(0.5)
    d_corr = 2 * 632.8e-9 / theta
    d_corr_pix = int(d_corr / dx)
    # d_corr_pix = 40
    speckle_size = 3
    diffusers = create_diffuser(shape // 2, d_corr_pix, speckle_size)
    # diffusers = generate_diffuser(shape, d_corr_pix)
    if i % 2 != 0:
        diffuser = fields[i]
        difuuser_phase = torch.angle(diffuser) - torch.angle(only_miror)
        diffuser = torch.exp(1j * difuuser_phase)
        diffuser = diffuser[shape // 4:3 * shape // 4, shape // 4:3 * shape // 4]
        diffusers = diffuser.unsqueeze(0)
    phase_values = shifted_phase(diffusers[0])
    print(f"Min phase4: {torch.min(phase_values).item()}, Max phase4: {torch.max(phase_values).item()}")
    ac = autocorrelation(phase_values)
    d_corr_estimated_pixels = find_d_corr(ac)
    d_corr_estimated = d_corr_estimated_pixels * dx
    theta_estimated = 2 * np.round(np.rad2deg(632.8e-9 / d_corr_estimated),3)
    d_corr_estimated_test = np.round(1e6 * 2 * 632.8e-9 / np.deg2rad(theta_estimated) ,3)
    ax = axes[i]
    im = ax.imshow(phase_values.numpy(), cmap='jet')
    ax.set_title(f'd_corr={np.round(1e6 * d_corr_estimated, 3)} um\nEstimated theta: {theta_estimated}')
    ax.axis('off')

fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', shrink=0.6)
plt.tight_layout()
plt.show()

a=5
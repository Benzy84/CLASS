import os
import numpy as np
from PIL import Image
from tkinter import filedialog
import re
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from scipy.ndimage import label, find_objects, center_of_mass

matplotlib.use('TkAgg')
plt.ion()

def draw_circle(image, center, radius):
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
    image[mask] = [0, 0, 255]  # Color the circle in blue

def load_array_or_image(file_path):
    if file_path.endswith('.npy'):
        return np.load(file_path)
    else:
        image = Image.open(file_path)
        # Convert to grayscale if the image is RGB
        if image.mode == 'RGB':
            image = image.convert('L')
        return np.array(image)

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

def count_saturation(image_array):
    if image_array.dtype == np.uint8:
        return np.sum(image_array == 255)
    elif image_array.dtype == np.uint16:
        return np.sum(image_array == 65535)
    else:
        raise ValueError("Unsupported image depth")

# Ask for the directory
pth = filedialog.askdirectory(title='Select Folder')

# Collect the files and sort them using natural sort
files = [f for f in os.listdir(pth) if f.endswith(('.npy', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
files.sort(key=natural_sort_key)

saturation_counts = []

# Use tqdm to add a progress bar for the loading loop
for file in tqdm(files, desc='Processing images'):
    file_path = os.path.join(pth, file)
    image_array = load_array_or_image(file_path)

    saturation_count = count_saturation(image_array)
    saturation_counts.append((file, saturation_count, image_array))

if len(saturation_counts) > 0:
    plt.figure()

    # Find the image with the highest number of saturated pixels
    file, max_saturation, max_saturation_image = max(saturation_counts, key=lambda x: x[1])

    # Normalize the image to the range [0, 255] for display
    if max_saturation_image.dtype == np.uint16:
        normalized_image = (max_saturation_image / 65535 * 255).astype(np.uint8)
        saturated_mask = (max_saturation_image == 65535)
    elif max_saturation_image.dtype == np.uint8:
        normalized_image = max_saturation_image
        saturated_mask = (max_saturation_image == 255)
    else:
        raise ValueError("Unsupported image depth")

    # Convert normalized grayscale image to RGB for display
    highlighted_image = np.stack([normalized_image] * 3, axis=-1)

    # Highlight the saturated pixels in red
    highlighted_image[saturated_mask] = [255, 0, 0]  # Color the saturated pixels in red

    # Find the largest group of saturated pixels
    labeled_array, num_features = label(saturated_mask)
    print(f"Number of features: {num_features}")  # Debugging statement
    if num_features > 0:
        largest_component_slice = max(find_objects(labeled_array),
                                      key=lambda x: (x[0].stop - x[0].start) * (x[1].stop - x[1].start))
        print(f"Largest component slice: {largest_component_slice}")  # Debugging statement
        largest_component_mask = labeled_array[largest_component_slice] == labeled_array[largest_component_slice].max()
        print(f"Largest component mask: {largest_component_mask}")  # Debugging statement

        # Ensure the mask is not empty
        if np.any(largest_component_mask):
            # Calculate the center of mass of the largest component
            largest_component_coords = np.argwhere(largest_component_mask)
            centroid = center_of_mass(largest_component_mask)
            print(f"Centroid: {centroid}")  # Debugging statement

            # Adjust centroid to global coordinates
            centroid_global = (
                centroid[0] + largest_component_slice[0].start, centroid[1] + largest_component_slice[1].start)
            centroid_global = tuple(map(int, centroid_global))

            # Calculate the radius as the maximum distance from the centroid to any point in the component
            max_distance = np.max(np.sqrt(
                (largest_component_coords[:, 0] - centroid[0]) ** 2 + (largest_component_coords[:, 1] - centroid[1]) ** 2))

            # Draw a circle that encompasses the largest component
            draw_circle(highlighted_image, centroid_global, int(1 * max_distance))

            # Ensure the saturated pixels are still visible over the blue circle
            highlighted_image[saturated_mask] = [255, 0, 0]  # Re-color the saturated pixels in red
            plt.imshow(highlighted_image)
            plt.title(f"{file} with {max_saturation} saturated pixels highlighted (red), largest group (blue circle)")
            plt.show()

            print("The number of saturated pixels in each image:")
            for file, count, _ in saturation_counts:
                print(f"{file}: {int(count)} pixels")
else:
    print("No images with saturation found.")

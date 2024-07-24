import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import EllipseSelector, Button
from tkinter import filedialog, Tk
import random
from scipy.ndimage import shift


def crop_image_interactively_circle(image_array, title='Select Crop Area', crop_coords=None):
    if crop_coords is None:
        fig, ax = plt.subplots()
        ax.imshow(np.log(np.abs(image_array)) + 1, cmap='gray')
        ax.set_title(title)

        # Function to update the cropping circle
        ellipse = None

        def update_ellipse(eclick, erelease):
            nonlocal x1, y1, x2, y2, ellipse
            dx = abs(erelease.xdata - eclick.xdata)
            dy = abs(erelease.ydata - eclick.ydata)
            radius = min(dx, dy) / 2
            center_x = (eclick.xdata + erelease.xdata) / 2
            center_y = (eclick.ydata + erelease.ydata) / 2
            x1, y1 = center_x - radius, center_y - radius
            x2, y2 = center_x + radius, center_y + radius
            if ellipse is None:
                ellipse = ax.add_patch(plt.Circle((center_x, center_y), radius, fill=False, edgecolor='r', linewidth=2))
            else:
                ellipse.set_radius(radius)
                ellipse.set_center((center_x, center_y))
            fig.canvas.draw()

        # Function to confirm the selection
        def confirm_selection(event):
            nonlocal crop_coords
            crop_coords = (x1, y1, x2, y2)
            plt.close(fig)

        # Create the ellipse selector
        x1, y1, x2, y2 = 0, 0, 0, 0
        es = EllipseSelector(ax, update_ellipse, useblit=True,
                             button=[1, 3],  # Left click to start, right click to stop
                             minspanx=5, minspany=5,
                             spancoords='pixels',
                             interactive=True)

        # Add a confirm button
        ax_confirm = plt.axes([0.8, 0.05, 0.1, 0.075])
        button = Button(ax_confirm, 'Confirm')
        button.on_clicked(confirm_selection)

        plt.show()
        return crop_coords
    else:
        return image_array[int(y1):int(y2), int(x1):int(x2)], crop_coords


def create_mask(image_shape, crop_coords):
    mask = np.zeros(image_shape, dtype=np.uint8)
    center_x = (crop_coords[0] + crop_coords[2]) / 2
    center_y = (crop_coords[1] + crop_coords[3]) / 2
    radius = (crop_coords[2] - crop_coords[0]) / 2
    Y, X = np.ogrid[:image_shape[0], :image_shape[1]]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    mask[dist_from_center <= radius] = 1
    return mask, (center_x, center_y)


def center_image(image, mask_center, image_center):
    shift_x = image_center[1] - mask_center[1]
    shift_y = image_center[0] - mask_center[0]
    centered_image = shift(image, shift=(shift_y, shift_x), mode='constant', cval=0)
    return centered_image


def main():
    root = Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory(title='Select the folder containing the .npy files')

    file_list = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

    if not file_list:
        print("No .npy files found in the selected folder.")
        return

    sample_image_path = os.path.join(folder_path, random.choice(file_list))
    image_array = np.load(sample_image_path)
    crop_coords = crop_image_interactively_circle(image_array)

    mask, mask_center = create_mask(image_array.shape, crop_coords)
    image_center = (image_array.shape[0] // 2, image_array.shape[1] // 2)

    random_files = random.sample(file_list, min(9, len(file_list)))
    plt.figure(figsize=(10, 10))
    for i, file_name in enumerate(random_files):
        image = np.load(os.path.join(folder_path, file_name))
        masked_image = image * mask
        centered_masked_image = center_image(masked_image, mask_center, image_center)
        plt.subplot(3, 3, i + 1)
        plt.imshow(np.abs(centered_masked_image), cmap='gray')
        plt.axis('off')
    plt.show()

    new_folder_path = os.path.join(folder_path, 'masked_images')
    os.makedirs(new_folder_path, exist_ok=True)

    for file_name in file_list:
        image = np.load(os.path.join(folder_path, file_name))
        masked_image = image * mask
        centered_masked_image = center_image(masked_image, mask_center, image_center)
        new_file_path = os.path.join(new_folder_path, file_name)
        np.save(new_file_path, centered_masked_image)


if __name__ == '__main__':
    main()

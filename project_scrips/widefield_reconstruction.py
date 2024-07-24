import cv2
import numpy as np
import os
import re
from tkinter import filedialog, messagebox, Tk
import matplotlib.patches as patches
import re
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')  # Set a different backend, like 'TkAgg'
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
import time


global latest_adjusted_image  # Declare the global variable
latest_adjusted_image = None  # Initialize it to None


def on_confirm(event):
    global confirmed, bbox
    confirmed = True
    # Update bbox with the new position and size of the rectangle
    x_min, y_min = rect.get_x(), rect.get_y()
    x_max, y_max = x_min + rect.get_width(), y_min + rect.get_height()
    bbox = (x_min, y_min, x_max, y_max)
    plt.close(fig)  # Close the figure to resume execution

def update(val):
    global last_update_time
    global latest_adjusted_image  # Declare as global inside the function

    current_time = time.time()
    if current_time - last_update_time > update_interval:
        alpha = slider_alpha.val
        beta = slider_beta.val
        adjusted_img = adjust_contrast_brightness(normalized_reconstructed_image, alpha, beta)
        latest_adjusted_image = adjusted_img  # Update the global variable
        ax2.imshow(adjusted_img, cmap='gray')
        fig.canvas.draw_idle()
        last_update_time = current_time

def adjust_contrast_brightness(image, alpha, beta):
    # alpha > 1 increases contrast
    # beta > 0 increases brightness
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)  # normalize to range [0,1]
    normalized_image = (normalized_image * 255).astype(np.uint8)  # scale to range [0,255]
    return normalized_image


class DraggableRectangle:
    def __init__(self, rect, callback):
        self.rect = rect
        self.press = None
        self.callback = callback

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.rect.axes: return
        contains, attrd = self.rect.contains(event)
        if not contains: return
        x0, y0 = self.rect.xy
        self.press = x0, y0, event.xdata, event.ydata

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.rect.axes: return
        x0, y0, xpress, ypress = self.press
        dx = int(event.xdata - xpress)
        dy = int(event.ydata - ypress)
        self.rect.set_x(x0+dx)
        self.rect.set_y(y0+dy)

        self.rect.figure.canvas.draw()

    def on_release(self, event):
        'on release we reset the press data'
        self.press = None
        self.rect.figure.canvas.draw()
        self.callback(self.rect)  # Call the callback function

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)


def extract_step_size_from_folder(folder_path):
    # Regex pattern to match the folder name and capture the step size
    pattern = r'(\d{2}\.\d{2}\.\d{4} \d{2}-\d{2}-\d{2} pixels step is (\d+))'
    match = re.search(pattern, folder_path)
    if match:
        return int(match.group(2))
    else:
        raise ValueError("Folder name does not contain step size information.")



def parse_filename(filename):
    # Extracts indices from the filename
    match = re.search(r'index=\[(-?\d+),(-?\d+)\]', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return None

def shift_image(image, shift_x, shift_y):
    # Shifts the image by the given amount
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted_image

def find_matrix_size(indices):
    # Determines the size of the matrix based on the indices
    max_index = np.max(indices, axis=0)
    return [max_index[0] + 1, max_index[1] + 1]

def find_illuminated_area(image):
    # Normalize the image
    max_val = np.max(image)
    normalized_image = image / max_val if max_val > 0 else image

    # Apply a threshold to separate the illuminated area
    _, thresh = cv2.threshold(normalized_image, threshold_value, 1, cv2.THRESH_BINARY)
    thresh = (thresh * 255).astype(np.uint8)  # Convert back to 8-bit image for contour detection

    # Find the bounding rectangle of all white pixels
    white_pixels = np.where(thresh == 255)
    x_min, y_min = np.min(white_pixels[1]), np.min(white_pixels[0])
    x_max, y_max = np.max(white_pixels[1]), np.max(white_pixels[0])

    # Calculate the center and radius
    center = ((x_min + x_max) // 2, (y_min + y_max) // 2)

    # Use the threshold to find the bounding square
    center_x, center_y = center
    half_size = step_size // 2  # Half of the step size

    # Calculate the square boundaries differently for even and odd step_size
    if step_size % 2 == 0:  # Even step_size
        x_min = center_x - half_size
        y_min = center_y - half_size
        x_max = center_x + half_size
        y_max = center_y + half_size
    else:  # Odd step_size
        x_min = center_x - half_size
        y_min = center_y - half_size
        x_max = center_x + half_size + 1  # Add 1 to ensure the size is always step_size
        y_max = center_y + half_size + 1  # Add 1 to ensure the size is always step_size

    # Ensure the square doesn't go outside the image boundaries
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, image.shape[1])
    y_max = min(y_max, image.shape[0])

    # Now, you can use the extracted area defined by (x_min, y_min, x_max, y_max) with a consistent size of step_size

    return (x_min, y_min, x_max, y_max), thresh


def apply_mask(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    mask = np.zeros_like(image)
    mask[y_min:y_max, x_min:x_max] = 255
    return cv2.bitwise_and(image, image, mask=mask)


def reconstruct_image(folder_path):
    global fig, rect, confirmed, bbox
    indices_list = []
    images = {}

    frame_00_processed = False
    for filename in os.listdir(folder_path):
        if filename.startswith("frame 00"):
            image = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
            bbox, thresh_image = find_illuminated_area(image)
            x_min, y_min, x_max, y_max = bbox

            fig, ax = plt.subplots()
            confirmed = False
            ax.imshow(image, cmap='gray',)

            # Create a draggable rectangle with initial position from bbox
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color="pink")
            ax.add_patch(rect)

            draggable_rect = DraggableRectangle(rect, lambda r: None)
            draggable_rect.connect()

            # Add a confirmation button
            ax_confirm = plt.axes([0.81, 0.05, 0.1, 0.075])
            btn_confirm = Button(ax_confirm, 'Confirm')
            btn_confirm.on_clicked(on_confirm)

            plt.show()

            if not confirmed:
                raise Exception("Confirmation not received.")

            # Use the updated bbox for further processing
            x_min, y_min, x_max, y_max = bbox
            frame_00_processed = True
            break

    if not frame_00_processed:
        raise Exception("No 'frame 00' file found.")

    # Process other files
    filenames = [f for f in os.listdir(folder_path) if f.endswith(".tiff") and not f.startswith("frame 00")]
    for filename in tqdm(filenames, desc="Processing Images",  unit="image"):
        indices = parse_filename(filename)
        indices_list.append(indices)
        image = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        # bbox, _ = find_illuminated_area(image)

        # Extract the illuminated area from the image
        # x_min, y_min, x_max, y_max = bbox
        illuminated_image = image[y_min:y_max, x_min:x_max]
        images[indices] = illuminated_image

    if not indices_list:
        raise ValueError("No valid images found in the folder.")

    # Summing up the masked images
    final_image = None
    # Determine the size of the final image
    indices_array = np.array(indices_list)
    matrix_size = find_matrix_size(indices_array)
    final_image_size = [matrix_size[0] * step_size, matrix_size[1] * step_size]
    final_image = np.zeros(final_image_size, dtype=np.uint8)

    # Place each image in the correct location in the final image
    for indices, illuminated_image in images.items():
        # Calculate the starting position for each image based on its indices
        y_start, x_start = np.array(indices) * step_size

        # Invert the y-coordinate
        y_start = final_image_size[0] - y_start - step_size

        # Calculate the end positions
        y_end, x_end = y_start + step_size, x_start + step_size

        # Check if the indices are within the bounds of final_image
        if y_end > final_image.shape[0] or x_end > final_image.shape[1]:
            continue  # Skip this image as it does not fit within final_image

        # Place the image at the calculated position
        final_image[y_start:y_end, x_start:x_end] = illuminated_image

    return final_image


# Usage
root = Tk()
root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
folder_path = filedialog.askdirectory()  # show an "Open" dialog box and return the path to the selected folder
step_size = extract_step_size_from_folder(folder_path)
threshold_value = 0.5
reconstructed_image = reconstruct_image(folder_path)
normalized_reconstructed_image = normalize_image(reconstructed_image)
adjusted_normalized_reconstructed_image = adjust_contrast_brightness(normalized_reconstructed_image, 10, 0)




if reconstructed_image is not None:
    last_update_time = 0
    update_interval = 0.1  # Update the image every 0.1 seconds at most

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Display the reconstructed image in the first subplot
    ax1.imshow(reconstructed_image, cmap='gray')
    ax1.set_title('Reconstructed Image')
    ax1.axis('off')  # Turn off axis

    # Display the normalized image in the second subplot
    ax2.imshow(normalized_reconstructed_image, cmap='gray')
    ax2.set_title('Normalized Image')
    ax2.axis('off')  # Turn off axis

    # Add sliders for adjusting alpha and beta
    ax_alpha = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_beta = plt.axes([0.25, 0.06, 0.65, 0.03], facecolor='lightgoldenrodyellow')

    slider_alpha = Slider(ax_alpha, 'Alpha', 0.1, 30.0, valinit=1.0)
    slider_beta = Slider(ax_beta, 'Beta', -100, 100, valinit=0)

    # Update the image when the sliders are changed
    slider_alpha.on_changed(update)
    slider_beta.on_changed(update)

    plt.show()

    input("Press Enter to continue...")

    if latest_adjusted_image is not None:
        cv2.imwrite('Adjusted_Reconstructed_Image.tiff', latest_adjusted_image)  # Save the latest adjusted image
    a=5
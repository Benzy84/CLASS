import os
import numpy as np
from PIL import Image
from tkinter import filedialog
import re
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

matplotlib.use('TkAgg')
plt.ion()

def load_array_or_image(file_path):
    if file_path.endswith('.npy'):
        return np.load(file_path)
    else:
        # Load the image and keep its original mode and depth
        image = Image.open(file_path)
        return np.array(image)


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]


def count_saturation(image_array, desc='Loading fields'):
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

saturation_counts = np.array([])

for file in tqdm(files, desc='Loading files'):
    file_path = os.path.join(pth, file)
    image_array = load_array_or_image(file_path)

    saturation_count = count_saturation(image_array)
    saturation_counts = np.append(saturation_counts, saturation_count)

plt.figure()
plt.plot(saturation_counts)
plt.show()
if len(saturation_counts) > 0:
    print("The number of saturated pixels in each image:")
    for file, count in zip(files, saturation_counts):
        print(f"{file}: {int(count)} pixels")
else:
    print("No images with saturation found.")

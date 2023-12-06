import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def plot_combined_rgb_channels_normalized(directory1, directory2, augmentation_type):
    def normalize_histogram(values):
        total_pixels = sum(values)
        return [count / total_pixels for count in values]

    def compute_normalized_histogram(image_path):
        image = Image.open(image_path)
        image = image.convert('RGB')
        r, g, b = image.split()

        r_values = list(r.getdata())
        g_values = list(g.getdata())
        b_values = list(b.getdata())

        # Compute histograms
        r_hist, _ = np.histogram(r_values, bins=256, range=(0, 256))
        g_hist, _ = np.histogram(g_values, bins=256, range=(0, 256))
        b_hist, _ = np.histogram(b_values, bins=256, range=(0, 256))

        # Normalize histograms
        r_hist_normalized = normalize_histogram(r_hist)
        g_hist_normalized = normalize_histogram(g_hist)
        b_hist_normalized = normalize_histogram(b_hist)

        return r_hist_normalized, g_hist_normalized, b_hist_normalized

    # Get a list of all files in the first directory
    file_list1 = os.listdir(directory1)
    image_files1 = [f for f in file_list1 if f.lower().endswith('.jpg')]

    # Get a list of all files in the second directory
    file_list2 = os.listdir(directory2)
    image_files2 = [f for f in file_list2 if f.lower().endswith('.jpg')]

    # Initialize arrays to store normalized histograms for each directory
    r_hist_all1, g_hist_all1, b_hist_all1 = [], [], []
    r_hist_all2, g_hist_all2, b_hist_all2 = [], [], []

    # Process each image in the first directory
    for image_file in image_files1:
        image_path = os.path.join(directory1, image_file)
        r_hist, g_hist, b_hist = compute_normalized_histogram(image_path)
        r_hist_all1.append(r_hist)
        g_hist_all1.append(g_hist)
        b_hist_all1.append(b_hist)

    # Process each image in the second directory
    for image_file in image_files2:
        image_path = os.path.join(directory2, image_file)
        r_hist, g_hist, b_hist = compute_normalized_histogram(image_path)
        r_hist_all2.append(r_hist)
        g_hist_all2.append(g_hist)
        b_hist_all2.append(b_hist)

    # Plot the combined and normalized RGB channels for both directories
    plt.figure(figsize=(10, 4))
    plt.suptitle(f'RGB Channels - {augmentation_type}', fontsize=16)

    plt.subplot(131)
    plt.title('Red Channel')
    plt.plot(np.mean(r_hist_all1, axis=0), color='red', label='Young')
    plt.plot(np.mean(r_hist_all2, axis=0), color='blue', label='Old')
    plt.legend()

    plt.subplot(132)
    plt.title('Green Channel')
    plt.plot(np.mean(g_hist_all1, axis=0), color='red', label='Young')
    plt.plot(np.mean(g_hist_all2, axis=0), color='blue', label='Old')
    plt.legend()

    plt.subplot(133)
    plt.title('Blue Channel')
    plt.plot(np.mean(b_hist_all1, axis=0), color='red', label='Young')
    plt.plot(np.mean(b_hist_all2, axis=0), color='blue', label='Old')
    plt.legend()

    plt.tight_layout()
    plt.show()

augmentation_types = ["greyscale", "blurred", "color_jittered", "edge_enhanced", "enhanced", "flipped", "negative", "noisy",
                      "rotated_clockwise", "rotated_counterclockwise", "split_mirror", "zoomed"]
plot_combined_rgb_channels_normalized(
        f"C:/Users/Rog Tuf/Desktop/MACHINE LEARNING/Young_Image_Pixel",
        f"C:/Users/Rog Tuf/Desktop/MACHINE LEARNING/Old_Image_Pixel",
        "Original"
    )

for augmentation_type in augmentation_types:
    plot_combined_rgb_channels_normalized(
        f"C:/Users/Rog Tuf/Desktop/MACHINE LEARNING/Young_Image_Pixel_Augmented/{augmentation_type}",
        f"C:/Users/Rog Tuf/Desktop/MACHINE LEARNING/Old_Image_Pixel_Augmented/{augmentation_type}",
        augmentation_type
    )

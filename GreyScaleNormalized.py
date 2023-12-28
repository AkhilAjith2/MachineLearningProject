import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def calculate_grayscale_content(image):
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the histogram of the grayscale image
    hist = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])

    # Normalize the histogram
    hist /= hist.sum()

    return hist


def plot_histograms(directory_path1, directory_path2, augmentation_type):
    # Get the list of image files in the directories
    image_files_group1 = [f for f in os.listdir(directory_path1) if f.endswith('.jpg')]
    image_files_group2 = [f for f in os.listdir(directory_path2) if f.endswith('.jpg')]

    # Initialize histograms
    hist_group1 = np.zeros((256, 1))
    hist_group2 = np.zeros((256, 1))

    # Calculate histograms for the first group
    for image_file in image_files_group1:
        image_path = os.path.join(directory_path1, image_file)
        image = cv2.imread(image_path)
        hist_group1 += calculate_grayscale_content(image)

    # Calculate histograms for the second group
    for image_file in image_files_group2:
        image_path = os.path.join(directory_path2, image_file)
        image = cv2.imread(image_path)
        hist_group2 += calculate_grayscale_content(image)

    # Plot both histograms on the same plot
    plt.suptitle(f'GreyScale - {augmentation_type}', fontsize=16)
    plt.plot(hist_group1, color='red', label='Young')
    plt.plot(hist_group2, color='blue', label='Old')
    plt.title('Grayscale Content Histogram')
    plt.xlabel('Intensity Value')
    plt.ylabel('Normalized Frequency')
    plt.legend()
    plt.show()


augmentation_types = ["blurred", "color_jittered", "edge_enhanced", "enhanced", "flipped", "negative", "noisy",
                      "rotated_clockwise", "rotated_counterclockwise", "split_mirror", "zoomed"]
plot_histograms(
        f"C:/Users/Rog Tuf/Desktop/MACHINE LEARNING/Young_Image_Pixel",
        f"C:/Users/Rog Tuf/Desktop/MACHINE LEARNING/Old_Image_Pixel",
        "Original"
    )

for augmentation_type in augmentation_types:
    plot_histograms(
        f"C:/Users/Rog Tuf/Desktop/MACHINE LEARNING/Young_Image_Pixel_Augmented/{augmentation_type}",
        f"C:/Users/Rog Tuf/Desktop/MACHINE LEARNING/Old_Image_Pixel_Augmented/{augmentation_type}",
        augmentation_type
    )

# for augmentation_type in augmentation_types:
#     plot_histograms(
#         f"C:/Users/Rog Tuf/Desktop/MACHINE LEARNING/Young_Image_Pixel_GreyScale_Augmented/{augmentation_type}",
#         f"C:/Users/Rog Tuf/Desktop/MACHINE LEARNING/Old_Image_Pixel_GreyScale_Augmented/{augmentation_type}",
#         augmentation_type
#     )
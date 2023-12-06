import os
import cv2
import numpy as np
from tqdm import tqdm  # For progress bar, install with: pip install tqdm
from PIL import Image, ImageFilter

# Directory containing your image files
input_directory = "C:/Users/Rog Tuf/Desktop/MACHINE LEARNING/Young_Image_Pixel_Augmented/greyscale"

# Output directory for augmented images
output_directory = "C:/Users/Rog Tuf/Desktop/MACHINE LEARNING/Young_Image_Pixel_GreyScale_Augmented"
os.makedirs(output_directory, exist_ok=True)

# Function to rotate images
def rotate_image(image, angle, direction="clockwise"):
    rows, cols, _ = image.shape
    if direction == "clockwise":
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), -angle, 1)
    else:
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

# Function to flip images
def flip_image(image, flip_code):
    flipped_image = cv2.flip(image, flip_code)
    return flipped_image

# Function to add salt and pepper noise to images
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)

    # Salt noise
    salt_mask = np.random.rand(*image.shape[:2]) < salt_prob
    noisy_image[salt_mask] = 255

    # Pepper noise
    pepper_mask = np.random.rand(*image.shape[:2]) < pepper_prob
    noisy_image[pepper_mask] = 0

    return noisy_image

# Function to split and mirror images
def split_and_mirror_image(image):
    height, width, _ = image.shape
    split_point = width // 2

    left_half = image[:, :split_point, :]
    mirrored_right_half = cv2.flip(left_half, 1)

    augmented_image = np.hstack([left_half, mirrored_right_half])

    return augmented_image

# Augmentation parameters
rotation_angle = 30
flip_code = 1  # 0: vertical flip, 1: horizontal flip
salt_prob = 0.01
pepper_prob = 0.01

# Function to enhance contrast in images
def enhance_contrast(image, alpha=1.2, beta=10):
    enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced_image

def color_jittering(image, saturation_factor=0.5, brightness_factor=0.5):
    # Convert BGR image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Adjust saturation
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)

    # Adjust brightness
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * brightness_factor, 0, 255)

    # Convert back to BGR
    jittered_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return jittered_image

saturation_factor = 1.5  # Increase or decrease based on your preference
brightness_factor = 1  # Increase or decrease based on your preference

# Function to perform zoom augmentation on images
def zoom_image(image, zoom_factor_range=(0.8, 1.2)):
    if image is None:
        return None

    zoom_factor = np.random.uniform(*zoom_factor_range)

    # Get the center of the image
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2

    # Calculate new dimensions after zooming
    new_width = int(image.shape[1] * zoom_factor)
    new_height = int(image.shape[0] * zoom_factor)

    # Calculate the cropping box
    crop_x1 = max(center_x - new_width // 2, 0)
    crop_x2 = min(center_x + new_width // 2, image.shape[1])
    crop_y1 = max(center_y - new_height // 2, 0)
    crop_y2 = min(center_y + new_height // 2, image.shape[0])

    # Crop and resize the image
    zoomed_image = image[crop_y1:crop_y2, crop_x1:crop_x2, :]

    # Resize the image back to its original dimensions
    zoomed_image = cv2.resize(zoomed_image, (image.shape[1], image.shape[0]))

    return zoomed_image

# Augmentation parameters for zooming
zoom_factor_range = (0.8, 1.2)  # Adjust the range based on your preference

# Function to apply edge enhancement to images
def edge_enhancement(image):
    return cv2.filter2D(image, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))

# Function to apply blurring to images
def blur_image(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def convert_to_greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Loop through each image in the input directory
for filename in tqdm(os.listdir(input_directory)):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust file extensions as needed
        image_path = os.path.join(input_directory, filename)

        # Read the image
        original_image = cv2.imread(image_path)

        # Check if the image is successfully loaded
        if original_image is None:
            print(f"Failed to load image: {image_path}")
            continue

        augmentation_folder = "_".join([
            f"rotated{rotation_angle}",
            f"flipped{flip_code}",
            f"salt{salt_prob}",
            f"pepper{pepper_prob}",
            f"enhanced",
            f"split_mirror",
            f"color_jittered{saturation_factor}_{brightness_factor}",
            f"zoomed{zoom_factor_range[0]}_{zoom_factor_range[1]}",
            f"edge_enhanced",
            f"blurred",
            f"negative"
            # f"greyscale",
        ])

        # Apply augmentations
        rotated_image_clockwise = rotate_image(original_image, rotation_angle, direction="clockwise")
        rotated_image_counterclockwise = rotate_image(original_image, rotation_angle, direction="counterclockwise")
        flipped_image = flip_image(original_image, flip_code)
        noisy_image = add_salt_and_pepper_noise(original_image, salt_prob, pepper_prob)
        enhanced_image = enhance_contrast(original_image)
        augmented_image = split_and_mirror_image(original_image)
        color_jittered_image = color_jittering(original_image, saturation_factor, brightness_factor)
        zoomed_image = zoom_image(original_image, zoom_factor_range)
        edge_enhanced_image = edge_enhancement(original_image)
        blurred_image = blur_image(original_image)
        negative_image = cv2.bitwise_not(original_image)
        # greyscale_image = convert_to_greyscale(original_image)

        output_augmentation_directory = os.path.join(output_directory, augmentation_folder)
        os.makedirs(output_augmentation_directory, exist_ok=True)

        # Save flipped image to a separate folder
        rotated_clockwise_folder = os.path.join(output_augmentation_directory, "rotated_clockwise")
        os.makedirs(rotated_clockwise_folder, exist_ok=True)
        cv2.imwrite(os.path.join(rotated_clockwise_folder, f"{filename}_rotated_clockwise.jpg"),rotated_image_clockwise)

        # Save flipped image to a separate folder
        rotated_counterclockwise_folder = os.path.join(output_augmentation_directory, "rotated_counterclockwise")
        os.makedirs(rotated_counterclockwise_folder, exist_ok=True)
        cv2.imwrite(os.path.join(rotated_counterclockwise_folder, f"{filename}_rotated_counterclockwise.jpg"),
                    rotated_image_counterclockwise)

        # Save flipped image to a separate folder
        flipped_folder = os.path.join(output_augmentation_directory, "flipped")
        os.makedirs(flipped_folder, exist_ok=True)
        cv2.imwrite(os.path.join(flipped_folder, f"{filename}_flipped.jpg"), flipped_image)

        # Save noisy image to a separate folder
        noisy_folder = os.path.join(output_augmentation_directory, "noisy")
        os.makedirs(noisy_folder, exist_ok=True)
        cv2.imwrite(os.path.join(noisy_folder, f"{filename}_noisy.jpg"), noisy_image)

        # Save enhanced image to a separate folder
        enhanced_folder = os.path.join(output_augmentation_directory, "enhanced")
        os.makedirs(enhanced_folder, exist_ok=True)
        cv2.imwrite(os.path.join(enhanced_folder, f"{filename}_enhanced.jpg"), enhanced_image)

        # Save split and mirrored image to a separate folder
        split_mirror_folder = os.path.join(output_augmentation_directory, "split_mirror")
        os.makedirs(split_mirror_folder, exist_ok=True)
        cv2.imwrite(os.path.join(split_mirror_folder, f"{filename}_split_mirror.jpg"), augmented_image)

        # Save color-jittered image to a separate folder
        color_jittered_folder = os.path.join(output_augmentation_directory, "color_jittered")
        os.makedirs(color_jittered_folder, exist_ok=True)
        cv2.imwrite(os.path.join(color_jittered_folder, f"{filename}_color_jittered.jpg"), color_jittered_image)

        # Repeat this pattern for other augmentations...

        # Save zoomed image to a separate folder
        zoomed_folder = os.path.join(output_augmentation_directory, "zoomed")
        os.makedirs(zoomed_folder, exist_ok=True)
        cv2.imwrite(os.path.join(zoomed_folder, f"{filename}_zoomed.jpg"), zoomed_image)

        # Save edge-enhanced image to a separate folder
        edge_enhanced_folder = os.path.join(output_augmentation_directory, "edge_enhanced")
        os.makedirs(edge_enhanced_folder, exist_ok=True)
        cv2.imwrite(os.path.join(edge_enhanced_folder, f"{filename}_edge_enhanced.jpg"), edge_enhanced_image)

        # Save blurred image to a separate folder
        blurred_folder = os.path.join(output_augmentation_directory, "blurred")
        os.makedirs(blurred_folder, exist_ok=True)
        cv2.imwrite(os.path.join(blurred_folder, f"{filename}_blurred.jpg"), blurred_image)

        # Save negative image to a separate folder
        negative_folder = os.path.join(output_augmentation_directory, "negative")
        os.makedirs(negative_folder, exist_ok=True)
        cv2.imwrite(os.path.join(negative_folder, f"{filename}_negative.jpg"), negative_image)

        # # Save greyscale image to a separate folder
        # greyscale_folder = os.path.join(output_augmentation_directory, "greyscale")
        # os.makedirs(greyscale_folder, exist_ok=True)
        # cv2.imwrite(os.path.join(greyscale_folder, f"{filename}_greyscale.jpg"), greyscale_image)

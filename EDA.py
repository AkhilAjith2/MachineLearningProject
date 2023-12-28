import os
import matplotlib.pyplot as plt

def count_images_per_age(root_folder):
    age_counts = {}

    for age_folder in os.listdir(root_folder):
        age_path = os.path.join(root_folder, age_folder)
        if os.path.isdir(age_path):
            image_count = sum(len(files) for _, _, files in os.walk(age_path))
            age_counts[int(age_folder)] = image_count

    return age_counts

def plot_histogram(age_counts):
    ages = list(age_counts.keys())
    counts = list(age_counts.values())

    plt.bar(ages, counts, color='blue')
    plt.xlabel('Age')
    plt.ylabel('Number of Pictures')
    plt.title('Number of Pictures per Age')
    plt.show()

if __name__ == "__main__":
    root_folder = "C:/Users/Rog Tuf/Desktop/Classifed_dataset"
    age_counts = count_images_per_age(root_folder)
    plot_histogram(age_counts)

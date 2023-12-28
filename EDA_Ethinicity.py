import os
import matplotlib.pyplot as plt

# Path to the main directory containing the images
main_path = "C:/Users/Rog Tuf/Desktop/Classifed_dataset"

ethnicities = ["Black", "Asian", "White", "Indian", "Others"]

for ethnicity in ethnicities:
    age_counts = []

    for age in range(1, 101):
        age_path = os.path.join(main_path, str(age), ethnicity)

        if os.path.exists(age_path):
            # Counting the number of image files in the age and ethnicity directory
            count = sum([len(files) for _, _, files in os.walk(age_path)])
            age_counts.append((age, count))

    # Unpacking age and count data for plotting
    ages, counts = zip(*age_counts)

    # Plotting histogram for each ethnicity
    plt.figure(figsize=(10, 6))
    plt.bar(ages, counts, color='skyblue')
    plt.xlabel('Age')
    plt.ylabel('Image Count')
    plt.title(f'Image Count for {ethnicity.capitalize()} ethnicity')
    plt.grid(axis='y')
    plt.show()
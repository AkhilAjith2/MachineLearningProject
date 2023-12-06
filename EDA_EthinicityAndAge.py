import os
import matplotlib.pyplot as plt

# Path to the main directory containing the images
main_path = "C:/Users/Rog Tuf/Desktop/Classifed_dataset"

ethnicities = ["Black", "Asian", "White", "Indian", "Others"]
genders = ["male", "female"]

fig, axs = plt.subplots(len(ethnicities), len(genders), figsize=(12, 16))

for i, ethnicity in enumerate(ethnicities):
    for j, gender in enumerate(genders):
        gender_counts = []

        for age in range(1, 101):
            gender_path = os.path.join(main_path, str(age), ethnicity, gender)

            if os.path.exists(gender_path):
                # Counting the number of image files in the age, ethnicity, and gender directory
                count = sum([len(files) for _, _, files in os.walk(gender_path)])
                gender_counts.append((age, count))

        # Unpacking age and count data for plotting
        ages, counts = zip(*gender_counts)

        # Plotting histogram for each ethnicity and gender
        ax = axs[i, j]
        ax.bar(ages, counts, color='skyblue')
        ax.set_xlabel('Age')
        ax.set_ylabel('Image Count')
        ax.set_title(f'{ethnicity.capitalize()} - {gender.capitalize()}')
        ax.grid(axis='y')

plt.tight_layout()
plt.show()
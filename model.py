import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

data_directory = "C:/Users/Rog Tuf/Desktop/Classified_dataset"  # Replace with the path to your dataset

# Initialize empty lists to store data and labels
data = []
labels = []

# Loop through age folders
for age_folder in os.listdir(data_directory):
    age_path = os.path.join(data_directory, age_folder)

    # Loop through race folders
    for race_folder in os.listdir(age_path):
        race_path = os.path.join(age_path, race_folder)

        # Loop through gender folders
        for gender_folder in os.listdir(race_path):
            gender_path = os.path.join(race_path, gender_folder)

            # Loop through images
            for image_file in os.listdir(gender_path):
                image_path = os.path.join(gender_path, image_file)

                # Check if the current item is a file
                if os.path.isfile(image_path):
                    # Read and preprocess the image
                    image = load_img(image_path, target_size=(64, 64))
                    image = img_to_array(image)
                    image = image / 255.0  # Normalize the pixel values

                    # Append the image and corresponding label to the lists
                    data.append(image)
                    labels.append(int(age_folder))

# Convert lists to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# One-hot encode the age labels
labels = to_categorical(labels, num_classes=101)  # Assuming age ranges from 0 to 100

# Split the data into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(101, activation="softmax"))  # Output layer with 101 units for age prediction

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(trainX, trainY, validation_data=(testX, testY), epochs=50, batch_size=32)

model.save("age_prediction_model_2.keras")

# import os
# import cv2
# import numpy as np
# from sklearn.model_selection import train_test_split
# from keras.utils import to_categorical
# from keras.preprocessing.image import img_to_array, load_img
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
#
# data_directory = "C:/Users/Rog Tuf/Desktop/Classified_dataset"
#
# data = []
# labels = []
# race_mapping = {'White': 0, 'Black': 1, 'Asian': 2, 'Indian': 3, 'Others': 4}
#
# # Loop through age folders
# for age_folder in os.listdir(data_directory):
#     age_path = os.path.join(data_directory, age_folder)
#
#     # Loop through race folders
#     for race_folder in os.listdir(age_path):
#         race_path = os.path.join(age_path, race_folder)
#
#         # Loop through gender folders
#         for gender_folder in os.listdir(race_path):
#             gender_path = os.path.join(race_path, gender_folder)
#
#             # Loop through images
#             for image_file in os.listdir(gender_path):
#                 image_path = os.path.join(gender_path, image_file)
#
#                 # Check if the current item is a file
#                 if os.path.isfile(image_path):
#                     # Read and preprocess the image
#                     image = load_img(image_path, target_size=(64, 64))
#                     image = img_to_array(image)
#                     image = image / 255.0  # Normalize the pixel values
#
#                     # Append the image and corresponding label to the lists
#                     data.append(image)
#
#                     # Assign labels based on the order: gender, race, age
#                     gender_label = 1 if gender_folder.lower() == 'male' else 0
#                     race_label = race_mapping[race_folder]
#
#                     # Concatenate gender and race labels with age
#                     labels.append([gender_label, race_label, int(age_folder)])
#
# # Convert lists to NumPy arrays
# data = np.array(data, dtype="float32")
# labels = np.array(labels)
#
# # Split the data into training and testing sets
# (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
#
# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation="relu"))
# model.add(Dense(101, activation="softmax"))  # Output layer with 101 units for age prediction
#
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
#
# model.fit(trainX, trainY, validation_data=(testX, testY), epochs=10, batch_size=32)
#
# model.save("age_prediction_model_without25-28.keras")
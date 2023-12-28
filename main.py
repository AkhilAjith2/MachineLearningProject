import os
import cv2
import numpy as np
from keras.models import load_model
from facenet_pytorch import MTCNN, InceptionResnetV1
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.image import resize

# model = load_model("age_prediction_model_2.keras")
age_model = load_model("C:/Users/Rog Tuf/PycharmProjects/Input/age_only_model_100epoch_10patience_by_ages(80)_subset_130.keras")
# Create an instance of the MTCNN detector
mtcnn = MTCNN(keep_all=True, min_face_size=20, thresholds=[0.6, 0.7, 0.8])
# Create an instance of the FaceNet model
face_net = InceptionResnetV1(pretrained='vggface2').eval()


def preprocess_face_image(face_image, target_size=(200, 200)):
    # Check if the face image has non-zero dimensions
    if face_image.shape[0] == 0 or face_image.shape[1] == 0:
        return None  # Skip processing for invalid face images

    # Resize the face image to the target size
    face_image = resize(img_to_array(face_image), target_size)

    # Normalize pixel values to the range [0, 1]
    face_image = face_image / 255.0

    # Expand dimensions to create a batch of size 1
    face_image = np.expand_dims(face_image, axis=0)

    return face_image


def predict_age(face_image):
    # Preprocess the face image for the model
    processed_face = preprocess_face_image(face_image)

    # Use the age and gender model to predict age and gender
    predictions = age_model.predict(processed_face)
    # Extract the age group index (adjust indices based on your model output)
    age = predictions[0][0]

    return int(age)


def detect_bounding_box(frame):
    height, width = frame.shape[:2]

    # Convert the frame to RGB (required by facenet_pytorch)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using the MTCNN model
    boxes, _ = mtcnn.detect(frame_rgb)

    faces = []
    if boxes is not None:
        # Check if boxes is a 2D array
        if len(boxes.shape) == 2:
            for box in boxes:
                (startX, startY, endX, endY) = box.astype("int")

                # Extract the face region for age and gender prediction
                face_image = frame[startY:endY, startX:endX]

                # Age and gender prediction
                age, gender = predict_age(face_image)

                # Draw bounding box
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 4)

                # Display age and gender over the bounding box
                cv2.putText(
                    frame,
                    f"Age: {age}, Gender: {gender}",
                    (startX, startY + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

                faces.append((startX, startY, endX - startX, endY - startY))
        else:
            # Use the single box directly
            box = boxes.astype("int")
            (startX, startY, endX, endY) = box

            # Extract the face region for age and gender prediction
            face_image = frame[startY:endY, startX:endX]

            # Age and gender prediction
            age, gender = predict_age(face_image)

            # Draw bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 4)

            # Display age and gender over the bounding box
            cv2.putText(
                frame,
                f"Age: {age}, Gender: {gender}",
                (startX, startY + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            faces.append((startX, startY, endX - startX, endY - startY))

    return faces

def process_webcam():
    video_capture = cv2.VideoCapture(0)
    while True:
        result, video_frame = video_capture.read()
        if result is False:
            break
        faces = detect_bounding_box(video_frame)
        cv2.imshow("My Face Detection Project", video_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def process_video(input_path, output_path):
    video_capture = cv2.VideoCapture(input_path)

    if not video_capture.isOpened():
        print("Error opening video file")
        return

    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width, frame_height))

    while True:
        result, video_frame = video_capture.read()

        if not result:
            break

        faces = detect_bounding_box(video_frame)
        out.write(video_frame)

        cv2.imshow("My Face Detection Project", video_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

 # For video file
# video_input_path = "C:/Users/David Abraham/Downloads/Faces of the World - 10 YEARS OF TRAVEL (1) (online-video-cutter.com).mp4"
# video_output_path = "C:/Users/David Abraham/Downloads/edited_video.mp4"
# process_video(video_input_path, video_output_path)

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, filename)
            img = cv2.imread(image_path)

            faces = detect_bounding_box(img)

            output_path = os.path.join(output_dir, f"output_{filename}")
            cv2.imwrite(output_path, img)

    cv2.destroyAllWindows()

# image_input_dir = "C:/Users/David Abraham/Desktop/Classifed_dataset/39/Indian/male"
# image_output_dir = "C:/Users/David Abraham/Desktop/face_recog_test"
# process_images(image_input_dir, image_output_dir)
import os
import cv2
import numpy as np

# Specify the correct paths to the model files
prototxt_path = "C:/Users/Rog Tuf/PycharmProjects/Input/deploy.prototxt"
caffemodel_path = "C:/Users/Rog Tuf/PycharmProjects/Input/res10_300x300_ssd_iter_140000.caffemodel"

face_net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

def predict_age(face_image):
    # TODO: Implement age prediction logic (replace this with a real age prediction model)
    return "NAN"  # Placeholder value

def detect_bounding_box(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face region for age prediction
            face_image = frame[startY:endY, startX:endX]

            # Age prediction
            age = predict_age(face_image)

            # Draw bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 4)

            # Display age over the bounding box
            cv2.putText(
                frame,
                f"Age: {age}",
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

        cv2.imshow("Face Detection", video_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Example usage for webcam
# process_webcam()

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

        cv2.imshow("Face Detection", video_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

# For video file
# video_input_path = "C:/Users/Rog Tuf/Desktop/video.mp4"
# video_output_path = "C:/Users/Rog Tuf/Desktop/edited_video.mp4"
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

image_input_dir = "C:/Users/Rog Tuf/Desktop/Old"
image_output_dir = "C:/Users/Rog Tuf/Desktop/Old_detected"
process_images(image_input_dir, image_output_dir)

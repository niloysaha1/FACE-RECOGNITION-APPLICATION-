import os
import cv2
import numpy as np
from PIL import Image

# Path to the Haar Cascade for face detection
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Initialize LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Path to dataset directory
path = "dataSet"

def get_images_with_ids(path):
    # Collect image paths from the dataset directory
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []

    for single_image_path in image_paths:
        # Open image, convert to grayscale
        faceImg = Image.open(single_image_path).convert("L")
        faceNp = np.array(faceImg, np.uint8)

        # Extract ID from the image filename
        id = int(os.path.split(single_image_path)[-1].split(".")[1])
        print(f"Processing ID: {id}")

        # Append face image and ID to lists
        faces.append(faceNp)
        ids.append(id)

        # Display the image being processed (optional)
        cv2.imshow("Training on image...", faceNp)
        cv2.waitKey(10)

    return np.array(ids), faces

# Call the function and get the ids and face arrays
ids, faces = get_images_with_ids(path)

# Train the recognizer on the dataset
recognizer.train(faces, ids)

# Ensure directory exists before saving model
save_dir = "recognizer/trainingdata"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the trained model
recognizer.save(f"{save_dir}/trainingdata.yml")

# Close any OpenCV windows
cv2.destroyAllWindows()
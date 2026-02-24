import cv2
import numpy as np
import os
import sqlite3

# Load face cascade and recognizer
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/trainingdata/trainingdata.yml")

# Initialize camera
cam = cv2.VideoCapture(0)


# Function to retrieve profile from SQLite database
def getprofile(id):
    conn = sqlite3.connect("sqlite.db")
    cursor = conn.execute("SELECT * FROM students WHERE ID=?", (id,))
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile


# Main loop for face detection and recognition
while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    # Process each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Predict ID and confidence for the detected face
        id, conf = recognizer.predict(gray[y:y + h, x:x + w])

        # Retrieve the profile from the database
        profile = getprofile(id)
        print(profile)  # Optional: Print profile for debugging

        # Display profile info on the image
        if profile is not None:
            cv2.putText(img, "Name: " + str(profile[1]), (x, y + h + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
            cv2.putText(img, "Age: " + str(profile[2]), (x, y + h + 45), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)

    # Display the image with detections and profile information
    cv2.imshow("FACE", img)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
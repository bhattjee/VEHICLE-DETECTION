import cv2
import urllib.request
import os

# Download Haar cascade files if they don't exist
def download_cascade(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists")

# URLs for OpenCV Haar cascade classifiers (raw files from GitHub)
face_cascade_url = "haarcascade_frontalface_default.xml"
eye_cascade_url = "haarcascade_eye_tree_eyeglasses.xml"


# Replace 0 with the correct camera index if you have multiple cameras
cap = cv2.VideoCapture(0)

# Load the face cascade classifier (now using downloaded files)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade2 = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

# Check if the cascades loaded successfully
if face_cascade.empty():
    print("Error loading face cascade")
    exit()
if face_cascade2.empty():
    print("Error loading eye cascade")
    exit()

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error opening camera")
    exit()

# Define color for bounding box (green)
green = (0, 255, 0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is read correctly
    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    # Convert frame to grayscale for faster face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces2 = face_cascade2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw green squares around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), green, 2)
    for (x, y, w, h) in faces2:
        cv2.rectangle(frame, (x, y), (x+w, y+h), green, 2)

    # Display the resulting frame
    cv2.imshow('Camera with Face Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
import cv2
import dlib
import os
import matplotlib.pyplot as plt

# Load the models
haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
dlib_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')


def load_image_from_local(image_path):
    """Load image from the local file system and convert to OpenCV format."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image from {image_path}")
    return image


def detect_faces_opencv(image):
    """Detect faces using OpenCV's Haar cascade."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = haar_face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    return faces


def detect_faces_dlib(image):
    """Detect faces using Dlib's CNN-based face detector."""
    faces = dlib_detector(image, 1)
    return [(face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()) for face in faces]


def process_image(image_path):
    """Process image by detecting faces."""
    image = load_image_from_local(image_path)
    if image is None:  # Skip processing if the image couldn't be loaded
        print(f"Skipping image processing for: {image_path}")
        return

    faces_opencv = detect_faces_opencv(image)
    faces_dlib = detect_faces_dlib(image)

    # Draw the detected faces using OpenCV (blue rectangles)
    for (x, y, w, h) in faces_opencv:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box for OpenCV detected face

    # Draw the detected faces using Dlib (red rectangles)
    for (x, y, w, h) in faces_dlib:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box for Dlib detected face

    # Display the image with bounding boxes
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def process_images_in_directory(directory_path):
    """Process all images in the specified directory."""
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Processing image: {filename}")
            process_image(file_path)
        else:
            print(f"Skipping non-image file: {filename}")


# Specify the path to the folder containing the images
image_directory = 'images/train'
process_images_in_directory(image_directory)

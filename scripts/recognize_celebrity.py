import cv2
import numpy as np
import joblib

# Load the model and label encoder
model = joblib.load('../model/celebrity_face_recognition_model.pkl')
label_encoder = joblib.load('../model/label_encoder.pkl')

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')

def detect_face(image):
    """
    Detect a face in the input image and return the cropped face.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None

    # Take the first detected face
    x, y, w, h = faces[0]
    cropped_face = image[y:y+h, x:x+w]
    return cropped_face

def recognize_celebrity_from_face(cropped_face, model, label_encoder):
    """
    Recognize the celebrity using the trained SVM model.
    """
    resized_face = cv2.resize(cropped_face, (32, 32))  # Ensure consistent size
    face_flattened = resized_face.flatten().reshape(1, -1)

    # Predict the celebrity
    prediction = model.predict(face_flattened)
    probabilities = model.predict_proba(face_flattened)
    confidence = np.max(probabilities)  # Maximum probability as confidence score

    celebrity = label_encoder.inverse_transform(prediction)[0]
    return celebrity, confidence

def identify_celebrity(image_path):
    """
    Full pipeline to detect and recognize a celebrity from an image path.
    """
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Image at {image_path} could not be loaded. Please check the file path.")

    # Detect face
    face = detect_face(img)
    if face is None:
        raise ValueError("No face detected in the image. Please try another image.")

    # Recognize celebrity
    celebrity, confidence = recognize_celebrity_from_face(face, model, label_encoder)
    return celebrity, confidence

if __name__ == "__main__":
    image_path = input("Enter the path of the image: ")
    try:
        celebrity, confidence = identify_celebrity(image_path)
        print(f"Celebrity: {celebrity}, Confidence: {confidence:.2f}")
    except Exception as e:
        print(f"Error: {e}")

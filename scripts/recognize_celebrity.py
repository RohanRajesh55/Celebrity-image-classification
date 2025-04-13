import os
import cv2
import numpy as np
import joblib
import argparse

# Get the current directory of this script
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

# Build paths according to the updated file structure
MODEL_PATH = os.path.join(CURRENT_DIR, '..', 'models', 'saved', 'celebrity_face_recognition_model.pkl')
LE_PATH = os.path.join(CURRENT_DIR, '..', 'models', 'saved', 'label_encoder.pkl')
HAAR_PATH = os.path.join(CURRENT_DIR, '..', 'haarcascades', 'haarcascade_frontalface_default.xml')

# Load the pre-trained model and label encoder using joblib
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LE_PATH)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(HAAR_PATH)

def detect_face(image):
    """
    Detect a face in the input image using Haar cascades and return the cropped face.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        return None

    # Take the first detected face and return the cropped region
    x, y, w, h = faces[0]
    cropped_face = image[y:y+h, x:x+w]
    return cropped_face

def recognize_celebrity_from_face(cropped_face, model, label_encoder):
    """
    Recognize the celebrity from the cropped face image.
    Resize the face to 32x32 (as during training), flatten it, and predict.
    """
    # Resize the cropped face to the size used during training
    resized_face = cv2.resize(cropped_face, (32, 32))
    face_flattened = resized_face.flatten().reshape(1, -1)
    
    # Predict the class and compute confidence score
    prediction = model.predict(face_flattened)
    probabilities = model.predict_proba(face_flattened)
    confidence = np.max(probabilities)
    
    celebrity = label_encoder.inverse_transform(prediction)[0]
    return celebrity, confidence

def identify_celebrity(image_path):
    """
    The full pipeline: load an image, detect a face,
    and recognize the celebrity.
    Returns the original image, predicted celebrity, and confidence.
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}.")
    
    face = detect_face(image)
    if face is None:
        raise ValueError("No face detected in the image. Please try a different image.")
    
    celebrity, confidence = recognize_celebrity_from_face(face, model, label_encoder)
    return image, celebrity, confidence

def display_result(image, celebrity, confidence):
    """
    Displays the image along with the prediction in a new canvas.
    The canvas includes an additional margin at the bottom to show the text clearly.
    """
    # Prepare the prediction text
    text = f"{celebrity} ({confidence:.2f})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    line_type = cv2.LINE_AA

    # Get text size and baseline
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    padding = 10
    margin = text_height + baseline + 2 * padding

    # Create a new canvas with extra space at the bottom for the text
    h, w, channels = image.shape
    canvas = np.zeros((h + margin, w, 3), dtype=np.uint8)
    canvas[:h, :] = image

    # Fill the margin area with a background color (black)
    canvas[h:, :] = (0, 0, 0)

    # Position the text centrally in the margin area
    text_x = (w - text_width) // 2
    text_y = h + padding + text_height

    # Draw the text on the canvas
    cv2.putText(canvas, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness, line_type)

    # Create a resizable window and display the canvas
    window_name = "Recognition Result"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    cv2.imshow(window_name, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Recognize a celebrity from an image.")
    parser.add_argument("image_path", help="Path to the input image file")
    args = parser.parse_args()
    
    try:
        image, celebrity, confidence = identify_celebrity(args.image_path)
        print(f"Celebrity: {celebrity}, Confidence: {confidence:.2f}")
        display_result(image, celebrity, confidence)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
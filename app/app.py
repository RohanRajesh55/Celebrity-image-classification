from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import cv2
from werkzeug.utils import secure_filename
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Set upload folder and allowed extensions
UPLOAD_FOLDER = './uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model and label encoder
model = joblib.load('../model/celebrity_face_recognition_model.pkl')
label_encoder = joblib.load('../model/label_encoder.pkl')

# Load Haar Cascade for face detection
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
    resized_face = cv2.resize(cropped_face, (32, 32))
    face_flattened = resized_face.flatten().reshape(1, -1)

    # Predict the celebrity
    prediction = model.predict(face_flattened)
    probabilities = model.predict_proba(face_flattened)
    confidence = np.max(probabilities)

    celebrity = label_encoder.inverse_transform(prediction)[0]
    return celebrity, confidence

@app.route('/')
def index():
    """
    Render the main webpage.
    """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """
    Handle image upload and process it to identify the celebrity.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Load the image
    img = cv2.imread(file_path)
    if img is None:
        return jsonify({"error": "Invalid image format"}), 400

    # Detect face and recognize celebrity
    face = detect_face(img)
    if face is None:
        return jsonify({"error": "No face detected in the image"}), 400

    celebrity, confidence = recognize_celebrity_from_face(face, model, label_encoder)

    # Return the result, including the image path
    return jsonify({
        "celebrity": celebrity,
        "confidence": round(confidence, 2),
        "image_url": f"/uploads/{filename}"
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve uploaded files to the frontend.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

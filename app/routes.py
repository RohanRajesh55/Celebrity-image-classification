from flask import Blueprint, current_app, render_template, request, jsonify, send_from_directory
import os
import cv2
from werkzeug.utils import secure_filename
import numpy as np

# Create a blueprint for the main routes
main = Blueprint('main', __name__)

def allowed_file(filename):
    """Check if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def detect_face(image):
    """Detect a face in the input image using the preloaded Haar cascade and return the cropped face."""
    face_cascade = current_app.config.get('face_cascade')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return image[y:y+h, x:x+w]

def recognize_celebrity_from_face(cropped_face):
    """
    Resize the cropped face to 32x32 pixels (the size used during training),
    flatten it, and use the preloaded model and label encoder to predict the celebrity.
    Returns the predicted celebrity and the confidence score.
    """
    model = current_app.config.get('model')
    label_encoder = current_app.config.get('label_encoder')
    resized_face = cv2.resize(cropped_face, (32, 32))
    face_flattened = resized_face.flatten().reshape(1, -1)
    prediction = model.predict(face_flattened)
    probabilities = model.predict_proba(face_flattened)
    confidence = np.max(probabilities)
    celebrity = label_encoder.inverse_transform(prediction)[0]
    return celebrity, confidence

@main.route('/', methods=['GET'])
def index():
    """Render the main page (index.html)."""
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload():
    """
    Handle the AJAX file upload from the front end.
    Save the uploaded file, perform face detection and celebrity recognition,
    then return a JSON response with the result and uploaded image URL.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = cv2.imread(file_path)
        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        face = detect_face(img)
        if face is None:
            return jsonify({"error": "No face detected in the image"}), 400

        celebrity, confidence = recognize_celebrity_from_face(face)
        image_url = f"/static/uploads/{filename}"
        return jsonify({
            "celebrity": celebrity,
            "confidence": round(confidence, 2),
            "image_url": image_url
        })

    return jsonify({"error": "Invalid file type"}), 400

@main.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve the uploaded files (images) from the upload directory."""
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)

# Integrated run logic: This allows you to start the app by running this file directly.
if __name__ == "__main__":
    from app import create_app  # Import the factory function from app/__init__.py
    app = create_app()         # Create the Flask application instance
    app.register_blueprint(main)  # Register our blueprint (if not already registered in __init__)
    app.run(debug=True)
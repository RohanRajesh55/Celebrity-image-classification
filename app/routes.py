from flask import Blueprint, render_template, request, jsonify, url_for, send_from_directory, current_app
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np

main = Blueprint("main", __name__)

def allowed_file(filename):
    # Check for allowed file extensions.
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {"png", "jpg", "jpeg"}

def detect_face(image):
    # Perform face detection using the Haar cascade loaded in the app config.
    face_cascade = current_app.config["face_cascade"]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return image[y:y+h, x:x+w]

def recognize_celebrity_from_face(cropped_face):
    model = current_app.config["model"]
    label_encoder = current_app.config["label_encoder"]
    resized_face = cv2.resize(cropped_face, (32, 32))
    face_flattened = resized_face.flatten().reshape(1, -1)
    prediction = model.predict(face_flattened)
    probabilities = model.predict_proba(face_flattened)
    confidence = np.max(probabilities)
    celebrity = label_encoder.inverse_transform(prediction)[0]
    return celebrity, confidence

@main.route("/")
def index():
    return render_template("index.html")

@main.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        
        image = cv2.imread(file_path)
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400
        
        face = detect_face(image)
        if face is None:
            return jsonify({"error": "No face detected in the image"}), 400
        
        celebrity, confidence = recognize_celebrity_from_face(face)
        image_url = url_for("static", filename="uploads/" + filename)
        return jsonify({
            "celebrity": celebrity,
            "confidence": round(confidence, 2),
            "image_url": image_url
        }), 200
    
    return jsonify({"error": "Invalid file type"}), 400

@main.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)
from flask import Flask
import os
import joblib
import cv2

def create_app():
    app = Flask(__name__)
    
    # Set the secret key from an environment variable; fallback used for development.
    app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'my_super_secret_key_2025!@#')
    
    # Configure the uploads folder (static/uploads) for serving uploaded images.
    upload_folder = os.path.join(app.root_path, 'static', 'uploads')
    app.config['UPLOAD_FOLDER'] = upload_folder
    os.makedirs(upload_folder, exist_ok=True)
    
    # Define the folder for model artifacts.
    # Note: We now use the "models/saved" folder.
    model_folder = os.path.join(app.root_path, '..', 'models', 'saved')
    
    # Build absolute paths for the model and label encoder.
    model_path = os.path.abspath(os.path.join(model_folder, 'celebrity_face_recognition_model.pkl'))
    label_encoder_path = os.path.abspath(os.path.join(model_folder, 'label_encoder.pkl'))
    
    # Build absolute path for the Haar cascade file.
    haar_path = os.path.abspath(os.path.join(app.root_path, '..', 'haarcascades', 'haarcascade_frontalface_default.xml'))
    
    # Check that the files exist; if not, raise a clear error.
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please ensure that 'celebrity_face_recognition_model.pkl' exists in the 'models/saved' folder."
        )
    
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(
            f"Label encoder file not found at {label_encoder_path}. Please ensure that 'label_encoder.pkl' exists in the 'models/saved' folder."
        )
    
    if not os.path.exists(haar_path):
        raise FileNotFoundError(
            f"Haar cascade file not found at {haar_path}. Please ensure that 'haarcascade_frontalface_default.xml' exists in the 'haarcascades' folder."
        )
    
    # Load the pre-trained model and label encoder using joblib.
    app.config['model'] = joblib.load(model_path)
    app.config['label_encoder'] = joblib.load(label_encoder_path)
    
    # Load the Haar cascade classifier for face detection.
    app.config['face_cascade'] = cv2.CascadeClassifier(haar_path)
    
    # Register the blueprint for routes defined in app/routes.py.
    from app.routes import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    return app
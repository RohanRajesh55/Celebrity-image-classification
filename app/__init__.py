from flask import Flask
import os
import joblib
import cv2

def create_app():
    app = Flask(__name__)
    
    # Set the secret key (use an environment variable in production)
    app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'my_super_secret_key_2025!@#')
    
    # Configure the uploads folder
    upload_folder = os.path.join(app.root_path, 'static', 'uploads')
    app.config['UPLOAD_FOLDER'] = upload_folder
    os.makedirs(upload_folder, exist_ok=True)
    
    # Define the model folder and construct file paths for the artifacts
    model_folder = os.path.join(app.root_path, '..', 'models', 'saved')
    model_path = os.path.abspath(os.path.join(model_folder, 'celebrity_face_recognition_model.pkl'))
    label_encoder_path = os.path.abspath(os.path.join(model_folder, 'label_encoder.pkl'))
    haar_path = os.path.abspath(os.path.join(app.root_path, '..', 'haarcascades', 'haarcascade_frontalface_default.xml'))
    
    # Ensure all required files are present
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}.")
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder file not found at {label_encoder_path}.")
    if not os.path.exists(haar_path):
        raise FileNotFoundError(f"Haar cascade file not found at {haar_path}.")
    
    # Load artifacts and save them in the config
    app.config['model'] = joblib.load(model_path)
    app.config['label_encoder'] = joblib.load(label_encoder_path)
    app.config['face_cascade'] = cv2.CascadeClassifier(haar_path)
    
    # Register the blueprint from routes.py
    from app.routes import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    return app
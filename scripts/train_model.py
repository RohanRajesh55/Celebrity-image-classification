import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle

# Define paths
RAW_DATA_DIR = '../datasets/raw/'        # Raw images
CROPPED_DATA_DIR = '../datasets/cropped/'  # Cropped images to be saved
MODEL_DIR = '../model/'                  # Folder to save the trained model

# Ensure directories exist
os.makedirs(CROPPED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_eye.xml')

# Function to crop the face if at least two eyes are detected
def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 2:  # If at least 2 eyes are detected, crop and return
            return roi_color
    return None

# Preprocessing: Create cropped images
count = 1
for celeb_name in os.listdir(RAW_DATA_DIR):
    celeb_path = os.path.join(RAW_DATA_DIR, celeb_name)
    
    if os.path.isdir(celeb_path):  # Ensure it's a directory
        print(f"Processing images for: {celeb_name}")
        
        # Loop through each image in the celebrity folder
        for img_name in os.listdir(celeb_path):
            img_path = os.path.join(celeb_path, img_name)
            cropped_face = get_cropped_image_if_2_eyes(img_path)
            
            if cropped_face is not None:  # Save only valid cropped faces
                cropped_folder = os.path.join(CROPPED_DATA_DIR, celeb_name)
                os.makedirs(cropped_folder, exist_ok=True)  # Create subfolder for celebrity
                
                cropped_file_name = f"{celeb_name}{count}.jpg"  # Unique name for each file
                cropped_file_path = os.path.join(cropped_folder, cropped_file_name)
                
                success = cv2.imwrite(cropped_file_path, cropped_face)
                if success:
                    print(f"Saved cropped image: {cropped_file_path}")
                else:
                    print(f"Failed to save cropped image: {cropped_file_path}")
                
                count += 1

# Prepare data for model training
X = []
y = []

# Loop through cropped directory to get images for training
for celeb_name in os.listdir(CROPPED_DATA_DIR):
    celeb_folder = os.path.join(CROPPED_DATA_DIR, celeb_name)
    
    if os.path.isdir(celeb_folder):
        # Get list of images in the folder
        image_files = os.listdir(celeb_folder)
        if not image_files:  # Skip empty folders
            print(f"Skipping empty folder: {celeb_folder}")
            continue
        
        print(f"Processing folder: {celeb_folder}")
        for img_name in image_files:
            img_path = os.path.join(celeb_folder, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                resized_img = cv2.resize(img, (32, 32))  # Resize to standard size (32x32)
                X.append(resized_img.flatten())  # Flatten for SVM input
                y.append(celeb_name)

# Check if we have any data for training
if not X or not y:
    raise ValueError("No images found for training. Ensure cropped folders contain valid images.")

# Convert data to numpy arrays
X = np.array(X)
y = np.array(y)

# Encode labels (celebrity names)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train SVM model
print("Training model...")
model = SVC(kernel='linear', probability=True)
model.fit(X, y_encoded)

# Save the trained model and label encoder
model_path = os.path.join(MODEL_DIR, 'celebrity_face_recognition_model.pkl')
le_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')

with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)

with open(le_path, 'wb') as le_file:
    pickle.dump(le, le_file)

print(f"Model saved at: {model_path}")
print(f"Label encoder saved at: {le_path}")

import os
import cv2
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Get the current directory of this script
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

# Define paths according to the updated file structure
CROPPED_DATA_DIR = os.path.join(CURRENT_DIR, '..', 'datasets', 'cropped')
MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models', 'saved')

# Ensure the models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Prepare data for model training by processing each celebrity's cropped images
X = []
y = []

print("Loading cropped images for training...")
for celeb_name in os.listdir(CROPPED_DATA_DIR):
    celeb_folder = os.path.join(CROPPED_DATA_DIR, celeb_name)
    if os.path.isdir(celeb_folder):
        image_files = os.listdir(celeb_folder)
        if not image_files:
            print(f"Skipping empty folder: {celeb_folder}")
            continue
        print(f"Processing folder: {celeb_folder}")
        for img_name in image_files:
            img_path = os.path.join(celeb_folder, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                # Resize image to a standard size (32x32) and flatten it for classifier input
                resized_img = cv2.resize(img, (32, 32))
                X.append(resized_img.flatten())
                y.append(celeb_name)

# Verify that there is data to train on
if not X or not y:
    raise ValueError("No images found for training. Please ensure cropped folders contain valid images.")

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Encode labels so that celebrity names become numeric classes
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Define a set of candidate models for comparison
models = {
    'SVM': SVC(kernel='linear', probability=True, random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
}

results = {}

# Train each model and evaluate its performance on the validation set
for name, clf in models.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    results[name] = acc
    print(f"Accuracy for {name}: {acc:.4f}")

# Determine the best-performing model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
best_acc = results[best_model_name]
print(f"\nBest model selected: {best_model_name} with an accuracy of {best_acc:.4f}")

# Save the best model and label encoder
model_path = os.path.join(MODEL_DIR, 'celebrity_face_recognition_model.pkl')
le_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')

with open(model_path, 'wb') as model_file:
    pickle.dump(best_model, model_file)

with open(le_path, 'wb') as le_file:
    pickle.dump(le, le_file)

print(f"Model saved at: {model_path}")
print(f"Label encoder saved at: {le_path}")
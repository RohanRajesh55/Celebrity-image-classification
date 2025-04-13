# Celebrity Face Recognition

This project is a machine learning-powered application designed to recognize celebrity faces in uploaded images. The app uses computer vision techniques like Haar cascades for face detection and a trained classification model (e.g., SVM or others) for identification. Additionally, it offers an interactive **Flask-based web interface** that allows users to upload images and receive recognition results instantly.

---

## 🚀 Features

- **Face Detection:** Utilizes OpenCV's Haar cascades for identifying faces in images.
- **Celebrity Identification:** Recognizes faces of celebrities based on a trained machine learning model.
- **Interactive Web App:** Includes a Flask-based app with a simple and user-friendly interface for uploading images and viewing results.
- **Pre-Trained Models:** Includes pre-trained models (e.g., SVM) for immediate use.
- **Modular Structure:** Separates concerns into face detection, classification, and web interface components, making it easy to modify or extend.

---

## 🗂️ Project Structure

```plaintext
Celebrity-image-classification/
├── app/
│   ├── __init__.py            # Initializes Flask app and loads configurations
│   ├── routes.py              # Handles routes and logic for web functionality
│   ├── templates/
│   │   └── index.html         # Web app template (HTML for UI)
│   └── static/
│       └── uploads/           # Directory for storing uploaded images
├── haarcascades/
│   └── haarcascade_frontalface_default.xml   # Face detection model
├── models/
│   └── saved/
│       ├── celebrity_model.pkl     # Pre-trained machine learning model
│       └── label_encoder.pkl       # Encoded labels for celebrity names
├── run.py                     # Script to launch Flask app
├── requirements.txt           # Python dependencies list
└── README.md                  # Project documentation (this file)
```

---

## ⚙️ Installation and Setup

### Prerequisites

- **Python 3.8+** is recommended.
- The `pip` package manager should be installed.

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/RohanRajesh55/Celebrity-image-classification.git
   cd Celebrity-image-classification
   ```

2. **Set Up a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate     # For Windows: venv\Scripts\activate
   ```

3. **Install Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify the Directory Structure**
   Ensure the following files are correctly placed:
   - `haarcascades/haarcascade_frontalface_default.xml`
   - `models/saved/celebrity_model.pkl`
   - `models/saved/label_encoder.pkl`

---

## 🧪 How to Run

1. **Start the Web Application**
   Run the Flask app:

   ```bash
   python run.py
   ```

   Once started, visit [http://localhost:5000](http://localhost:5000) in your browser to access the app.

2. **Upload Images**
   - Use the upload form to select an image.
   - The app will detect faces and classify them into celebrity labels, showing the results along with confidence scores.

---

## 🛠️ Troubleshooting

- **File Not Found Errors:** Ensure required files (e.g., Haar cascade, models) are in their correct paths.
- **Dependency Issues:** Check that all required Python packages are installed properly via `requirements.txt`.
- **Server Not Running:** Confirm that the Flask app is started with `python run.py` and there are no port conflicts.

---

## 📝 Contributing

Contributions are welcome! Feel free to fork the repository, make changes, and open a pull request. For major changes, please open an issue first to discuss your ideas.

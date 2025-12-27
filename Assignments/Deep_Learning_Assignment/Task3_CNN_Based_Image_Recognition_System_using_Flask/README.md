# Task 3: CNN-Based Face Recognition System using Flask 🤖

This project is an end-to-end **CNN-based face recognition system** using Python and Flask, covering everything from data collection to deployment. It recognizes faces of **three people**: my own images captured via OpenCV under various lighting, angles, and expressions, and two celebrities (images sourced from the [Processed Celebrity Face Image Dataset on Kaggle](https://www.kaggle.com/datasets/biyoukjabbarimanjili/processed-celebrity-face-image-dataset)).

The dataset is structured into **train and test sets** with 200+ images per class after augmentation for the celebrities. The model was trained using a custom CNN and achieved **~91% accuracy on the test set**, showing strong performance while highlighting real-world challenges in live predictions.

## 🔹 Repository Contents
- `cnn_training.ipynb` → CNN model training  
- `collect_images.ipynb` → Capture personal images using OpenCV  
- `augment_images.ipynb` → Data augmentation for celebrity images  
- `live_camera_prediction.ipynb` → Optional live camera testing  
- `app.py` → Flask web app for image upload and prediction  
- `face_recognition_model.h5` → Trained CNN model  

> **Note:** Dataset folders and personal images are excluded for privacy.

## 🔹 Key Features
- Upload images for **face recognition**  
- Confidence-based predictions to **reduce misclassification**  
- Demonstrates **real-world deployment** of a deep learning model  
- Highlights differences between **test set performance and live predictions**

This project gave me hands-on experience with **data collection, augmentation, CNN training, and deployment**, while learning the practical challenges of building a robust face recognition system.


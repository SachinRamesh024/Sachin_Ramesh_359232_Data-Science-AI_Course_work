from flask import Flask, request, render_template_string
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__, static_url_path='/uploads', static_folder='uploads')
model = load_model("face_recognition_model.h5")
class_names = ["sachin", "tom_cruise", "will_smith"]

HOME_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Recognition System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f7fa;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            max-width: 480px;
            width: 100%;
            padding: 40px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 32px;
        }
        
        .header h1 {
            color: #1a1a1a;
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        .header p {
            color: #6b7280;
            font-size: 14px;
            font-weight: 400;
        }
        
        .upload-section {
            margin-bottom: 24px;
        }
        
        .upload-area {
            border: 2px dashed #d1d5db;
            border-radius: 8px;
            padding: 32px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            background: #fafafa;
        }
        
        .upload-area:hover {
            border-color: #2563eb;
            background: #f0f7ff;
        }
        
        .upload-area.dragover {
            border-color: #2563eb;
            background: #eff6ff;
        }
        
        #file-input {
            display: none;
        }
        
        .upload-icon {
            width: 48px;
            height: 48px;
            margin: 0 auto 16px;
            background: #e5e7eb;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .upload-icon svg {
            width: 24px;
            height: 24px;
            color: #6b7280;
        }
        
        .upload-text {
            color: #374151;
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 4px;
        }
        
        .upload-hint {
            color: #9ca3af;
            font-size: 12px;
        }
        
        .file-info {
            display: none;
            margin-top: 16px;
            padding: 12px;
            background: #f3f4f6;
            border-radius: 6px;
            font-size: 13px;
            color: #374151;
        }
        
        .preview-container {
            margin-top: 20px;
            display: none;
        }
        
        .preview-image {
            width: 100%;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
        }
        
        .btn {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .btn-primary {
            background: #2563eb;
            color: white;
            display: none;
            margin-top: 16px;
        }
        
        .btn-primary:hover {
            background: #1d4ed8;
        }
        
        .btn-primary:active {
            transform: scale(0.98);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Face Recognition System</h1>
            <p>Upload an image to identify the person</p>
        </div>
        
        <form id="upload-form" action="/predict" method="POST" enctype="multipart/form-data">
            <div class="upload-section">
                <div class="upload-area" id="upload-area">
                    <div class="upload-icon">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
                        </svg>
                    </div>
                    <div class="upload-text">Click to upload or drag and drop</div>
                    <div class="upload-hint">PNG, JPG or JPEG</div>
                    <input type="file" name="file" id="file-input" accept="image/*" required>
                </div>
                <div class="file-info" id="file-info"></div>
            </div>
            
            <div class="preview-container" id="preview-container">
                <img class="preview-image" id="preview-image" alt="Preview">
            </div>
            
            <button type="submit" class="btn btn-primary" id="submit-btn">Analyze Image</button>
        </form>
    </div>
    
    <script>
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        const fileInfo = document.getElementById('file-info');
        const submitBtn = document.getElementById('submit-btn');
        const previewContainer = document.getElementById('preview-container');
        const previewImage = document.getElementById('preview-image');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        
        fileInput.addEventListener('change', function(e) {
            if (this.files && this.files[0]) {
                const file = this.files[0];
                fileInfo.textContent = file.name;
                fileInfo.style.display = 'block';
                submitBtn.style.display = 'block';
                
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImage.src = e.target.result;
                    previewContainer.style.display = 'block';
                }
                reader.readAsDataURL(file);
            }
        });
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            fileInput.files = e.dataTransfer.files;
            fileInput.dispatchEvent(new Event('change'));
        });
    </script>
</body>
</html>
'''

RESULT_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recognition Result</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f7fa;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            max-width: 520px;
            width: 100%;
            padding: 40px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 32px;
        }
        
        h2 {
            color: #1a1a1a;
            font-size: 22px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        .subtitle {
            color: #6b7280;
            font-size: 14px;
        }
        
        .result-section {
            margin-bottom: 24px;
        }
        
        .result-card {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 24px;
        }
        
        .result-label {
            font-size: 12px;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
            font-weight: 500;
        }
        
        .result-value {
            font-size: 28px;
            font-weight: 600;
            color: {{ value_color }};
            margin-bottom: 20px;
        }
        
        .confidence-section {
            margin-top: 16px;
        }
        
        .confidence-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .confidence-label {
            font-size: 13px;
            color: #6b7280;
            font-weight: 500;
        }
        
        .confidence-value {
            font-size: 13px;
            color: #374151;
            font-weight: 600;
        }
        
        .confidence-bar {
            width: 100%;
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: {{ bar_color }};
            border-radius: 4px;
            transition: width 0.8s ease-out;
            width: 0;
        }
        
        .image-container {
            margin: 24px 0;
            text-align: center;
        }
        
        .result-image {
            max-width: 100%;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
        }
        
        .btn {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            background: #2563eb;
            color: white;
            text-decoration: none;
            display: block;
            text-align: center;
        }
        
        .btn:hover {
            background: #1d4ed8;
        }
        
        .btn:active {
            transform: scale(0.98);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>Recognition Complete</h2>
            <p class="subtitle">Analysis results</p>
        </div>
        
        <div class="result-section">
            <div class="result-card">
                <div class="result-label">Identified Person</div>
                <div class="result-value">{{ label }}</div>
                
                <div class="confidence-section">
                    <div class="confidence-header">
                        <span class="confidence-label">Confidence</span>
                        <span class="confidence-value">{{ confidence_text }}</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" id="confidence-fill"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="image-container">
            <img src="/uploads/{{ filename }}" alt="Analyzed Image" class="result-image">
        </div>
        
        <a href="/" class="btn">Analyze Another Image</a>
    </div>
    
    <script>
        setTimeout(() => {
            document.getElementById('confidence-fill').style.width = '{{ confidence }}%';
        }, 100);
    </script>
</body>
</html>
'''

@app.route("/")
def home():
    return render_template_string(HOME_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    pred = model.predict(img_array)
    confidence = np.max(pred)
    label_idx = np.argmax(pred)
    
    if confidence < 0.7:
        label = "Unknown"
        value_color = "#dc2626"
        bar_color = "#dc2626"
    else:
        label = class_names[label_idx]
        value_color = "#2563eb"
        bar_color = "#2563eb"
    
    confidence_percent = confidence * 100
    
    return render_template_string(
        RESULT_TEMPLATE,
        label=label,
        confidence=confidence_percent,
        confidence_text=f"{confidence_percent:.1f}%",
        filename=file.filename,
        value_color=value_color,
        bar_color=bar_color
    )

if __name__ == "__main__":
    app.run(debug=True)
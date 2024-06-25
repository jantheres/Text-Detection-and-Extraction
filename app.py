import os
import numpy as np
from flask import Flask, render_template, request, send_file
import cv2
import pytesseract

app = Flask(__name__)

# Path to Tesseract executable (if necessary)
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Check if an image file is present in the request
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')
    
    file = request.files['file']
    
    # If no file is selected
    if file.filename == '':
        return render_template('index.html', message='No image selected')

    # Read image file string data
    img_string = file.read()

    # Convert string data to numpy array
    nparr = np.frombuffer(img_string, np.uint8)

    # Decode image using OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image to a reasonable size for OCR processing
    resized_image = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    # Use pytesseract to do OCR on the preprocessed image
    extracted_text = pytesseract.image_to_string(blurred_image)

    # Save extracted text to a file
    text_file_path = os.path.join(UPLOAD_FOLDER, 'extracted_text.txt')
    with open(text_file_path, 'w', encoding='utf-8') as f:
        f.write(extracted_text)

    return render_template('index.html', message='OCR Result:', extracted_text=extracted_text)

@app.route('/download')
def download():
    text_file_path = os.path.join(UPLOAD_FOLDER, 'extracted_text.txt')
    return send_file(text_file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

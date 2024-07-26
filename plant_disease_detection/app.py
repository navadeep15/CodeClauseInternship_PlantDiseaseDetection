from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('C:\\Users\\navad\\OneDrive\\Desktop\\vs code\\plant_disease_detection\\Healthy_powdert_rusty.h5')

# Define the labels
labels = ['Healthy', 'Powdery', 'Rust']

# Image upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def process_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Process the image and make predictions
            img = process_image(file_path)
            predictions = model.predict(img)
            predicted_label = labels[np.argmax(predictions)]
            
            return render_template('result.html', label=predicted_label, filename=file.filename)
    return render_template('upload.html')

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')

# Load the trained CNN model
model = tf.keras.models.load_model(r'C:\Users\USER\Desktop\Full_HAR_Project\backend\cnn_har_model.keras')

# Load class names (label encoder stored as a NumPy array)
class_names = np.load(r'C:\Users\USER\Desktop\Full_HAR_Project\backend\label_encoder.npy', allow_pickle=True)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route for image upload
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(file).convert('RGB')
        image = image.resize((224, 224))  # Match model input size
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array)
        predicted_class_index = np.argmax(prediction)
        predicted_label = class_names[predicted_class_index]

        return jsonify({'predicted_activity': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

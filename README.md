# Human Activity Recognition with CNN

This project is a Human Activity Recognition web application using CNNs, Flask, and a stylish frontend. Users can upload images or videos to classify human actions in real-time.

## Features
- Upload Image or Video
- Predicts Human Activity using a trained CNN model
- Model: TensorFlow-based CNN
- Live preview and prediction
- Prediction History (client-side)
- Clean, responsive UI

## Run Locally

1. Clone the repository
2. Install dependencies from `requirements.txt`
3. Run the Flask server
4. Open in browser at `http://localhost:5000/`

## Folder Structure

```
Human-Activity-Recognition-CNN/
├── backend/
│   ├── app.py
│   ├── cnn_har_model.keras
│   ├── label_encoder.npy
│   ├── requirements.txt
├── frontend/
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/style.css
│       └── js/script.js
```

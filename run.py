from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from extract_bottleneck_features import *
from glob import glob
from keras.preprocessing import image


app = Flask(__name__)

# Load your trained model
model = load_model('saved_models/weights.best.InceptionV3.hdf5')
ResNet50_model = ResNet50(weights='imagenet')
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
dog_names = [item[20:-1] for item in sorted(glob("data/dogImages/train/*/"))]

def detect_and_predict_breed(img_path):
    # Check if it's a dog
    if dog_detector(img_path):
        return f"This dog's breed is most likely {predict_dog_breed(img_path).split('.', 1)[1]}."
    # check if it has a human
    elif face_detector(img_path):
        return f"This human looks like a {predict_dog_breed(img_path).split('.', 1)[1]} dog."
    else:
        return "Error!! Neither a human nor a dog were detected."


def predict_dog_breed(img_path):
    # Extract bottleneck features for the given img_path.
    bottleneck_feature = extract_InceptionV3(preprocess_input(path_to_tensor(img_path)))

    # Obtain predicted vector for the image
    predicted_vector = model.predict(bottleneck_feature)

    # Return the dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img, verbose=0))

def path_to_tensor(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        # Normalize the image tensor
        return np.expand_dims(x, axis=0)
    except IOError:
        print(f"Warning: Skipping corrupted image {img_path}")
        return None

def paths_to_tensor(img_paths):
    batch_tensors = []
    for img_path in img_paths:
        tensor = path_to_tensor(img_path)
        if tensor is not None:
            batch_tensors.append(tensor[0])
    return np.array(batch_tensors)

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

@app.route('/', methods=['GET'])
def index():
    # Render the main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Ensure the uploads directory exists
        uploads_dir = os.path.join('uploads')
        os.makedirs(uploads_dir, exist_ok=True)  # Creates the directory if it does not exist

        # Save the file to the uploads directory
        filepath = os.path.join(uploads_dir, file.filename)
        file.save(filepath)

        # Make prediction
        result = detect_and_predict_breed(filepath)

        # Optional: Remove the saved file after prediction
        os.remove(filepath)

        return result


if __name__ == '__main__':
    app.run(debug=True)
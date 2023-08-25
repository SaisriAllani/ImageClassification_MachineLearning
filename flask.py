from flask import Flask, render_template, request
import cv2
import numpy as np
from sklearn.decomposition import PCA
import joblib
import random
import base64
import pathlib
import os

app = Flask(__name__)
image_size = (224, 224)
pca = joblib.load('pca_model.pkl')
clf = joblib.load('random_search_model.pkl')


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    return image


def predict_image(image):
    preprocessed_image = preprocess_image(image)
    flattened_image = preprocessed_image.flatten()
    pca_image = pca.transform([flattened_image])
    predicted_label = clf.predict(pca_image)[0]
    return predicted_label


@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            predicted_label = predict_image(image)

            # Convert the image to base64 encoding
            _, img_encoded = cv2.imencode('.jpg', image)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')

            return render_template('rupa.html', image_base64=img_base64, predicted_label=predicted_label)
    return render_template('rupa.html', image_base64=None, predicted_label=None)


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify, send_file, make_response, redirect, flash
import os
import cv2
import numpy as np
import pickle

covid_model = pickle.load(open('covid19_model.pkl', 'rb'))


UPLOAD_FOLDER = './assets/uploads'
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def predict_cancer(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (224, 224))
    image = image/255.0
    predictions = covid_model.predict(image)
    return jsonify({'prediction': np.argmax(predictions)})


@app.route('/predict', methods=['POST'])
def predict():
    f = request.files['file']
    f.save(os.path.join(UPLOAD_FOLDER, f.filename))
    return predict_cancer("./assets/uploads/" + f.filename)


if __name__ == '__main__':
    app.run(debug=True)

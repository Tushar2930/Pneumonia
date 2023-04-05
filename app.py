from flask import Flask, render_template, request, jsonify, send_file, make_response, redirect, flash
import os
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import keras.utils as image


# covid_model = pickle.load(open('covid19_model.pkl', 'rb'))
covid_model = load_model('model.h5')


UPLOAD_FOLDER = './assets/uploads'
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/hospital/covid')
def covid():
    return render_template('form.html')


def predict_cancer(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (64, 64))
    disease_class = ['Covid-19', 'Non Covid-19']
    img = img/255.0
    x = np.expand_dims(img, axis=0)
    custom = covid_model.predict(x)
    print(custom[0])
    a = custom[0]
    ind = np.argmax(a)
    print('Prediction:', disease_class[ind], 'with probability of ', np.max(
        custom[0])*100, '%')
    os.remove(file_path)
    return jsonify({'prediction': disease_class[ind], 'probability': np.max(custom[0])*100})


@app.route('/predict_covid', methods=['POST'])
def predict():
    f = request.files['file']
    f.save(os.path.join(UPLOAD_FOLDER, f.filename))
    return predict_cancer("./assets/uploads/" + f.filename)


if __name__ == '__main__':
    app.run(debug=True)

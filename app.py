import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow_hub as hub


st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache()
def load_model():
    model = tf.keras.models.load_model("Pneumonia_model.h5", custom_objects={
                                       'KerasLayer': hub.KerasLayer})
    return model


def app():
    # load model and deploy it predicts whether user has covid or not by taking image as input
    model = load_model()
    st.title("Pneumonia Detection")
    st.write(
        "Upload an image of your chest X-ray to detect whether you have Pneumonia or not")
    file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
    if file is not None:

        image = tf.keras.preprocessing.image.load_img(
            file, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        img = image/255.0
        x = np.expand_dims(img, axis=0)
        custom = model.predict(x)
        custom_labels = np.argmax(custom)
        classes = ['PNEUMONIA', 'NORMAL']
        st.write("The image is classified as: ", classes[custom_labels])
        # image = tf.keras.preprocessing.image.load_img(
        #     file, target_size=(224, 224))
        # st.image(image, caption='Uploaded Image.', use_column_width=True)
        # st.write("")
        # st.write("Classifying...")
        # image = tf.keras.preprocessing.image.img_to_array(image)
        # image = tf.keras.applications.mobilenet.preprocess_input(image)
        # image = tf.expand_dims(image, 0)
        # prediction = model.predict(image)
        # if prediction == 0:
        #     st.write("You have Covid-19")
        # else:
        #     st.write("You don't have Covid-19")


if __name__ == '__main__':
    app()

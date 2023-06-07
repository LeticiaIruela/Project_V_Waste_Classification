import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, EfficientNetB0
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from datetime import datetime
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import src.models as md
from PIL import Image

st.set_page_config(page_title="Garbage Classification", page_icon=":sunglasses")
st.title("Garbage Classification")
st.header("Model")
st.markdown("fdfdfd")

model = tf.keras.models.load_model('models/model_0306_efficientnetB0_retrain2.h5')

st.title("Waste classificator")
st.write("Upload your photo and obtain your waste type!")

# Cargar imagen del usuario
uploaded_file = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Guardar la imagen en una ubicación temporal
    image = Image.open(uploaded_file).convert('RGB')
    image_path = "temp_image.jpg"
    image.save(image_path)

    # Preprocesar y hacer la predicción de la imagen
    predicted_class, confidence, predictions = md.preprocess_and_predict(image_path, model)

    # Cargar el DataFrame con las etiquetas de clase
    df_labels = pd.read_csv("datasets/df_f_new.csv")  # Reemplaza "ruta_del_dataframe.csv" con la ruta correcta

    garbage_types_labels = {
    0: 'battery',
    1: 'biological',
    2: 'brown-glass',
    3: 'cardboard',
    4: 'green-glass',
    5: 'metal',
    6: 'paper',
    7: 'plastic',
    8: 'trash',
    9: 'white-glass'
}

    # Obtener la etiqueta de clase predicha
    predicted_label = df_labels.loc[df_labels["Category"] == predicted_class, "Category"].iloc[0]

    garbage_type = garbage_types_labels[predicted_class]


    # Mostrar la imagen cargada
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Mostrar la clase predicha y la probabilidad
    st.write("Prediction:", predicted_label)
    st.write("Garbagge waste type:", garbage_type)
    st.write("Confidence (%):", confidence)
    st.write("Prediccions:", predictions)









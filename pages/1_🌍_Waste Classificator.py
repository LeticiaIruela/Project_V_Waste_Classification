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

st.set_page_config(page_title="Garbage Classification", page_icon=":seedling:")

st.header(":recycle: Waste classificator Model")
st.markdown("""
## Waste Classifier

Welcome to the Waste Classifier! This powerful tool uses advanced machine learning techniques to identify and classify different types of waste based on uploaded images.

### How it Works

1. Upload an image of the waste you want to classify. Accepted formats are JPG, JPEG, and PNG.

2. The classifier will analyze the image using a pre-trained model and provide predictions on the type of waste present.

3. You will receive the predicted waste category along with a confidence score, indicating the model's certainty in its prediction.

4. Additionally, you'll find recommendations on how to dispose of the waste properly, ensuring a more sustainable approach to waste management.

### Why is Waste Classification Important?

Proper waste classification plays a crucial role in effective waste management. By accurately identifying different types of waste, we can implement appropriate recycling, composting, and disposal methods, minimizing environmental impact and promoting sustainability.

### Join us in the Waste Classification Journey!

Start contributing to a cleaner environment today by using the Waste Classifier. Together, we can make a difference and build a greener future.

Upload your image now and let the Waste Classifier do the rest!
""")

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
    garbage_containers = {
        0: 'it goes to the green recycling points',
        1: 'goes to organic container',
        2: 'it goes to green container',
        3: 'it goes to blue container',
        4: 'it goes to green container',
        5: 'it goes to the green recycling points',
        6: 'it goes to blue container',
        7: 'it goes to yellow container',
        8: 'it goes to grey container',
        9: 'it goes to yellow container'
    }
    # Obtener la etiqueta de clase predicha
    predicted_label = df_labels.loc[df_labels["Category"] == predicted_class, "Category"].iloc[0]

    garbage_type = garbage_types_labels[predicted_class]
    garbage_container = garbage_containers[predicted_class]

    # Mostrar la imagen cargada
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Mostrar la clase predicha y la probabilidad
    st.write("Prediction:", predicted_label)
    st.write("Garbagge waste type:", garbage_type)
    st.write("Confidence (%):", confidence)
    st.write("Prediccions:", predictions)

# Variables globales
image = None

# Crear un botón para cargar otra imagen
btn_reload = st.button("Añadir otra imagen")

# Verificar si se hizo clic en el botón
if btn_reload:
    # Borrar la imagen anterior
    image = None

# Mostrar el botón solo si no se ha cargado una imagen o se hizo clic en "Añadir otra imagen"
if image is None or btn_reload:
    unique_key = "file_uploader_" + str(datetime.now())
    uploaded_file = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"], key=unique_key)
    if uploaded_file is not None:
        # Guardar la imagen en una ubicación temporal
        image = Image.open(uploaded_file).convert('RGB')
        image_path = "temp_image.jpg"
        image.save(image_path)

        # Preprocesar y hacer la predicción de la imagen
        predicted_class, confidence, predictions = md.preprocess_and_predict(image_path, model)

        # Cargar el DataFrame con las etiquetas de clase
        df_labels = pd.read_csv("datasets/df_f_new.csv")  # Reemplaza "ruta_del_dataframe.csv" con la ruta correcta

        # Obtener la etiqueta de clase predicha
        predicted_label = df_labels.loc[df_labels["Category"] == predicted_class, "Category"].iloc[0]

        garbage_type = garbage_types_labels[predicted_class]

        # Mostrar la imagen cargada
        st.image(image, caption="Imagen cargada", use_column_width=True)

        # Mostrar la clase predicha y la probabilidad
        st.write("Prediction:", predicted_label)
        st.write("Garbage waste type:", garbage_type)
        st.write("Confidence (%):", confidence)
        st.write("Prediccions:", predictions)

        # Mostrar el contenedor adecuado
        st.write("Container:")
        for prediction, container in garbage_containers.items():
            if prediction == predicted_class:
                st.write(container)





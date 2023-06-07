import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
from sklearn.utils import shuffle
from keras.preprocessing import image
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score
import re
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score
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


# Preprocessing
def display_random_image(data_directory):
    subfolders = os.listdir(data_directory)
    random_subfolder = random.choice(subfolders)
    subfolder_path = os.path.join(data_directory, random_subfolder)

    random_file = random.choice(os.listdir(subfolder_path))

    image_path = os.path.join(subfolder_path, random_file)
    random_image = Image.open(image_path)
    print('Image path:', image_path)
    
    plt.imshow(random_image)
    plt.show()

def create_dataframe(data_directory, garbage_types_labels):
    image_list = []
    categories_list = []

    for category, category_code in garbage_types_labels.items():
        category_path = os.path.join(data_directory, category)
        images = os.listdir(category_path)
        image_list += images
        categories_list += [category_code] * len(images)

    df = pd.DataFrame({'Image': image_list, 'Category': categories_list})
    return df

def file_rename(df, col_name):
    df[col_name] = df[col_name].astype(str).apply(lambda x: x[:re.search("\d", str(x)).start()] + '/' + str(x))
    return df

def display_random_image_from_df(df, data_directory):
    random_row = random.randint(0, len(df) - 1)
    sample = df.iloc[random_row]
    random_image = Image.open(data_directory + sample['Image'])
    
    print(sample['Image'])
    plt.imshow(random_image)
    plt.show()

#Modeling

def create_model(path, name, img_height, img_width, train_generator, validation_generator, epochs):
    num_classes = 10

    start = datetime.now()

    # Crear el modelo base
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Definir el modelo final
    model = Model(inputs=base_model.input, outputs=predictions)

    # Congelar las capas del modelo base
    for layer in base_model.layers:
        layer.trainable = False

    # Compilar el modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    history=model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

    try:
        model.save(f"{path}/{name}.h5")
    except:
        print("Modelo no guardado")

    end = datetime.now()

    print(f"It took {end-start} time")
    
    return history

from PIL import Image
import numpy as np

def preprocess_and_predict1(image_path, model):
    # Cargar la imagen en color
    image = Image.open(image_path).convert('RGB')

    # Redimensionar la imagen a las dimensiones requeridas por el modelo
    img_width, img_height = 224, 224  # Usar las dimensiones adecuadas para tu modelo
    image = image.resize((img_width, img_height))

    # Convertir la imagen en un arreglo numpy
    image_array = np.array(image)

    # Añadir una dimensión adicional para el batch
    image_array = np.expand_dims(image_array, axis=0)

    # Preprocesar la imagen (normalización, etc.) de acuerdo con el modelo
    image_array = image_array.astype('float32')
    image_array /= 255

    # Realizar la predicción con el modelo
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    class_labels = ['battery', 'biological', 'brown-glass', 'cardboard', 'green-glass', 'metal', 'paper', 'plastic', 'trash', 'white-glass']
    predicted_label = class_labels[predicted_class]

    return predicted_label, confidence, predictions

def preprocess_and_predict(image_path, model):
    # Cargar la imagen en color
    image = Image.open(image_path).convert('RGB')

    # Redimensionar la imagen a las dimensiones requeridas por el modelo
    img_width, img_height = 224, 224  # Usar las dimensiones adecuadas para tu modelo
    image = image.resize((img_width, img_height))

    # Convertir la imagen en un arreglo numpy
    image_array = np.array(image)

    # Añadir una dimensión adicional para el batch
    image_array = np.expand_dims(image_array, axis=0)

    # Preprocesar la imagen (normalización, etc.) de acuerdo con el modelo
    image_array = image_array.astype('float32')
    image_array /= 255

    # Realizar la predicción con el modelo
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    return predicted_class, confidence, predictions



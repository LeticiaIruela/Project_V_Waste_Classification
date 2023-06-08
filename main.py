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


st.set_page_config(
    page_title="Garbage Classification",
    page_icon=":seedling:",
    layout="wide"
)

# Set background color
page_bg = """
<style>
body {
    background-color: #e6f7e9;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Main title and objective description
st.title("Welcome to the Waste Classifier")
st.markdown("""
The Waste Classifier is a powerful tool that utilizes advanced machine learning techniques to identify and classify different types of waste based on user-uploaded images.
""")

# Add an image
image_path = "fig/SAFE.png"  # Replace with the actual path to your image
image = Image.open(image_path)
image = image.resize((500, 500))  # Reduce the size of the image
st.image(image, use_column_width=True)






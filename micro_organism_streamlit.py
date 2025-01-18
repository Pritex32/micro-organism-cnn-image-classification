from keras.models import load_model
import streamlit as st
import numpy as np

import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.preprocessing import image


model=load_model("micro_organism_nn_model.h5")

st.title('micro organism image classification'.upper())
uploaded_img = st.file_uploader('Upload an Image', type=['png', 'jpg', 'jpeg'])

if uploaded_img is not None:
    # Preprocess the uploaded image
    img = image.load_img(uploaded_img, target_size=(64, 64))  # Resize to input size
    img_array = np.array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Display the uploaded image
    st.image(uploaded_img, caption='Uploaded Image', use_column_width=True)

    if st.button('Classify'):
        # Make predictions
        predictions = model.predict(img_array)
        class_names = ['Amoeba', 'Euglena', 'Hydra', 'Paramecium', 
                       'Rod_bacteria', 'Spherical_bacteria', 
                       'Spiral_bacteria', 'Yeast']

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions, axis=1)[0]

        # Map the index to the class name
        predicted_class_name = class_names[predicted_class_index]

        # Display the prediction
        st.success(f'The uploaded image is classified as: {predicted_class_name}')
    else:
        st.text('please upload file')

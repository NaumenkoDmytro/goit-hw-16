import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


model_scratch = load_model('model_scratch.h5') 
model_vgg = load_model('model_vgg.h5')  

st.title("Image classification using neural networks")

model_option = st.selectbox(
    "Select a model to classify:",
    ("First CNN model (28x28x1)", "The second model is VGG16 (32x32x3)")
)

def preprocess_image(image_data, target_size, model_option):
    try:
        if model_option == "First CNN model (28x28x1)":
            img = image.load_img(image_data, target_size=target_size, color_mode='grayscale')
            img_array = image.img_to_array(img)
        else:
            img = image.load_img(image_data, target_size=target_size, color_mode='grayscale')
            img_array = image.img_to_array(img)
            img_array = np.repeat(img_array, 3, axis=2)
        
        
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array
    except Exception as e:
        st.error(f"Error while processing image: {e}")
        return None


uploaded_file = st.file_uploader("Upload an image for classification", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_data = uploaded_file.read()
    st.image(image_data, caption='Input image', use_column_width=True)
    
    if model_option == "First CNN model (28x28x1)":
        target_size = (28, 28)
    else:
        target_size = (32, 32)
    
    img_array = preprocess_image(uploaded_file, target_size, model_option)
    
    if img_array is not None:
        if model_option == "First CNN model (28x28x1)":
            prediction = model_scratch.predict(img_array)
            history = np.load('history_scratch.npy', allow_pickle='TRUE').item()     
        else:
            prediction = model_vgg.predict(img_array)
            history = np.load('history_vgg.npy', allow_pickle='TRUE').item()  

           
        predicted_class = np.argmax(prediction, axis=1)
        class_probabilities = prediction[0]
        
        class_names = ['Class 0', 'Class 1', 'Class 2','Class 3','Class 4','Class 5','Class 6','Class 7','Class 8','Class 9']  
        
       
        st.subheader("Classification results")
        st.write(f"Predicted class: {class_names[predicted_class[0]]}")
        st.write("Probabilities for each class:")
        for idx, prob in enumerate(class_probabilities):
            st.write(f"{class_names[idx]}: {prob * 100:.2f}%")

        
        sst.subheader("Graphs learning models")
        
        
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(history['loss'], label='Training losses')
        ax_loss.plot(history['val_loss'], label='Validation losses')
        ax_loss.set_title('Loss Function')
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        st.pyplot(fig_loss)
        
        
        fig_acc, ax_acc = plt.subplots()
        ax_acc.plot(history['accuracy'], label='Training losses')
        ax_acc.plot(history['val_accuracy'], label='Validation losses')
        ax_acc.set_title('Model accuracy')
        ax_acc.set_xlabel('Epochs')
        ax_acc.set_ylabel('Loss')
        ax_acc.legend()
        st.pyplot(fig_acc)

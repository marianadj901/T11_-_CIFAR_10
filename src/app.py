import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Classificação CIFAR-10 (1 canal)")

# Carregar modelo salvo
model = tf.keras.models.load_model("cnn_best.h5")
classes = ["avião","automóvel","pássaro","gato","cervo","cachorro","sapo","cavalo","navio","caminhão"]

file = st.file_uploader("Envie uma imagem", type=["jpg","png"])
if file:
    img = Image.open(file).resize((32,32)).convert("L")
    x = np.array(img)/255.0
    x = x.reshape(1,32,32,1)
    pred = model.predict(x)[0]
    st.bar_chart(pred)
    st.write("Classe prevista:", classes[np.argmax(pred)])

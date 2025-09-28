import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2

# Caminho do modelo treinado (ajuste se necessário)
MODEL_PATH = "cnn_best.h5"

# Carregar modelo
model = keras.models.load_model(MODEL_PATH)

# Classes do CIFAR-10
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

st.title("🔎 Classificação de Imagens - CIFAR-10")
st.write("Faça upload de uma imagem e veja a predição do modelo treinado.")

# Upload de imagem
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Ler imagem
    img = image.load_img(uploaded_file, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Mostrar imagem
    st.image(uploaded_file, caption="Imagem carregada", use_column_width=True)

    # Fazer previsão
    predictions = model.predict(img_array)
    pred_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    st.subheader("Resultado da previsão")
    st.write(f"**Classe prevista:** {class_names[pred_class]} ({confidence:.2f}%)")

    # Mostrar gráfico de probabilidades
    fig, ax = plt.subplots()
    ax.bar(class_names, predictions[0])
    ax.set_xticklabels(class_names, rotation=45)
    st.pyplot(fig)

    # Grad-CAM para visualização
    last_conv_layer = model.get_layer(index=-4)
    grad_model = keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    # Redimensionar para o tamanho original da imagem
    heatmap = cv2.resize(heatmap.numpy(), (32, 32))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

    st.subheader("Grad-CAM (regiões mais importantes)")
    st.image(superimposed_img, channels="BGR", use_column_width=True)

import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image

# ------------------------------
# Configuração
# ------------------------------
st.set_page_config(page_title="CIFAR-10 Classificação", layout="centered")
st.title("🔍 Classificação CIFAR-10 (1 canal) com Grad-CAM")

# Carregar modelo salvo (ajuste o nome do arquivo se necessário)
MODEL_PATH = "cnn_best.h5"
model = keras.models.load_model(MODEL_PATH)

classes = ["avião","automóvel","pássaro","gato","cervo",
           "cachorro","sapo","cavalo","navio","caminhão"]

# ------------------------------
# Função Grad-CAM
# ------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
    return heatmap.numpy()

def overlay_heatmap(heatmap, img, alpha=0.4, cmap="jet"):
    heatmap = np.uint8(255 * heatmap)
    cmap = plt.get_cmap(cmap)
    cmap_colors = cmap(np.arange(256))[:, :3]
    heatmap_color = cmap_colors[heatmap]
    heatmap_color = keras.preprocessing.image.array_to_img(heatmap_color)
    heatmap_color = heatmap_color.resize((img.size[0], img.size[1]))
    superimposed_img = Image.blend(img.convert("RGB"), heatmap_color, alpha)
    return superimposed_img

# ------------------------------
# Upload da imagem
# ------------------------------
file = st.file_uploader("📂 Envie uma imagem (.jpg ou .png)", type=["jpg", "png"])

if file:
    # Pré-processamento
    img = Image.open(file).resize((32, 32)).convert("L")
    x = np.array(img) / 255.0
    x = x.reshape(1, 32, 32, 1)

    # Predição
    pred = model.predict(x)[0]
    pred_class = np.argmax(pred)

    # Mostrar imagem original
    st.image(img, caption="Imagem enviada", width=150)

    # Mostrar predições
    st.bar_chart(pred)
    st.write(f"🔮 Classe prevista: **{classes[pred_class]}**")

    # Grad-CAM
    st.subheader("🌡️ Grad-CAM (ativação visual)")
    heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name="conv2d")
    gradcam_img = overlay_heatmap(heatmap, img)

    st.image(gradcam_img, caption="Grad-CAM", use_column_width=True)

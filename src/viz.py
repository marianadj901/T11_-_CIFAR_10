# src/viz.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------
# 1) Função para plotar curvas de treino/validação
# ------------------------------------------------------------
def plot_curves(history, title="Curvas de treino"):
    """
    Plota curvas de acurácia e perda a partir do history do Keras.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Acurácia
    axes[0].plot(history.history['accuracy'], label='Treino')
    axes[0].plot(history.history['val_accuracy'], label='Validação')
    axes[0].set_title('Acurácia')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Acurácia')
    axes[0].legend()

    # Perda
    axes[1].plot(history.history['loss'], label='Treino')
    axes[1].plot(history.history['val_loss'], label='Validação')
    axes[1].set_title('Perda')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    return fig

# ------------------------------------------------------------
# 2) Função para matriz de confusão
# ------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names, title="Matriz de Confusão"):
    """
    Plota matriz de confusão normalizada.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    ax.set_title(title)
    return fig

# ------------------------------------------------------------
# 3) Função para mostrar Grad-CAM
# ------------------------------------------------------------
def plot_gradcam(img, heatmap, alpha=0.4, title="Grad-CAM"):
    """
    Sobrepõe heatmap (Grad-CAM) na imagem original.
    """
    import cv2

    # Normalizar heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Redimensionar heatmap para a imagem
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Sobrepor
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img)
    ax[0].set_title("Imagem Original")
    ax[0].axis("off")

    ax[1].imshow(superimposed_img)
    ax[1].set_title(title)
    ax[1].axis("off")

    plt.tight_layout()
    return fig

import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import cifar10
import numpy as np
from models import create_mlp, create_cnn

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["mlp", "cnn"], required=True)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# Fixar seeds
tf.random.set_seed(args.seed)
np.random.seed(args.seed)

# Dados
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[:,:,:,0:1].astype("float32")/255.0  # apenas canal R
x_test = x_test[:,:,:,0:1].astype("float32")/255.0

num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

if args.model == "mlp":
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))
    model = create_mlp((32*32,), num_classes)
else:
    model = create_cnn((32,32,1), num_classes)

# Compilar
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint(f"{args.model}_best.h5", save_best_only=True)
]

# Treinar
history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=args.epochs,
    batch_size=64,
    callbacks=callbacks
)

# Avaliar
loss, acc = model.evaluate(x_test, y_test)
print(f"Teste - Loss: {loss:.4f}, Acurácia: {acc:.4f}")

# Salvar modelo final
model.save(f"{args.model}_final.h5")

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import os

# =========================
# 1. Dataset path
# =========================

DATASET_DIR = "dataset/paddies"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# =========================
# 2. Load dataset
# =========================

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names

print("\nPaddy Classes:")
for i, name in enumerate(class_names):
    print(i, name)

# =========================
# 3. Improve data pipeline
# =========================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# =========================
# 4. Data augmentation
# =========================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15)
])

# =========================
# 5. MobileNetV2
# =========================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

# =========================
# 6. Build model
# =========================

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# =========================
# 7. Compile
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# 9. Train
# =========================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

# =========================
# 10. Save model
# =========================

os.makedirs("models", exist_ok=True)

model.save(
    "models/paddy_disease_model.keras"
)

print("\n================================")
print("Paddy model training completed!")
print("Model saved successfully.")
print("================================")
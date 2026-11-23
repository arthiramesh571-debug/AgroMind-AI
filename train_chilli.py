import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# =========================
# 1. SETTINGS
# =========================

DATASET_PATH = "dataset/chilli"
MODEL_PATH = "models/chilli_disease_model.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

# =========================
# 2. LOAD DATASET
# =========================

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names

print("\nClasses:", class_names)

# =========================
# 3. CLASS WEIGHTS
# =========================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=np.concatenate([
        y.numpy()
        for _, y in train_ds
    ])
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(class_weights_array)
}

print("Class weights:", class_weights)

# =========================
# 4. PERFORMANCE
# =========================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# =========================
# 5. DATA AUGMENTATION
# =========================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15)
])

# =========================
# 6. BASE MODEL
# =========================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# =========================
# 7. BUILD MODEL
# =========================

inputs = tf.keras.Input(shape=(224, 224, 3))

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
# 8. COMPILE
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# 9. CALLBACKS
# =========================

os.makedirs("models", exist_ok=True)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# =========================
# 10. TRAIN
# =========================

print("\nStarting Chilli disease model training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)

# =========================
# 11. FINAL RESULT
# =========================

loss, accuracy = model.evaluate(
    val_ds,
    verbose=0
)

print("\n===================================")
print("CHILLI MODEL TRAINING COMPLETED")
print("===================================")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
print(f"Model saved at: {MODEL_PATH}")
print("Classes:", class_names)
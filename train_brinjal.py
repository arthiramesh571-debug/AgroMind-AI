
import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Dataset location
DATASET_DIR = r"C:\Users\Arthi.R\Downloads\archive"

# Model output location
OUTPUT_DIR = r"brinjal_training"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

print("Checking Brinjal dataset...")

# Load images from the five class folders
train_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names

print("\nBrinjal classes:")
for name in class_names:
    print("-", name)

# Save class names for later prediction
with open(os.path.join(OUTPUT_DIR, "class_names.json"), "w") as f:
    json.dump(class_names, f, indent=4)

# Improve input pipeline efficiency
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Load pretrained MobileNetV2 feature extractor
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("\nStarting Brinjal Fruit Disease Model training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Save trained model
model_path = os.path.join(OUTPUT_DIR, "brinjal_fruit_model.keras")
model.save(model_path)

# Save training results
results = {
    "classes": class_names,
    "epochs": EPOCHS,
    "final_training_accuracy": float(history.history["accuracy"][-1]),
    "final_validation_accuracy": float(history.history["val_accuracy"][-1])
}

with open(os.path.join(OUTPUT_DIR, "training_results.json"), "w") as f:
    json.dump(results, f, indent=4)

print("\nTraining completed!")
print("Model saved to:", model_path)
print("Final training accuracy:",
      round(results["final_training_accuracy"] * 100, 2), "%")
print("Final validation accuracy:",
      round(results["final_validation_accuracy"] * 100, 2), "%")
print("\nNote: Validation accuracy is not independent test accuracy.")

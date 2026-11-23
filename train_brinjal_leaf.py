
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = Path("brinjal_leaf_dataset")
MODEL_DIR = Path("brinjal_leaf_training")
MODEL_DIR.mkdir(exist_ok=True)

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42

tf.keras.utils.set_random_seed(SEED)

print("Loading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Classes:", class_names)

with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

# Calculate class weights using training labels
labels = np.concatenate([
    y.numpy() for _, y in train_ds
])

classes = np.unique(labels)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=labels
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, weights)
}

print("Class weights:", class_weights)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(
    1000, seed=SEED
).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.1),
])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(
    len(class_names), activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_DIR / "brinjal_leaf_model.keras",
        monitor="val_loss",
        save_best_only=True
    ),
]

print("Starting training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks,
    class_weight=class_weights
)

results = {
    "final_train_accuracy": float(history.history["accuracy"][-1]),
    "final_validation_accuracy": float(history.history["val_accuracy"][-1]),
    "epochs_completed": len(history.history["accuracy"]),
    "class_names": class_names,
    "image_size": list(IMG_SIZE),
    "class_weights": class_weights
}

with open(MODEL_DIR / "training_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nTraining complete!")
print(json.dumps(results, indent=2))
print("Model folder:", MODEL_DIR.resolve())

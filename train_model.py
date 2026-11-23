import os
import random
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_DIR = "dataset/tomatoes"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "tomato_disease_model.keras")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SIZE = 0.20
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. FIND ALL IMAGES
# ============================================================

print("\nScanning tomato dataset...")

class_names = sorted([
    folder
    for folder in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, folder))
])

print(f"\nFound {len(class_names)} disease classes:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")


image_paths = []
labels = []

valid_extensions = (".jpg", ".jpeg", ".png")

for class_index, class_name in enumerate(class_names):

    image_folder = os.path.join(
        DATASET_DIR,
        class_name,
        "color",
        "images"
    )

    if not os.path.exists(image_folder):
        print(f"Warning: folder not found: {image_folder}")
        continue

    for filename in os.listdir(image_folder):

        if filename.lower().endswith(valid_extensions):

            image_paths.append(
                os.path.join(image_folder, filename)
            )

            labels.append(class_index)


print(f"\nTotal images found: {len(image_paths)}")


# ============================================================
# 3. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    image_paths,
    labels,
    test_size=VALIDATION_SIZE,
    random_state=SEED,
    stratify=labels
)

print(f"Training images: {len(X_train)}")
print(f"Validation images: {len(X_val)}")


# ============================================================
# 4. IMAGE LOADING FUNCTION
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(image, tf.float32)

    return image, label


# ============================================================
# 5. DATA AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


def augment_image(image, label):

    image = augmentation(image, training=True)

    return image, label


# ============================================================
# 6. CREATE TF.DATA DATASETS
# ============================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
)

train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.map(
    augment_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.shuffle(
    1000,
    seed=SEED
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)


val_dataset = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
)

val_dataset = val_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# 7. HANDLE CLASS IMBALANCE
# ============================================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(class_weights_array)
}

print("\nClass weights:")

for index, weight in class_weights.items():
    print(
        f"{class_names[index]}: {weight:.2f}"
    )


# ============================================================
# 8. BUILD MOBILE NET V2 MODEL
# ============================================================

print("\nBuilding AI model...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


model = models.Sequential([

    layers.Input(
        shape=(224, 224, 3)
    ),

    layers.Rescaling(
        1.0 / 127.5,
        offset=-1
    ),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(
        len(class_names),
        activation="softmax"
    )
])


# ============================================================
# 9. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\nStarting training...")
print("Please wait. This may take some time.\n")


history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    class_weight=class_weights
)


# ============================================================
# 11. SAVE MODEL
# ============================================================

model.save(MODEL_PATH)

print("\n===================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("===================================")

print(f"\nModel saved to:")
print(MODEL_PATH)

print(
    f"\nFinal training accuracy: "
    f"{history.history['accuracy'][-1] * 100:.2f}%"
)

print(
    f"Final validation accuracy: "
    f"{history.history['val_accuracy'][-1] * 100:.2f}%"
)
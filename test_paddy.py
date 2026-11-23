import tensorflow as tf
import numpy as np
import sys

MODEL_PATH = "models/paddy_disease_model.keras"

class_names = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro"
]

model = tf.keras.models.load_model(MODEL_PATH)

image_path = sys.argv[1]

image = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)

image_array = tf.keras.utils.img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)

predictions = model.predict(image_array, verbose=0)

predicted_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_index]
confidence = predictions[0][predicted_index] * 100

print("\n==============================")
print("PADDY DISEASE PREDICTION")
print("==============================")
print("Predicted Disease:", predicted_class)
print("Confidence:", round(confidence, 2), "%")
print("==============================")
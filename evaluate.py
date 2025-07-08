import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ────────── GPU CHECK ──────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_visible_devices(gpus[0], "GPU")
        print(f"[INFO] Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("[WARNING] No GPU detected – evaluation will run on CPU.")

# ────────── CONFIGURATION ──────────
IMG_SIZE    = (160, 160)
BATCH_SIZE  = 32
TEST_DIR    = "celebrity_dataset_split/test"
MODEL_PATH  = "celebrity_cnn_model.keras"

# ────────── LOAD MODEL ──────────
print(f"[INFO] Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("[INFO] Model loaded successfully.\n")

# ────────── PREPARE TEST DATA ──────────
print(f"[INFO] Loading test data from: {TEST_DIR}")
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

class_labels = list(test_generator.class_indices.keys())
filenames    = test_generator.filenames
y_true       = test_generator.classes

# ────────── ASSERTION TO CHECK CLASS MATCH ──────────
assert model.output_shape[-1] == len(class_labels), (
    f"Mismatch between model output classes ({model.output_shape[-1]}) "
    f"and test classes ({len(class_labels)})!"
)

# ────────── PREDICT ──────────
print("[INFO] Running predictions on test set …")
predictions = model.predict(test_generator, verbose=0)
y_pred      = np.argmax(predictions, axis=1)

# ────────── CLASSIFICATION REPORT ──────────
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_labels, zero_division=0))

# ────────── CONFUSION MATRIX ──────────
print("\n[INFO] Generating confusion matrix …")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels,
)
plt.title("Confusion Matrix – Celebrity Face Recognition")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
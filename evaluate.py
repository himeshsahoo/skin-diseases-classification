import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = "skin_disease_model.keras"
TEST_DIR = "train"   # change to "test" if you create a separate test folder
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

model = tf.keras.models.load_model(MODEL_PATH)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="categorical"
)

class_names = test_ds.class_names

test_ds = test_ds.map(
    lambda x, y: (
        tf.cast(x, tf.float32) / 255.0,
        y
    )
)

loss, acc = model.evaluate(test_ds)
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Loss: {loss:.4f}")

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

print("\nClassification Report\n")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


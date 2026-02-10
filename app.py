import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, url_for
from PIL import Image

MODEL_PATH = "skin_disease_model.keras"
TRAIN_DIR = "train"
UPLOAD_FOLDER = "static/uploads"
IMG_SIZE = (224, 224)

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the trained model once at startup
model = tf.keras.models.load_model(MODEL_PATH)


def get_class_names(train_dir: str):
    """Infer class names from the subfolders in the training directory."""
    if not os.path.isdir(train_dir):
        return []
    class_dirs = [
        d
        for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ]
    # ImageDataGenerator uses alphabetical order for class indices
    return sorted(class_dirs)


class_names = get_class_names(TRAIN_DIR)


def preprocess_image(image_path):
    """Load image from path and prepare for model prediction."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img).astype("float32")
    # MobileNetV2 expects values in [-1, 1]
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="out_relu", pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if isinstance(preds, list):
             preds = preds[0]
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_visualizations(image_path, heatmap):
    # Load the original image
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.resize(jet, IMG_SIZE)

    # Create an image with RGB colorized heatmap
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    
    # Superimpose the heatmap on original image
    superimposed_img = jet * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    
    # Save heatmap image
    heatmap_filename = "heatmap_" + os.path.basename(image_path)
    heatmap_path = os.path.join(app.config["UPLOAD_FOLDER"], heatmap_filename)
    cv2.imwrite(heatmap_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

    # --- Bounding Box Generation ---
    # Use relative thresholding: Keep pixels > 60% of max intensity
    # This tightens the box around the most relevant area
    max_val = np.max(heatmap)
    threshold_val = 0.6 * max_val
    _, binary_map = cv2.threshold(heatmap, threshold_val, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding box on a copy of original image
    bbox_img = img.copy()
    if contours:
        # Filter contours by size (ignore small noise < 5% of image area)
        image_area = IMG_SIZE[0] * IMG_SIZE[1]
        valid_contours = [c for c in contours if cv2.contourArea(c) > 0.05 * image_area]
        
        if valid_contours:
            # Get the largest valid contour
            c = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        elif contours:
             # Fallback to largest contour if all are small
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save bounding box image
    bbox_filename = "bbox_" + os.path.basename(image_path)
    bbox_path = os.path.join(app.config["UPLOAD_FOLDER"], bbox_filename)
    cv2.imwrite(bbox_path, bbox_img)
    
    return heatmap_filename, bbox_filename


@app.route("/", methods=["GET", "POST"])
def upload_and_predict():
    error = None
    prediction = None
    confidence = None
    topk = None
    original_image = None
    heatmap_image = None
    bbox_image = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "No file part in the request."
        else:
            file = request.files["file"]
            if file.filename == "":
                error = "No file selected."
            else:
                try:
                    # Save the uploaded file
                    filename = file.filename
                    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    file.save(file_path)
                    
                    original_image = filename

                    # Preprocess and Predict
                    image_array = preprocess_image(file_path)
                    preds = model.predict(image_array)
                    probs = preds[0]
                    idx = int(np.argmax(probs))

                    if class_names and 0 <= idx < len(class_names):
                        prediction = class_names[idx]
                    else:
                        prediction = f"Class index {idx}"

                    confidence = float(probs[idx])

                    # Top 3 predictions
                    top_indices = np.argsort(probs)[::-1][:3]
                    topk = [
                        (
                            class_names[i]
                            if class_names and 0 <= i < len(class_names)
                            else f"Class index {i}",
                            float(probs[i]),
                        )
                        for i in top_indices
                    ]
                    
                    # Generate Visualizations
                    heatmap = make_gradcam_heatmap(image_array, model, last_conv_layer_name="Conv_1")
                    heatmap_filename, bbox_filename = save_visualizations(file_path, heatmap)
                    
                    heatmap_image = heatmap_filename
                    bbox_image = bbox_filename

                except Exception as exc:  # noqa: BLE001
                    error = f"Error processing image: {exc}"
                    prediction = None
                    # Also log to console
                    import traceback
                    traceback.print_exc()

    return render_template(
        "index.html",
        error=error,
        prediction=prediction,
        confidence=confidence,
        topk=topk,
        original_image=original_image,
        heatmap_image=heatmap_image,
        bbox_image=bbox_image
    )


if __name__ == "__main__":
    # Run the Flask development server
    app.run(debug=True)


# Skin Disease Classification AI

A Flask-based web application that classifies skin diseases using a deep learning model (MobileNetV2). It features **Grad-CAM** heatmaps and **Bounding Box** localization to explain the model's predictions.

## Features
- **Image Classification**: Predicts skin disease classes from uploaded images.
- **Visual Explainability**:
    - **Grad-CAM Heatmap**: Shows where the model is looking.
    - **Bounding Box**: Automatically highlights the lesion area using Otsu's thresholding.
- **Web Interface**: Clean, responsive UI built with HTML/CSS.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/himeshsahoo/skin-diseases-classification.git
    cd skin-disease-classification
    ```

2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the application**:
    ```bash
    python app.py
    ```

2.  Open your browser and navigate to `http://127.0.0.1:5000`.

3.  Upload a skin image and click **Analyze Image**.

## Deployment

This project is ready for deployment on platforms like **Render** or **Railway**.

1.  **Push to GitHub**:
    ```bash
    git remote add origin https://github.com/himeshsahoo/skin-diseases-classification.git
    git push -u origin master
    ```

2.  **Deploy on Render**:
    - Create a new **Web Service**.
    - Connect your GitHub repository.
    - Render will automatically detect the `Procfile` and deploy using `gunicorn`.

## Project Structure
- `app.py`: Main Flask application and logic.
- `requirements.txt`: Python dependencies.
- `Procfile`: Deployment configuration.
- `static/`: CSS and uploaded images.
- `templates/`: HTML templates.
- `skin_disease_model.keras`: Trained model file (ensure this exists locally).

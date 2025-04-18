import os
import numpy as np
import tensorflow as tf
import cv2
import base64
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
cnn = tf.keras.models.load_model('trained_plant_disease_model.keras')

# Validation set (to get class names)
validation_set = tf.keras.utils.image_dataset_from_directory(
    r"C:\Users\DELL\Downloads\valid",
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)
class_names = validation_set.class_names

# Configure the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif'}

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/index')
def home():
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image is uploaded
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the uploaded image
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = tf.keras.preprocessing.image.load_img(filepath, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to a batch.

        # Model prediction
        predictions = cnn.predict(input_arr)
        result_index = np.argmax(predictions)
        model_prediction = class_names[result_index]

        # Convert image to base64 for display in HTML
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, img_encoded = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        # Render the result page with the prediction
        return render_template('result.html', prediction=model_prediction, image_base64=img_base64)

    return redirect(request.url)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

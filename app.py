from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "chest_pneumonia_detection_model.h5"  
model = tf.keras.models.load_model(MODEL_PATH)

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    prediction = model.predict(img_array)
    return "Pneumonia" if prediction[0][0] > 0.5 else "Normal"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join("static/uploads", file.filename)
            file.save(file_path)
            result = predict_disease(file_path)
            return render_template('index.html', result=result, image=file_path)
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import io
import base64

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "chest_pneumonia_detection_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

def predict_disease(img):
    img = image.load_img(io.BytesIO(img.read()), target_size=(150, 150))
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
            # Convert image to base64
            img_bytes = file.read()
            encoded_img = base64.b64encode(img_bytes).decode('utf-8')
            img_data = f"data:image/jpeg;base64,{encoded_img}"

            # Predict disease
            result = predict_disease(io.BytesIO(img_bytes))
            
            return render_template('index.html', result=result, image=img_data)

    return render_template('index.html', result=None, image=None)

if __name__ == '__main__':
    app.run(debug=True)

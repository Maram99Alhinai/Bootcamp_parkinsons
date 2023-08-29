from flask import Blueprint, render_template, request, current_app
from keras.models import load_model
from PIL import Image
import numpy as np
import os

model = Blueprint('model', __name__)
classifier = load_model('classifier1.h5')

@model.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        image = request.files.get('image')
        if image:
            new_image_path = os.path.join(current_app.root_path,'images',image.filename)
            image.save(new_image_path)
            
            def preprocess_img(image_path):
                new_image = Image.open(image_path).convert("RGB")
                new_image = new_image.resize((128, 128))
                new_image_array = np.array(new_image)
                new_image_array = np.expand_dims(new_image_array, axis=0)
                new_image_array = new_image_array.astype('float32') / 255.0
                return new_image_array
            
            prediction = classifier.predict(preprocess_img(new_image_path))
            return render_template("home.html", prediction=prediction)
            
    return render_template("home.html")

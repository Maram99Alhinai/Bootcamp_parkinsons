from flask import Blueprint, render_template, request, current_app
from keras.models import load_model
from PIL import Image
import numpy as np
import os
import cv2
from skimage.morphology import closing, disk
from utils.process_images import *
from utils.process_data import *
import pickle
import joblib

model = Blueprint('model', __name__)

@model.route('/', methods=['GET', 'POST'])
def home():      
    return render_template("home.html")

@model.route('/try2', methods=['GET', 'POST'])
def pred2():
    if request.method == 'POST':
        model_filename = 'rf_model_wave.joblib'  # Replace with the correct filename
        loaded_model = joblib.load(model_filename)
        image = request.files.get('image')
        if image:
            new_image_path = os.path.join(current_app.root_path,'images',image.filename)
            image.save(new_image_path)
            def process_new_image(new_image_path):
                new_image = read_and_thresh(new_image_path, resize=False)

                # Compute the same interaction features as during training
                mean_thickness = np.mean(stroke_thickness(closing(label_sort(new_image) > 0, disk(1))))
                std_thickness = np.std(stroke_thickness(closing(label_sort(new_image) > 0, disk(1))))
                num_pixels = sum_pixels(skeleton_drawing(new_image))
                num_ep = number_of_end_points(new_image, k_nn)
                num_inters = number_of_intersection_points(new_image, k_nn)

                # Create a feature array matching the training data
                new_features = np.array([
                    mean_thickness, std_thickness, num_pixels, num_ep, num_inters,
                    mean_thickness * std_thickness, mean_thickness * num_pixels, mean_thickness * num_ep, mean_thickness * num_inters,
                    std_thickness * num_pixels, std_thickness * num_ep, std_thickness * num_inters,
                    num_pixels * num_ep, num_pixels * num_inters, num_ep * num_inters
                ]).reshape(1, -1)

                return new_features
            new_features = process_new_image(new_image_path)


            predicted_probabilities = loaded_model.predict_proba(new_features)
            if predicted_probabilities[0][1] < 0.87:
                predictions="Healthy"
                return render_template("pred2.html",prediction=predictions)
            else:
                predictions='You May have Parkinson. Please visit your doctor to be sure'
                return render_template("pred2.html",prediction=predictions)
            
    return render_template("pred2.html")



@model.route('/try1', methods=['GET', 'POST'])
def pred1():
    if request.method == 'POST':
        loaded_model = joblib.load('logistic_regression_model.pkl')
        MDVP1 = request.form.get('MDVP:Fo')
        MDVP2 = request.form.get('MDVP:Flo')
        MDVP3 = request.form.get('MDVP:Jitter')
        MDVP4 = request.form.get('MDVP:Jitter(Abs)')
        MDVP5 = request.form.get('MDVP:Shimmer')
        RPDE = request.form.get('RPDE')
        DFA = request.form.get('DFA')
        spread1 = request.form.get('spread1')
        spread2 = request.form.get('spread2')
        D2 = request.form.get('D2')
        PPE = request.form.get('PPE')

        data = {
        'MDVP:Fo(Hz)': [float(MDVP1)],
        'MDVP:Flo(Hz)': [float(MDVP2)],
        'MDVP:Jitter(%)': [float(MDVP3)],
        'MDVP:Jitter(Abs)': [float(MDVP4)],
        'MDVP:Shimmer': [float(MDVP5)],
        'RPDE': [float(RPDE)],
        'DFA': [float(DFA)],
        'spread1': [float(spread1)],
        'spread2': [float(spread2)],
        'D2': [float(D2)],
        'PPE': [float(PPE)],}
        input_series = pd.DataFrame(data).squeeze()
        predicted_class = loaded_model.predict([input_series])
        if predicted_class < 0.5:
            predictions="Healthy"
            return render_template("pred1.html",prediction=predictions)
        else:
            predictions='You May have Parkinson. Please visit your doctor to be sure'
            return render_template("pred1.html",prediction=predictions)
          
    return render_template("pred1.html")

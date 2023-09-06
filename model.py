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

model = Blueprint('model', __name__)

with open('rf_model.pickle', 'rb') as model_file:
    loaded_rf_model = pickle.load(model_file)

@model.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        image = request.files.get('image')
        if image:
            new_image_path = os.path.join(current_app.root_path,'images',image.filename)
            image.save(new_image_path)
            def process_single_image(image_path):
                # Load and preprocess the image
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                clean_img = closing(thresh_img > 0, disk(1))

                # Calculate stroke thickness
                thickness = stroke_thickness(clean_img)
                mean_thickness = np.mean(thickness)
                std_thickness = np.std(thickness)

                # Calculate number of pixels, end points, and intersection points
                num_pixels = np.sum(clean_img)
                num_ep = number_of_end_points(clean_img, k_nn)
                num_inters = number_of_intersection_points(clean_img, k_nn)

                # Calculate interactions between features
                interactions = {
                    'mean_thickness_std_thickness': mean_thickness * std_thickness,
                    'mean_thickness_num_pixels': mean_thickness * num_pixels,
                    'mean_thickness_num_ep': mean_thickness * num_ep,
                    'mean_thickness_num_inters': mean_thickness * num_inters,
                    'std_thickness_num_pixels': std_thickness * num_pixels,
                    'std_thickness_num_ep': std_thickness * num_ep,
                    'std_thickness_num_inters': std_thickness * num_inters,
                    'num_pixels_num_ep': num_pixels * num_ep,
                    'num_pixels_num_inters': num_pixels * num_inters,
                    'num_ep_num_inters': num_ep * num_inters
                }

                # Create a DataFrame with the processed features
                feature_data = {
                    'mean_thickness': mean_thickness,
                    'std_thickness': std_thickness,
                    'num_pixels': num_pixels,
                    'num_ep': num_ep,
                    'num_inters': num_inters,
                    **interactions  # Include interactions in the DataFrame
                }

                df = pd.DataFrame([feature_data])

                return df


            processed_df = process_single_image(new_image_path)
            predictions = loaded_rf_model.predict(processed_df )
            return render_template("home.html", prediction=predictions)
            
    return render_template("home.html")

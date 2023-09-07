# ParkiCareAI

ParkiCareAI is a repository dedicated to early detection of Parkinson's disease. This project was developed as part of a Data Science Bootcamp. The website aims to assist in the detection of Parkinson's disease, focusing on two key symptoms: hand shaking and speech changes. It employs two main models:

1. **Speech Model**: This model utilizes 11 features extracted from speech data collected by a telemonitoring device.

2. **Image Model**: The second model detects Parkinson's disease by analyzing images of hand-drawn waves.

## Contents

This repository contains the following components:

- **Dataset**: The dataset used to train the models.
- **Saved Models**: Pre-trained models for disease detection.
- **Code**: Code for training the models and creating the website.
- **Web Pages**: Code for the project's web interface.

## Usage

To run the website locally, follow these steps:

1. Install the required dependencies by running:

   ```bash
   pip install -r requirements.txt
   python main.py
   ```
This will launch the ParkiCareAI website, where you can input data for disease detection and explore its functionality.
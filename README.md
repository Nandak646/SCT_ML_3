**🐱🐶 Cat vs Dog Classifier**

This repository contains an interactive web application that classifies images of cats 🐱 and dogs 🐶 using a Support Vector Machine (SVM).

Users can upload images (jpg / png) or PDFs containing images. The app predicts whether the file contains a cat or dog and displays the result in colored cards (green for cats, blue for dogs).

##🚀 Features

Predicts Cat or Dog from uploaded images or PDFs.

Supports multiple file uploads at once.

Colored cards for visual clarity of predictions.

Summary table showing all uploaded files and their predicted labels.

Built using Python, OpenCV, scikit-learn, joblib, Streamlit, and pdf2image.

##🛠 How It Works

###Model Training

Preprocesses images (resize to 64x64, grayscale).

Trains a linear SVM classifier on Cat and Dog images.

Saves the trained model for fast prediction.

###Web App Deployment

Users upload images or PDFs.

App predicts Cat 🐱 or Dog 🐶.

Displays uploaded images and predictions in a professional interface.

This project demonstrates image preprocessing, machine learning model training, and interactive deployment using Streamlit.

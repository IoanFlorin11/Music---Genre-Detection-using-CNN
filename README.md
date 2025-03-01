Music Genre Detection using CNN

Overview

This project focuses on automatic music genre classification using deep learning and machine learning techniques. It employs a Convolutional Neural Network (CNN) with Transfer Learning and traditional machine learning classifiers trained on extracted audio features. The goal is to predict the genre of a given audio file using the GTZAN Dataset - Music Genre Classification.

We explore two approaches:

CNN-Based Classification - Uses spectrogram images of audio files as input to a pre-trained VGG-16 model.

Feature-Based Classification - Extracts key audio features (MFCC, Chroma, Spectral features) and applies traditional machine learning algorithms such as Logistic Regression, Random Forest, SVM, and Gradient Boosting.

Dataset

The project uses the GTZAN Dataset, which consists of 1000 audio files (each 30 seconds long) categorized into 10 different genres:

Blues

Classical

Country

Disco

Hip-hop

Jazz

Metal

Pop

Reggae

Rock

Project Structure

To run this project, execute the following steps in order:

# 1. Download the dataset
# (GTZAN Dataset - https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

# 2. Generate Spectrograms
Run PSV_plot_spectrogram.ipynb

# 3. Train the CNN Model (Transfer Learning with VGG-16)
Run PSV_vgg_model_transfer_learning.ipynb

# 4. Extract Audio Features
Run PSV_feature_extraction.ipynb

# 5. Train Machine Learning Models & Evaluate Performance
Run PSV_model_building.ipynb

Results

The CNN-based model (VGG-16 Transfer Learning) achieved an accuracy of 71%.

The best feature-based model (XGBoost) achieved an accuracy of 72%.

The ensemble model (CNN + XGBoost) outperformed both, achieving an AUC score of 0.967.

Requirements

Ensure you have the following dependencies installed:

pip install tensorflow-gpu==1.3.0 keras==2.0.8 numpy==1.12.1 pandas==0.22.0 \
youtube-dl==2018.2.4 scipy==0.19.0 librosa==0.5.1 tqdm==4.19.1 Pillow==4.1.1

Future Improvements

Increase Dataset Size: More training data could further enhance model performance.

Fine-Tuning CNN Model: Experiment with different CNN architectures and hyperparameters.

Optimize Feature Extraction: Identify and select the most relevant features for machine learning classifiers.

Acknowledgements

GTZAN Dataset - Kaggle

Deep Learning & ML Libraries: TensorFlow, Keras, Librosa, Scipy, Pandas, NumPy


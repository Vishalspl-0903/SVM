Datasets Used in the Project
This file describes the datasets used for each application in the project and provides guidance on how to obtain and use them.

1. Sentiment Analysis (IMDB Reviews)
Dataset: IMDB Reviews Dataset
Description: A labeled dataset of 50,000 movie reviews for binary sentiment analysis (positive or negative).
Usage:
Preprocess the text using regular expressions to clean unnecessary characters and whitespace.
Use TF-IDF vectorization for feature extraction before training the SVM.
2. Real vs AI-Generated Images
Dataset: CIFAKE Dataset
Description: A dataset comprising 120,000 images divided into real and AI-generated categories.
Usage:
Resize all images to a uniform size (e.g., 128x128 pixels) and convert them to grayscale.
Use Histogram of Oriented Gradients (HOG) for feature extraction.
Train the SVM using the extracted features with the RBF kernel for classification.
3. Hand Gesture Recognition
Dataset: Hand Gesture Dataset
Description: Contains images of 20 different hand gestures, split into 16,000 training and 4,000 testing images.
Usage:
Preprocess the images with OpenCV (resize and normalize).
Extract HOG features and train a Sigmoid kernel SVM for multiclass classification.
4. Medical Image Classification
Dataset: Chest X-ray Dataset
Description: Chest X-ray images labeled into Normal, Pneumonia, and COVID-19 categories.
Usage:
Preprocess images to ensure consistent resolution and quality.
Extract HOG features and train an SVM classifier using the RBF kernel.

# PixelSense-Fashion-Image-Classification-using-Deep-Learning
![Python](https://img.shields.io/badge/Python-3.8+-black?style=flat&logo=python)
![DL](https://img.shields.io/badge/Deep%20Learning-Convolutional%20Neural%20Network-purple)
![Dataset](https://img.shields.io/badge/Dataset-Fashion--MNIST-blue)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

PixelSense is a deep learning project focused on classifying grayscale fashion images into predefined apparel categories using neural networks. The project compares a baseline Artificial Neural Network (ANN) with a Convolutional Neural Network (CNN) to highlight the importance of spatial feature learning in image classification tasks.
The dataset used is **Fashion-MNIST**, which contains 28×28 grayscale images across 10 clothing categories.

---

##  Objectives
- Build a baseline ANN model using flattened pixel values
- Develop a CNN to capture spatial features from images
- Compare model performance using accuracy, confusion matrix, and classification report
- Demonstrate why CNNs outperform ANNs for image-based tasks

---

##  Dataset
- **Name:** Fashion-MNIST  
- **Images:** 28×28 grayscale  
- **Classes:** 10 apparel categories  
- **Train/Test Split:** Predefined in dataset  

---

##  Models Implemented
###  Artificial Neural Network (ANN)
- Flattened image input
- Dense layers with ReLU activation
- Softmax output layer
- Used as a baseline model

###  Convolutional Neural Network (CNN)
- Convolution + MaxPooling layers
- Preserves spatial structure of images
- Dense layers for classification
- Achieved significantly better performance on visually similar classes

---

##  Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

---

##  Results Summary
- ANN performed well on visually distinct classes (shoes, bags)
- ANN struggled with similar upper-body garments (shirts, pullovers)
- CNN improved classification accuracy and reduced misclassification
- CNN demonstrated superior generalization and feature extraction

---

##  Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

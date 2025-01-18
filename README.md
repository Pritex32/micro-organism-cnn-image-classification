# micro-organism-cnn-image-classification

## deployed on streamlit cloud link [https://micro-organism-cnn-image-classification-hlh7ygaohbzgkzdm6zjph4.streamlit.app/]

# Project Overview
Author: Prisca
Objective: Develop a model to classify images of microorganisms based on features

# 2. DatasetSource
The dataset is rom kaggle
# 3. Dataset Preparation
ImageDataGenerator from Keras is used to preprocess the images, including:
Rescaling pixel values to a range of [0, 1].
Splitting the dataset into training and validation subsets (validation_split=0.2).
Image Dimensions: Images are resized to (64, 64) for uniformity.
Batch Size: 32 images per batch.
# 4. Visualization
The script visualizes individual images using OpenCV and Matplotlib.
# 5. Model Architecture
The CNN model is defined with the following layers:

Convolutional Layers:
3 layers with filter sizes of 32, 64, and 64, respectively.
Kernel size: (3, 3) with ReLU activation.
Pooling Layers: Max pooling applied after each convolution layer (2, 2).
Flatten Layer: Converts feature maps into a 1D vector.
Dense Layers:
First Dense layer: 128 neurons with ReLU activation and dropout (rate: 0.3).
Output Dense layer: Number of neurons equal to the number of classes (softmax activation).
# 6 Compilation and Training
Optimizer: Adam.
Loss Function: Categorical cross-entropy (suitable for multi-class classification).
Metrics: Accuracy.
Epochs: 50.
Validation Split: 20%.
# 7. Evaluation
The model achieves 87% accuracy on both training and validation sets, suggesting good generalization.

# 8. Image Prediction
A sample image is loaded, preprocessed, and passed through the trained model.
Predictions are mapped to corresponding class names, which include:
Amoeba, Euglena, Hydra, Paramecium, Rod bacteria, Spherical bacteria, Spiral bacteria, and Yeast.

# 9. Model Saving
The model is saved in two formats:
.keras
.h5

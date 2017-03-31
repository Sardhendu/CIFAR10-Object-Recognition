# CIFAR10
CIFAR10- Object Recognition


The repository contains implementation and evaluation of several Models for CIFAR10 Object recognition

Below are some Feature Extraction and Modules implemented.

    1. Feature Extraction: 
            --> RGB 
            --> Standarized Image
            --> Edge Features.
            --> Histogram of oriented Gradients
            --> ZCA whitened
    
    2. Models:
            --> K-nearest Neighbors
            --> Logistic Regression
            --> Support Vector Machines
            --> Deep Neural Networks
            --> Convolutional Neural Networks
            --> Convolutional Neural Networks with Inception Module
            
    3. Evaluation:
            --> Model Accuracy
            --> Confusion Matrix

Note: The majority of the code resides inside the MODEL folder. For Simplicity and deep understanding of several techniques/model, we emlploy and evaluate the models for only 2 classes (Airplane and Cat). However the code can be easily be extented for all the 10 labels, which would require a little bit of hyperparameter tuning.
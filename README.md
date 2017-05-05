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


Paper/Code References:

    1. http://cs231n.github.io/convolutional-networks/
    2. ImageNet Classification with Deep Convolutional Neural Networks - Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
    3. Maxout Networks : Ian J. Goodfellow David Warde-Farley Mehdi Mirza, Aaron Courville Yoshua Bengio
    4. Dropout: A Simple Way to Prevent Neural Networks from Overfitting - Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov
    5. Going deeper with convolutions - Christian Szegedy, Pierre Sermanet, Wei Liu , Yangqing Jia , Dumitru Erhan , Scott Reed , Dragomir Anguelov , Vincent Vanhoucke , Andrew Rabinovich
from __future__ import division
from __future__ import print_function

import numpy as np



def reshape_data(dataset, labels=None, imageSize=32, numChannels=1, sample_size=None):
    if sample_size:
        dataset = dataset[:sample_size].reshape((-1,imageSize,imageSize,numChannels)) # To reshape the
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        if labels:
            numLabels = len(np.unique(labels))
            labels = (np.arange(numLabels) == labels[:,None]).astype(np.float32)
    else:
        dataset = dataset.reshape((-1,imageSize,imageSize,numChannels)) # To reshape the
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        if labels:
            numLabels = len(np.unique(labels))
            labels = (np.arange(numLabels) == labels[:,None]).astype(np.float32)
    return dataset, labels


def accuracy(predictions, labels):
    # Both the predictions and the labels should be in One-Hot vector format.
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])


def confusionMatrix(predictions, labels):
    # Both the predictions and the labels should be in One-Hot vector format.
    return (pd.crosstab(np.argmax(labels, 1), np.argmax(predictions, 1), rownames=['True'], colnames=['Predicted'], margins=True))
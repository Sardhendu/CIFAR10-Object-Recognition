from __future__ import division
import numpy as np


def accuracy(prediction, labels, labels_one_hot = None):
	# The input labels are a One-Hot Vector
	if labels_one_hot:
		return (100.0 * np.sum(np.argmax(prediction, 1) == np.argmax(labels, 1)) / prediction.shape[0])
	else:
		return (100.0 * np.sum(np.argmax(prediction, 1) == np.reshape(labels, [-1])) / prediction.shape[0])

def confusionMatrix():
	pass

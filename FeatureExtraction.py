from __future__ import print_function
import os, sys, glob
import numpy as np
import pandas as pd
"""
	About me: This module contain different feature extraction techniques
				This modele is called by the DataPreparation modele whilw creating batches

"""


import cv2
import pickle
from IPython.display import display, Image
from scipy import ndimage



def image_standarize(image_pxlvals):
    return(image_pxlvals - 255.0/2)/255.0

def feature_pixelSTD(parent_path,file_names_arr,image_size=32, min_num_image=None):
	# Declare an nd array of the lenght and size provided
	dataset = np.ndarray(shape=(len(file_names_arr), 
                                image_size, 
                                image_size
                               ),
                         dtype=np.float32)
	print (dataset.shape)
	for num_images, image in enumerate(file_names_arr):
		image_file_dir = os.path.join(parent_path, image)

		try:
			image_orig = cv2.imread(image_file_dir)
			image_gray = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
			# print (image_gray.shape)
			# image_pixels = ndimage.imread(image_file_dir).astype(float)

			image_standarized = image_standarize(image_gray)
			# print (image_standarized.shape)
			if image_standarized.shape != (image_size, image_size):
				raise Exception('Unexpected image shape: %s' % str(image_standarized.shape))

			dataset[num_images, :, :] = image_standarized

		except IOError as e:
			print('Could not read:', image, ':', e, '- hence skipping.')

	# dataset = dataset[0:num_images+1, :, :]  # Will combine all the image pixels for 1 type Example All the 'a'
	print('Complete Training/Crossvalidation dataset :', dataset.shape)
	print('Mean:', np.mean(dataset))
	print('Standard deviation:', np.std(dataset))
	return dataset
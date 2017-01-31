"""
About me: This module will go through all the folders representing each labels of the CIFAR10 dataset, slice and dice the data and create batches (10 batches), where batches are subset of data.

	1. All the batches have equal proportion of labels:
		For example batch 1 of size 5000 contains 500 dataset pertaining to label CAR, 500 dataset pertaining to cat and so on.

"""


from __future__ import print_function
import os, sys, glob
import numpy as np
import pandas as pd

import cv2
import pickle
from IPython.display import display, Image
from scipy import ndimage

from FeatureExtraction import feature_pixelSTD

from Tools import GlobalVariables

         
    
def create_dataset(dataPath, featureType, max_num_images=None,force_dump=None):
	for no,images_path in enumerate(dataPath):
		if max_num_images:
			file_names_arr = os.listdir(images_path)[0:max_num_images]
		else:
			file_names_arr = os.listdir(images_path)

		dataset = feature_pixelSTD(images_path,file_names_arr,image_size=32)

		# Store the image as a pickle file in the directory
		pickle_filename = os.path.basename(os.path.abspath(images_path))
		path_to_folder = conf['parent_dir']+'/'+featureType

		if not os.path.exists(path_to_folder):
			os.makedirs(path_to_folder)

		pickle_image_dir = os.path.dirname(os.path.abspath(images_path))+'/'+featureType+'/'+pickle_filename+'.pickle'
		# print (pickle_image_dir)
		if os.path.exists(pickle_image_dir) and not force_dump:
			print ('The path already exists, you should force the dump')
		else:
			try:
				with open(pickle_image_dir, 'wb') as f:
					pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
			except Exception as e:
				print('Unable to save data to', pickle_image_dirpickle_image_dir, ':', e)



class BuildDataset():

	def __init__(self):
		self.imageSize = 32

	def initialize_arrays(self, num_images):
		dataset = np.ndarray((num_images, self.imageSize, self.imageSize), dtype=np.float32)
		labels = np.ndarray(num_images, dtype=np.int32)
		return dataset, labels

	# You should provide the max_num_image. And the num_of_images per class label should not exceed it.
	# The below code created a big Training set with all the 
	def train_test(self, numBatches, max_num_images, featureType, test_percntg=10):
		root_dir = conf['parent_dir']+featureType
		label_categories = [files for files in os.listdir(root_dir) if files.endswith(".pickle")]
		# print (label_categories)

		if not os.path.exists(root_dir):
			print ('The path %s doesnt exists ', root_dir)
			# break

		test_size_per_class = int(test_percntg/100 * max_num_images)
		train_size_per_class = int(max_num_images - test_size_per_class)
		# print (train_size_per_class, test_size_per_class)

		num_test_datapoints = int(np.ceil(test_size_per_class * len(label_categories)))
		num_train_datapoints = int(train_size_per_class * len(label_categories))
		
		self.testDataset, self.testLabels = self.initialize_arrays(num_test_datapoints)
		self.trainDataset, self.trainLabels = self.initialize_arrays(num_train_datapoints)

		# print (testDataset.shape, testLabels.shape)
		# print (trainDataset.shape, trainLabels.shape)

		start_trn, start_tst = 0, 0
		end_trn, end_tst  = train_size_per_class, test_size_per_class
		end = train_size_per_class + test_size_per_class

		self.label_dict = {}
		for label_id, label_file in enumerate(label_categories):
			# print (start_trn ,start_tst, end_trn, end_tst)
			self.label_dict[label_id] = label_file
			try:
				with open (os.path.join(root_dir,label_file), 'rb') as f:
					dataMatrix = pickle.load(f)
					# print (dataMatrix.shape)

					np.random.shuffle(dataMatrix)
					self.testDataset[start_tst:end_tst,:,:] = dataMatrix[0:test_size_per_class,:,:]
					self.trainDataset[start_trn:end_trn,:,:] = dataMatrix[test_size_per_class:end,:,:]
					self.testLabels[start_tst:end_tst] = label_id
					self.trainLabels[start_trn:end_trn] = label_id

					start_tst += test_size_per_class
					end_tst += test_size_per_class
					start_trn += train_size_per_class
					end_trn += train_size_per_class
			except Exception as e:
				print('Unable to process data from', pickle_files, ':', e)
				raise

		print ('The training Data set size is : ', self.trainDataset.shape)
		print ('The training Labels size is : ', self.trainLabels.shape)
		print ('The test Data set size is : ', self.testDataset.shape)
		print ('The test Labels size is : ', self.testLabels.shape)
		# return 	label_dict, trainDataset, testDataset, trainLabels, testLabels


	def generate_batches(self, indices_arrays):
		# The below code woll just provide the initiator function with the training dataset and labels for the input index values.
		for indices in indices_arrays:
			yield self.trainDataset[indices], self.trainLabels[indices]  # , 


def dumpBatches(featureType, create_pickle_file='N', create_train_test_valid='N', test_percntg=0):
	if create_pickle_file == 'Y':
		objGV = GlobalVariables()
		create_dataset(objGV.dataPaths, featureType=featureType, force_dump='y')

	if create_train_test_valid == 'Y':
		batchPath = conf['parent_dir']+featureType+'/batchPath/'
		numBatches = 10
		obj = BuildDataset()
		obj.train_test(numBatches=10, max_num_images=5000, featureType=featureType, test_percntg=test_percntg)
		
		if (len(np.unique(obj.trainLabels))%numBatches == 0): 
			# The below three lines of code will just create a dummy array and jumble the indices so we get equal mumber of labels in all the batches.
			dummy_arr = np.arange(len(obj.trainLabels))
			# print (dummy_arr)
			nwshape = (numBatches, int(len(dummy_arr)/numBatches))
			batch_indices = np.reshape(np.reshape(dummy_arr, nwshape).T, nwshape)
			# print (list(batch_indices[0]))

			for no, (trnBatchData, trnBatchLabel) in enumerate(obj.generate_batches(batch_indices)):
				print (trnBatchData.shape, trnBatchLabel.shape)
				
				if not os.path.exists(batchPath):
					os.makedirs(batchPath)

				with open(batchPath+'batch'+str(no)+'.pickle', 'wb') as f:
					batch = {
							'batchData': trnBatchData,
							'batchLabels': trnBatchLabel
							}
					pickle.dump(batch, f, pickle.HIGHEST_PROTOCOL)
		else:
			print ('For equal distribution please provide a numBatches that multiple of the number of Class %d'%len(np.unique(trainLabels)))



# dumpBatches(featureType='featureSTD', create_pickle_file='N', create_train_test_valid='Y', test_percntg=0)


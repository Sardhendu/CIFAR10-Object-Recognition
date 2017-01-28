from __future__ import print_function
import os, sys, glob
import numpy as np
import pandas as pd

import cv2
import pickle
from IPython.display import display, Image
from scipy import ndimage


parent_dir = '/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/'

bird_datapath = '/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/trainDataBird/'
dog_datapath = '/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/trainDataDog/'
cat_datapath = '/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/trainDataCat/'
horse_datapath = '/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/trainDataHorse/'
deer_datapath = '/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/trainDataDeer/'
frog_datapath = '/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/trainDataFrog/'
truck_datapath = '/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/trainDataTruck/'
airplane_datapath = '/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/trainDataAirplane/'
ship_datapath = '/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/trainDataShip/'
auto_datapath = '/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/trainDataAuto/'

dataPaths = [bird_datapath,dog_datapath,cat_datapath,horse_datapath,deer_datapath,
frog_datapath,truck_datapath,airplane_datapath,ship_datapath,auto_datapath]


def image_standarize(image_pxlvals):
    return(image_pxlvals - 255.0/2)/255.0

def feature_pixelSTD(parent_path,file_names_arr,image_size=32, min_num_image=None):
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
         
    
def create_dataset(dataPath, featureType, max_num_images=None,force_dump=None):
	for no,images_path in enumerate(dataPath):
		if max_num_images:
			file_names_arr = os.listdir(images_path)[0:max_num_images]
		else:
			file_names_arr = os.listdir(images_path)

		dataset = feature_pixelSTD(images_path,file_names_arr,image_size=32)

		# Store the image as a pickle file in the directory
		pickle_filename = os.path.basename(os.path.abspath(images_path))
		path_to_folder = parent_dir+'/'+featureType

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
	def train_test(self, numBatches, max_num_images, featureType, test_percntg=20):
		root_dir = parent_dir+featureType
		label_categories = os.listdir(root_dir)

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
		# return 	label_dict, trainDataset, testDataset, trainLabels, testLabels


	def generate_batches(self, indices_arrays):
		for indices in indices_arrays:
			yield self.trainDataset[indices], self.trainLabels[indices]  # , 




def dumpBatches(featureType, create_pickle_file='N', create_train_test_valid='N'):
	if create_pickle_file == 'Y':
		create_dataset(dataPaths, featureType=featureType, force_dump='y')

	if create_train_test_valid == 'Y':
		batchPath = parent_dir+featureType+'/batchPath/'
		numBatches = 10
		obj = BuildDataset()
		obj.train_test(numBatches=10, max_num_images=5000, featureType=featureType)
		if (len(np.unique(obj.trainLabels))%numBatches == 0): 

			dummy_arr = np.arange(len(obj.trainLabels))
			nwshape = (numBatches, int(len(dummy_arr)/numBatches))

			batch_indices = np.reshape(np.reshape(dummy_arr, nwshape).T, nwshape)
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


# Get the batches from the disk and see if the plots looks good
# This just Generated batches one after another to the calling mechanism
def batch_file_iterator(batch_dir, batch_filename_arr):
    for no, batch_file in enumerate(batch_filename_arr):
        with open(os.path.join(batch_dir,batch_file), 'rb') as f:
            data = pickle.load(f)
            trnBatchData = data['batchData']
            trnBatchLabels = data['batchLabels']
        yield trnBatchData, trnBatchLabels




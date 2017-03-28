"""
About Me: The Input unzipped folder for CIFAR10 has the images of all the labels together. This module just segregates them and puts them into seperate folder. So ultimately the output of this module will be 10 folders each representing one label of the CIFAR10 data set.
"""


import os, sys, glob
import numpy as np
import pandas as pd
import shutil


if True:
	# input your Folder structure Here:
	parent_dir = '/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/'

	# dataPath is the folder path to all the images
	dataPath = '/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/trainData/'
	# labelsPath is the csv file containing the image file name and the corresponding label
	labelsPath = '/Users/sam/All-Program/App-DataSet/Kaggle-Challenges/CIFAR-10/trainLabels.csv'

	# The below are different folder path, where you want to put your Class segregated data.
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




labels = pd.read_csv(labelsPath, sep=',')

# labels_df = pd.DataFrame(labels)
# print (labels.head())

labels.groupby('label').agg({'label':np.size})

bird_image_index = np.array(labels_df[labels_df['label'] == 'bird']['id'])
dog_image_index = np.array(labels_df[labels_df['label'] == 'dog']['id'])
cat_image_index = np.array(labels_df[labels_df['label'] == 'cat']['id'])
horse_image_index = np.array(labels_df[labels_df['label'] == 'horse']['id'])
deer_image_index = np.array(labels_df[labels_df['label'] == 'deer']['id'])
frog_image_index = np.array(labels_df[labels_df['label'] == 'frog']['id'])
truck_image_index = np.array(labels_df[labels_df['label'] == 'truck']['id'])
airplane_image_index = np.array(labels_df[labels_df['label'] == 'airplane']['id'])
ship_image_index = np.array(labels_df[labels_df['label'] == 'ship']['id'])
auto_image_index = np.array(labels_df[labels_df['label'] == 'automobile']['id'])

label_indices = [bird_image_index,dog_image_index,cat_image_index,horse_image_index,deer_image_index,
frog_image_index,truck_image_index,airplane_image_index,ship_image_index,auto_image_index]

label_paths = [bird_datapath,dog_datapath,cat_datapath,horse_datapath,deer_datapath,
frog_datapath,truck_datapath,airplane_datapath,ship_datapath,auto_datapath]
# print (bird_image_index[0:10])

for path,indices in zip(label_paths, label_indices):
    print ('Done: ', path)
    for files in glob.glob(dataPath+'*.png'):
        filename = os.path.basename(files).split('.')[0]
    #     print (filename)
        if int(filename) in indices:
            shutil.copy(files, path)
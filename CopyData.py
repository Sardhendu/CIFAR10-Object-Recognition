import os, sys, glob
import numpy as np
import pandas as pd
import shutil


if True:
	from Tools import GlobalVariables
	# print (conf["bird_datapath"])
	obj_GV = GlobalVariables()
	dataPath = obj_GV.dataPath
	labelsPath = obj_GV.labelsPath
	
	bird_datapath = obj_GV.bird_datapath
	dog_datapath = obj_GV.dog_datapath
	cat_datapath = obj_GV.cat_datapath
	horse_datapath = obj_GV.horse_datapath
	deer_datapath = obj_GV.deer_datapath
	frog_datapath = obj_GV.frog_datapath
	truck_datapath = obj_GV.truck_datapath
	airplane_datapath = obj_GV.airplane_datapath
	ship_datapath = obj_GV.ship_datapath
	auto_datapath = obj_GV.auto_datapath


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
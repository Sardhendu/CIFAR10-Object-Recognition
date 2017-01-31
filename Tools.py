from __future__ import division
import numpy as np
import os, sys
import pickle


class GlobalVariables():
    def __init__(self):
        # print ('plplplplplp')
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        import config
        conf = config.get_config_settings()
        self.rawDataPath = conf["rawDataPath"]
        self.rawLabelsPath = conf["rawLabelsPath"]
        self.bird_datapath = conf["bird_datapath"]
        self.dog_datapath = conf['dog_datapath']
        self.cat_datapath = conf['cat_datapath']
        self.horse_datapath = conf['horse_datapath']
        self.deer_datapath = conf['deer_datapath']
        self.frog_datapath = conf['frog_datapath']
        self.truck_datapath = conf['truck_datapath']
        self.airplane_datapath = conf['airplane_datapath']
        self.ship_datapath = conf['ship_datapath']
        self.auto_datapath = conf['auto_datapath']

        self.dataPaths = [self.bird_datapath,self.dog_datapath,self.cat_datapath,self.horse_datapath,self.deer_datapath,self.frog_datapath,self.truck_datapath,self.airplane_datapath,self.ship_datapath,self.auto_datapath]
        self.stdBatchPath = conf['stdBatchPath']







def accuracy(prediction, labels, labels_one_hot = None):
	# The input labels are a One-Hot Vector
	if labels_one_hot:
		return (100.0 * np.sum(np.argmax(prediction, 1) == np.argmax(labels, 1)) / prediction.shape[0])
	else:
		return (100.0 * np.sum(np.argmax(prediction, 1) == np.reshape(labels, [-1])) / prediction.shape[0])

def confusionMatrix():
	pass


# objGV = GlobalVariables()
# print (objGV.stdBatchPath)

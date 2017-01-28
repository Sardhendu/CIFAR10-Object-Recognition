from __future__ import print_function
import os, sys, glob
import numpy as np
import pandas as pd


from DataPreparation import dumpBatches


__main__(featureType = 'featureSTD',
	create_pickle_file = 'N', 
	create_train_test_valid = 'N')
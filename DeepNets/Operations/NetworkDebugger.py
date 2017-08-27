from __future__ import division
from __future__ import print_function

import numpy as np
# import tensorflow as tf

class SanityCheck():
    def __init__(self):
        pass
    
    @staticmethod
    def checkStartingLoss(lossIN, numLabels):
        print (lossIN)
        print (-np.log2(1/numLabels))
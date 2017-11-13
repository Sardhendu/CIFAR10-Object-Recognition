from __future__ import division
from __future__ import print_function

import numpy as np
# import tensorflow as tf

class SanityCheck():
    def __init__(self):
        pass
    
    @staticmethod
    def checkStartingLoss(lossIN, numLabels):
        l1 = lossIN
        l2 = -np.log2(1/numLabels)
        if int(l1) != int(l2):
            raise ValueError ('The loss in the first iteration doesnt equal the -log/numLabels, Check your code!!')
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from keras import backend as krs
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
import fr_utils

'''
Inception: With convolutional networks a lot depends on the patch/filter size you use to stride through the
image. VGG net showed that having one patch but deep networks could provide better output. The confusion arises, what is a good pathc 3x3 or 5x5. Inception says, why not use all and stack them together.

Another advantage of Inception is the use of 1x1 path, which can be seen as a natwork inside a network.
'''

def inceptionBlock_c1(inp, blockNum):
    '''
    :param inp: The input image
        
        """
            This is the first convolve of the inception block. It has a single convolution module,
            1) convolve through 64 1x1 filter : application to shrink accross channels (mini network inside a network)
            
            if your input data is in 32x32x3 format, pass data_format='channel_last' and axis = -1 (normalize accross channels)
            if your input data is in 3x32x32 format, pass data_format='channel_first' and axis = 1 (normalize accross channels)
            
            momentum = moving average (exponential average) beta value typically more than 0.9
            epsilon = To avoid division by zero
            
        """
    '''
    no = blockNum
    X = Conv2D(filters=64, kernel_size=(1,1), data_format='channels_last', name='inception_c11_l%s_conv'%str(no))(inp)
    X = BatchNormalization(axis=1, momentum=0.99, epsilon=1e-5, name='inception_c1_l%s_bn'%str(no))(X)
    X = Activation('relu')(X)
    
    return X

def inceptionBlock_c2(inp, blockNum):
    no = blockNum
    X = Conv2D(filters=96, kernel_size=(1,1), data_format='channel_last', name='inception_c21_l%s_conv'%str(no))(inp)
    X = BatchNormalization(axis=1, momentum=0.99, epsilon=1e-5, name='inception_c21_l%s_bn' % str(no))(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters=128, kernel_size=(3,3), data_format='channel_last', name='inception_c22_l%s_conv'%str(no))(X)
    X = BatchNormalization(axis=1, momentum=0.99, epsilon=1e-5, name='inception_c22_l%s_bn' % str(no))(X)
    X = Activation('relu')(X)
    
    return X

def inceptionBlock_c3(inp, blockNum):
    no = blockNum
    X = Conv2D(filters=16, kernel_size=(1, 1), data_format='channel_last', name='inception_c31_l%s_conv' % str(no))(inp)
    X = BatchNormalization(axis=1, momentum=0.99, epsilon=1e-5, name='inception_c31_l%s_bn' % str(no))(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters=32, kernel_size=(5, 5), data_format='channel_last', name='inception_c32_l%s_conv' % str(no))(X)
    X = BatchNormalization(axis=1, momentum=0.99, epsilon=1e-5, name='inception_c32_l%s_bn' % str(no))(X)
    X = Activation('relu')(X)
    
    return X

def inceptionBlock_c4(inp, blockNum):
    no = blockNum
    X = MaxPooling2D(pool_size=3, strides=2, data_format='channel_last', name='inception_c41_l%s_maxpool' % str((no)))(inp)

    X = Conv2D(filters=32, kernel_size=(1, 1), data_format='channel_last', name='inception_c41_l%s_conv' % str(no))(X)
    X = BatchNormalization(axis=1, momentum=0.99, epsilon=1e-5, name='inception_c41_l%s_bn' % str(no))(X)
    X = Activation('relu')(X)

    # padding: ((top_pad, bottom_pad), (left_pad, right_pad))
    X = ZeroPadding2D(padding=((3, 4), (3, 4)), data_format='channels_last')(X)
    
def inceptionConcat(inp, blockNum):
    X_c1 = inceptionBlock_c1(inp, blockNum)
    X_c2 = inceptionBlock_c2(inp, blockNum)
    X_c3 = inceptionBlock_c3(inp, blockNum)
    X_c4 = inceptionBlock_c4(inp, blockNum)
    
    inception = concatenate([X_c4, X_c2, X_c3, X_c1], axis=1)

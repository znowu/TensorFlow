from __future__ import absolute_import, print_function, division,unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D
from tensorflow.keras import Model

# define the layers beforehand for ease
def conv3(filters):
    return Conv2D(filters= filters, kernel_size= 3, strides= 1, padding= 'same',
                  activation= 'relu', kernel_initializer= tf.random_normal_initializer)

def conv1(filters):
    return Conv2D(filters= filters, kernel_size= 1, strides= 1, padding= 'same',
                  activation= 'relu', kernel_initializer= tf.random_normal_initializer)

def deconv(filters):
    return Conv2DTranspose(filters= filters, kernel_size= 2, strides=2, padding= 'same',
                           activation= 'relu', kernel_initializer= tf.random_normal_initializer)

# define model (specify the number of classes)
class UNet(Model):
    def __init__(self, config):
        super().__init__()
        self.classes= config["classes"]
        self.levels = 5
        self.starting_filters= 64
        self.pool = MaxPool2D(2,2)

        # contraction branch
        self.contraction = {
            level: [conv3(self.starting_filters*pow(2, level)),
                    conv3(self.starting_filters*pow(2, level) )]
            for level in range(self.levels)
        }

        # upsampling branch
        self.upsampling = {
            level: [deconv(self.starting_filters*pow(2, level ) ),
                    conv3(self.starting_filters*pow(2, level)),
                    conv3(self.starting_filters*pow(2, level) ) ]
            for level in range(self.levels-1, -1, -1)
        }
        self.fcn = conv1(filters= classes)
        self.logit = tf.keras.layers.Softmax(-1 )

    def __call__(self, x):
        layers_to_concat = []
        for level in range(self.levels):
            block = self.contraction[level]
            for layer in block:
                x = layer(x)
            layers_to_concat.append(x)

        for level in range(self.levels-1, -1, -1):
            block = self.upsampling[level]
            x = block[0](x)
            x = tf.concat([layers_to_concat[level], x], -1)
            for layer in block[1:]:
                x = layer(x)

        return x

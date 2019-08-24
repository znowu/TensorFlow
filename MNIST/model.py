from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras import Model

class Network(Model):
    def __init__(self, config):
        super().__init__()
        # layers
        self.conv= Conv2D(filters=config['conv_filters'] , kernel_size= config['conv_kernel'],
                          strides= config['conv_strides'], activation= 'relu')
        self.dense1 = Dense(config['dense1'], activation= 'relu')
        self.dense2 = Dense(config['dense2'], activation= 'softmax')
        self.flatten = Flatten()

        # training actors and metrics
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.train_loss = tf.keras.metrics.Mean()
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # test metrics
        self.test_loss = tf.keras.metrics.Mean()
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    def call(self, x):
        x = self.flatten( self.conv(x) )
        x = self.dense2( self.dense1(x) )
        return x

#-*- coding: utf-8 -*-
import os
import sys
import time
import copy
import math

import numpy
import argparse

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from convolution import WsdConvPoolLayer

class cnnmodel(object):

    def __init__(self, rng, window_radius, learning_rate, batch_size, nkerns, loginput_num, vector_size):

        self.window_radius = window_radius
        self.vector_size = vector_size
        self.batch_size = batch_size
        self.nkerns = nkerns
        self.loginput_num = loginput_num
        self.learning_rate = learning_rate
        self.x = T.matrix('x')   
        self.y = T.ivector('y')

        self.layer0_input = self.x.reshape((batch_size, 1, 2*window_radius+1, vector_size))
        self.layer0 = []
        for i in range(1,2*window_radius+2):
            ph = 2*window_radius+2-i
            #print ph
            self.layer0.append(WsdConvPoolLayer(
                rng,
            input=self.layer0_input,
            image_shape=(batch_size, 1, 2*window_radius+1, vector_size),
            filter_shape=(nkerns, 1, i, vector_size),
            poolsize=(ph, 1)
            ))
        self.layer0_output = theano.tensor.concatenate([self.layer0[i].output for i in range(0, len(self.layer0))])
        self.layer1_input = self.layer0_output.flatten(1)

        input_size = nkerns*(2*window_radius+1)

        self.layer1 = HiddenLayer(
            rng,
            input=self.layer1_input,
            #n_in=(2*window_radius+1)*(vector_size+1-filter_width+1-pool_width),
            n_in=input_size,
            n_out=loginput_num,
            activation=T.tanh
        )

        self.layer2 = LogisticRegression(input=self.layer1.output, n_in=loginput_num, n_out=100)






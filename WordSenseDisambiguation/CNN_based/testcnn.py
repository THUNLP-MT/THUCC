#-*- coding: utf-8 -*-
import os
import sys
import time
import copy
import math
import json
import codecs

import numpy
import argparse
import cPickle
import traceback
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from convolution import WsdConvPoolLayer
from cnnmodel import cnnmodel
from datafetch import sentence2vector, todict

def testcnn_one(sentence,position):
	
	result = []

	for i in range(position,position+1):

		if True:
			try:
				fileopen = False
				savefile = open('model//cnn//'+sentence[i])
				fileopen = True
				model = cPickle.load(savefile)
				data_x = [sentence2vector(sentence, model.window_radius, model.vector_size,i)]

				test_set_x = theano.shared(numpy.asarray(data_x,
												   dtype=theano.config.floatX),
									 borrow=True)

				index = T.lscalar()
				output_model = theano.function(
					[index],
					[model.layer2.y_pred],
					givens={
						model.x: test_set_x[index:(index+1)]
					}
				)

				result.append(output_model(0)[0][0])
			except:
				if not fileopen:
					print 'model for '+sentence[i]+' does not exist'
				else:
					traceback.print_exc()
					print 'unknown error'
	return result 


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='testcnn')
	parser.add_argument('-w','--wenyan')
	parser.add_argument('-d', '--dictionary')
	parser.add_argument('-i', '--index')

	args = parser.parse_args()
	wenyan = args.wenyan.decode('utf-8')

	result = testcnn_one(wenyan, int(args.index))
	print result

	dic = json.load(codecs.open(args.dictionary, 'r', 'utf-8'))
	dic = todict(dic)
	print dic[wenyan[int(args.index)]][result[0]]

	

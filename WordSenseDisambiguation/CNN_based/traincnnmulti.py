#-*- coding: utf-8 -*-
import os
import sys
import time
import copy
import math

import json
import numpy
import argparse
import cPickle
import codecs

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from convolution import WsdConvPoolLayer
from cnnmodel import cnnmodel
from datafetch import prepare_data, todict, process_corpus

def trainword(dic, train, keyword, dev=None, window_radius = 3, learning_rate = 0.1, n_epochs = 10,batch_size = 1,nkerns = 1,filter_height=3,filter_width = 50, pool_height=1,pool_width = 1, loginput_num = 50, vector_size = 50, sequence = 0):

	print 'train word:', keyword

	rng = numpy.random.RandomState(23455)
	datasets = prepare_data(dic, train, keyword, window_radius, vector_size, sequence = sequence)
	train_set_x, train_set_y, trainsentence = datasets[0]
	if dev:
		valid_set_x, valid_set_y, validsentence = prepare_data(dic, dev, keyword, window_radius, vector_size, sequence = sequence)[0]

	#if valid_set_x.get_value(borrow=True).shape[0] == 0:
	#	valid_set_x, valid_set_y, validsentence = train_set_x, train_set_y, trainsentence
	#if test_set_x.get_value(borrow=True).shape[0] == 0:
	#	test_set_x, test_set_y, testsentence = train_set_x, train_set_y, trainsentence
	senselist = datasets[1]

	n_train_batches = train_set_x.get_value(borrow=True).shape[0]
	#n_test_batches = test_set_x.get_value(borrow=True).shape[0]
	n_train_batches /= batch_size
	#n_test_batches /= batch_size
	if n_train_batches <= 0:
		return
	if dev:
		n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
		n_valid_batches /= batch_size
		if n_valid_batches == 0:
			dev = None
		print n_train_batches, n_valid_batches#, n_test_batches

	index = T.lscalar()

	model = cnnmodel(rng, window_radius, learning_rate, batch_size, nkerns, loginput_num, vector_size)

	cost = model.layer2.negative_log_likelihood(model.y)

	if dev:
		validate_model = theano.function(
			[index],
			model.layer2.errors(model.y),
			givens={
				model.x: valid_set_x[index * batch_size: (index + 1) * batch_size],
				model.y: valid_set_y[index * batch_size: (index + 1) * batch_size]
			}
		)

		output_model = theano.function(
			[index],
			[model.layer2.y_pred],
			givens={
				model.x: valid_set_x[index * batch_size: (index + 1) * batch_size]
			}
		)

	params = model.layer2.params + model.layer1.params #+ layer0.params
	for i in range(0, len(model.layer0)):
		params += model.layer0[i].params

	grads = T.grad(cost, params)

	updates = [
		(param_i, param_i - learning_rate * grad_i)
		for param_i, grad_i in zip(params, grads)
	]

	train_model = theano.function(
		[index],
		cost,
		updates=updates,
		givens={
			model.x: train_set_x[index * batch_size: (index + 1) * batch_size],
			model.y: train_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)

	print '... training'
	# early-stopping parameters
	patience = max(n_train_batches*5,20000)  # look as this many examples regardless
	patience_increase = 2  # wait this much longer when a new best is
						   # found
	improvement_threshold = 0.995  # a relative improvement of this much is
								   # considered significant
	validation_frequency = min(n_train_batches, patience / 4)
								  # go through this many
								  # minibatche before checking the network
								  # on the validation set; in this case we
								  # check every epoch

	best_validation_loss = numpy.inf
	best_params = 0
	best_iter = 0
	test_score = 0.
	start_time = time.clock()

	epoch = 0
	done_looping = False

	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1
		for minibatch_index in xrange(n_train_batches):

			iter = (epoch - 1) * n_train_batches + minibatch_index

			if iter % 100 == 0:
				print 'training @ iter = ', iter
			cost_ij = train_model(minibatch_index)

			if dev and (iter + 1) % validation_frequency == 0:

				# compute zero-one loss on validation set
				validation_losses = [validate_model(i) for i
									 in xrange(n_valid_batches)]
				#for index in range(0, n_valid_batches):
				#	print output_model(index)
				#	print valid_set_y[index * batch_size: (index + 1) * batch_size].eval()
				this_validation_loss = numpy.mean(validation_losses)
				print('epoch %i, minibatch %i/%i, validation error %f %%' %
					  (epoch, minibatch_index + 1, n_train_batches,
					   this_validation_loss * 100.))

				# if we got the best validation score until now
				if this_validation_loss < best_validation_loss:

					#improve patience if loss improvement is good enough
					if this_validation_loss < best_validation_loss *  \
					   improvement_threshold:
						patience = max(patience, iter * patience_increase)

					# save best validation score and iteration number
					best_validation_loss = this_validation_loss
					best_iter = iter
					savefile = open('model//cnn//'+keyword, 'wb')
					cPickle.dump(model,savefile,-1)

			if patience <= iter:
				done_looping = True
				break

	end_time = time.clock()
	print('Optimization complete.')
	
	if dev:
		print('Best validation score of %f %% obtained at iteration %i, '
			  'with test performance %f %%' %
			  (best_validation_loss * 100., best_iter + 1, test_score * 100.))
	else:
		savefile = open('model//cnn//'+keyword, 'wb')
		cPickle.dump(model,savefile,-1)
	print >> sys.stderr, ('The code for file ' +
						  os.path.split(__file__)[1] +
						  ' ran for %.2fm' % ((end_time - start_time) / 60.))
	return [best_validation_loss * 100., test_score * 100.]

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='traincnn')
	parser.add_argument('-w', '--window_radius', action="store",dest="window_radius", type=int,default=4)
	parser.add_argument('-fh', '--filter_height', action="store",dest="filter_height", type=int,default=1)
	parser.add_argument('-fw', '--filter_width', action="store",dest="filter_width", type=int,default=250)
	parser.add_argument('-ph', '--pool_height', action="store",dest="pool_height", type=int,default=1)
	parser.add_argument('-pw', '--pool_width', action="store",dest="pool_width", type=int,default=1)
	parser.add_argument('-pn', '--no_pool', action="store_true")
	parser.add_argument('-b', '--batch_size', action="store",dest="batch_size", type=int,default=1)
	parser.add_argument('-nk', '--nkerns', action="store",dest="nkerns", type=int,default=50)
	parser.add_argument('-n', '--n_epochs', action="store",dest="n_epochs", type=int,default=100)
	parser.add_argument('-ln', '--loginput_num', action="store",dest="loginput_num", type=int,default=2000)
	parser.add_argument('-l', '--learning_rate', action="store",dest="learning_rate", type=float,default=0.01)
	parser.add_argument('-v', '--vector_size', action="store", dest="vector_size",type=int,default=250)
	parser.add_argument('--dic')
	parser.add_argument('--train')
	parser.add_argument('--dev')
	os.system('mkdir -p model/cnn')
	args = parser.parse_args()
	window_radius = args.window_radius
	learning_rate = args.learning_rate
	nkerns = args.nkerns
	n_epochs = args.n_epochs
	batch_size = args.batch_size
	filter_height = args.filter_height
	filter_width = args.filter_width
	pool_width = args.pool_width
	pool_height = args.pool_height
	loginput_num = args.loginput_num
	vector_size = args.vector_size
	no_pool = args.no_pool
	acc = []
	validaverage = 0
	testaverage = 0
	dic = json.load(codecs.open(args.dic, 'r', 'utf-8'))
	dic = todict(dic)
	train = codecs.open(args.train, 'r', 'utf-8').read().split('\n')
	if train[-1] == '':
		del train[-1]
	trainwordlist = process_corpus(train)
	if args.dev:
		dev = codecs.open(args.dev, 'r', 'utf-8').read().split('\n')
		if dev[-1] == '':
			del dev[-1]
		devwordlist = process_corpus(dev)
	else:
		dev = None
		devwordlist = None
		
	for keyword in trainwordlist.keys():
		if os.path.exists('model//cnn//'+keyword):
			print 'model for', keyword, 'exists'
			continue
		if not dic.has_key(keyword):
			print 'no dictionary for', keyword 
			continue
		train_word = [train[i] for i in trainwordlist[keyword]]
		if dev and devwordlist.has_key(keyword):
			dev_word = [dev[i] for i in devwordlist[keyword]]
		else:
			dev_word = None
		trainword(dic, train_word, keyword, dev_word, window_radius, learning_rate, n_epochs, batch_size,nkerns,filter_height,filter_width,pool_height,pool_width,loginput_num, vector_size, 0)
	exit()

	content = str(acc)
	content += '\r\n'+ '==training parameters=='
	content += '\r\n'+ 'window_radius: '+str(window_radius)
	content += '\r\n'+ 'vector_size: '+str(vector_size)
	content += '\r\n'+ 'filter_height: '+str(filter_height)
	content += '\r\n'+ 'filter_width: '+str(filter_width)
	content += '\r\n'+ 'pool_height: '+str(pool_height)
	content += '\r\n'+ 'pool_width: '+str(pool_width)
	content += '\r\n'+ 'no_pool: '+str(args.no_pool)
	content += '\r\n'+ 'nkerns: '+str(nkerns)
	content += '\r\n'+ 'loginput_num: '+str(loginput_num)
	content += '\r\n'+ 'learning_rate: '+str(learning_rate)
	content += '\r\n'+ 'n_epochs: '+str(n_epochs)
	content += '\r\n'+ 'batch_size: '+str(batch_size)
	content += '\r\n'
	content += '\r\n valid: '+str(100-validaverage)
	content += '\r\n test: '+str(100-testaverage)
	output = open('result//trainmulti'+str(window_radius)+str(vector_size)+str(nkerns)+str(loginput_num)+str(learning_rate)+args.keyword.decode('utf-8')+'.txt', 'wb')
	output.write(content)
	output.close()

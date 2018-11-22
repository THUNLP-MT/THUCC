#-*- coding: utf-8 -*-
import numpy
import argparse
import theano
import codecs
import json
import theano.tensor as T
import gensim, logging

word2vec='word2vec/Word2Vec_250'

def sentence2vector(sentence, window_radius, vector_size, i):  
	word2vecmodel = gensim.models.Word2Vec.load(word2vec)			   
	dataarray = numpy.array([])
	for j in range(i-window_radius,i+window_radius+1):
		if j < 0 or j >= len(sentence):
			dataarray = numpy.hstack((dataarray,numpy.array([0]*vector_size)))
		else:
			dataarray = numpy.hstack((dataarray,word2vecmodel[sentence[j]]))
	#print dataarray
	return dataarray

def normalize(a):
	sqr = 0
	for i in range(0,len(a)):
		sqr += a[i]*a[i]
	result = [a[i]/sqr for i in range(0, len(a))]
	return result

def getsense(sentence, i):
	if len(sentence['senses'][i]) > 0 and sentence['senses'][i] != '[]':
		predictsense = ''
		tagnum = 0
		for sense in sentence['senses'][i]:
			if 'dict' in sense['tagger']:
				predictsense = sense['sense']
				tagnum = maxtagnum
			elif len(sense['tagger']) > tagnum:
				predictsense = sense['sense']
				tagnum = len(sense['tagger'])
		return predictsense
	else:
		return ''

def todict(dic):
	result = {}
	for i in dic:
		result[i['word']] = i['senses']
	return result

def process_corpus(sens):
	result = {}
	for i in range(len(sens)): 
		e = sens[i].split('\t')
		if len(e) < 3: 
			continue
		try:
			sen, pos, sense = e 
		except:
			print sens[i]
			exit()
		pos = int(pos)
		keyword = sen[pos]
		if result.has_key(keyword):
			result[keyword].append(i)
		else:
			result[keyword] = [i]
	return result

def prepare_data(dic, corpus, keyword, window_radius, vector_size, sequence = 0, nomralized = False, border = False, showsentence = False, outputtxt = False):
	#dic = json.loads(codecs.open(dic, 'r', 'utf-8').read())
	#dic = todict(dic)
	#corpus = codecs.open(corpus, 'r', 'utf-8').read().split('\n')
	model = gensim.models.Word2Vec.load(word2vec)

	senselist = dic[keyword]
	sensecount = [0] * len(senselist)

	outputcontent = ''
	data_x = []
	data_y = []
	data_sentence = []
	for snum in range(len(corpus)):
		if len(corpus[snum].split('\t')) != 3:
			break
		sentence, index, sense = corpus[snum].split('\t')
		sen = sentence
		index = int(index)
		i = index
		text = sentence
		word = text[index]
		wordsense = sense
		if wordsense in senselist:
			try:
				senseindex = senselist.index(wordsense)
				sensecount[senseindex] += 1
				dataarray = numpy.array([])
				for j in range(i-window_radius,i+window_radius+1):
					if j < 0 or j >= len(text):
						dataarray = numpy.hstack((dataarray,numpy.array([0]*vector_size)))
						sen = sen+' '
					else:
						sen = sen+text[j]
						if nomralized:
							dataarray = numpy.hstack((dataarray,normalize(model[text[j]])))
						else:
							dataarray = numpy.hstack((dataarray,model[text[j]]))
			except:
				continue
		else:
			continue
		data_y.append(senseindex)
		data_x.append(dataarray)
		data_sentence.append(sentence)

	print 'sensenum: '+str(len(senselist))
	print 'traindatanum: '+str(len(data_x))

	trainnumpydata_x = []
	trainnumpydata_y = []
	trainsentence = []
	
	'''
	testnumpydata_x = []
	testnumpydata_y = []
	testsentence = []	   

	validnumpydata_x = []
	validnumpydata_y = []
	validsentence = []
	'''

	for i in range(0, len(data_x)):
		'''
		if i % 5 == (3+sequence) % 5:
			testnumpydata_x.append(data_x[i])
			testnumpydata_y.append(numpy.int64(data_y[i]))
			testsentence.append(data_sentence[i])
		elif i % 5 == (4+sequence) % 5:
			validnumpydata_x.append(data_x[i])
			validnumpydata_y.append(numpy.int64(data_y[i]))
			validsentence.append(data_sentence[i])
		else:
		'''
		trainnumpydata_x.append(data_x[i])
		trainnumpydata_y.append(numpy.int64(data_y[i]))
		trainsentence.append(data_sentence[i])

	train_set =  (trainnumpydata_x,trainnumpydata_y)
	#test_set = (testnumpydata_x,testnumpydata_y)
	#valid_set = (validnumpydata_x,validnumpydata_y)

	def shared_dataset(data_xy, borrow=True):
		data_x, data_y = data_xy
		shared_x = theano.shared(numpy.asarray(data_x,
											   dtype=theano.config.floatX),
								 borrow=borrow)
		shared_y = theano.shared(numpy.asarray(data_y,
											   dtype=theano.config.floatX),
								 borrow=borrow)
		'''
		print shared_x, shared_y
		print len(data_x), len(data_y)
		print type(data_x[1]), type(data_y[1])
		print data_x[1].shape
		'''
		# When storing data on the GPU it has to be stored as floats
		# therefore we will store the labels as ``floatX`` as well
		# (``shared_y`` does exactly that). But during our computations
		# we need them as ints (we use labels as index, and if they are
		# floats it doesn't make sense) therefore instead of returning
		# ``shared_y`` we will have to cast it to int. This little hack
		# lets ous get around this issue
		return shared_x, T.cast(shared_y, 'int32')

	#test_set_x, test_set_y = shared_dataset(test_set)
	#valid_set_x, valid_set_y = shared_dataset(valid_set)
	train_set_x, train_set_y = shared_dataset(train_set)

	rval = (train_set_x, train_set_y, trainsentence)
	#rval = [(train_set_x, train_set_y, trainsentence), (valid_set_x, valid_set_y, validsentence),
	#		(test_set_x, test_set_y, testsentence)]
	return (rval, senselist)




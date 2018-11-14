import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy
import re
from matplotlib import font_manager

fontP = font_manager.FontProperties()
fontP.set_family('SimHei')
fontP.set_size(14)

def parse_numpy(string):
	string = string.replace('[', ' ').replace(']', ' ').replace(',', ' ')
	string = re.sub(' +',' ', string)
	result = numpy.fromstring(string, sep=' ')
	return result

def normalize(matrix):
	matrix = numpy.abs(matrix)
	total = numpy.sum(matrix, axis=1)
	return matrix/numpy.expand_dims(total, axis=1)

def visualize(src, trg, rlv, params):
	src_words = src.split(' ')
	src_words.append('<eos>')
	trg_words = trg.split(' ')
	trg_words.append('<eos>')

	len_t = len(trg_words)
	len_s = len(src_words)

        '''
	src_idx = re.findall('src_idx: (\[[\s\S]*?\])', result)
	if len(src_idx) > 0:
		src_idx = parse_numpy(src_idx[0])
	else:
		src_idx = numpy.zeros([len_s,])
	trg_idx = re.findall('trg_idx: (\[[\s\S]*?\])', result)
	if len(trg_idx) > 0:
		trg_idx = parse_numpy(trg_idx[0])
	else:
		trg_idx = numpy.zeros([len_t,])
        '''

	#rlv = numpy.reshape(rlv, [trg_idx.shape[0] ,src_idx.shape[0]])
	rlv = numpy.reshape(rlv, [len(trg_words) ,len(src_words)])
	rlv = rlv[:len(trg_words), :len(src_words)]
	rlv = normalize(rlv)

	plt.matshow(1-rlv, cmap='Greys')

	fontname = "Times"
	#plt.colorbar()
	plt.xticks(range(len(src_words)), src_words, fontsize=14, family=fontname,rotation='vertical')
	plt.yticks(range(len(trg_words)), trg_words, fontsize=14, family=fontname)

	matplotlib.rcParams['font.family'] = "Times"
	plt.savefig('rlv.pdf')
        print('save visualization to rlv.pdf')

	#plt.show()
	return 




'''
result = open(sys.argv[1], 'r').read()
src = re.findall('src: (.*?)\n', result)[0]
src = src.decode('utf-8')

trg = re.findall('trg: (.*?)\n', result)[0]

rlv = re.findall('result: ([\s\S]*)', result)[0]

rlv = parse_numpy(rlv)
'''






#print src_words, trg_words, rlv



#-*- coding: utf-8 -*-
from testcnn import testcnn_one
import os
import sys

import json
import numpy
import argparse
import codecs

from datafetch import prepare_data, todict, process_corpus

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='traincnn')
	parser.add_argument('--dic')
	parser.add_argument('--test')

	args = parser.parse_args()

	correct = 0 
	total = 0
	test = codecs.open(args.test, 'r', 'utf-8').read().split('\n')
	if test[-1] == '':
		del test[-1]
	
	dic = json.load(codecs.open(args.dic, 'r', 'utf-8'))
	dic = todict(dic)

	for line in test:
		sen, pos, ans = line.split('\t')
		pos = int(pos)
		result_cnn = testcnn_one(sen, pos)
		if len(result_cnn) > 0:
			print('succeed')
			result_idx = result_cnn[0]
		else:
			print('failed')
			result_idx = 0
		result = dic[sen[pos]][result_idx]
		if result == ans:
			correct += 1
		total += 1
	print 'correct:', correct, '/', total
	print 'acc:', correct*1.0/total

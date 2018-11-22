#-*- coding: utf-8 -*-
import codecs
import traceback
import json
import re
import numpy as np
import numpy
import os
import sys
import string
import argparse

TsinghuaAligner = 'TsinghuaAligner/TsinghuaAligner'
thulac = 'thulac/thulac'
align_model = 'align' # the path to the directory containing TsinghuaAligner models

parser = argparse.ArgumentParser(description='WSD with alignment')
parser.add_argument('-b', '--baihua')
parser.add_argument('-w', '--wenyan')
parser.add_argument('-i', '--index')

def parse_align(align):
	try:
		pos, p = align.split('/')
		pos_s, pos_t = pos.split('-')
		return [string.atoi(pos_s), string.atoi(pos_t), string.atof(p)]
	except:
		return [-1,-1,-1]

def getsense_align(baihua, wenyan, index):
	os.system('mkdir -p tmp')
	baihua = segment(baihua)
	wenyan = wenyan.replace('$','').replace('_','').strip()
	ini = align_model + '/TsinghuaAligner.ini'
	src = 'tmp/source_solve'
	trg = 'tmp/target_solve'
	align = 'tmp/alignment_solve'
	align_command = 'nohup ' + TsinghuaAligner + ' --ini-file %s --src-file %s --trg-file %s --agt-file %s --posterior 1' % (ini, src, trg, align)
	srcf = codecs.open(src,'w','utf-8')
	srcf.write(baihua)
	srcf.close()
	trgf = codecs.open(trg, 'w','utf-8')
	trgf.write(' '.join(wenyan))
	trgf.close()
	os.system(align_command)
	alignmentf = codecs.open(align, 'r','utf-8')
	alignment = alignmentf.read()
	alignmentf.close()
	aligns = alignment.split(' ')
	pos_align = -1
	maxp = 0.
	for align in aligns:
		align_info = parse_align(align)
		if align_info[1] == index and align_info[2] > maxp:
			pos_align = align_info[0]
			maxp = align_info[2]
	if pos_align == -1:
		return ''
	else:
		return baihua.split(' ')[pos_align]

def segment(text):
	os.system('mkdir -p tmp')
	segfile = codecs.open('tmp/toseg', 'w', 'utf-8')
	segfile.write(text)
	segfile.close()
	os.system(thulac+' -seg_only < tmp/toseg > tmp/segmented')
	result = codecs.open('tmp/segmented', 'r', 'utf-8').read()
	os.system('rm tmp/toseg')
	os.system('rm tmp/segmented')
	return result.replace('\n','')
			 

if __name__ == '__main__':
	args = parser.parse_args()
	baihua = args.baihua.decode('utf-8')
	wenyan = args.wenyan.decode('utf-8')
	index = int(args.index)
	sense = getsense_align(baihua, wenyan, index)
	print '释义：', sense
	os.system('rm -r tmp')
	os.system('rm nohup.out')



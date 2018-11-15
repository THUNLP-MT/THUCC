#!python3

import os
import re
import pickle
from collections import defaultdict
from util import *

def defaultdictFunc():
	return defaultdict(int)

processFindingRE = [
				'(了|出)?(?P<core>.*?)的?$',
				'(作者|诗人|词人|自己|他|她)?(?P<core>.*)',
				'的?(?P<core>.*)'
				]
processFindingPatterns = [re.compile(t) for t in processFindingRE]

def processFinding(string):
	for p in processFindingPatterns:
		m = p.match(string)
		string = m.group('core')
	return string

def train():
	patterns = [re.compile(t) for t in set(templates)]
	keywords = defaultdict(int)
	keywords_characters = defaultdict(defaultdictFunc)
	
	os.chdir(dataDir)
	filenames = os.listdir()
	for filename in filenames:
		m = re.match(r'yuanwen([0-9]+).txt', filename)
		if m:
			ID = m.group(1)
			shangxiFilename = 'shangxi'+ID+'.txt'
			with open(filename, 'r', encoding='utf-8') as fYuanwen, \
				open(shangxiFilename, 'r', encoding='utf-8') as fShangxi:
				yuanwenContent = fYuanwen.read()
				shangxiContent = fShangxi.read()
				shangxiContentClauses = re.split('\W+', shangxiContent)
				poemUnits = generatePoemUnits(yuanwenContent)
				for shangxiContentClause in shangxiContentClauses:
					for p in patterns:
						s = p.search(shangxiContentClause)
						if s:
							keyword = s.group(0)
							if len(keyword) > 6:
								keywords[keyword] += 1
								for c in poemUnits:
									keywords_characters[keyword][c] += 1
							else:
								print(keyword)
	return keywords, keywords_characters

if __name__ == '__main__':
	keywords, keywords_characters = train()
	with open(modelFile, 'wb') as modelFileObj:
		pickle.dump((keywords, keywords_characters), modelFileObj)
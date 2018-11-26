#!python3

import pickle
from train import defaultdictFunc	# for pickle
from util import modelFile
from decode import computeScore, generateAppreciation
import sys

class PoemAppreciationAPI:
	def __init__(self):
		with open(modelFile, 'rb') as modelFileObj:
			self.keywords, self.keywords_characters = pickle.load(modelFileObj)
	
	def appreciate(self, poem, numClauses=3, XMLin=False, XMLout=False):
		if XMLin:
			poem = poem.split(' |||| ')[0]
		scores = computeScore(poem, self.keywords, self.keywords_characters)
		#print(scores[:10])
		result = generateAppreciation(scores, numClauses)
		if XMLout:
			result = '<answer org="THU">'+result+'</answer>'
		return result

if __name__ == '__main__':
	p = PoemAppreciationAPI()
	print(sorted(p.keywords.items(), key=lambda x: x[1], reverse=True))
	print(sorted(p.keywords_characters['表现了凄凉哀怨的心境'].items(), key=lambda x: x[1], reverse=True))
	#print(sorted(p.keywords_characters['营造出一种清新轻松的情调氛围'].items(), key=lambda x: x[1], reverse=True))
	#print(sorted(p.keywords_characters['表达的人类的向美本能和情感'].items(), key=lambda x: x[1], reverse=True))
	#print(sorted(p.keywords_characters['抒发了自己的政治感慨'].items(), key=lambda x: x[1], reverse=True))
	#poem = '韦曲花无赖，家家恼煞人。绿樽须尽日，白发好禁春。石角钩衣破，藤梢刺眼新。何时占丛竹，头戴小乌巾。'
	#poem = '两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。' #'床前明月光，疑是地上霜。举头望明月，低头思故乡。'
	poem = sys.argv[1]
	result = p.appreciate(poem)
	print(result)

#!python3

import math
import re
from util import isPunctuation, generatePoemUnits

def computeScore(poem, keywords, keywords_characters):
	scores = []
	for keyword, keyword_count in keywords.items():
		keywords_characters_sum = sum(keywords_characters[keyword].values())
		score = sum((-1000 if keywords_characters[keyword][c] == 0 else math.log(keywords_characters[keyword][c])) 
					- math.log(keywords_characters_sum) 
				for c in generatePoemUnits(poem))
		score += math.log(keyword_count)
		scores.append((keyword, score))
	return sorted(scores, key=lambda x: x[1], reverse=True)

ProperNouns = set('刘兰芝 莺莺 碧桃 苏小小 嵇喜 裴迪 李白 长吉 永州 长安 三峡'.split())

SentimentLexicon = [
				'激愤 不平 悲愤 抑郁 幽怨 离别 沉痛 哀怨 忧愤 悲惨 悲凉 猜疑 孤独'.split(),
				'清新 轻松 美妙 愉悦 欣喜 悠闲 自得 完美 热爱 舒适 祥和 欢乐 快乐 美丽 向美'.split()
				]

def identifySentiment(s):
	result = -1
	for i in range(len(SentimentLexicon)):
		for sentiment_word in SentimentLexicon[i]:
			found = s.find(sentiment_word)
			if found != -1:
				result = i
				return result
	return result

def removeProperNouns(s):
	for p in ProperNouns:
		s = re.sub(p+'的?', '', s)
	return s

def generateAppreciation(scores, num):
	num = min(num, len(scores))
	verbs = set()
	global_sentiment = -1
	result = '本诗'
	for i in range(len(scores)):
		clause = scores[i][0] #output_templates[i % len(output_templates)].replace('(.*?)', scores[i][0])
		verb = clause[0:2]
		sentiment = identifySentiment(clause)
		if not verb in verbs and (global_sentiment == -1 or sentiment == -1 or global_sentiment == sentiment):
			verbs.add(verb)
			if global_sentiment == -1:
				global_sentiment = sentiment
			result += clause
			if len(verbs) < num:
				result += '，'
			else:
				result += '。'
				break
	#postprocess
	result = removeProperNouns(result)
	return result


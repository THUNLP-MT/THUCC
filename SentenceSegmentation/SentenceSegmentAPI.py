#!python3
from callCRF import callCRF
from retrieve import retrieve
import re
import sys

class SentenceSegmentAPI:
	def __init__(self):
		with open('punctuationRemoved.all', 'r', encoding='UTF-8') as databaseF:
			self.database = [line.strip() for line in databaseF]
		with open('preprocessed.all', 'r', encoding='UTF-8') as answerF:
			self.answer = [line.strip() for line in answerF]
	
	def segment(self, data, XMLin=False, XMLout=False):
		if XMLin:
			match = re.search('<term>(.*?)</term>', data)
			if match:
				data = match.group(1)
				data = re.sub('（.*?）', '', data)
				data = data.strip()
			else:
				return 'Error format.'
		
		#print(data)
		result = retrieve(data, self.database, self.answer)
		if result:
			#print('Matched in database.')
			pass
		else:
			result = callCRF(data)
		
		if XMLout:
			result = '<answer org="THU">'+result+'</answer>'
		return result

if __name__ == '__main__':
	s = SentenceSegmentAPI()
	#data = '忽逢桃花林夹岸数百步中无杂树芳草鲜美落英缤纷渔人甚异之'
	#print(s.segment(data))
	#data = '曰士之仕也犹农夫之耕也农夫岂为出疆舍其耒耜哉'i
	data = sys.argv[1]
	print(s.segment(data))

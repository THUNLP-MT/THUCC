from SentenceSeg import isPunctuation
import sys

def retrieve(toSeg, database, answer):
	n = len(database)
	index = -1
	databaseStart = -1
	for i in range(n):
		databaseStart = database[i].find(toSeg)
		if databaseStart != -1:
			index = i
			break
	if index == -1:
		return None
	answerLine = answer[index]
	databaseLine = database[index]
	
	trans = []
	j = 0
# 	print(len(toSeg), index, len(databaseLine), len(answerLine))
	for i in range(len(databaseLine)):
		while isPunctuation(answerLine[j]) or answerLine[j] == '“' or answerLine[j] == '”':
			j = j+1
# 			print(i, j)
		trans.append(j)
# 		print(i, databaseLine[i], j, answerLine[j])
		j = j+1
	trans.append(j)
	
	answerStart = trans[databaseStart]
	answerEnd = trans[databaseStart + len(toSeg)]
	result = []
	for c in answerLine[answerStart:answerEnd]:
		if isPunctuation(c):
			result.append('/')
		elif c == '“' or c == '”':
			pass
		else:
			result.append(c)
	return ''.join(result)

if __name__ == '__main__':
	toSeg = sys.argv[1]
	with open('punctuationRemoved.all', 'r', encoding='UTF-8') as databaseF:
		database = [line.strip() for line in databaseF]
	with open('preprocessed.all', 'r', encoding='UTF-8') as answerF:
		answer = [line.strip() for line in answerF]
	result = retrieve(toSeg, database, answer)
	print(result)
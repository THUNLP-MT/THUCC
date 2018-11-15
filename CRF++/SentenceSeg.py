import re, os

def preprocess(topDir, outputFilename):
	outputBuffer = []
	for filename in os.listdir(topDir):
		with open(topDir+os.sep+filename, 'r', encoding='UTF-8') as fin:
			for line in fin:
				line = line.strip().replace(' ', '')
				line = line.replace('（', '')
				line = line.replace('）', '')
				line = line.replace('“', '')
				line = line.replace('”', '')
				line = line.replace('‘', '')
				line = line.replace('’', '')
				line = line.replace('!', '！')
				line = line.replace('?', '？')
				line = line.replace('[', '【')
				line = line.replace(']', '】')
				line = re.sub(r'\(.*\)', '', line)
				lines = line.split('　　')
				for candidate in lines:
					if candidate != '':
						outputBuffer.append(candidate.replace('　', ''))
	with open(outputFilename, 'w', encoding='UTF-8') as fout:
		fout.write('\n'.join(outputBuffer) + '\n')

def isPunctuation(c):
	return c in {'，', '。', '！', '？', '：', '；', '、', '/'}

def removePunctuation(inputFilename, outputFilename):
	table = str.maketrans('', '', '，。！？：；、')
	with open(inputFilename, 'r', encoding='UTF-8') as fin, open(outputFilename, 'w', encoding='UTF-8', newline='') as fout:
		for line in fin:
			line = line.translate(table)
			fout.write(line)
			
def convertToCrfFormat(inputFilename, outputFilename):
	totalTokenCount = 0
	with open(inputFilename, 'r', encoding='UTF-8') as fin, open(outputFilename, 'w', encoding='UTF-8', newline='') as fout:
		for line in fin:
			line = line.strip()
			buf = []
			for c in line:
				if isPunctuation(c):
					#write out buf
					#first recognize rare words in 【】
					combinedBuf = []
					rareWord = ''
					inRareWord = False
					for cc in buf:
						if cc == '【':
							inRareWord = True
							rareWord += cc
						elif cc == '】':
							inRareWord = False
							rareWord += cc
							combinedBuf.append(rareWord)
							rareWord = ''
						else:
							if inRareWord:
								rareWord += cc
							else:
								combinedBuf.append(cc)
					#write out according to its length
					count = len(combinedBuf)
					totalTokenCount += count
					if count == 1:
						fout.write(combinedBuf[0] + '\tS\n')
					else:
						for i in range(count):
							if i == 0:
								fout.write(combinedBuf[i] + '\tB\n')
							elif i == count-1:
								fout.write(combinedBuf[i] + '\tE\n')
							else:
								fout.write(combinedBuf[i] + '\tM\n')
					#clear the buf
					buf.clear()
				else:
					buf.append(c)
			if len(buf) > 0:
				print('Warning: the last character is not punctuation?', line)
			fout.write('\n')
	print('Total token count =', totalTokenCount)
	
if __name__ == '__main__':
# 	preprocess('data', 'preprocessed.all')
	removePunctuation('preprocessed.all', 'punctuationRemoved.all')
# 	preprocess('corrupted_data', 'preprocessed.1')
# 	convertToCrfFormat('preprocessed.1', 'data.1')
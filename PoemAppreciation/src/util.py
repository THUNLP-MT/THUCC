#!python3

dataDir = '../data'
modelFile = '/data/disk1/share/wangshuo/PoemAppreciation/model/model.pickle'
unitOption = 1

# template_verbs = {
# 				'表现了',
# 				'抒发了',
# 				'传达出',
# 				'渲染了',
# 				'表达了',
# 				'传递了',
# 				'刻画了',
# 				'隐喻了',
# 				'传达了'
# 				}
# 
# template_nouns = {
# 				'的心境',
# 				'',
# 				'',
# 				'',
# 				'',
# 				'',
# 				'',
# 				'',
# 				'',
# 				''
# 				}

templates = [
			'表现(.*?)心境',
			'抒发(.*?)感慨',
			'传达(.*?)意境',
			'表现(.*?)意境',
			'渲染(.*?)气氛',
			'传达(.*?)心境',
			'表达(.*?)情感',
			'传递(.*?)之情',
			'刻画(.*?)形象',
			'隐喻(.*?)之情',
			'表达(.*?)心情',
			'表现(.*?)之情',
			'表达(.*?)情感',
			'寄托(.*?)情感',
			'营造(.*?)意境',
			'抒发(.*?)情感',
			'揭露(.*?)事实',
			'寄寓(.*?)情感',
			'表达(.*?)思想',
			'寄托(.*?)情怀',
			'营造(.*?)氛围',
			'抒发(.*?)情怀',
			'创造(.*?)境界'
			]

output_templates = [
			'表现了(.*?)的心境',
			'抒发了(.*?)的感慨',
			'传达出(.*?)的意境',
			'表现了(.*?)的意境',
			'渲染了(.*?)的气氛',
			'传达了(.*?)的心境',
			'表达了(.*?)的情感',
			'传递了(.*?)之情',
			'刻画了(.*?)的形象',
			'隐喻了(.*?)之情',
			'表达了(.*?)的心情',
			'表现了(.*?)之情',
			'表达了(.*?)的情感',
			'寄托了(.*?)的情感',
			'营造了(.*?)的意境',
			'抒发了(.*?)的情感',
			'揭露了(.*?)的事实',
			'寄寓了(.*?)的情感',
			'表达了(.*?)的思想',
			'寄托了(.*?)的情怀',
			'营造了(.*?)的氛围',
			'抒发了(.*?)的情怀',
			'创造了(.*?)的境界'
			]

def isPunctuation(c):
	return c in {'，', '。', '！', '？', '：', '；', '、', '/'}

def generatePoemUnits(poem):
	'''unitOption - 1: character. 2: bigram.'''
	if unitOption == 1:
		return [c for c in poem if not c.isspace() and not isPunctuation(c)]
	else:
		units = []
		for i in range(len(poem) - 1):
			c1 = poem[i]
			c2 = poem[i+1]
			if not c1.isspace() and not isPunctuation(c1) and not c2.isspace() and not isPunctuation(c2):
				units.append(c1 + c2)
		return units

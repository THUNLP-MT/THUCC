#  coding=utf8
import re
import os
import sys
import argparse
def is_prompt(s):
	s = s.decode('utf8')
	punc_list = [u"。",   u"，",   u"？",  u"！",  u"；",  u"；",  u"：",  u"_", u',', u'.',u';',u':']
	for char in s:
		if char not in punc_list:
			#print char
			return 1
	return 0

def search_poem(name,filename):
	f = open(filename)
	res = ''
	for line in f:
		if line.strip().split('\t') == name:
			return line.strip().split('\t')[0]
		if line.strip().split('\t')[1].find(name) != -1:
			#print "search_poem",line.strip().split('\t')[1],line.strip().split('\t')[0]
			res =  line.strip().split('\t')[0]
			
	#print res
	return res



def poem_retrieval(s):
	

	cmd_text = "java -jar PoemRetrieval.jar ./PoemRetrieval/data/index '" + s + "'"
	r = os.popen(cmd_text)
	text = r.read()
	r.close()
	index = text.find("<answer")
	index_end = text.find('</answer>')
	if index != -1:
		text = text[index:index_end + 9].replace('\r\n','').replace('\n\r','').replace('\n','').replace('\t','')
		reg = r"<answer .*>(.+?)</answer>"
		text = re.findall(reg,text)[0]
	else:
		reg = r'<blank num.*>(.+?)</blank>'
		s_list = re.findall(reg,s)
		if len(s_list) == 0:
			pass
		else:
			s = s_list[0]
		reg = r"“(.+?)”"
		index = re.findall(reg,s)
		ques_list = []
		for i in index:
			if i.find('_') != -1:
				ques_list.append(i)
		reg = r'"(.+?)"'
		index = re.findall(reg,s)
		for i in index:
			if i.find('_') != -1:
				ques_list.append(i)
		#print len(ques_list)
		text = ''
		if len(ques_list) != 0:
			for ques in ques_list:
				#print ques
				has_prompt = is_prompt(ques)
				#print 'has_prompt', has_prompt
				if has_prompt == 1:
					cmd_text = "java -jar PoemRetrieval.jar ./PoemRetrieval/data/index '<question><blank> " + ques + " </blank></question>'"
					r = os.popen(cmd_text)
					text_tmp = r.read()
					r.close()
					
					index = text_tmp.find("<answer")
					index_end = text_tmp.find("</answer>")
					if index != -1:
						text_tmp = text_tmp[index:index_end + 9].replace('\r\n','').replace('\n\r','').replace('\n','').replace('\t','')
						reg = r"<answer .*>(.+?)</answer>"
						text_tmp = re.findall(reg,text_tmp)[0]
						text = text + text_tmp + ' '	
				else:
					
					reg = r"《(.+?)》"
					index = s.find(ques)
					name_list = re.findall(reg, s[:index])
					if len(name_list) != 0:
						name = name_list[-1]
					else:
						name_list = re.findall(reg,s)
						if len(name_list) != 0:
							name = name_list[0]
						else:
							name = ''
					
					res = search_poem(name, "poem.txt")
					text = text + res + ' '
		else:

			n = len(s)
			index = s.find('_')
			index_end = s[::-1].find('_')
			if index != -1 and index_end != -1:
				index_end = n - index_end -1
			ques = s[index:index_end]
			has_prompt = is_prompt(ques)
			if has_prompt == 1:
				cmd_text = "java -jar PoemRetrieval.jar ./PoemRetrieval/data/index '<question><blank> " + ques + " </blank></question>'"
				r = os.popen(cmd_text)
				text_tmp = r.read()
				r.close()
				index = text_tmp.find("<answer")
				index_end = text_tmp.find("</answer>")
				if index != -1:
					text_tmp = text_tmp[index:index_end + 9].replace('\r\n','').replace('\n\r','').replace('\n','').replace('\t','')
					reg = r"<answer .*>(.+?)</answer>"
					text_tmp = re.findall(reg,text_tmp)[0]
					text = text_tmp + ' '
			else:
				reg = r"《(.+?)》"
				index = s.find(ques)
				name_list = re.findall(reg, s[:index])
				if len(name_list) != 0:
					name = name_list[-1]
				else:
					name_list = re.findall(reg,s)
					if len(name_list) != 0:
						name = name_list[0]
					else:
						name = ''		
				res = search_poem(name, "poem.txt")
				text = res
	return '<answer org="THU">\n\t' + text + '\n</answer>'
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i","--input-file",help = "input file")
	parser.add_argument("-o","--output-file",help="output file")
	args = parser.parse_args()
	if args.input_file:
		filename = args.input_file
	if args.output_file:
		filename_2 = args.output_file
	s = open(filename).read().strip().replace('\t','').replace('\n','').replace('\n\r','').replace('\r\n','')
	print s
	text = poem_retrieval(s)
	f = open(filename_2,'w')
	f.write(text)




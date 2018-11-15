from bottle import route, run, static_file, post, request
from PoemAppreciationAPI import PoemAppreciationAPI
from train import defaultdictFunc	# for pickle


@route('/appreciate/<filename>')
def server_static(filename):
	return static_file(filename, root='./WebRoot')

@post('/performPoemAppreciation')
def performPoemAppreciation():
	data = request.forms.getunicode('input')
	print(data)
	return s.appreciate(data)

s = PoemAppreciationAPI()
run(host='0.0.0.0', port=5490, debug=False)

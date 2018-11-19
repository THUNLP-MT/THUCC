from bottle import route, run, static_file, post, request
from SentenceSegmentAPI import SentenceSegmentAPI

@route('/segment/<filename>')
def server_static(filename):
	return static_file(filename, root='./WebRoot')

@post('/performSentenceSegment')
def performSentenceSegment():
	data = request.forms.getunicode('input')
	print(data)
	return s.segment(data, True, False)

s = SentenceSegmentAPI()
run(host='0.0.0.0', port=5489, debug=False)

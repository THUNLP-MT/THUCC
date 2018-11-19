import subprocess, io

def callCRF(data):
	with open('tempInput.txt', 'w', encoding='UTF-8') as f:
		for c in data:
			f.write(c + '\n')
	output = subprocess.check_output('install/bin/crf_test -m model tempInput.txt', shell=True)
	tokens = output.decode().split('\n')
	result = io.StringIO()
	for token in tokens:
		parts = token.split()
		ncols = len(parts)
		if ncols == 0:
			result.write('\n')
		else:
			result.write(parts[0])
			if parts[ncols-1] == 'E' or parts[ncols-1] == 'S':
				result.write('/')
	return result.getvalue()

if __name__ == '__main__':
	print(callCRF('又南三百里曰藟山其上有玉'))
	

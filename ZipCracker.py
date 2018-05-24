from zipfile import ZipFile

file_name = "CODE-BOOK.zip"

def numtochar(n):
	if n < 26:
		n += 65
	elif n < 52: 
		n += 71
	else:
		n -= 4
	return chr(n)

def numtostring(n):
	s = ''
	while n != 0:
		m = n%62
		n /= 62
		s += numtochar(m)
	return s[::-1]

print('Extracting all the files...')

temp = 0
while temp != -1:
	print('PASSWORD TRYING: '+numtostring(temp))
	with ZipFile(file_name, 'r') as zip:
		try:
			zip.extractall(pwd=numtostring(temp))
			zip.printdir()
			print('PASSWORD MATCHED: '+numtostring(temp)+'\n')
		except:
			temp = temp+1
		else:
			temp =- 1

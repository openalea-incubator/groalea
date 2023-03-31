import sys
import (
	urllib.request, 
	urllib.error, 
	urllib.parse
)
from urllib.error import URLError, HTTPError

#read graph
f = file("beech.xeg")
graph = f.read()
f.close()

#read xl
f = file("beech.xl")
xl = f.read()
f.close()

#prepare data
url = 'http://localhost:58070'
data = urllib.parse.urlencode({'graph': graph, 'xlcode': xl, 'command': 'init'})

#send
req = urllib.request.Request(url)
try:
	fd = urllib.request.urlopen(req, data)
	#answer
	while 1:
		data = fd.read()
		if not len(data):
			break
	sys.stdout.write(data)
except HTTPError as e:
	print("Error code: "+str(e.code))
	print(e.read())
except URLError as e:
	print("Error: "+str(e.reason))

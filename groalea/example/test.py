import sys, urllib2, urllib
from urllib2 import URLError, HTTPError

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
data = urllib.urlencode({'graph': graph, 'xlcode': xl, 'command': 'init'})

#send
req = urllib2.Request(url)
try:
	fd = urllib2.urlopen(req, data)
	#answer
	while 1:
		data = fd.read()
		if not len(data):
			break
	sys.stdout.write(data)
except HTTPError, e:
	print "Error code: "+str(e.code)
	print e.read()
except URLError, e:
	print "Error: "+str(e.reason)



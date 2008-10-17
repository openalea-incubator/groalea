# Author: C. Pradal
# Licence : GPL
from openalea.groimp.client import *
import openalea.groimp.graphio as io
from openalea.plantgl.all import Viewer

fn_graph = 'sample.xeg'
fn_code = 'sample.xl'

fn_graph = 'graph1.xml'
fn_code = 'code1.xl'

f=open(fn_graph)
xml_graph = f.read()
f.close()

f=open(fn_code)
xlcode = f.read()
f.close()

#g, s = io.xml2graph(xml_graph)
#Viewer.display(s)

conn = connexion('localhost', '58070')
out_graph = simulation(xlcode, xml_graph, 'run', conn)

f = open('test_color.xeg','w')
f.write(out_graph)
f.close()

print out_graph

#g, s = io.xml2graph(out_graph)
#Viewer.display(s)

#gxml = io.graph2xml(g)

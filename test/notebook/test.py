from openalea.plantgl.all import *
from openalea.mtg.aml import *

from openalea import groalea
from openalea.groalea.mappletConverter import convert
from openalea.groalea.graphio import graph2xml, xml2graph, xmlFile2graph

from openalea.deploy import shared_data
from path import Path

data_dir = Path('../../share/interface-example/')
sti_fn, sti_fs_fn, st_fn = data_dir.glob('*.xeg')

from openalea.groalea.topology import spanning_mtg

for filename in (sti_fn, sti_fs_fn, st_fn):
    graph, scene = xmlFile2graph(filename)
    g = spanning_mtg(graph)
    #g.display()

#Viewer.display(scene)
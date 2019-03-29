from openalea.plantgl.all import *
from openalea.mtg.aml import *

from openalea import groalea
from openalea.groalea.mappletConverter import convert
from openalea.groalea.graphio import (graph2xml, xml2graph,
                                      xmlFile2graph, getSceneXEG,
                                      getMTGRootedGraph)

from openalea.deploy import shared_data
from path import Path
import time

data_dir = Path('../../share/interface-example/')
sti_fn, sti_fs_fn, st_fn, _, _ = data_dir.glob('*.xeg')

from openalea.groalea.topology import spanning_mtg

scenes = []
for filename in (sti_fn, st_fn):
    graph, scene = xmlFile2graph(filename, onlyTopology=False)
    xeg_graph = getSceneXEG(graph)
    _g, scene = xml2graph(xeg_graph)
    scenes.append(scene)

    graph, scene = xmlFile2graph(filename, True)
    graph = getMTGRootedGraph(graph)
    g = spanning_mtg(graph)

    print g.display()
    #g.display()


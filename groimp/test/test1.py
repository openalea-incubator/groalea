# Author: C. Pradal
# Licence : GPL
from openalea.groimp.graphio import *
from openalea.plantgl.all import *

def test0():
    fn = "SampleFile.xml"
    parser = Parser()
    g, scene = parser.parse(fn)
    assert len(g) == 9
    assert g.nb_edges() == 8
    edge_type = g.edge_property("edge_type")
    assert len(edge_type) == 8
    
    return g
    
def test1():
    fn = "SampleFile.xml"
    parser = Parser()
    g, scene = parser.parse(fn)

    assert len(scene) == 6
    assert 7.72 < surface(scene) <7.73, surface(scene)
    assert 0.661 < volume(scene) < 0.662, volume(scene)

def test2():
    fn = "SampleFile.xml"
    parser = Parser()
    g, scene = parser.parse(fn)

    dump = Dumper()
    f = open('tmp.xml', 'w')
    txt = dump.dump(g)
    f.write(txt)
    f.close()
    
    p2 = Parser()
    g1, scene2 = parser.parse('tmp.xml')

    assert len(g) == len(g1)
    assert len(scene) == len(scene2)

def test3():
    fn = 'gi_graph.xeg'
    parser = Parser()
    g, scene = parser.parse(fn)

    assert len(g) == 3
    assert g.nb_edges() == 2

def __test4():
    fn = 'sample.xeg'
    parser = Parser()
    g, scene = parser.parse(fn)

def test5():
    # Test read colors
    fn = 'test_color.xeg'
    parser = Parser()
    g, scene = parser.parse(fn)


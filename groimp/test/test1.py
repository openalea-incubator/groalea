# Author: C. Pradal
# Licence : GPL
from openalea.groimp.graphio import *

def test0():
    fn = "SampleFile.xml"
    parser = Parser()
    g = parser.parse(fn)
    assert len(g) == 9
    assert g.nb_edges() == 8
    edge_type = g.edge_property("edge_type")
    assert len(edge_type) == 8
    
    return g
    

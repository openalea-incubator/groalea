
import numpy as np
import re

import xml.etree.ElementTree as xml
from collections import defaultdict

from openalea.plantgl.all import *
from openalea.mtg.aml import *

from xegadj import mtgvidstore



###################################################################################################

class Dumper(object):

    def dump(self, graph):
        self._graph = graph
        self.graph()
        adj_doc = mtgvidstore(self.doc)
        return xml.tostring(adj_doc)

    def SubElement(self, *args, **kwds):
        elt = xml.SubElement(*args, **kwds)
        if not elt.text:
            elt.text = '\n\t'
        elt.tail = '\n\t'
        return elt

    def graph(self):
        self.doc = xml.Element('graph')
        self.doc.tail = '\n'
        self.doc.text = '\n\t'
        # add root
        root = self._graph.root
        self.SubElement(self.doc, 'root', dict(root_id=str(root)))
        # universal types
        # Define the specific types in xeg
        # <type name='toto'>

        self.universal_node()

        for vid in self._graph.vertices():
            self.node(vid)

        for eid in self._graph.edges():
            self.edge(eid)

    def universal_node(self):
        # test
        _types = self._graph._types
        #_types['Boid']=['sphere']
        attrib = {}
        if _types:
            for t, extends in _types.iteritems():
                attrib['name'] = t
                user_type = self.SubElement(self.doc, 'type', attrib)
                for t in extends:
                    attrib['name'] = t
                    self.SubElement(user_type, 'extends', attrib)

    def node(self, vid):
        g = self._graph

        pname = g.vertex_property('name')
        ptype = g.vertex_property('type')
        properties = g.vertex_property('parameters')

        if vid == g.root and vid not in pname:
            # The root node has been only declared
            # by <root root_id="1"/>
            return

        attrib = {}
        attrib['id'] = str(vid)
        attrib['name'] = pname[vid]
        attrib['type'] = ptype[vid]
        node = self.SubElement(self.doc, 'node', attrib)

        t=None
        try:
            t = properties.get(vid)['transform']
        except KeyError:
            pass
        if t:
            transfo = self.SubElement(node,
                                      'property',
                                      {'name': 'transform'})
            matrix = self.SubElement(transfo, 'matrix')
            s = '\n'
            for i in range(4):
                c = tuple(t.getRow(i))
                s += '\t\t\t%.5f %.5f %.5f %.5f\n' % c
            matrix.text = s + '\n' +'\t'
                
        c3 = g.vertex_property('color').get(vid)
        if c3:
            ctu = (c3.clampedRed(), c3.clampedGreen(), c3.clampedBlue())
            color = self.SubElement(node,'property', {'name': 'color'})
            rgb = self.SubElement(color, 'rgb')
            h = '\n'
            s = '\t\t\t%.5f %.5f %.5f\n'%ctu
            rgb.text = h + s +'\t'
            

        pdicts = properties.get(vid, [])
        if type(pdicts) is list:
            for pdict in pdicts:
                for (name, value) in pdicts.iteritems():
                    if name != 'transform':
                        attrib = {'name': name, 'value': str(value)}
                        self.SubElement(node, 'property', attrib)
        else:
            for (name, value) in properties.get(vid, []).iteritems():
                if name != 'transform':
                    attrib = {'name': name, 'value': str(value)}
                    self.SubElement(node, 'property', attrib)

    def edge(self, eid):
        edge_type_conv = {}
        edge_type_conv['<'] = 'successor'
        edge_type_conv['+'] = 'branch'
        edge_type_conv['/'] = 'decomposition'
        g = self._graph
        edge_type = g.edge_property('edge_type').get(eid)
        attrib = {}
        attrib['id'] = str(eid)
        attrib['src_id'] = str(g.source(eid))
        attrib['dest_id'] = str(g.target(eid))
        if edge_type:
            attrib['type'] = edge_type_conv[edge_type]

        self.SubElement(self.doc, 'edge', attrib)


##########################################################################

# Wrapper functions for OpenAlea usage.

def graph2xml(graph):
    dump = Dumper()
    return dump.dump(graph)

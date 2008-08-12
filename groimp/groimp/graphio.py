import xml.etree.ElementTree as xml
from openalea.core.graph.property_graph import PropertyGraph
import openalea.plantgl.all as pgl

class Parser(object):
    def __init__(self, fn):
        self.fn = fn

    def parse(self, fn):
        doc = xml.parse(fn)
        root = doc.getroot()
        self.dispatch(root)
        return self._graph

    def dispatch(self, elt):
        return self.__getattribute__(elt.tag)(elt.getchildren(), **elt.attrib)

    def graph(self, elements):
        """
        A graph is a set of nades and edges.
        """ 
        graph = self._graph = PropertyGraph()

        graph.add_vertex_property("name")
        graph.add_vertex_property("geometry")
        graph.add_edge_property("edge_type")
        
        for elt in elements:
            self.dispatch(elt)

    def node(self, geometries, id, name=None):
        id = int(id)
        if not name:
            name = str(id)

        graph = self._graph

        graph.add_vertex(id)
        graph.vertex_property('name')[id] = name

        assert len(geometries) == 1, "A node as at least one shape."

        shape = self.dispatch(geometries[0]) 
        
        if shape:
            graph.vertex_property('geometry')[id] = shape

    def sphere(self, elements, radius=0):
        return pgl.Sphere(radius=float(radius))

    def box(self, elements, length, width, height):
        length, width, height= float(length), float(width), float(height)
        size = pgl.Vector3(length/2, width/2, height/2)
        return pgl.Translated((0,0,height), pgl.Box(size))

    def cone(self, elements, radius, height):
        radius, height = float(radius), float(height)
        return pgl.Cone(radius=radius, height=height)

    def cylinder(self, elements, radius, height):
        radius, height = float(radius), float(height)
        return pgl.Cylinder(radius=radius, height=height)

    def transform(self, elements, **kwds):
        matrix = elements[0]
        assert matrix.tag == 'matrix'
        m4 = map(float, matrix.text.strip().split())
        m4 = pgl.Matrix4(m4[:4], m4[4:8], m4[8:12], m4[12:])
        return m4

    def edge(self, elements, src_id, dest_id, type, id=None):
        graph = self._graph
        if id: id = int(id)
        graph.add_edge(edge=(int(src_id),int(dest_id)), eid=id)

    def scenegraph(self, root):
        # traverse the graph
        scene = pgl.Scene()
        geometry = self._graph.vertex_property('geometry')
        
        transfo = []
        shape = geometry.get(root)
        if is_matrix(shape):
            transfo.append(shape)

def is_matrix(shape):
    return type(shape) == pgl.Matrix4

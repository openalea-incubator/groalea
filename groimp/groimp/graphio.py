# TODO
# 1. add a header
# 2. separate graphe parsing and scenegraph generation
# 3. Error management
# 4. Documentation
# 5. Compute properties when it is possible (sphere, ...)
# 6. 2D draw of the graph

import xml.etree.ElementTree as xml
from openalea.core.graph.property_graph import PropertyGraph
import openalea.plantgl.all as pgl
from StringIO import StringIO

class RootedGraph(PropertyGraph):
    def _set_root(self, root):
        self._root = root

    def _get_root(self):
        return self._root

    root = property(_get_root, _set_root)

class Parser(object):
    edge_type_name = {'successor':'<', 'branch':'+'}
    geometries = ['sphere', 'box', 'cone', 'cylinder']

    def parse(self, fn):
        self._graph = None
        self._scene = None

        doc = xml.parse(fn)
        root = doc.getroot()
        self.dispatch(root)
        self.scenegraph()
        return self._graph, self._scene

    def dispatch(self, elt):
        return self.__getattribute__(elt.tag)(elt.getchildren(), **elt.attrib)

    def dispatch2(self, method_name, args):
        try:
            return self.__getattribute__(method_name)(**args)
        except:
            # method name has to be defined as a type 
            return self.universal_node(method_name, **args)

    def graph(self, elements):
        """
        A graph is a set of nades and edges.
        """ 
        graph = self._graph = RootedGraph()
        self._edges = {}
        graph._types = {}

        graph.add_vertex_property("name")
        graph.add_vertex_property("type")
        graph.add_vertex_property("parameters")
        graph.add_vertex_property("color")
        graph.add_vertex_property("geometry")
        graph.add_vertex_property("transform")
        graph.add_edge_property("edge_type")
        
        for elt in elements:
            self.dispatch(elt)

        # add the edges to the graph, when all the nodes have been added.
        if graph.root not in graph:
            graph.add_vertex(graph.root)

        self._add_edges()

    def type(self, elts, name):
        # Add this to the graph...
        self._graph._types[name] = []
        for elt in elts:
            print elt.tag
            if elt.tag == 'extends':
                elt.attrib['type_name'] = name
                self.dispatch(elt)

    def extends(self, elts, name, type_name):
        self._graph._types[type_name].append(name)

    implements = extends

    def root(self, elts, root_id):
        self._graph.root = int(root_id)

    def node(self, properties, id, type, name=None):
        id = int(id)
        if not name:
            name = str(id)

        graph = self._graph

        graph.add_vertex(id)
        if name:
            graph.vertex_property('name')[id] = name
        graph.vertex_property('type')[id] = type

        # Hack to separate transformation (without value) 
        # from other properties (with value)
        transfos = [p for p in properties \
                      if p.attrib['name'] == 'transform']
        colors = [p for p in properties \
                      if p.attrib['name'] == 'color']
        properties = [p for p in properties \
                      if p.attrib['name'] not in ('transform', 'color')]

        args = self._get_args(properties)
        graph.vertex_property('parameters')[id] = args
        
        if type == 'node':
            # special case.
            shape, transfo = None, None
        else:
            print 
            shape, transfo = self.dispatch2(type, args) 
        
        assert len(transfos) <= 1
        if transfos:
            assert transfo is None
            transfo = self.transform(transfos[0].getchildren())

        color = None
        if colors:
            color = self.color(colors[0].getchildren())
            
        if shape:
            graph.vertex_property('geometry')[id] = shape
        if transfo:
            graph.vertex_property('transform')[id] = transfo 
        if color:
            graph.vertex_property('color')[id] = color

    def sphere(self, radius=0, **kwds):
        return pgl.Sphere(radius=float(radius)), None

    def box(self, length, width, height, **kwds):
        length, width, height= float(length), float(width), float(height)
        size = pgl.Vector3(length/2, width/2, height/2)
        return (pgl.Translated((0,0,height/2), pgl.Box(size)), 
               pgl.Matrix4.translation(pgl.Vector3(0,0,height)))

    def cone(self, radius, height, **kwds):
        radius, height = float(radius), float(height)
        return (pgl.Cone(radius=radius, height=height),
                pgl.Matrix4.translation(pgl.Vector3(0,0,height)))

    def cylinder(self, radius, height, **kwds):
        radius, height = float(radius), float(height)
        return (pgl.Cylinder(radius=radius, height=height),
               pgl.Matrix4.translation(pgl.Vector3(0,0,height)))

    def transform(self, elements, **kwds):
        matrix = elements[0]
        assert matrix.tag == 'matrix'
        m4 = map(float, matrix.text.strip().split())
        m4 = pgl.Matrix4(m4[:4], m4[4:8], m4[8:12], m4[12:])
        m4 = m4.transpose()
        return m4

    def color(self, elements, **kwds):
        rgb = elements[0]
        assert rgb.tag == 'rgb'
        color = pgl.Color3(*( int(float(x)*255) for x in rgb.text.strip().split()) )
        return color

    def edge(self, elements, src_id, dest_id, type, id=None):
        # we add the edges at the end of the process
        edges = self._edges

        if id: id = int(id)
        edge_type = self.edge_type_name.get(type,type)

        edges[id] = (int(src_id),int(dest_id))
        self._graph.edge_property("edge_type")[id] = edge_type
        #graph.add_edge(edge=(int(src_id),int(dest_id)), eid=id)

    def _add_edges(self):
        edges = self._edges
        graph = self._graph
        
        for eid, edge in edges.iteritems():
            graph.add_edge(edge=edge, eid=eid)

    def universal_node(self, type_name, **kwds):
        _types = self._graph._types
        assert type_name in _types, (type_name, _types)
        print 'universal %s'%type_name, _types[type_name]

        # look for the first geometric methods.
        types = [type_name]
        for method in types:
            if method in self.geometries:
                return self.__getattribute__(method)(**kwds)
            else:
                types.extend(_types.get(method,[]))
           

    def scenegraph(self):
        # traverse the graph
        g = self._graph
        if self._scene:
            return self._scene

        self._graph.add_vertex_property('final_geometry')
        final_geometry = g.vertex_property("final_geometry")
 
        transfos = [pgl.Matrix4()]
        
        list(self.traverse(g.root, transfos))

        self._scene = pgl.Scene(final_geometry.values())
        return self._scene
    
    def traverse(self, vid, transfos):
        
        g = self._graph
        edge_type = g.edge_property("edge_type")
        transform = g.vertex_property("transform")

        m = transfos[-1]
        assert vid in g

        # visitor
        self._local2global(vid, m)

        local_t = transform.get(vid)
        if local_t:
            m = local_t * m

        for eid in g.out_edges(vid):
            target_vid = g.target(eid)
            if edge_type[eid] in ['<', '+']:
                for new_vid in self.traverse(target_vid, transfos+[m]):
                    yield new_vid

    def _local2global(self, vid, matrix):
        g = self._graph
        geometry = g.vertex_property("geometry")
        colors = g.vertex_property("color")
        final_geometry = g.vertex_property("final_geometry")
        shape = geometry.get(vid)
        color = colors.get(vid)
        if shape:
            if color:
                shape = pgl.Shape(transform4(matrix, shape), pgl.Material(color))
            else: 
                shape = pgl.Shape(transform4(matrix, shape))
            shape.id = vid
            final_geometry[vid] = shape

    def _get_args( self, properties ):
        return dict([(p.attrib['name'], p.attrib['value']) for p in properties])

def is_matrix(shape):
    return type(shape) == pgl.Matrix4

def transform4(matrix, shape):
    """
    Return a shape transformed by a Matrix4.
    """ 
    scale, (a,e,r), translation = matrix.getTransformation2()
    shape = pgl.Translated( translation, 
                            pgl.Scaled( scale, 
                                        pgl.EulerRotated(a,e,r, 
                                                         shape)))
    return shape


##########################################################################

class Dumper(object):

    def dump(self, graph):
        self._graph = graph
        self.graph()
        return xml.tostring(self.doc)

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
        if not g.vertex_property('geometry').get(vid):
            t = g.vertex_property('transform').get(vid)
            if t:
                transfo = self.SubElement(node, 
                    'property', 
                    {'name':'transform'})
                matrix = self.SubElement(transfo, 'matrix')
                s='\n'
                for i in range(4):
                    c = tuple(t.getRow(i))
                    s += '\t\t\t%.5f %.5f %.5f %.5f\n'%c
                matrix.text = s+'\n' 

        for (name, value) in properties.get(vid,[]).iteritems():
            attrib = {'name':name, 'value':str(value)}
            self.SubElement(node, 'property', attrib)
        
    def edge(self, eid):
        edge_type_conv = {}
        edge_type_conv['<'] = 'successor'
        edge_type_conv['+'] = 'branch'
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

def xml2graph(xml_graph):
    """
    Convert a xml string to a graph and scene graph.
    """
    f = StringIO(xml_graph)
    parser = Parser()
    g, scene = parser.parse(f)
    f.close()
    return g, scene
    

def graph2xml(graph):
    dump = Dumper()
    return dump.dump(graph)


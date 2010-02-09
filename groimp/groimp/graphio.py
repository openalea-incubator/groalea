# TODO

# 1. Implement the full specification
# 2. Test all the cases with several examples
# 3. Implement a loop in OpenAlea
# 4. Use the PlantGL turtle

# 2.1. add a header
# 2.2. separate graph parsing and scenegraph generation
# 2.3. Error management
# 2.4. Documentation
# 2.5. Compute properties when it is possible (sphere, ...)
# 2.6. 2D draw of the graph

from StringIO import StringIO
from math import radians

import xml.etree.ElementTree as xml

from openalea.core.graph.property_graph import PropertyGraph
import openalea.plantgl.all as pgl
Vector3 = pgl.Vector3

class RootedGraph(PropertyGraph):
    def _set_root(self, root):
        self._root = root

    def _get_root(self):
        return self._root

    root = property(_get_root, _set_root)

class Parser(object):
    edge_type_name = {'successor':'<', 'branch':'+'}
    geometries = ['Sphere', 'Box', 'Cone', 'Cylinder', 'Frustum', 
                  'sphere', 'box', 'cone', 'cylinder', 'frustum', 
                  'parallelogram', 'Parallelogram',
                  'F', 'RL', 'RU', 'RH', 'AdjustLU']

    def parse(self, fn):
        self.trash = []
        self._graph = None
        self._scene = None
        # Turtle intialisation
        self._turtle_diameter = -1.
        self._turtle_color = -1

        doc = xml.parse(fn)
        root = doc.getroot()
        self.has_type = False
        self.types(doc.findall('type'))

        self.dispatch(root)
        self.scenegraph()

        return self._graph, self._scene

    def dispatch(self, elt):
        #print 'Dispatch :', elt.tag, elt.attrib 
        #return self.__getattribute__(elt.tag)(elt.getchildren(), **elt.attrib)
        try:
            return self.__getattribute__(elt.tag)(elt.getchildren(), **elt.attrib)
        except Exception, e:
            print e
            raise Exception("Unvalid element %s"%elt.tag)
            

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
        graph._types = self._types

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

    def types(self, elts):
        """ Construct the entire hierarchy of types.
        This is done before parsing the graph.
        """
        self._types = {'Axiom':[]}
        for elt in elts:
            self.type(elt.getchildren(), **elt.attrib)

        self.has_type = True
        # Look recursively to know what is the geometric type
        def geom(name):
            if name in self.geometries:
                return name
            else:
                for ex_type in self._types.get(name,[]):
                    n = geom(ex_type)
                    if n is not None:
                        return n
            return

        self._geoms = {}
        for name in self._types:
            n = geom(name)
            if n is not None:
                self._geoms[name] = geom(name)


    def type(self, elts, name):
        # Add this to the graph...
        if self.has_type == True:
            return

        self._types[name] = []
        for elt in elts:
            #print elt.tag
            if elt.tag == 'extends':
                elt.attrib['type_name'] = name
                self.dispatch(elt)

    def extends(self, elts, name, type_name):
        self._types[type_name].append(name)

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
        
        if type in ['node', 'Axiom']:
            # special case.
            shape, transfo = None, None
        else:
            shape, transfo = self.dispatch2(type, args) 
        
        assert len(transfos) <= 1
        if transfos:
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

    Node = node

    def Sphere(self, radius=1., **kwds):
        return pgl.Sphere(radius=float(radius)), None


    def Box(self, depth=1., width=1., height=1., **kwds):
        depth, width, height= float(depth), float(width), float(height)
        size = pgl.Vector3(depth/2, width/2, height/2)
        return (pgl.Translated((0,0,height/2), pgl.Box(size)), 
               pgl.Matrix4.translation(pgl.Vector3(0,0,height)))

    def Cone(self, radius=1., height=1., bottom_open= False, **kwds):
        # TODO: Implement bottom_open (bool)
        radius, height = float(radius), float(height)
        solid = not bottom_open
        return (pgl.Cone(radius=radius, height=height, solid=solid),
                pgl.Matrix4.translation(pgl.Vector3(0,0,height)))

    def Cylinder(self, radius=1., height=1., bottom_open=False, top_open=False, **kwds):
        #radius, height = float(radius)*10, float(height)*10
        radius, height = float(radius), float(height)
        solid = not(bool(bottom_open) and bool(top_open))
        return (pgl.Cylinder(radius=radius, height=height, solid=solid),
               pgl.Matrix4.translation(pgl.Vector3(0,0,height)))

    def Frustum(self, radius=1., height=1., taper=0.5, **kwds):
        radius, height, taper = float(radius), float(height), float(taper)
        bottom_open = kwds.get('bottom_open', False)
        top_open = kwds.get('top_open', False)
        solid = not(bool(bottom_open) and bool(top_open))

        return (pgl.Frustrum(radius=radius, height=height, taper=taper, solid=solid),
                pgl.Matrix4.translation(pgl.Vector3(0,0,height)))

    def Parallelogram(self,length, **kwds):
        length = float(length) 
        pts = [Vector3(0,0,0), Vector3(length,0,0),Vector3(length, length,0),Vector3(0,length, 0)]
        index = [(0,1,2), (0,2,3)]
        return (pgl.TriangleSet(pts, index), None)

    sphere = Sphere
    box = Box
    cone = Cone
    cylinder = Cylinder
    frustrum = Frustum
    parallelogram = Parallelogram
    
    # Turtle implementation:
    # F0, M, M0, RV, RG, AdjustLU
    def F(self, length=1., diameter=-1., turtle_color=-1, **kwds):
        height = float(length)
        diameter = float(diameter)
        if diameter == -1.:
            diameter = self._turtle_diameter
            if diameter == -1:
                diameter = 0.1
        radius = diameter/2.

        return (pgl.Cylinder(radius=radius, height=height),
               pgl.Matrix4.translation(pgl.Vector3(0,0,height)))

    def F0(self, **kwds):
        return (None, pgl.Matrix4())
    M = M0 = F0

    def RL(self, angle, **kwds):
        # Rotation around the x axis
        angle = radians(float(angle))
        matrix = pgl.Matrix3.axisRotation(Vector3(1,0,0), angle)
        return (None, pgl.Matrix4(matrix))

    def RU(self, angle, **kwds):
        # Rotation around negative y axis
        angle = radians(float(angle))
        matrix = pgl.Matrix3.axisRotation(pgl.Vector3(0,-1,0), angle)
        return (None, pgl.Matrix4(matrix))

    def RH(self, angle, **kwds):
        # Rotation around the z axis
        angle = radians(float(angle))
        matrix = pgl.Matrix3.axisRotation(pgl.Vector3(0,0,1), angle)
        return (None, pgl.Matrix4(matrix))

    def RV(self, strength, **kwds):
        """ Gravitropism. """
        # TODO: Implement this method
        return (None, pgl.Matrix4())

    def RG(self, **kwds):
        """ Maximal gravitropism such that local z-direction points downwards. """
        # TODO: Implement this method
        return (None, pgl.Matrix4())

    def AdjustLU(self, **kwds):
        """ Rotate around local z-axis such that local y-axis points upwards as far as possible."""
        return (None, -1)

    def L(self, length=1., **kwds):
        """ Set the turtle state to the given length. """
        self._turtle_length = float(length)
        return (None, None)

    def D(self, diameter=0.1, **kwds):
        """ Set the turtle state to the given diameter. """
        self._turtle_diameter = float(diameter)
        return (None, None)

    def P(self, color = 14, **kwds):
        """ Set the turtle state to the given color. """
        self._turtle_color = int(color)
        return (None, None)

    def turtle_color(self, color):
        color = int(color)
        if 0 <= color <= 15:
            color = 14
        

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

    Edge = edge

    def _add_edges(self):
        edges = self._edges
        graph = self._graph
        
        for eid, edge in edges.iteritems():
            graph.add_edge(edge=edge, eid=eid)

    def universal_node(self, type_name, **kwds):
        _types = self._graph._types
        #assert type_name in _types, (type_name, _types)
        #print 'universal %s'%type_name, _types[type_name]
        if type_name not in _types:
            if type_name.title() not in _types:
                raise "Unknow object type %s. Known objects are %s."%(type_name,
                                                         sorted(_types.keys()))
            else: 
                type_name = type_name.title()

        # look for the first geometric methods.
        method = self._geoms.get(type_name)
        if method:
            return self.__getattribute__(method)(**kwds)
        else:
            #print '%s has no geometric object associated.'%(type_name,)
            if type_name not in self.trash:
                self.trash.append(type_name)
            return None, None


    def scenegraph(self):
        # traverse the graph
        g = self._graph
        if self._scene:
            return self._scene

        self.visited = set()

        self._graph.add_vertex_property('final_geometry')
        final_geometry = g.vertex_property("final_geometry")
 
        transfos = [pgl.Matrix4()]
        
        self.traverse2(g.root)
        self._scene = pgl.Scene(final_geometry.values())
        return self._scene
    
    def traverse2(self, vid):
        from openalea.container.traversal.graph import breadth_first_search
        
        g = self._graph
        edge_type = g.edge_property("edge_type")
        transform = g.vertex_property("transform")

        transfos = {g.root : pgl.Matrix4()}
        
        def parent(vid):
            for eid in g.in_edges(vid):
                if edge_type[eid] in ['<', '+']:
                    return g.source(eid)
            return vid


        for v in breadth_first_search(g, vid):
            if parent(v) == v and v != g.root:
                print "ERRRORRRR"
                print v
                continue

            m = transfos[parent(v)]
            # Transform the current shape with the stack of transfos m from the root.
            # Store the result in the graph.
            self._local2global(v, m)
            local_t = transform.get(v)

            if local_t == -1:
                #TODO : AdjustLU
                t = m.getColumn(3)
                t = Vector3(t.x, t.y, t.z)
                
                x, y, z = Vector3(1,0,0), Vector3(0,1,0), Vector3(0,0,1)
                X, Y, Z = m*Vector3(1,0,0), m*Vector3(0,1,0), m*Vector3(0,0,1)
                new_x = z ^ Z
                if pgl.normSquared(new_x) > 1e-3:
                    new_y = Z ^ new_x
                    new_x.normalize()
                    new_y.normalize()
                    m = pgl.Matrix4(pgl.BaseOrientation(new_x, new_y).getMatrix3())
                    m.translation(t)
                else:
                    print 'AdjustLU: The two vectors are Colinear'
            elif local_t:
                if local_t.getColumn(3) != pgl.Vector4(0,0,0,1):
                    print v
                    print m
                    m = m * local_t 
                    print m
                else:
                    m = m * local_t 
                    
            else:
                #print m
                pass
            transfos[v] = m

            if parent(v) == 7295 and v == 181:
                print m.getColumn(3)
        
    def traverse(self, vid, transfos):
        if vid in self.visited:
            return

        g = self._graph
        edge_type = g.edge_property("edge_type")
        transform = g.vertex_property("transform")

        m = transfos[-1]
        assert vid in g

        # visitor
        self._local2global(vid, m)

        local_t = transform.get(vid)
        if local_t:
            m = m* local_t

        # Do not traverse again this node
        self.visited.add(vid)

        for eid in g.out_edges(vid):
            target_vid = g.target(eid)
            if edge_type[eid] in ['<', '+']:
                for new_vid in self.traverse(target_vid, [m]):
                    yield new_vid

    def _local2global(self, vid, matrix):
        g = self._graph
        geometry = g.vertex_property("geometry")
        colors = g.vertex_property("color")
        final_geometry = g.vertex_property("final_geometry")
        shape = geometry.get(vid)
        color = colors.get(vid)
        edge_type = g.edge_property("edge_type")

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


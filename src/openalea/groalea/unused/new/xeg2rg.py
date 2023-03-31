

from math import sqrt
from math import cos
from math import sin
from math import radians
from copy import deepcopy
from StringIO import StringIO


import openalea.plantgl.all as pgl

from .utils import (TurtleState, FunctionalGeometry, 
                       rgb_color,
                       orientation, project3Dto2D, determinant, no_interior)

from .xegadj import mtgvidrestore

from .topology import RootedGraph

Vector3 = pgl.Vector3
Vector4 = pgl.Vector4
Color4Array = pgl.Color4Array


class Parser(object):

    edge_type_name = {'successor': '<', 'branch': '+', 'decomposition': '/'}
    geometries = ['Sphere', 'Box', 'Cone', 'Cylinder', 'Frustum',
                  'sphere', 'box', 'cone', 'cylinder', 'frustum',
                  'parallelogram', 'Parallelogram', 'TextLabel', 'textLabel', 'PointCloud', 'pointCloud',
                  'polygon', 'Polygon', 'BezierSurface', 'bezierSurface','ShadedNull', 
                  'F', 'F0', 'M', 'M0', 'RL', 'RU', 'RH', 'V', 'RV', 'RV0', 'RG', 'RD', 'RO', 'RP', 'RN', 'AdjustLU',
                  'L', 'LAdd', 'LMul', 'D', 'DAdd', 'DMul', 'P', 'Translate', 'Scale', 'Rotate']

    FUNCTIONAL = -10

    def parse(self, doc):
        self.trash = []
        self._graph = None
        #self._scene = None
        root = doc.getroot()

        self.has_type = False
        self.types(doc.findall('type'))

        self.dispatch(root)
        #self.scenegraph()
        #self.setFinalGeometry()

        #return self._graph, self._scene
        return self._graph


    def dispatch(self, elt):

        if len(list(elt)) > 1:
            list(elt)
        
        print "Dispatch elt : ", elt
        print "list(elt) : ", list(elt)
        print 'Dispatch elt.tag :', elt.tag 
        print 'Dispatch elt.attrib',elt.attrib 
        print "self.__getattribute__(elt.tag)", self.__getattribute__(elt.tag)
        print "elt.getchildren() : ", elt.getchildren()

        # return self.__getattribute__(elt.tag)(elt.getchildren(), **elt.attrib)
        #try:
        return self.__getattribute__(elt.tag)(list(elt), **elt.attrib)
#        except Exception, e:
#            print e
#            raise Exception("Invalid element %s" % elt.tag)

    def dispatch2(self, method_name, args):
        try:
            return self.__getattribute__(method_name)(**args)
        except:
            # method name has to be defined as a type
            return self.universal_node(method_name, **args)

    def graph(self, elements):
        """
        A graph is a set of nodes and edges.
        """

        graph = self._graph = RootedGraph()

        self._edges = {}
        graph._types = self._types

        # Add initial properties
        graph.add_vertex_property("name")
        graph.add_vertex_property("type")
        graph.add_vertex_property("parameters")
        graph.add_vertex_property("color")
        graph.add_vertex_property("geometry")
        graph.add_vertex_property("transform")
        graph.add_vertex_property("turtle_state")
        graph.add_edge_property("edge_type")

        # Process / Parse all the xml nodes contains directly inside the <graph> node.
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
        # problem: add axiom as a type here to RootedGraph during XEG->RootedGraph,
        #          then it will also be add to XEG during RootedGraph->XEG 
        #self._types = {'Axiom': []}
        self._types = {}
        for elt in elts:
            self.type(elt.getchildren(), **elt.attrib)

        self.has_type = True
        # Look recursively to know what is the geometric type

        def geom(name):
            if name in self.geometries:
                return name
            else:
                for ex_type in self._types.get(name, []):
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
        if self.has_type is True:
            return

        self._types[name] = []
        for elt in elts:
            # print elt.tag
            if elt.tag == 'extends':
                elt.attrib['type_name'] = name
                self.dispatch(elt)

    def extends(self, elts, name, type_name):
        self._types[type_name].append(name)

    implements = extends

    def root(self, elts, root_id):
        self._graph.root = int(root_id)

    def node(self, properties, id, type, name=None):
        """
        TODO: Write an exhaustive list of examples here

        <node id="3" name="" type="L">
            <property name="length" value="6.0"/>
        </node>


        """
        # print "node id, type, name = ", id, type, name
        # print "current turtle_length = ", self._turtle_length
        #if type not in ['Tree', 'GrowthUnit', 'Internode', 'Metamer']:
            #return

        id = int(id)
        if not name:
            name = str(id)

        graph = self._graph

        self._current_turtle = TurtleState()

        graph.add_vertex(id)
        if name:
            graph.vertex_property('name')[id] = name
        graph.vertex_property('type')[id] = type

        # Hack to separate transformation (without value)
        # from other properties (with value)
        transfos = [p for p in properties
                    if p.attrib['name'] == 'transform']
        colors = [p for p in properties
                  if p.attrib['name'] == 'color']
        properties2 = [p for p in properties
                       if p.attrib['name'] not in ('transform', 'color')]

        # TODO : Improve the design

        if type in ['P', 'PointCloud']:
            args = self._get_args(properties)
        else:
            args = self._get_args(properties2)

        graph.vertex_property('parameters')[id] = args

        if type in ['node', 'Axiom']:
            # special case.
            shape, transfo = None, None
        else:
            # call dispatch2 for node conversion
            shape, transfo = self.dispatch2(type, args)
            #shape, transfo = None, None

        # End of TODO

        assert len(transfos) <= 1

        if transfos:
            transfo = self.transform(transfos[0].getchildren())

        color = None

        # Store the turtle state for this node, ONLY
        if colors:
            if len(colors) > 0:
                if len(colors[0].getchildren()) != 0:
                    color = self.color(colors[0].getchildren())
            else:
                raise Exception("color is null!!!")

        if shape:
            graph.vertex_property('geometry')[id] = shape
        if transfo:
            graph.vertex_property('transform')[id] = transfo
        if color:
            graph.vertex_property('color')[id] = color

        if self._current_turtle:
            graph.vertex_property('turtle_state')[id] = self._current_turtle

        self._current_turtle = None                   
        

    Node = node

    def Sphere(self, radius=1., **kwds):
        print "#### get in Sphere func"
        return pgl.Sphere(radius=float(radius)), None

    def Box(self, depth=1., width=1., height=1., **kwds):
        depth, width, height = float(depth), float(width), float(height)
        size = Vector3(depth / 2, width / 2, height / 2)
        return (pgl.Translated((0, 0, height / 2), pgl.Box(size)),
                pgl.Matrix4.translation(Vector3(0, 0, height)))

    def Cone(self, radius=1., height=1., bottom_open=False, **kwds):
        # TODO: Implement bottom_open (bool)
        radius, height = float(radius), float(height)
        solid = not bottom_open
        return (pgl.Cone(radius=radius, height=height, solid=solid),
                pgl.Matrix4.translation(Vector3(0, 0, height)))

    def Cylinder(self, radius=1., height=1., bottom_open=False, top_open=False, color=None, **kwds):
        if color:
            self._current_turtle.color = self.color(color)
        # radius, height = float(radius)*10, float(height)*10
        radius, height = float(radius), float(height)
        solid = not(bool(bottom_open) and bool(top_open))
        return (pgl.Cylinder(radius=radius, height=height, solid=solid),
                pgl.Matrix4.translation(Vector3(0, 0, height)))

    def Frustum(self, radius=1., height=1., taper=0.5, **kwds):
        radius, height, taper = float(radius), float(height), float(taper)
        bottom_open = kwds.get('bottom_open', False)
        top_open = kwds.get('top_open', False)
        solid = not(bool(bottom_open) and bool(top_open))

        return (pgl.Frustrum(radius=radius, height=height, taper=taper, solid=solid),
                pgl.Matrix4.translation(Vector3(0, 0, height)))

    def Parallelogram(self, length=1., width=0.5, **kwds):
        length = float(length)
        width = float(width)
        # pts = [Vector3(0,0,0), Vector3(length,0,0),Vector3(length, width,0),Vector3(0,width, 0)]
        pts = [Vector3(0, 0, 0), Vector3(width, 0, 0), Vector3(width, 0, length), Vector3(0, 0, length)]
        index = [(0, 1, 2), (0, 2, 3)]
        return (pgl.TriangleSet(pts, index), None)

    def TextLabel(self, caption="Default TextLabel", **kwds):
        caption = str(caption)
        return (pgl.Text(caption), None)

    def PointCloud(self, color, points, pointSize, **kwds):
        points = str(points)
        color = str(color)
        points = [float(num) for num in points.split(",")]
        colorlist = [float(num) for num in color.split(",")]
        pointSize = int(pointSize)
        if pointSize <= 0:
            pointSize = 1
        items, chunk = points, 3
        point3Array = zip(*[iter(items)] * chunk)
        idx4 = pgl.Index4(int(colorlist[0] * 255), int(colorlist[1] * 255),
                          int(colorlist[2] * 255), int(colorlist[3] * 255))
        lidx4, v3array = [], []
        for item in point3Array:
            v3array.append(Vector3(item))
            lidx4.append(idx4)
        c4array = Color4Array(lidx4)
        return (pgl.PointSet(v3array, c4array, pointSize), None)

    def Polygon(self, vertices, **kwds):
        """ TODO: Move code to geometry """
        points = str(vertices)
        points = [float(num) for num in points.split(",")]
        items, chunk = points, 3
        p3list = zip(*[iter(items)] * chunk)
        p2list = project3Dto2D(p3list)
        pd2list = []
        for i in range(len(p2list)):
            pd2list.append({i: p2list[i]})
        indexlist = []
        poly_orientation = orientation(p2list)

        while len(pd2list) >= 3:
            for cur in range(len(pd2list)):
                prev = cur - 1
                nex = (cur + 1) % len(pd2list)  # Wrap around on the ends
                # By definition, at least there are two ears;
                # we will iterate at end only if poly_orientation
                # was incorrect.
                pcur, pprev, pnex = pd2list[cur].values()[0], pd2list[prev].values()[0], pd2list[nex].values()[0]
                det = determinant(pcur, pprev, pnex)
                inside = no_interior(pprev, pcur, pnex, pd2list, poly_orientation)
                if (det == poly_orientation) and inside:
                    # Same orientation as polygon
                    # No points inside
                    # Add index of this triangle to the index list
                    index = pd2list[prev].keys()[0], pd2list[cur].keys()[0], pd2list[nex].keys()[0]
                    indexlist.append(index)
                    # Remove the triangle from the polygon
                    del(pd2list[cur])
                    break
        return (pgl.TriangleSet(pgl.Point3Array(p3list), indexlist), None)

    def BezierSurface(self, uCount, data, dimension, **kwds):
        print "#### get in BezierSurface func"
        points = str(data)
        dimension = int(dimension)
        points = [float(num) for num in points.split(",")]
        items, chunk = points, dimension 
        pdlist = zip(*[iter(items)] * chunk)
        
        
        p4m = pgl.Point4Matrix(dimension,dimension)
      
        its, pice = pdlist, 4
        pdmrlst = zip(*[iter(its)] * pice)
        for i in range(len(pdmrlst)):
            for j in range(len(pdmrlst[i])):
                p4m.__setitem__((i,j),pdmrlst[i][j])
                
        return (pgl.BezierPatch(p4m), None)
        
        
    def ShadedNull(self, transform, **kwds):
        print "pass null in"

        transform = str(transform)
        transform = [float(num) for num in transform.split(",")]
        
        items, chunk = transform, 4
        m4rlist = zip(*[iter(items)] * chunk)
        m4 = pgl.Matrix4()

        for i in range(len(m4rlist)):
            for j in range(len(m4rlist[i])):
                m4.__setitem__((i,j),m4rlist[i][j])

        #self._current_turtle.color = self.color(color)
                
        print "pass null out"

        return (None, m4)


    sphere = Sphere
    box = Box
    cone = Cone
    cylinder = Cylinder
    frustrum = Frustum
    parallelogram = Parallelogram
    textLabel = TextLabel
    pointCloud = PointCloud
    bezierSurface = BezierSurface


    # Turtle implementation:
    # F0, M, M0, RV, RG, AdjustLU
    def F(self, length=1., diameter=-1., color=14, **kwds):

        def f(turtle):
            height = turtle.length
            radius = turtle.diameter /2.
            color = turtle.color
            return (pgl.Cylinder(radius=radius, height=height),
            pgl.Matrix4.translation(Vector3(0, 0, height)))

        length = float(length)
        diameter = float(diameter)
        color3 = rgb_color(color)

        self._current_turtle.length = length
        self._current_turtle.diameter = diameter
        self._current_turtle.set_diameter = True
        self._current_turtle.color = color3

        return (FunctionalGeometry(f), self.FUNCTIONAL)

    def F0(self, **kwds):
        def f(turtle):
            height = turtle.length
            radius = turtle.diameter /2.
            color = turtle.color
            return (pgl.Cylinder(radius=radius, height=height),
                    pgl.Matrix4.translation(Vector3(0, 0, height)))

        self._current_turtle.set_diameter = True
        self._current_turtle.set_length = True

        return (FunctionalGeometry(f), self.FUNCTIONAL)

    def M(self, length=1., **kwds):
        height = float(length)
        return (None, pgl.Matrix4.translation(Vector3(0, 0, height)))

    def M0(self, **kwds):

        def f(turtle):
            height = turtle.length
            return (None,
                    pgl.Matrix4.translation(Vector3(0, 0, height)))

        self._current_turtle.set_length = True
        return (FunctionalGeometry(f), self.FUNCTIONAL)

    def RL(self, angle, **kwds):
        # Rotation around the x axis
        angle = radians(float(angle))
        matrix = pgl.Matrix3.axisRotation(Vector3(1, 0, 0), angle)
        return (None, pgl.Matrix4(matrix))

    def RU(self, angle, **kwds):
        # Rotation around negative y axis <-- NO, Wrong idea
        angle = radians(float(angle))
        matrix = pgl.Matrix3.axisRotation(Vector3(0, 1, 0), angle)
        return (None, pgl.Matrix4(matrix))

    def RH(self, angle, **kwds):
        # Rotation around the z axis
        angle = radians(float(angle))
        matrix = pgl.Matrix3.axisRotation(Vector3(0, 0, 1), angle)
        return (None, pgl.Matrix4(matrix))

    def V(self, argument=0., **kwds):
        self._current_turtle.tropism = float(argument)
        return (None, None)

    def RV(self, argument=1., **kwds):
        """ Gravitropism. """
        self._current_turtle.tropism = float(argument)
        return (None, -2)

    def RV0(self, **kwds):
        return (None, -2)

    def RG(self, **kwds):
        """ Maximal gravitropism such that local z-direction points downwards. """
        self._current_turtle.tropism = 1e10
        return (None, -2)

    def RD(self, direction, strength, **kwds):
        self._current_turtle.tropism = float(strength)
        direction = str(direction)
        self._current_turtle.direction = tuple([float(num) for num in direction.split(",")])
        return (None, -3)

    def RO(self, direction, strength, **kwds):
        self.RD(direction, strength, **kwds)
        return (None, -4)

    def RP(self, target, strength, **kwds):
        self._current_turtle.tropism = float(strength)
        target = str(target)
        self._current_turtle.tropism_target = tuple([float(num) for num in target.split(",")])
        return (None, -5)

    RN = RP

    def AdjustLU(self, **kwds):
        """ Rotate around local z-axis such that local y-axis points upwards as far as possible."""
        return (None, -1)

    def L(self, length=1., **kwds):
        """ Set the turtle state to the given length. """
        self._current_turtle.length = float(length)
        return (None, None)

    def LAdd(self, argument=1., **kwds):
        self._current_turtle.length_add = float(argument)
        return (None, None)

    def LMul(self, argument=1., **kwds):
        self._current_turtle.length_mul = float(argument)
        return (None, None)

    def D(self, diameter=0.1, **kwds):
        """ Set the turtle state to the given diameter. """
        self._current_turtle.diameter = float(diameter)
        return (None, None)

    def DAdd(self, argument=1., **kwds):
        self._current_turtle.diameter_add = float(argument)
        return (None, None)

    def DMul(self, argument=1., **kwds):
        self._current_turtle.diameter_mul = float(argument)
        return (None, None)

    def P(self, color=14, **kwds):
        """ Set the turtle state to the given color. """
        color3 = rgb_color(color)
        self._current_turtle.color = color3
        return (None, None)

    def Translate(self, translateX=0., translateY=0., translateZ=0., **kwds):
        # TODO: put the code in geometry
        tx = float(translateX)
        ty = float(translateY)
        tz = float(translateZ)
        return (None, pgl.Matrix4.translation(Vector3(tx, ty, tz)))

    def Scale(self, scaleX=0., scaleY=0., scaleZ=0., **kwds):
        # TODO: put the code in geometry

        sx = float(scaleX)
        sy = float(scaleY)
        sz = float(scaleZ)
        matrix = pgl.Matrix3.scaling(Vector3(sx, sy, sz))
        return (None, pgl.Matrix4(matrix))

    def Rotate(self, rotateX=0., rotateY=0., rotateZ=0., **kwds):
        # TODO: put the code in geometry

        rx = radians(float(rotateX))
        ry = radians(float(rotateY))
        rz = radians(float(rotateZ))
        mx = pgl.Matrix3.axisRotation(Vector3(1, 0, 0), rx)
        my = pgl.Matrix3.axisRotation(Vector3(0, 1, 0), ry)
        mz = pgl.Matrix3.axisRotation(Vector3(0, 0, 1), rz)
        matrix = mx * my * mz
        return (None, pgl.Matrix4(matrix))

    def transform(self, elements, **kwds):
        # TODO: put the code in geometry

        matrix = elements[0]
        assert matrix.tag == 'matrix'
        m4 = map(float, matrix.text.strip().split())
        m4 = pgl.Matrix4(m4[:4], m4[4:8], m4[8:12], m4[12:])
        m4 = m4.transpose()
        return m4

    def color(self, elements, **kwds):
        rgb = elements[0]
        assert rgb.tag == 'rgb'
        color = pgl.Color3(*(int(float(x) * 255) for x in rgb.text.strip().split()))
        self._current_turtle.color = color
        return color

    def edge(self, elements, src_id, dest_id, type, id=None):
        # we add the edges at the end of the process
        edges = self._edges

        if id:
            id = int(id)
        edge_type = self.edge_type_name.get(type, type)

        edges[id] = (int(src_id), int(dest_id))
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
        # print 'universal %s'%type_name, _types[type_name]
        if type_name not in _types:
            if type_name.title() not in _types:
                raise Exception("Unknow object type %s. Known objects are %s." % (type_name,
                                                                                  sorted(_types.keys())))
            else:
                type_name = type_name.title()

        # look for the first geometric methods.
        method = self._geoms.get(type_name)
        if method:
            return self.__getattribute__(method)(**kwds)
        else:
            # print '%s has no geometric object associated.'%(type_name,)
            if type_name not in self.trash:
                self.trash.append(type_name)
            return None, None


    def _get_args(self, properties):
        return dict([(p.attrib['name'], p.attrib['value']) for p in properties])

##########################################################################################################

def xml2graph(xeg_fn):
    """
    Convert a xml string to a rootedgraph and scene graph.
    """
    
    #of = open(xml_graph_file_abs, "r")
    #fc = of.read()
    #f = StringIO(xml_graph_file_abs)
    doc = mtgvidrestore(xeg_fn)
    parser = Parser()
    g = parser.parse(doc)
    #g = adjustFromGroIMP(g)
    #g = adjustmentToMtg(g)
    #g = upscaling4Light(g)
    #f.close()
    return g




def xml2graph_old(xml_graph):
    """

    Convert a xml string to a graph and scene graph.
    """
    f = StringIO(xml_graph)
    parser = Parser()
    g, scene = parser.parse(f)
    f.close()
    return g, scene

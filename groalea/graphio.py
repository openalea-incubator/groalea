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


# 3. Add enum like FUNCTIONAL

from StringIO import StringIO
from math import radians
from math import sqrt
from math import cos
from math import sin
from math import pi
from math import pow
from copy import deepcopy
import xml.etree.ElementTree as xml
import threading
import Queue


from openalea.mtg.io import *
from openalea.core.graph.property_graph import PropertyGraph
import openalea.plantgl.all as pgl


from .geometry import (TurtleState, FunctionalGeometry, rgb_color,
                       is_matrix, transform4, frame,
                       orientation, project3Dto2D, determinant, no_interior,
                       grotation, directionalTropism, orthogonalTropism, adjust_lu)

from .topology import (RootedGraph, spanning_mtg)

from .mappletConverter import offset

from pprint import pprint


Vector3 = pgl.Vector3
Vector4 = pgl.Vector4
Color4Array = pgl.Color4Array
msidShapeidDic = None


class Parser(object):
    edge_type_name = {'successor': '<', 'branch': '+', 'decomposition': '/'}
    geometries = ['Sphere', 'Box', 'Cone', 'Cylinder', 'Frustum',
                  'sphere', 'box', 'cone', 'cylinder', 'frustum',
                  'parallelogram', 'Parallelogram', 'TextLabel', 'textLabel', 'PointCloud', 'pointCloud',
                  'Polygon', 'polygon', 'BezierSurface', 'bezierSurface', 'ShadedNull', 'shadedNull', 
                  'F', 'F0', 'M', 'M0', 'RL', 'RU', 'RH', 'V', 'Vl', 'VlAdd', 'VlMul', 'VAdd', 'VMul','RV', 'RV0', 'RG', 'RD', 'RO', 'RP', 'RN', 'AdjustLU',
                  'L', 'Ll', 'LlAdd', 'LlMul', 'LAdd', 'LMul', 'D', 'Dl', 'DlAdd', 'DlMul', 'DAdd', 'DMul', 'P', 'Translate', 'Scale', 'Rotate', 'NURBSCurve', 'nURBSCurve',
                  'NURBSSurface', 'nURBSSurface', 'Supershape', 'supershape', 'HeightField', 'heightField']

    FUNCTIONAL = -10

    def parse(self, fn, onlyTopology):
        self.onlyTopology = onlyTopology
        self.trash = []
        self._graph = None
        self._scene = None
        doc = xml.parse(fn)
        root = doc.getroot()
        self.has_type = False
        self.types(doc.findall('type'))
        self.dispatch(root)
        
        if not onlyTopology:
            self.scenegraph()

        return self._graph, self._scene


    def dispatch(self, elt):
        
        #print "Dispatch elt : ", elt
        #print "list(elt) : ", list(elt)
        #print 'Dispatch elt.tag :', elt.tag 
        #print 'Dispatch elt.attrib',elt.attrib 
        #print "self.__getattribute__(elt.tag)", self.__getattribute__(elt.tag)
        #print "elt.getchildren() : ", elt.getchildren()

        if len(list(elt)) > 1:
            list(elt)

        return self.__getattribute__(elt.tag)(list(elt), **elt.attrib)


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
        #print "pass graph(self, elements) function"
        #print "elements : ", elements

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
	        th = threading.Thread(target=self.dispatch, args=(elt,))
	        th.start()
	        th.join()
            #self.dispatch(elt)

        # add the edges to the graph, when all the nodes have been added.
        #print "graph.root", type(graph)
        #print graph
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
            #x = elt.getchildren()
            #y = **elt.attrib
	        #th = threading.Thread(target=self.type, args=(x, y,))
	        #th.start()
	        #th.join()
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

        if type in ['node', 'Axiom', 'MtgVertex']:
            # special case.
            shape, transfo = None, None
        else:
            #shape, transfo = self.dispatch2(type, args)
            
            # Here to switch off geometrical type mapping (groimp type to plantgl type)
            if not self.onlyTopology :
                shape, transfo = self.dispatch2(type, args)
            else:
                shape, transfo = None, None
  

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


    def Cylinder(self, radius=1., length=1., base_open=False, top_open=False, **kwds):
        radius, length = float(radius), float(length)
        solid = not(bool(base_open) and bool(top_open))
        return (pgl.Cylinder(radius, length, solid),
                pgl.Matrix4.translation(Vector3(0, 0, length)))

    def Frustum(self, radius=1., height=1., taper=0.5, **kwds):
        radius, height, taper = float(radius), float(height), float(taper)
        bottom_open = kwds.get('bottom_open', False)
        top_open = kwds.get('top_open', False)
        solid = not(bool(bottom_open) and bool(top_open))

        return (pgl.Frustum(radius=radius, height=height, taper=taper, solid=solid),
                pgl.Matrix4.translation(Vector3(0, 0, height)))

    def Parallelogram(self, length=1., width=0.5, **kwds):
        length = float(length)
        width = float(width)
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


    def NURBSCurve(self, ctrlpoints, dimension, degree, **kwds):
        ctrlpoints = str(ctrlpoints)
        ctrlpoints = [float(num) for num in ctrlpoints.split(",")]
        dimension = int(dimension)
        degree = int(degree)
        items, chunk = ctrlpoints, dimension
        pointArray = zip(*[iter(items)] * chunk)

        if (dimension == 2):
            v4array = []
            for item in pointArray:
                v4array.append(pgl.Vector2(item))

            parray = pgl.Point3Array(0)
            for item in v4array:
                parray.append(Vector3(item,1))

            return (pgl.NurbsCurve2D(parray, degree), None)
        elif (dimension == 3):
            v4array = []
            for item in pointArray:
                v4array.append(pgl.Vector3(item))

            parray = pgl.Point4Array(0)
            for item in v4array:
                parray.append(Vector4(item,1))
            return (pgl.NurbsCurve(parray, degree), None)


    def NURBSSurface(self, ctrlpoints, uSize, vSize, uDegree, vDegree, dimension, **kwds):
        ctrlpoints = str(ctrlpoints)
        ctrlpoints = [float(num) for num in ctrlpoints.split(",")]
        dimension = int(dimension)
        uSize = int(uSize)
        vSize = int(vSize)
        uDegree = int(uDegree)
        vDegree = int(vDegree)
        items, chunk = ctrlpoints, dimension
        pointArray = zip(*[iter(items)] * chunk) 
        v4array = []

        if (dimension == 2):        
            for item in pointArray:
                v4array.append(Vector4(item,0,1))
        elif (dimension == 3):
            for item in pointArray:
                v4array.append(Vector4(item,1)) 
        elif (dimension == 4):
            v4array = pointArray

        # create uSize x vSize matrix
        matrixArray = [v4array[i:i+uSize] for i in xrange(0, len(v4array), uSize)]

        return pgl.NurbsPatch(matrixArray, uDegree, vDegree), None

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
        
        
    def ShadedNull(self, transform=None, color=None, **kwds):
        #print "pass null in"

        if transform :
            transform = str(transform)
            transform = [float(num) for num in transform.split(",")]
        
            items, chunk = transform, 4
            m4rlist = zip(*[iter(items)] * chunk)
            m4 = pgl.Matrix4()

            for i in range(len(m4rlist)):
                for j in range(len(m4rlist[i])):
                    m4.__setitem__((i,j),m4rlist[i][j])

        else:
            m4 = None 

        if color:
            self._current_turtle.color = color(color)
                
        #print "pass null out"

        return (None, m4)



    def Supershape(self, a, b, m1, m2, n11, n12, n13, n21, n22, n23, **kwds):
        a = float(a)
        b = float(b)
        m1 = float(m1)
        m2 = float(m2)
        n11 = float(n11)
        n12 = float(n12)
        n13 = float(n13)
        n21 = float(n21)
        n22 = float(n22)
        n23 = float(n23)
        
        verts = []
        faces = []
        scale = 1.0
     
        Unum = 20
        Vnum = 20
    
        Uinc = pi / (Unum/2)
        Vinc = (pi/2)/(Vnum/2)
 
        #fill verts array
        theta = -pi
        for i in range (0, Unum + 1):
            phi = -pi/2
            r1 = Superformula(theta, a, b, m1, n11, n12, n13)
            for j in range(0,Vnum + 1):
                r2 = Superformula(phi, a, b, m2, n21, n22, n23)
                x = scale * (r1 * cos(theta) * r2 * cos(phi))
                y = scale * (r1 * sin(theta) * r2 * cos(phi))
                z = scale * (r2 * sin(phi))
 
                vert = (x,y,z) 
                verts.append(vert)

                phi = phi + Vinc

            theta = theta + Uinc
 
        #define faces
        count = 0
        for i in range (0, (Vnum + 1) *(Unum)):
            if count < Vnum:
                A = i
                B = i+1
                C = (i+(Vnum+1))+1
                D = (i+(Vnum+1))
 
                face = (A,B,C,D)
                faces.append(face)
 
                count = count + 1
            else:
                count = 0    

        return pgl.QuadSet(pgl.Point3Array(verts), faces), None

    def HeightField(self, heightValues, usize, vsize, zerolevel, scale, water, **kwds):
        usize = int(usize)
        vsize = int(vsize)
        zerolevel = float(zerolevel)
        scale = float(scale)
        water = str(water)
        water = water.lower() == 'true'

        heightValues = str(heightValues)
        heightValues = [float(num) for num in heightValues.split(",")]

        verts = []
        faces = []

        nu = usize
        nv = vsize
        p = (0, 0, 0)	

        for v in range(nv):
            for u in range(nu):
                p = heightFieldGetVertex(v*usize + u, usize, vsize, heightValues, zerolevel, scale, water)
                verts.append(p)

        n = 0
        for v in range(nv):
            for u in range(nu):
                if v < nv-1 and u < nu-1:
                    face = (n, n+1, n+1+nu, n+nu)
                    faces.append(face)
                n = n+1         

        return pgl.QuadSet(pgl.Point3Array(verts), faces), None

    def MyNURBSCurve(self, ctrlpoints, dimension, **kwds):
        dimension = int(dimension)
        points = str(ctrlpoints)
        points = [float(num) for num in points.split(",")]
        items, chunk = points, dimension
        plist = zip(*[iter(items)] * chunk)

        ctlplist = []
        for i in range(len(plist)):
            ctlplist.append(plist[i]+(1,))

        if dimension == 2:		
            return (pgl.NurbsCurve2D(ctlplist), None)
        elif dimension == 3:
            return (pgl.NurbsCurve(ctlplist), None)


    sphere = Sphere
    box = Box
    cone = Cone
    cylinder = Cylinder
    frustum = Frustum
    parallelogram = Parallelogram
    textLabel = TextLabel
    pointCloud = PointCloud
    polygon = Polygon
    bezierSurface = BezierSurface
    shadedNull = ShadedNull
    nURBSCurve = NURBSCurve
    nURBSSurface = NURBSSurface
    supershape = Supershape
    heightField = HeightField



    # Turtle implementation:
    # F0, M, M0, RV, RG, AdjustLU
    def F(self, length, diameter=-1., fcolor=-1, **kwds):
        def f(turtle):
            height = length
            radius = turtle.diameter /2.
            color = turtle.color
	    if radius !=0 and height !=0 : 
                return (pgl.Cylinder(radius=radius, height=height),
                pgl.Matrix4.translation(Vector3(0, 0, height)))
            else:
                return (None, pgl.Matrix4.translation(Vector3(0, 0, height)))

        length = float(length)
        diameter = float(diameter)
        color = int(fcolor)
        self._current_turtle.diameter = diameter
        self._current_turtle.color = color
        self._current_turtle.node_type = 'F'

        return (FunctionalGeometry(f), self.FUNCTIONAL)

    def F0(self, **kwds):
        def f(turtle):
            height = turtle.length
            radius = turtle.diameter /2.
            color = turtle.color
	    if radius !=0 and height !=0 : 
                return (pgl.Cylinder(radius=radius, height=height),
                pgl.Matrix4.translation(Vector3(0, 0, height)))
            else:
                return (None, pgl.Matrix4.translation(Vector3(0, 0, height)))

        self._current_turtle.node_type = 'F0'

        return (FunctionalGeometry(f), self.FUNCTIONAL)

    def M(self, length, **kwds):
        height = float(length)
        return (None, pgl.Matrix4.translation(Vector3(0, 0, height)))

    def M0(self, **kwds):
        def f(turtle):
            height = turtle.length
            return (None,
                    pgl.Matrix4.translation(Vector3(0, 0, height)))

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

    def V(self, argument, **kwds): 
        self._current_turtle.set_tropism_value = float(argument)
	self._current_turtle.set_tropism = True
        return (None, None)

    def Vl(self, argument, **kwds):
        self._current_turtle.localTropism = float(argument)
        self._current_turtle.set_localTropism = True
        return (None, None)

    def VlAdd(self, argument, **kwds):
        self._current_turtle.tropism_ladd = float(argument)
        return (None, None)

    def VlMul(self, argument, **kwds):
        self._current_turtle.tropism_lmul = float(argument)
        return (None, None)

    def VAdd(self, argument, **kwds):
        self._current_turtle.tropism_add = float(argument)
	#self._current_turtle.set_tropism = True
        self._current_turtle.tropism_op = 'add'
        return (None, None)

    def VMul(self, argument, **kwds):
        self._current_turtle.tropism_mul = float(argument)
	#self._current_turtle.set_tropism = True
        self._current_turtle.tropism_op = 'mul'
        return (None, None)

    def RV(self, argument, **kwds):
        """ Gravitropism. """
        self._current_turtle.tropism_rv = float(argument)
	#self._current_turtle.set_tropism = True
        return (None, -6)

    def RV0(self, **kwds):
	#self._current_turtle.set_tropism = True
        return (None, -2)

    def RG(self, **kwds):
        """ Maximal gravitropism such that local z-direction points downwards. """
        self._current_turtle.tropism = 1e10
        return (None, -2)

    def RD(self, direction, strength, **kwds):
        self._current_turtle.tropism = float(strength)
        direction = str(direction)
        self._current_turtle.tropism_direction = tuple([float(num) for num in direction.split(",")])
        return (None, -3)

    def RO(self, direction, strength, **kwds):
        self._current_turtle.tropism = float(strength)
        direction = str(direction)
        self._current_turtle.tropism_direction = tuple([float(num) for num in direction.split(",")])
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

    def L(self, length, **kwds):
        """ Set the turtle state to the given length. """
        self._current_turtle.set_length_value = float(length)
	self._current_turtle.set_length = True
        return (None, None)

    def Ll(self, argument, **kwds):
        self._current_turtle.localLength = float(argument)
        self._current_turtle.set_localLength = True
        return (None, None)

    def LlAdd(self, argument, **kwds):
        self._current_turtle.length_ladd = float(argument)
        return (None, None)

    def LlMul(self, argument, **kwds):
        self._current_turtle.length_lmul = float(argument)
        return (None, None)

    def LAdd(self, argument, **kwds):
        self._current_turtle.length_add = float(argument)
	#self._current_turtle.set_length = True
        self._current_turtle.length_op = 'add'
        return (None, None)

    def LMul(self, argument, **kwds):
        self._current_turtle.length_mul = float(argument)
	#self._current_turtle.set_length = True
        self._current_turtle.length_op = 'mul'
        return (None, None)

    def D(self, diameter, **kwds):
        """ Set the turtle state to the given diameter. """
        self._current_turtle.set_diameter_value = float(diameter)
        self._current_turtle.set_diameter = True
        return (None, None)

    def Dl(self, argument, **kwds):
        self._current_turtle.localDiameter = float(argument)
        self._current_turtle.set_localDiameter = True
        return (None, None)

    def DlAdd(self, argument, **kwds):
        self._current_turtle.diameter_ladd = float(argument)
        return (None, None)

    def DlMul(self, argument, **kwds):
        self._current_turtle.diameter_lmul = float(argument)
        return (None, None)

    def DAdd(self, argument, **kwds):
        self._current_turtle.diameter_add = float(argument)
        #self._current_turtle.set_diameter = True
        self._current_turtle.diameter_op = 'add'
        return (None, None)

    def DMul(self, argument, **kwds):
        self._current_turtle.diameter_mul = float(argument)
        #self._current_turtle.set_diameter = True
        self._current_turtle.diameter_op = 'mul'
        return (None, None)

    def P(self, color=14, **kwds):
        """ Set the turtle state to the given color. """
        self._current_turtle.set_color_value = int(color)
        self._current_turtle.set_color = True
        return (None, None)

    def color(self, elements, **kwds):
        rgb = elements[0]
        assert rgb.tag == 'rgb'
        color = pgl.Color3(*(int(float(x) * 255) for x in rgb.text.strip().split()))
        self._current_turtle.shaded_color = color
        return color

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

    def scenegraph(self, rg=None):

        def storeShapelist(final_geometry):
            """
            store the shape in the order of the sid number, which corresponds to
            metamer toplogy applied in mappletConverter to convert MTG+Scene to XEG
            """
            vid_set = set()
            for sid in final_geometry.keys():
                vid_set.add(sid/10**offset)
    
            
            vid_sid_dic = {}
            for vid in vid_set:
                sidspvid = []
                for sid in final_geometry.keys():
                    if sid/10**offset == vid:
                        sidspvid.append(sid)
                vid_sid_dic[vid]=sidspvid

            shapess=[]
            for sidspvid in vid_sid_dic.values():
                sidspvid.sort()
                shapes = []
                for sid in sidspvid:
                    shapes.append(final_geometry[sid])
                shapess.append(pgl.Scene(shapes))

            return shapess
  

        # traverse the graph
        if rg != None:
            self._graph = rg

        g = self._graph

        if self._scene:
            return self._scene
        else:
            self._scene = pgl.Scene()

        self.visited = set()

        #self._graph.add_vertex_property('final_geometry')
        g.add_vertex_property('final_geometry')
        final_geometry = g.vertex_property("final_geometry")

        transfos = [pgl.Matrix4()]

        self.traverse2(g.root)

        shapesc_list = storeShapelist(final_geometry)
        #self._scene.merge(pgl.Scene(final_geometry.values()))
        self._scene = pgl.Scene(shapesc_list)

        
        self._scene.save("/home/groimp/temps/try_st_id1.bgeom")
        return self._scene

    def traverse2(self, vid):
        from openalea.container.traversal.graph import breadth_first_search

        g = self._graph

        edge_type = g.edge_property("edge_type")
        transform = g.vertex_property("transform")
        local_turtles = g.vertex_property("turtle_state")

        transfos = {g.root: pgl.Matrix4()}

        # CPL
        turtles = {g.root: TurtleState()}

        def parent(vid):
            for eid in g.in_edges(vid):
                if edge_type[eid] in ['<', '+', '/']:
                    return g.source(eid)
            return vid

        def update_turtle(v, ts):
            lt = local_turtle = local_turtles.get(v, TurtleState())
            #global_turtle = ts.combine(local_turtle) <-- wrong idea
            global_turtle = local_turtle.combine(ts)
            return global_turtle


        #for v in breadth_first_search(g, vid):

        def func_search(v, g, transfos, turtles, transform):
            pid = parent(v)

            if not (pid == v and v != g.root):
                    #print "ERRRORRRR"
                    #print v
                    #continue

                #print "v:",v
                # print "parent(v)", parent(v)
                # print "transfos", transfos
                #m = transfos[parent(v)]

                m = transfos.get(pid)

                if not m:
                    m=pgl.Matrix4()

                # CPL
                ts = turtles.get(pid, TurtleState())
                gt = global_turtle = update_turtle(v, ts)
                #print "v, gt.set_locTm, gt.locTm, gt.set_tm_val= ", v, gt.set_localTropism, gt.localTropism, gt.set_tropism_value
                #print "gt.tropism", gt.tropism
                local_t = transform.get(v)
                if local_t == self.FUNCTIONAL:
                    # Get the functional shape to compute the transformation and geometry
                    # with the turtle state
                    local_t = self.f_shape(v, global_turtle)

                #print "every m : ", m

                # Transform the current shape with the stack of transfos m from the root.
                # Store the result in the graph.
                self._local2global(v, m, global_turtle.color)


                #print "local_t : ", local_t

                if local_t == -1:
                    m = adjust_lu(m)
                elif local_t == -2:
                    #RV0 and RG
                    local_m = grotation(m, gt.tropism)
                    m = m * local_m
                elif local_t == -6:
                    #RV
                    local_m = grotation(m, gt.tropism_rv)
                    m = m * local_m
                elif local_t == -3:
                    # RD
                    local_m = directionalTropism(m, gt.tropism_direction, gt.tropism)
                    m = m * local_m
                elif local_t == -4:
                    # RO
                    local_m = orthogonalTropism(m, gt.tropism_direction, gt.tropism)
                    m = m * local_m
                elif local_t == -5:
                    #RP and RN
                    local_m = positionalTropism(m, gt.tropism_target, gt.tropism)
                    m = m * local_m
                elif local_t:
                    if local_t.getColumn(3) != Vector4(0, 0, 0, 1):
                        m = m * local_t
                    else:
                        m = m * local_t
                else:
                    pass

                transfos[v] = m
                turtles[v] = global_turtle

        for v in breadth_first_search(g, vid):
	        th = threading.Thread(target=func_search, args=(v, g, transfos, turtles, transform,))
	        th.start()
	        th.join()
            

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
            m = m * local_t

        # Do not traverse again this node
        self.visited.add(vid)

        for eid in g.out_edges(vid):
            target_vid = g.target(eid)
            if edge_type[eid] in ['<', '+', '/']:
                for new_vid in self.traverse(target_vid, [m]):
                    yield new_vid

    def f_shape(self, vid, t):
        g = self._graph
        geometry = g.vertex_property("geometry")
        transform = g.vertex_property("transform")

        shape = geometry.get(vid)
        geom, transfo = shape(t)

        geometry[vid] = geom
        transform[vid] = transfo
        return transfo

    def _local2global(self, vid, matrix, color):
        g = self._graph
        geometry = g.vertex_property("geometry")
        colors = g.vertex_property("color")
        final_geometry = g.vertex_property("final_geometry")
        shape = geometry.get(vid)
        edge_type = g.edge_property("edge_type")


        if shape:
            if color:
                shape = pgl.Shape(transform4(matrix, shape), pgl.Material(color))
            else:
                shape = pgl.Shape(transform4(matrix, shape))
            shape.id = self._getShapeid(vid)
            final_geometry[vid] = shape

        if color:
            colors[vid] = color

    def _getShapeid(self, vid):
        global msidShapeidDic
        shapeid = 0
        #print msidShapeidDic
        #print "vid --=", vid
        if msidShapeidDic == None:
            pass
        else:
            try:
                #print "dic id =", vid/10**offset * 10**offset
                shapeid = int(msidShapeidDic[vid/10**offset * 10**offset])
            except KeyError:
                pass
        #print "shapeid =", shapeid  
        return shapeid

    def _get_args(self, properties):
        return dict([(p.attrib['name'], p.attrib['value']) for p in properties])


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
            if properties.get(vid):
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


def Superformula(theta, a, b, m, n1, n2, n3):
    tmp1 = abs((1.0/a) * cos(m * theta/4.0))
    tmp1 = pow(tmp1, n2)

    tmp2 = abs((1.0/b) * sin(m * theta/4.0))
    tmp2 = pow(tmp2, n3)

    tmp3 = tmp1 + tmp2
    tmp3 = pow(tmp3, -1.0/n1)
    return tmp3

def heightFieldGetVertex(index, usize, vsize, heightValues, zerolevel, scale, water):
    sx = usize
    sy = vsize
    y = index / sx
    x = index - y * sx

    p = (0, 0, 0)
    out_x = x * 1 / float(sx - 1)
    out_y = y * 1 / float(sy - 1)
    out_z = (heightValues[y*usize + x] - zerolevel) * scale

    if water and (height <= zerolevel):
        out_z = 0

    p = (out_x, out_y, out_z)
    return p


##########################################################################

# Wrapper functions for OpenAlea usage.


def xmlFile2graph(xeg_file_abname, onlyTopology=False):
    """
    Convert a xml string to a rootedgraph and scene graph.
    """
    
    of = open(xeg_file_abname, "r")
    f = of.read()
    #f = StringIO(xml_graph_file_abs)
    f = StringIO(f)
    parser = Parser()
    rootedgraph, scene = parser.parse(f, onlyTopology)
    #g = adjustFromGroIMP(graph)
    #g = adjustmentToMtg(g)
    #g = upscaling4Light(g)
    #f.close()
    return rootedgraph, scene


def graph2xml(graph):
    dump = Dumper()
    return dump.dump(graph)


def xml2graph(xml_graph, onlyTopology=False):
    """
    Convert a xml string to a graph and scene graph.
    """
    f = StringIO(xml_graph)
    parser = Parser()
    g, scene = parser.parse(f, onlyTopology)
    f.close()

    return g, scene


def xeg2MtgAndScene(xml_graph):
    """
    convert a xml string (an XEG data file) to a MTG object and a Scene object
    """
    import copy
    from openalea.mtg import MTG
    # firstly, get only the rooted graph with mapped node (geometrical) type (by dispatch2)
    rootedgraph, nullscene = xml2graph(xml_graph, True)
    crg, nullscene = xml2graph(xml_graph, True)

    # then, produce the scene and mtg by Parallel Computing in two threads
    scene_queue = Queue.Queue()
    mtg_queue = Queue.Queue()

    for i in range(2):
        if i == 0:
            th = threading.Thread(target=rootedgraph2scene, args=(rootedgraph, scene_queue,))
        else:
            th = threading.Thread(target=rootedgraph2mtg, args=(crg, mtg_queue,))
        th.start()
        th.join()
    
    return mtg_queue.get(), scene_queue.get()


def rootedgraph2scene(rootedgraph, scene_queue):

    # get the rooted graph having only the sub metamer scale and produce a scene object
    sceneXEG = getSceneXEG(rootedgraph)
    rg, scene = xml2graph(sceneXEG)
    scene_queue.put(scene)


def rootedgraph2mtg(crg, mtg_queue):

    # get the rooted graph having scales in mtg and produce a mtg object 
    mtgrg = getMTGRootedGraph(crg)
    if len(mtgrg._edges) == 0:
        mtg = None
    else:
        mtg = spanning_mtg(mtgrg)
    
    mtg_queue.put(mtg)
    

def upscaling4Light(rootedgraph):
    """
    aggregate light interception value from submetamer scale (0-many blades) to metamer scale (1 vertex) 
    using color to detact BezierSurface typed blades 
    """
    sids = rootedgraph._vertices.keys()
    sids.remove(rootedgraph.root)
    edgedic = rootedgraph._edges
    for sid in sids:  
        if rootedgraph.vertex_property("type")[sid] == "BezierSurface":
            rgb_color = rootedgraph.vertex_property("color")[sid] 
            if isGreen(rgb_color):
                print " BezierSurface node sid == ", sid
                for eid in edgedic.keys(): 
                    if edgedic[eid][1] == sid and rootedgraph.edge_property("edge_type")[eid]== "/":
                        msid = edgedic[eid][0]
                        rootedgraph.vertex_property("lightInterception")[mid] += rootedgraph.vertex_property("lightInterception")[sid]

    return rootedgraph


def isGreen(rgb_color):
    r=rgb_color.red
    g=rgb_color.green
    b=rgb_color.blue
    if (r*1.5<=g and b*1.5<=g and g!=0):
        return True



def getSceneRootedGraph(rootedgraph):
    """
    delete the scales from MTG and connet graph root to roots in geometric scale
    """
    
    g = rootedgraph
    
    # store the mapping between MTG vertex and shape id in scene
    storeMsidShapeidDic(g)

    # to allow resulting single scale XEG have "transform" as node's property
    # transform need to be put as paramters of nodes in rootedgraph
    transdic = g.vertex_property("transform") 
    for sid in transdic.keys():
        para = {'transform': transdic[sid]}
        g.vertex_property("parameters")[sid] = para
 
    g.remove_vertex_property("transform")
    
    #get roots of the geometric scale/finest scale 
    mtg_mpt = spanning_mtg(g)
    roots = mtg_mpt.roots(mtg_mpt.max_scale())

    #delete the scales from MTG
    sids = g._vertices.keys()
        # for error caused by that root has no name property
    sids.remove(g.root)
    for sid in sids:
        if g.vertex_property("type")[sid] == "MtgVertex":
            g.remove_vertex(sid)
     
    # connect graph root to roots of the remainning geometric scale       
    rootedgraph = g  
    oneSucessorAdded = False     
    for root in roots:
        edge = (rootedgraph.root, root) 
        edgeid = rootedgraph.add_edge(edge)
        if not oneSucessorAdded:
            rootedgraph.edge_property("edge_type")[edgeid] = "<"
            oneSucessorAdded = True
        else:
            rootedgraph.edge_property("edge_type")[edgeid] = "+"

    return g


def getSceneXEG(rootedgraph):
    """
    delete the scales from MTG and connet graph root to roots in geometric scale
    """
    from pprint import pprint 
    
    g = rootedgraph

    #pprint(vars(g))    
    
    # store the mapping between MTG vertex and shape id in scene
    storeMsidShapeidDic(g)

    # to allow resulting single scale XEG have "transform" as node's property
    # transform need to be put as paramters of nodes in rootedgraph
    transdic = g.vertex_property("transform") 
    for sid in transdic.keys():
        para = {'transform': transdic[sid]}
        g.vertex_property("parameters")[sid] = para
 
    g.remove_vertex_property("transform")
    
    #get roots of the geometric scale/finest scale 
    mtg_mpt = spanning_mtg(g)
    maxscale = mtg_mpt.max_scale()
    roots = mtg_mpt.roots(maxscale)

    #delete the scales from MTG
    sids = g._vertices.keys()
        # for error caused by that root has no name property
    sids.remove(g.root)
    for sid in sids:
        if g.vertex_property("type")[sid] == "MtgVertex":
            g.remove_vertex(sid)

    # delete the decomposition edges for GroIMP one scale case (one or multi tree)
    edgedic = g._edges
    for edgeid in edgedic.keys():
        if g.edge_property("edge_type")[edgeid] == "/":
            g.remove_edge(edgeid)
     
    # connect graph root to roots of the remainning geometric scale       
    rootedgraph = g
    oneSucessorAdded = False  
       
    for root in roots:
        edge = (rootedgraph.root, root) 
        edgeid = rootedgraph.add_edge(edge)
        if not oneSucessorAdded:
            rootedgraph.edge_property("edge_type")[edgeid] = "<"
            oneSucessorAdded = True
        else:
            rootedgraph.edge_property("edge_type")[edgeid] = "+"

    # convert the rootedgraph to xeg. Ready to be converted to scene graph
    single_scale_xeg = graph2xml(g)

    return single_scale_xeg

    	
def storeMsidShapeidDic(rootedgraph):
    """
    store the mapping between MTG vertex (super id of metamer: msid) and shape in scene (orignal "id")
    """
    global msidShapeidDic

    g = rootedgraph
    prodic = g.vertex_property('parameters')

    msidShapeidDic = {}
    for sid in prodic.keys():
        try:
            if prodic.get(sid)['id']:
                shapeid = prodic.get(sid)['id']
                msidShapeidDic[sid] = shapeid
        except KeyError:
            pass   


def getMTGRootedGraph(rootedgraph):
    """
    delete sub-metamer scale and set the sid of each remained node to original vid
    """
    sids = rootedgraph._vertices.keys()
    # for error caused by that root has no name property
    sids.remove(rootedgraph.root)
    for sid in sids:
        if rootedgraph.vertex_property("type")[sid] != "MtgVertex":
            rootedgraph.remove_vertex(sid)

    
    # set the sid of each remained node to original vid
    temp_sids = rootedgraph._vertices.keys()
    mtg_sids = sorted(temp_sids, key=int)
    mtg_sids_edgedic = rootedgraph._edges
    for mtg_sid in mtg_sids:
        #if mtg_sid == 100000:
            #print "sid = ", mtg_sid
        mtg_vid = mtg_sid/ 10**offset
        # for error caused by root == 0
        if mtg_vid != mtg_sid:
            #if mtg_sid == 100000:
                #print "sid = ", mtg_sid
            #if mtg_vid == 1000:
                #print "vid = ", mtg_vid
            rootedgraph._vertices[mtg_vid] = rootedgraph._vertices[mtg_sid]
            if rootedgraph._vertices[mtg_vid] == rootedgraph._vertices[mtg_sid]:
                del rootedgraph._vertices[mtg_sid]

    # set also the edge (for source and destination vetex) sid to vid
    for mtg_eid in mtg_sids_edgedic.keys():
        srcsid = mtg_sids_edgedic[mtg_eid][0]
        dstsid = mtg_sids_edgedic[mtg_eid][1]
        mtg_sids_edgedic[mtg_eid] = (srcsid/10**offset, dstsid/10**offset)


    # set also the parameters sid to vid    
    for skey in rootedgraph.vertex_property("parameters").keys():
        nkey = skey/ 10**offset
        if nkey != skey:
            rootedgraph.vertex_property("parameters")[nkey] = rootedgraph.vertex_property("parameters")[skey] 
            del rootedgraph.vertex_property("parameters")[skey]

    

    # remove properties does not belongs to the mtg (properties have been added to allow scene creation)
    non_mtg_pnames = ["name", "type", "color", "geometry", "transform", "turtle_state", "final_geometry"]
    for non_mtg_pname in non_mtg_pnames:
        if non_mtg_pname in rootedgraph._vertex_property.keys():
            rootedgraph.remove_vertex_property(non_mtg_pname)


    # get all the paramters back as properties of vertex/node
    for nkey in rootedgraph.vertex_property("parameters").keys():
        para_dic = rootedgraph.vertex_property("parameters")[nkey]
        pnames = para_dic.keys()

        for pname in pnames:
            if pname not in rootedgraph._vertex_property.keys():
                rootedgraph.add_vertex_property(pname)
            rootedgraph.vertex_property(pname)[nkey] = para_dic[pname]
    
    rootedgraph.remove_vertex_property("parameters")

    #resumeVidFromSid(rootedgraph)
    produceMTGContentfile(rootedgraph, "/home/groimp/temps/st_rootedgraph_content.txt")

    return rootedgraph



def resumeVidFromSid(rootedgraph):
    """
    set the sid of each remained node to original vid
    """

    # set the edge (for source and destination vetex) sid to vid
    mtg_sids_edgedic = rootedgraph._edges
    for mtg_eid in mtg_sids_edgedic.keys():
        srcsid = mtg_sids_edgedic[mtg_eid][0]
        dstsid = mtg_sids_edgedic[mtg_eid][1]
        mtg_sids_edgedic[mtg_eid] = (srcsid/10**offset, dstsid/10**offset)

    # set the node sid sid to vid
    mtg_sids = rootedgraph._vertices.keys()
    for mtg_sid in mtg_sids:
        mtg_vid = mtg_sid/ 10**offset
        # for error caused by root == 0
        if mtg_vid != mtg_sid:
            rootedgraph._vertices[mtg_vid] = rootedgraph._vertices[mtg_sid]
            del rootedgraph._vertices[mtg_sid]


    # set also the parameters sid to vid
    """    
    for skey in rootedgraph.vertex_property("parameters").keys():
        nkey = skey/ 10**offset
        if nkey != skey:
            rootedgraph.vertex_property("parameters")[nkey] = rootedgraph.vertex_property("parameters")[skey] 
            del rootedgraph.vertex_property("parameters")[skey]
    """

    return rootedgraph



def produceXEGfile(xeg_object, xeg_file_abname):
    """
    produce XEG file with given absolute name from xeg object
    """
    import sys
     
    orig_stdout = sys.stdout
    f = file(xeg_file_abname, 'w')
    sys.stdout = f

    from pprint import pprint 
    print(xeg_object)

    sys.stdout = orig_stdout
    f.close()


def produceMTGContentfile(mtg_object, mtg_file_abname):
    """
    produce mtg content text file with given absolute name from mtg object
    """
    
    import sys
     

    orig_stdout = sys.stdout
    f = file(mtg_file_abname, 'w')
    sys.stdout = f

    from pprint import pprint 
    pprint(vars(mtg_object))

    sys.stdout = orig_stdout
    f.close()


def produceMTGDisplayfile(mtg_object, mtg_file_abname):
    """
    produce mtg content text file with given absolute name from mtg object
    """
    
    import sys
     

    orig_stdout = sys.stdout
    f = file(mtg_file_abname, 'w')
    sys.stdout = f

    from pprint import pprint 
    pprint(mtg_object.display())

    sys.stdout = orig_stdout
    f.close()



def produceMTGfile(mtg_object, mtg_file_abname):
    """
    produce mtg file with given absolute name from mtg object
    """
    
    
    g = mtg_object 
    properties = [(p, 'REAL') for p in g.property_names() if p not in ['edge_type', 'index', 'label', '_line']]
    print properties
    mtg_lines = write_mtg(g, properties)
    f = open(mtg_file_abname, 'w')
    f.write(mtg_lines)
    f.close()

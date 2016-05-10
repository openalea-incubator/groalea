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
from copy import deepcopy
import xml.etree.ElementTree as xml

from openalea.core.graph.property_graph import PropertyGraph
import openalea.plantgl.all as pgl

from .geometry import (TurtleState, FunctionalGeometry, rgb_color,
                       is_matrix, transform4, frame,
                       orientation, project3Dto2D, determinant, no_interior,
                       grotation, directionalTropism, orthogonalTropism, adjust_lu)

Vector3 = pgl.Vector3
Vector4 = pgl.Vector4
Color4Array = pgl.Color4Array


class RootedGraph(PropertyGraph):
    """ A general graph with a root vertex. """
    def _set_root(self, root):
        self._root = root

    def _get_root(self):
        return self._root

    root = property(_get_root, _set_root)

class Parser(object):
    edge_type_name = {'successor': '<', 'branch': '+', 'decomposition': '/'}
    geometries = ['Sphere', 'Box', 'Cone', 'Cylinder', 'Frustum',
                  'sphere', 'box', 'cone', 'cylinder', 'frustum',
                  'parallelogram', 'Parallelogram', 'TextLabel', 'textLabel', 'PointCloud', 'pointCloud',
                  'polygon', 'Polygon', 'nURBSCurve', 'NURBSCurve',
                  'F', 'F0', 'M', 'M0', 'RL', 'RU', 'RH', 'V', 'Vl', 'VlAdd', 'VlMul','VAdd', 'VMul','RV', 'RV0', 'RG', 'RD', 'RO',        			  'RP', 'RN', 'AdjustLU',
                  'L', 'Ll', 'LlAdd', 'LlMul', 'LAdd', 'LMul', 'D', 'Dl', 'DlAdd', 'DlMul', 'DAdd', 'DMul', 'P', 'Translate', 'Scale', 			  'Rotate']

    FUNCTIONAL = -10

    def parse(self, fn):
        self.trash = []
        self._graph = None
        self._scene = None
        doc = xml.parse(fn)
        root = doc.getroot()
        self.has_type = False
        self.types(doc.findall('type'))
        self.dispatch(root)
        self.scenegraph()

        return self._graph, self._scene

    def dispatch(self, elt):
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
        self._types = {'Axiom': []}
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
            shape, transfo = self.dispatch2(type, args)

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

    def Cylinder(self, radius=1., height=1., bottom_open=False, top_open=False, **kwds):
        radius, height = float(radius), float(height)
        solid = not(bool(bottom_open) and bool(top_open))
        return (pgl.Cylinder(radius=radius, height=height, solid=solid),
                pgl.Matrix4.translation(Vector3(0, 0, height)))

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

	def NURBSCurve(self, ctrlpoints, dimension, **kwds):
		dimension = int(dimension)
		points = str(ctrlpoints)
        points = [float(num) for num in points.split(",")]
        items, chunk = points, dimension
        plist = zip(*[iter(items)] * chunk)

        for i in range(len(plist)):
			ctlplist.append(plist[i]+(1,))

        if dimension == 2:		
			return (pgl.NURBSCurve2D(ctlplist), None)
        elif demension == 3:
			return (pgl.NURBSCurve(ctlplist), None)

    sphere = Sphere
    box = Box
    cone = Cone
    cylinder = Cylinder
    frustrum = Frustum
    parallelogram = Parallelogram
    textLabel = TextLabel
    pointCloud = PointCloud
    #nURBSCurve = NURBSCurve

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

    def scenegraph(self):
        # traverse the graph
        g = self._graph
        if self._scene:
            return self._scene
        else:
            self._scene = pgl.Scene()

        self.visited = set()

        self._graph.add_vertex_property('final_geometry')
        final_geometry = g.vertex_property("final_geometry")

        transfos = [pgl.Matrix4()]

        self.traverse2(g.root)
        self._scene.merge(pgl.Scene(final_geometry.values()))
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
	    print "v, lt.set_locTm, lt.locTm, lt.set_tm_val= ", v, lt.set_localTropism, lt.localTropism, lt.set_tropism_value
            print "lt.tropism", lt.tropism
            #global_turtle = ts.combine(local_turtle) <-- wrong idea
            global_turtle = local_turtle.combine(ts)
            return global_turtle


        for v in breadth_first_search(g, vid):
            pid = parent(v)

            if pid == v and v != g.root:
                print "ERRRORRRR"
                print v
                continue

            print "v",v
            print "parent(v)", parent(v)

            m = transfos.get(pid)

            # CPL
            ts = turtles.get(pid, TurtleState())
            gt = global_turtle = update_turtle(v, ts)
            print "v, gt.set_locTm, gt.locTm, gt.set_tm_val= ", v, gt.set_localTropism, gt.localTropism, gt.set_tropism_value
            print "gt.tropism", gt.tropism
            local_t = transform.get(v)
            if local_t == self.FUNCTIONAL:
                # Get the functional shape to compute the transformation and geometry
                # with the turtle state
                local_t = self.f_shape(v, global_turtle)

            # Transform the current shape with the stack of transfos m from the root.
            # Store the result in the graph.
            self._local2global(v, m, global_turtle.color)

            # print "local_t : ", local_t
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
            shape.id = vid
            final_geometry[vid] = shape

        if color:
            colors[vid] = color

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
                                          {'name': 'transform'})
                matrix = self.SubElement(transfo, 'matrix')
                s = '\n'
                for i in range(4):
                    c = tuple(t.getRow(i))
                    s += '\t\t\t%.5f %.5f %.5f %.5f\n' % c
                matrix.text = s + '\n'

        for (name, value) in properties.get(vid, []).iteritems():
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

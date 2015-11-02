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
from math import sqrt
from math import cos
from math import sin

import xml.etree.ElementTree as xml

from openalea.core.graph.property_graph import PropertyGraph
import openalea.plantgl.all as pgl
Vector3 = pgl.Vector3
Vector4 = pgl.Vector4
Color4Array = pgl.Color4Array


class RootedGraph(PropertyGraph):
    def _set_root(self, root):
        self._root = root

    def _get_root(self):
        return self._root

    root = property(_get_root, _set_root)

class Parser(object):
    edge_type_name = {'successor':'<', 'branch':'+', 'decomposition':'/'}
    geometries = ['Sphere', 'Box', 'Cone', 'Cylinder', 'Frustum', 
                  'sphere', 'box', 'cone', 'cylinder', 'frustum', 
                  'parallelogram', 'Parallelogram', 'TextLabel', 'textLabel', 'PointCloud', 'pointCloud',
		  'polygon', 'Polygon',			
                  'F', 'F0', 'M', 'M0', 'RL', 'RU', 'RH', 'V', 'RV', 'RV0', 'RG', 'RD', 'RO', 'RP', 'RN', 'AdjustLU', 
		  'L', 'LAdd', 'LMul', 'D', 'DAdd', 'DMul', 'P', 'Translate', 'Scale', 'Rotate']
    #turtle_length = -1.
    #turtle_diameter = -1.
    #turtle_color = []
    #turtle_color_setflag = False
    #turtle_tropism = 0.	

    def parse(self, fn):
        self.trash = []
        self._graph = None
        self._scene = None

        # Turtle intialisation
        self._turtle_diameter = -1.
        self._turtle_color = []
	self._turtle_color_setflag = False
	self._turtle_length = -1.
	self._turtle_tropism = 0.
	self._tropism_drectionList = []
	self._tropism_target = []
	self._translateX = self._translateY = self._translateZ = 0.

	#print "pass parse intialisation"
        doc = xml.parse(fn)
        root = doc.getroot()
	
        self.has_type = False
        self.types(doc.findall('type'))
	#self.guiding_edgelist_produce(doc.findall('edge'))
        #print "#####pass to dispatch(root)"
        self.dispatch(root)
        self.scenegraph()

        return self._graph, self._scene

    #def guiding_edgelist_produce(self, elements):
	
	

    def dispatch(self, elt):
	#print "#####pass in dispatch(root)"
	#print "Dispatch elt : ", elt
        #print 'Dispatch elt.tag :', elt.tag 
	#print 'Dispatch elt.attrib',elt.attrib 
	#print "self.__getattribute__(elt.tag)", self.__getattribute__(elt.tag)
	#print "elt.getchildren() : ", elt.getchildren()
	if len(list(elt))>1:
	    list(elt)
	#print 'Dispatch get **', elt.getchildren()
        #return self.__getattribute__(elt.tag)(elt.getchildren(), **elt.attrib)
        try:
            return self.__getattribute__(elt.tag)(list(elt), **elt.attrib)
        except Exception, e:
            print e
            raise Exception("Invalid element %s"%elt.tag)
            

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
        print "pass graph(self, elements) function"
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
	
	#print "node id, type, name = ", id, type, name 
	#print "current turtle_length = ", self._turtle_length

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
        properties2 = [p for p in properties \
                      if p.attrib['name'] not in ('transform', 'color')]
	
        args = self._get_args(properties2)
	if type in ['P', 'PointCloud']:
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
	    if len(colors)>0:
		if len(colors[0].getchildren())!=0:
 		    color = self.color(colors[0].getchildren())
		
		if self._turtle_color_setflag == True:
		    color = self._turtle_color			
	    else:
		raise Exception("color is null!!!")
            
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
        size = Vector3(depth/2, width/2, height/2)
        return (pgl.Translated((0,0,height/2), pgl.Box(size)), 
               pgl.Matrix4.translation(Vector3(0,0,height)))

    def Cone(self, radius=1., height=1., bottom_open= False, **kwds):
        # TODO: Implement bottom_open (bool)
        radius, height = float(radius), float(height)
        solid = not bottom_open
        return (pgl.Cone(radius=radius, height=height, solid=solid),
                pgl.Matrix4.translation(Vector3(0,0,height)))

    def Cylinder(self, radius=1., height=1., bottom_open=False, top_open=False, **kwds):
        #radius, height = float(radius)*10, float(height)*10
        radius, height = float(radius), float(height)
        solid = not(bool(bottom_open) and bool(top_open))
        return (pgl.Cylinder(radius=radius, height=height, solid=solid),
               pgl.Matrix4.translation(Vector3(0,0,height)))

    def Frustum(self, radius=1., height=1., taper=0.5, **kwds):
        radius, height, taper = float(radius), float(height), float(taper)
        bottom_open = kwds.get('bottom_open', False)
        top_open = kwds.get('top_open', False)
        solid = not(bool(bottom_open) and bool(top_open))

        return (pgl.Frustrum(radius=radius, height=height, taper=taper, solid=solid),
                pgl.Matrix4.translation(Vector3(0,0,height)))

    def Parallelogram(self,length=1., width=0.5, **kwds):
        length = float(length)
	width = float(width) 
        #pts = [Vector3(0,0,0), Vector3(length,0,0),Vector3(length, width,0),Vector3(0,width, 0)]
	pts = [Vector3(0,0,0), Vector3(width,0,0),Vector3(width,0,length),Vector3(0,0,length)]
        index = [(0,1,2), (0,2,3)]
        return (pgl.TriangleSet(pts, index),None)

    def TextLabel(self, caption="Default TextLabel", **kwds):
	caption = str(caption)
	return (pgl.Text(caption), None)

    def PointCloud(self, color, points, pointSize, **kwds):
	points = str(points)
	color = str(color)
	points = [float(num) for num in points.split(",")]	
        colorlist = [float(num) for num in color.split(",")]
    	pointSize = float(pointSize)
	pointSize = int(pointSize)
    	if pointSize <= 0:
            pointSize = 1
    	items, chunk = points, 3
    	point3Array = zip(*[iter(items)]*chunk)
    	idx4 = pgl.Index4(int(colorlist[0]*255), int(colorlist[1]*255), int(colorlist[2]*255), int(colorlist[3]*255))
    	lidx4, v3array = [], []
	for item in point3Array:
	     v3array.append(Vector3(item))
	     lidx4.append(idx4)
	c4array = Color4Array(lidx4)
	return (pgl.PointSet(v3array, c4array, pointSize), None)


    def Polygon(self, vertices, **kwds):
	points = str(vertices)
	points = [float(num) for num in points.split(",")]
    	items, chunk = points, 3
    	p3list = zip(*[iter(items)]*chunk)
	p2list = self.project3Dto2D(p3list)	
	pd2list = []
	for i in range(len(p2list)):
    	    pd2list.append({i:p2list[i]})
	indexlist = []
	poly_orientation = self.orientation(p2list)
	
	while len(pd2list) >= 3:
	    for cur in range(len(pd2list)):
	        prev = cur - 1
		nex = (cur + 1) % len(pd2list) # Wrap around on the ends
			# By definition, at least there are two ears;
			# we will iterate at end only if poly_orientation
			# was incorrect.
		if self.determinant(pd2list[cur].values()[0], pd2list[prev].values()[0], pd2list[nex].values()[0]) == poly_orientation and \
			self.no_interior(pd2list[prev].values()[0], pd2list[cur].values()[0], pd2list[nex].values()[0], pd2list, 				poly_orientation):
				# Same orientation as polygon
				# No points inside
				# Add index of this triangle to the index list
		    index = pd2list[prev].keys()[0], pd2list[cur].keys()[0], pd2list[nex].keys()[0]
		    indexlist.append(index)
		    # Remove the triangle from the polygon
		    del(pd2list[cur])
		    break
	return (pgl.TriangleSet(pgl.Point3Array(p3list), indexlist),None)

    def project3Dto2D(self, p3list):
	v01 = Vector3((p3list[1][0]-p3list[0][0]), (p3list[1][1]-p3list[0][1]), (p3list[1][2]-p3list[0][2]))
	v12 = Vector3((p3list[2][0]-p3list[1][0]), (p3list[2][1]-p3list[1][1]), (p3list[2][2]-p3list[1][2]))
	vn = pgl.cross(v01, v12)

	p2s = []
	# cosTheta = A dot B/(|A|*|B|) => if A dot B ==0, then Theta == 90
	# if polygon not || y axis, project it to the y=0 plane   
	if pgl.dot(vn, Vector3(0,1,0)) != 0:
	    for i in range(len(p3list)):
    	        v = p3list[i][0], p3list[i][2]
    	        p2s.append(v)
	    
	else:
	    # if polygon || y axis and z axis (it will perpendicular x axis), project it to the x=0 plane
	    # if polygon || y axis and x axis (it will perpendicular z axis), project it to the z=0 plane
	    # if polygon || y axis, not || x and z (it will not perpendicular z and x axis), project it to the z=0 plane (or x=0 plane)
 	    if pgl.dot(vn, Vector3(0,0,1)) == 0:
	    	for i in range(len(p3list)):
    	            v = p3list[i][1], p3list[i][2]
    	            p2s.append(v)
	
	    else:
	    	for i in range(len(p3list)):
    	            v = p3list[i][0], p3list[i][1]
    	            p2s.append(v)
	return p2s


    def orientation(self, v):
	area = 0.0
     	# Compute the area (times 2) of the polygon
	for i in range(len(v)):
		area += v[i-1][0]*v[i][1] - v[i-1][1]*v[i][0]

	if area >= 0.0:
		return 0
	return 1


    def determinant(self, p1, p2, p3):
	determ = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])

	if determ >= 0:
		return 1
	return 0


    def no_interior(self, p1, p2, p3, v, poly_or):
	for p in v:
		if p.values()[0] == p1 or p.values()[0] == p2 or p.values()[0] == p3:
			# Don't bother checking against yourself
			continue
		if self.determinant(p1, p2, p.values()[0]) == poly_or or \
			self.determinant(p3, p1, p.values()[0]) == poly_or or \
			self.determinant(p2, p3, p.values()[0]) == poly_or:
				# This point is outside
				continue
		# The point is inside
		return False
	# No points inside this triangle
	return True

    def IsSameSide(A, B, C, P):
	AB = B - A
    	AC = C - A
    	AP = P - A
    	v1 = pgl.cross(AB, AC)
    	v2 = pgl.cross(AB, AP)
    	# v1 and v2 should point to the same direction
    	return pgl.dot(v1, v2) >= 0	

    def PointinTriangle(A, B, C, P):
	return IsSameSide(A, B, C, P) and IsSameSide(B, C, A, P) and IsSameSide(C, A, B, P)


    sphere = Sphere
    box = Box
    cone = Cone
    cylinder = Cylinder
    frustrum = Frustum
    parallelogram = Parallelogram
    textLabel = TextLabel
    pointCloud = PointCloud
    
    # Turtle implementation:
    # F0, M, M0, RV, RG, AdjustLU
    def F(self, length=1., diameter=-1., color=14, **kwds):
        height = float(length)	
	self._turtle_length = height
        diameter = float(diameter)
        if diameter == -1.:
            diameter = self._turtle_diameter
            if diameter == -1:
                diameter = self._turtle_diameter = 0.1
        radius = diameter/2.
	ega_color  = 	[(0, 0, 0),
		      	(0, 0, 170),
		      	(0, 170, 0),
			(0, 170, 170),
			(170, 0, 0),
			(170, 0, 170),
			(170, 85, 0),
			(170, 170, 170),
			(85, 85, 85),
			(85, 85, 255),
			(85, 255, 85),
			(85, 255, 255),
			(255, 85, 85),
			(255, 85, 255),
			(255, 255, 85),
			(255, 255, 255),]
	color3 = pgl.Color3(ega_color[int(color)])
	self._turtle_color = color3
	self._turtle_color_setflag = True
        return (pgl.Cylinder(radius=radius, height=height),
               pgl.Matrix4.translation(Vector3(0,0,height)))

    def F0(self, **kwds):
	height = self._turtle_length
	if self._turtle_diameter == -1:
	    self._turtle_diameter = 0.1
	radius = self._turtle_diameter/2.
	if height <= 0 :
	    height = 1.
	return (pgl.Cylinder(radius=radius, height=height),
               pgl.Matrix4.translation(Vector3(0,0,height)))
	
    def M(self, length=1., **kwds):
	height = float(length)	
	return (None, pgl.Matrix4.translation(Vector3(0,0,height)))

    def M0(self, **kwds):
	height = self._turtle_length
	if height <= 0 :
	    height = 1.
	return (None, pgl.Matrix4.translation(Vector3(0,0,height)))

    def RL(self, angle, **kwds):
        # Rotation around the x axis
        angle = radians(float(angle))
        matrix = pgl.Matrix3.axisRotation(Vector3(1,0,0), angle)
        return (None, pgl.Matrix4(matrix))

    def RU(self, angle, **kwds):
        # Rotation around negative y axis <-- NO, Wrong idea
        angle = radians(float(angle))
        matrix = pgl.Matrix3.axisRotation(Vector3(0,1,0), angle)
        return (None, pgl.Matrix4(matrix))

    def RH(self, angle, **kwds):
        # Rotation around the z axis
        angle = radians(float(angle))
        matrix = pgl.Matrix3.axisRotation(Vector3(0,0,1), angle)
        return (None, pgl.Matrix4(matrix))

    def V(self, argument=0., **kwds):
	self._turtle_tropism = float(argument)
	return (None, None) 

    def RV(self, argument=1., **kwds):
        """ Gravitropism. """
	self._turtle_tropism = float(argument)
	return (None, -2)

    def RV0(self, **kwds):
	return (None, -2)

    def RG(self, **kwds):
        """ Maximal gravitropism such that local z-direction points downwards. """
	self._turtle_tropism = 1e10
	return (None, -2)
			
    def RD(self, direction, strength, **kwds):
	self._turtle_tropism = float(strength)
	direction = str(direction)
	self._tropism_drectionList = [float(num) for num in direction.split(",")]	
	return (None, -3)

    def RO(self, direction, strength, **kwds):
	self._turtle_tropism = float(strength)
	direction = str(direction)
	self._tropism_drectionList = [float(num) for num in direction.split(",")]
	return (None, -4)

    def RP(self, target, strength, **kwds):
	self._turtle_tropism = float(strength)
	target = str(target)
	self._tropism_target = [float(num) for num in target.split(",")]
	return (None, -5)

    def RN(self, target, strength, **kwds):
	self._turtle_tropism = float(strength)
	t = str(target)
	self._tropism_target = [float(num) for num in t.split(",")]
	return (None, -5)
	

    def AdjustLU(self, **kwds):
        """ Rotate around local z-axis such that local y-axis points upwards as far as possible."""
        return (None, -1)

    def L(self, length=1., **kwds):
        """ Set the turtle state to the given length. """  	
	self._turtle_length = float(length)
        return (None, None)

    def LAdd(self, argument=1., **kwds):
	self._turtle_length += float(argument)
	return (None, None)

    def LMul(self, argument=1., **kwds):
	self._turtle_length *= float(argument)
	return (None, None)

    def D(self, diameter=0.1, **kwds):
        """ Set the turtle state to the given diameter. """
        self._turtle_diameter = float(diameter)
        return (None, None)

    def DAdd(self, argument=1., **kwds):
	self._turtle_diameter += float(argument)
	return (None, None)

    def DMul(self, argument=1., **kwds):
	self._turtle_diameter *= float(argument)
	return (None, None)

    def P(self, color=14, **kwds):
        """ Set the turtle state to the given color. """
	ega_color  = 	[(0, 0, 0),
		      	(0, 0, 170),
		      	(0, 170, 0),
			(0, 170, 170),
			(170, 0, 0),
			(170, 0, 170),
			(170, 85, 0),
			(170, 170, 170),
			(85, 85, 85),
			(85, 85, 255),
			(85, 255, 85),
			(85, 255, 255),
			(255, 85, 85),
			(255, 85, 255),
			(255, 255, 85),
			(255, 255, 255),]
	color3 = pgl.Color3(ega_color[int(color)])
	self._turtle_color = color3
	self._turtle_color_setflag = True
        #Parser.turtle_color = color3
	#Parser.turtle_color_setflag = True
        return (None, None)

    def Translate(self, translateX=0., translateY=0., translateZ=0., **kwds):
	tx = float(translateX); ty = float(translateY); tz = float(translateZ);
	return (None, pgl.Matrix4.translation(Vector3(tx,ty,tz)))

    def Scale(self, scaleX=0., scaleY=0., scaleZ=0., **kwds):
    	sx = float(scaleX); sy = float(scaleY); sz = float(scaleZ);
	matrix = pgl.Matrix3.scaling(Vector3(sx,sy,sz))
	return (None, pgl.Matrix4(matrix))

    def Rotate(self, rotateX=0., rotateY=0., rotateZ=0., **kwds):
	rx = radians(float(rotateX)); ry = radians(float(rotateY)); rz = radians(float(rotateZ));
	mx = pgl.Matrix3.axisRotation(Vector3(1,0,0), rx)
	my = pgl.Matrix3.axisRotation(Vector3(0,1,0), ry)
	mz = pgl.Matrix3.axisRotation(Vector3(0,0,1), rz)
	matrix = mx*my*mz
	return (None, pgl.Matrix4(matrix))  

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
                raise Exception("Unknow object type %s. Known objects are %s."%(type_name,
                                                         sorted(_types.keys())))
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
        else:
            self._scene = pgl.Scene()

        self.visited = set()

        self._graph.add_vertex_property('final_geometry')
        final_geometry = g.vertex_property("final_geometry")
 
        transfos = [pgl.Matrix4()]
        
        self.traverse2(g.root)
        self._scene.merge( pgl.Scene(final_geometry.values()))
        return self._scene
    
    def traverse2(self, vid):
        from openalea.container.traversal.graph import breadth_first_search
        
        g = self._graph

        edge_type = g.edge_property("edge_type")
        transform = g.vertex_property("transform")

        transfos = {g.root : pgl.Matrix4()}
        
        def parent(vid):
            for eid in g.in_edges(vid):
                if edge_type[eid] in ['<', '+', '/']:
                    return g.source(eid)
            return vid

	def grotation(m, strength):	    
	    t = m.getColumn(2)	 
            v0 = t.x; v1 = t.y; v2 = t.z;
	    q = 1/sqrt(t.x*t.x + t.y*t.y + t.z*t.z)
	    v0 *= q; v1 *= q; v2 *= q;
	    v02 = v0 * v0; v12 = v1 * v1; q = v02 + v12;
	    m00 = m11 = m22 = m10 = m20 = m01 = m21 = m02 = m12 = 0
            local_m = 0
	    if (q < 1e-10) or (v2*v2>0.99999):
	       if v2 * (v2- self._turtle_tropism) < 0:
		   m00 = m.getRow(0).x; m10 = -m.getRow(1).x; m20 = -m.getRow(2).x;
		   m01 = m.getRow(0).y; m11 = -m.getRow(1).y; m21 = -m.getRow(2).y;
		   m02 = m.getRow(0).z; m12 = -m.getRow(1).z; m22 = -m.getRow(2).z;
	       else:
		   m00 = m11 = m22 = 1; m10 = m20 = m01 = m21 = m02 = m12 = 0;
	    else:
	       n = 1/sqrt(1- 2 * strength * v2 + strength * strength)
	       m22 = (1 - strength * v2) * n
	       m02 = strength * v0 * n
	       m20 = -m02
	       m12 = strength * v1 * n
	       m21 = -m12

	       q = 1 / q;
	       m00 = (v12 + v02 * m22) * q;
	       m11 = (v02 + v12 * m22) * q;
	       m01 = m10 = v0 * v1 * (m22 - 1) * q;

	    vec1 = Vector4(m00, m10, m20, 0)
	    vec2 = Vector4(m01, m11, m21, 0)
	    vec3 = Vector4(m02, m12, m22, 0)
	    vec4 = Vector4(0, 0, 0, 1)
	    local_m = pgl.Matrix4(vec1, vec2, vec3, vec4)
	    return local_m

	def invTransformVector(t, v):
	    x = v[0]; y = v[1]; 
	    m00=t.getRow(0).x; m01=t.getRow(0).y; m02=t.getRow(0).z 
	    m10=t.getRow(1).x; m11=t.getRow(1).y; m12=t.getRow(1).z
	    m20=t.getRow(2).x; m21=t.getRow(2).y; m22=t.getRow(2).z
	    d0 = m11 * m22 - m12 * m21;
	    d1 = m12 * m20 - m10 * m22;
	    d2 = m10 * m21 - m11 * m20;
	    d = 1.0/ (m00 * d0 + m01 * d1 + m02 * d2);
	    v[0] = d0 * d * x + (m21 * m02 - m01 * m22) * d * y + (m01 * m12 - m02 * m11) * d * v[2];
	    v[1] = d1 * d * x + (m00 * m22 - m02 * m20) * d * y + (m10 * m02 - m00 * m12) * d * v[2];
	    v[2] = d2 * d * x + (m20 * m01 - m00 * m21) * d * y + (m00 * m11 - m01 * m10) * d * v[2];
	    return

	def setFromAxisAngle(x, y, z, angle):
	    n = sqrt(x*x + y*y + z*z)
	    n = 1/n; x *= n; y *= n; z *= n;	    
	    c = cos(angle); s = sin(angle); omc = 1.0 - c;
	    m00 = c + x*x*omc; m11 = c + y*y*omc; m22 = c + z*z*omc;
	    tmp1 = x*y*omc; tmp2 = z*s; m01 = tmp1 - tmp2; m10 = tmp1 + tmp2;
	    tmp1 = x*z*omc; tmp2 = y*s; m02 = tmp1 + tmp2; m20 = tmp1 - tmp2;
	    tmp1 = y*z*omc; tmp2 = x*s; m12 = tmp1 - tmp2; m21 = tmp1 + tmp2;	
	    m03 = m13 = m23 = m30 = m31 = m32 = 0; m33 = 1;
	    vec1 = Vector4(m00, m10, m20, m30)
	    vec2 = Vector4(m01, m11, m21, m31)
	    vec3 = Vector4(m02, m12, m22, m32)
	    vec4 = Vector4(m03, m13, m23, m33)
	    return pgl.Matrix4(vec1, vec2, vec3, vec4)

	def directionalTropism(m, direction, strength):    
	    t = m.getColumn(2)
	    x = direction[2] * t.y - direction[1] * t.z
	    y = direction[0] * t.z - direction[2] * t.x
	    z = direction[1] * t.x - direction[0] * t.y
	    vec3 = Vector3(x, y, z)
	    angle = strength * sqrt((x*x + y*y + z*z)/(t.x*t.x + t.y*t.y + t.z*t.z))
	    if (angle * angle) >= 1e-20:
		invTransformVector(m, vec3)
		return setFromAxisAngle(vec3.x, vec3.y, vec3.z, angle)
	    else:
		vec1 = Vector4(1,0,0,0); vec2 = Vector4(0,1,0,0); vec3 = Vector4(0,0,1,0); vec4 = Vector4(0,0,0,1)
		return pgl.Matrix4(vec1, vec2, vec3, vec4)

	def orthogonalTropism(m, direction, strength):		
	    t = m.getColumn(2)
	    x = direction[2] * t.y - direction[1] * t.z
	    y = direction[0] * t.z - direction[2] * t.x
	    z = direction[1] * t.x - direction[0] * t.y
	    vec3 = Vector3(x, y, z)
	    angle = -strength * (t.x*direction[0]+t.y*direction[1]+t.z*direction[2])/sqrt(t.x*t.x + t.y*t.y + t.z*t.z)
	    
	    if (angle * angle) >= 1e-20:
		invTransformVector(m, vec3)
		return setFromAxisAngle(vec3.x, vec3.y, vec3.z, angle)
	    else:
		vec1 = Vector4(1,0,0,0); vec2 = Vector4(0,1,0,0); vec3 = Vector4(0,0,1,0); vec4 = Vector4(0,0,0,1)
		return pgl.Matrix4(vec1, vec2, vec3, vec4)

	def positionalTropism(m, target, strength):
	    x = target[0] - m.getRow(0).w; y = target[1] - m.getRow(1).w; z = target[2] - m.getRow(2).w;
	    l = x*x + y*y + z*z
	    t = m.getColumn(2)
	    if l>0:
		xv = z*t.y - y*t.z; yv = x*t.z - z*t.x; zv = y*t.x - x*t.y		
		vec3 = Vector3(xv, yv, zv)
	    	angle = strength * sqrt((xv*xv + yv*yv + zv*zv)/(l*(t.x*t.x + t.y*t.y + t.z*t.z)))
	    	if (angle * angle) >= 1e-20:
		    invTransformVector(m, vec3)
		    return setFromAxisAngle(vec3.x, vec3.y, vec3.z, angle)
	    else:
		vec1 = Vector4(1,0,0,0); vec2 = Vector4(0,1,0,0); vec3 = Vector4(0,0,1,0); vec4 = Vector4(0,0,0,1)
		return pgl.Matrix4(vec1, vec2, vec3, vec4)	
		

        for v in breadth_first_search(g, vid):
            if parent(v) == v and v != g.root:
                print "ERRRORRRR"
                print v
                continue
	    #print "v",v
	    #print "parent(v)", parent(v)
	    #print "transfos", transfos
            #m = transfos[parent(v)]
	    m = transfos.get(parent(v))

	    #print "every m : ", m
            # Transform the current shape with the stack of transfos m from the root.
            # Store the result in the graph.
            self._local2global(v, m)
            local_t = transform.get(v)
		
	    #print "local_t : ", local_t
            if local_t == -1:
                #TODO : AdjustLU
                #debug
                #frame(m, self._scene, color=1)
                t = m.getColumn(3)
                t = Vector3(t.x, t.y, t.z)
                
                m3 = pgl.Matrix3(m)
                x, y, z = Vector3(1,0,0), Vector3(0,1,0), Vector3(0,0,1)
                X, Y, Z = m*Vector4(1,0,0,0), m*Vector4(0,1,0, 0), m*Vector4(0,0,1, 0)
                Z = Vector3(Z.x, Z.y, Z.z)
                new_x = z ^ Z 
                if pgl.normSquared(new_x) > 1e-3:
                    new_y = Z ^ new_x
                    new_x.normalize()
                    new_y.normalize()
                    m = pgl.BaseOrientation(new_x, new_y).getMatrix()
                    m = m.translation(t) * m 
                    #frame(m, self._scene, color=2)
                else:
                    print 'AdjustLU: The two vectors are Colinear'   		
	    elif local_t == -2:
	        #RV and RG
		local_m = grotation(m, self._turtle_tropism)
		m = m * local_m
	    elif local_t == -3:
		#RD
		local_m = directionalTropism(m, self._tropism_drectionList, self._turtle_tropism)
		m = m * local_m
	    elif local_t == -4:
		#RO
		local_m = orthogonalTropism(m, self._tropism_drectionList, self._turtle_tropism)
		m = m * local_m
	    elif local_t == -5:
		#RP and RN
		local_m = positionalTropism(m, self._tropism_target, self._turtle_tropism)
		m = m * local_m	
            elif local_t:
                if local_t.getColumn(3) != Vector4(0,0,0,1):
                    m = m * local_t 
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
            if edge_type[eid] in ['<', '+', '/']:
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
# Debug utility

def frame(matrix, scene, color=1):
    """ Add a frame to the scene.
    The frame is represented by the matrix.
    :param color: allow to distinguish between to identical frames.
    """
    if color == 1:
        r = pgl.Material(pgl.Color3(*(255,0,0)))
        g = pgl.Material(pgl.Color3(*(0,255,0)))
        b = pgl.Material(pgl.Color3(*(0,0,255)))
    else:
        r = pgl.Material(pgl.Color3(*(255,0,255)))
        g = pgl.Material(pgl.Color3(*(255,255,0)))
        b = pgl.Material(pgl.Color3(*(0,0,0)))
        
    cylinder = pgl.Cylinder(radius=0.005, height=1)
    #x = pgl.AxisRotated(Vector3(0,1,0), radians(90), cylinder)
    #y = pgl.AxisRotated(Vector3(1,0,0), radians(-90), cylinder)
    z = cylinder

    #geom_x = transform4(matrix, x)
    #scene.add(pgl.Shape(geom_x, r))
    #geom_y = transform4(matrix, y)
    #scene.add(pgl.Shape(geom_y, g))
    geom_z = transform4(matrix, z)
    scene.add(pgl.Shape(geom_z, b))
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
	dege_type_conv['/'] = 'decomposition'
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
    print "g====", g
    print "scene====", scene
    return g, scene
    

def graph2xml(graph):
    dump = Dumper()
    print "dump===",dump
    print "dump.dump(graph)====",dump.dump(graph)
    return dump.dump(graph)


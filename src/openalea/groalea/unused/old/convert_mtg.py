# -*- coding: utf-8 -*-
# -*- python -*-
#
#       Topological algorithms to convert MTG into xeg format.
#
#       groalea: GroIMP / OpenAlea Communication framework
#
#       Copyright 2015 Goettingen University - CIRAD - INRIA
#
#       File author(s): Christophe Pradal
#
#       File contributor(s):
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       OpenAlea WebSite : http://openalea.gforge.inria.fr
#
###############################################################################

"""

"""
from .topology import RootedGraph
from openalea.plantgl.all import *
from openalea.mtg.aml import *
import numpy as np
import re


def mtg2graph(mtgfile, bgeomfile):
    """ Convert an MTG into a Rooted graph.
    Just topology
    author: Christophe Pradal
    Add _types for types
    """
    mtg, vlist, metamerlist = mappletfiles_pre(mtgfile, bgeomfile)
    rootedgraph = rootedgraph_pre(mtg)

    # add a nid (XEG node id, starts from 1), add the nid-vid map to rootedgraph accordingly
    #nid = 1
    #rootedgraph._vid = {}

    for vid in vlist:
        # add and set vertices
        #if vid not in rootedgraph:
        #rootedgraph._vid[nid] = vid

        rootedgraph.add_vertex(vid)

        mtg_vertex_properties(mtg, vid, rootedgraph)

        metamer_id = Feature(vid, "id")
        metamer = metamerlist[metamer_id]
        geom_metamer_properties(metamer, vid, rootedgraph)

    # set edges and edge_property
    edgeid = 0
    for edge in mtg.edges(mtg.max_scale()):
        edgeid = edgeid + 1
        rootedgraph.add_edge(edge, edgeid)
        edge_type = mtg.EdgeType(edge[0], edge[1])
        rootedgraph.edge_property("edge_type")[edgeid] = edge_type

        if edge_type == "+":
            m = getVertexCoords(edge[0])
            h = getVertexCoords(edge[1])
            l = getVertexCoords(mtg.parent(edge[0]))
            if mtg.parent(edge[0]) == None:
                l = [0, 0, 0]
                # getVertexCoords(mtg.complex(edge[0]))
            pm = np.array(m)
            ph = np.array(h)
            pl = np.array(l)
            
            transmatrix = getMatrixRotateArbitraryLine(pl, pm, ph)
            rootedgraph.vertex_property("transform")[edge[0]] = transmatrix


    # add types
    types = {'':[]}
    rootedgraph._types = types

    # add
    return rootedgraph

def getMatrixRotateArbitraryLine(pl, pm, ph):

    v0 = pm - pl
    v1 = ph - pm
    cos = np.dot(v0,v1)/np.linalg.norm(v0)/np.linalg.norm(v1)
    sin = (1 - cos**2)**0.5
    u = v0[0]; v = v0[1]; w = v0[2]
    a = pl[0]; b = pl[1]; c = pl[2]

    c0 = [u**2 + (v**2 + w**2) * cos, u * v * (1 - cos) + w * sin, u * w * (1 - cos)- v * sin, 0]
    c1 = [u * v * (1 - cos) - w * sin, v**2 + (u**2 + w**2) * cos, v * w * (1 - cos) + u * sin, 0]
    c2 = [u * w * (1 - cos) + v * sin, v * w * (1 - cos) - u * sin, w**2 + (u**2 + v**2) * cos, 0]

    c30 = (a * (v**2 + w**2) - u * (b * v + c * w)) * (1 - cos) + (b * w - c * v) * sin
    c31 = (b * (w**2 + u**2) - v * (a * u + c * w)) * (1 - cos) + (c * u - a * w) * sin
    c32 = (c * (u**2 + v**2) - w * (a * u + b * v)) * (1 - cos) + (a * v - b * u) * sin
    c33 = 1  
    c3 = [c30, c31, c32, c33]

    return Matrix4(c0, c1, c2, c3)
    

def getVertexCoords(vid):
    x = Feature(vid, "XX")
    y = Feature(vid, "YY")
    z = Feature(vid, "ZZ")

    return [x, y, z]


def mappletfiles_pre(mtgfile, bgeomfile):
    mtg = MTG(mtgfile)
    max_scale = mtg.max_scale()
    vlist = mtg.vertices(max_scale)

    scene = Scene(bgeomfile)
    metamerlist = scene.todict()

    return mtg, vlist, metamerlist


def rootedgraph_pre(mtg):
    # one scale
    rootedgraph = RootedGraph()
    rootedgraph._types = None
    mtg2graph = {}
    # for vertices:
    """
        pname = g.vertex_property('name') label
        ptype = g.vertex_property('type') class_type
        properties = g.vertex_property('parameters') properties
		color = g.vertex_property('color') color
		geometry = g.g.vertex_property('geometry')
        geometry or transformation
    """
    # edges
    # edge_type = g.edge_property('edge_type')
    # add initial properties
    rootedgraph.add_vertex_property("name")
    rootedgraph.add_vertex_property("type")
    rootedgraph.add_vertex_property("parameters")
    rootedgraph.add_vertex_property("color")
    rootedgraph.add_vertex_property("geometry")
    rootedgraph.add_vertex_property("transform")
    rootedgraph.add_edge_property("edge_type")

    # add root
    '''
    rootedgraph.root = 1
    '''
    rootedgraph.root = mtg.component_roots_at_scale_iter(mtg.root, scale=mtg.max_scale()).next()
    '''
    if rootedgraph.root not in rootedgraph:
        rootedgraph.add_vertex(rootedgraph.root)
    '''
    return rootedgraph


def mtg_vertex_properties(mtg, vid, rootedgraph):
    # set name
    label = mtg[vid]["label"]
    rootedgraph.vertex_property("name")[vid] = label

    '''
    # set type
    match = re.match(r"([a-zA-Z]+)([0-9]+)", label, re.I)
    if match:
        rootedgraph.vertex_property("type")[vid] = match.groups()[0]
    '''
    # set parameters
    parameters = mtg.get_vertex_property(vid)
    for p in  ['edge_type', 'index', 'label']:
        if p in parameters:
            del parameters[p]

    rootedgraph.vertex_property("parameters")[vid] = parameters

    # set color
    defaultcolor = Color3(255,255,85)
    rootedgraph.vertex_property("color")[vid] = defaultcolor



def geom_metamer_properties(metamer, vid, rootedgraph):

    length = 0.0; radius = 0.0; typem = None; types = ""
    for i in range(len(metamer)):
        if i != 0:
          continue
        shape = metamer[i]
        temp = shape.geometry

        # compute geometry properties of shape in metamer for XEG node
        transcount = 0
        while isinstance(temp, Transformed):
            transcount = transcount + 1
            temp = temp.geometry

        if type(temp) is Cylinder:
           radius = temp.radius
           length = temp.height
           typem = temp
           types = 'F'
        else:
            raise Exception('Unexcepted type!!!!!')

    rootedgraph.vertex_property("geometry")[vid] = typem
    rootedgraph.vertex_property("type")[vid] = types
    paras = {'length': length, 'radius': radius}
    rootedgraph.vertex_property("parameters")[vid] = paras

    return


def geometry2turtle(geometry):
    """ Code from translation, oriented to xeg turtle commands.

    Examples:
      translation (x,y,z) rotation cylinder=> x-xp, y-yp, z-zp (xp, yp, zp for parent coordinate)
      Direction and length

      2 oriented => RU
      RU
    """
    pass




def transGeometry(mtg_vid, bgeom):
    """ Code to do the geometry transformation from MTG object/.mtg and .bgeom files.

    A topology vertex in .mtg file corresponds a geometrical metamer, which contais several shapes in .bgeom file:

    A shape is a transformed PlantGL geometry object:
        Cylinder shape is a Cylinder object, on which 2 transformations (Oriented, Translated) have been applied.
        BezierPatch shape is a BezierPatch object, on which 4 transformations (Oriented, Translated, 2 Scaled) have been applied.
    """
    scene = Scene(bgeom)
    shapes = scene.todict()
    for shape in shapes[mtg_vid]:
        temp = shape.geometry
        while isinstance(temp, Transformed):
            if type(temp) is Oriented:
                tm4 = temp.primary
            temp = temp.geometry


def trans_mtg(mtgfile, bgeomfile):
    mtg = MTG(mtgfile)
    max_scale = mtg.max_scale()
    vlist = VtxList(Scale=max_scale)

    scene = Scene(bgeomfile)
    metamerlist = scene.todict()

    mtg_properties = mtg.properties()
    mtg_properties_list = mtg_properties.keys()
    mtgproperty_diclist = []

    for vertex in vlist:
        try:
            vertex_id = Feature(vertex, "id")

            # compute properties of mtg vertex for XEG node
            dic = {}
            for p in mtg_properties_list:
                dic[p] = mtg[vertex_id][p]

            mtgproperty_diclist.append({vertex_id: dic})

            tm4list = []
            localtm4list = []
            geoproperty_diclist = []
            material_diclist = []

            metamer = metamerlist[vertex_id]

            for i in range(len(metamer)):
                shape = metamer[i]
                temp = shape.geometry

                # compute material properties of shape in metamer for XEG node
                material = shape.appearance
                material_elements_diclist.append(getMat4Shape(material))

                # compute geometry properties of shape in metamer for XEG node
                transcount = 0
                while isinstance(temp, Transformed):
                    transcount = transcount + 1
                    temp = temp.geometry

                geoproperty_diclist.append(getPro4Shape(temp))

                # compute a composite local transformation matrix for each XEG node/shape
                tm4 = np.identity(4)
                flag = transcount

                for j in range(transcount):
                    temp1 = shape
                    for k in range(flag):
                        temp1 = temp1.geometry
                    flag = flag - 1
                    tm4 = getTM4Geo(temp1) * tm4

                tm4list.append(tm4)
                localtm4 = np.identity(4)

                if i == 0:
                    localtm4 = tm4list[0]
                else:
                    localtm4 = tm4list[i] * tm4list[i-1].I

                localtm4list.append(localtm4)


                # compute elementary local transformation matrix and properties of corresponding turtle command for each XEG node/shape
                eltm4list = []
                eltstypelist = []
                ellocaltm4list = []
                for j in range(transcount):
                    temp1 = shape
                    for k in range(flag):
                        temp1 = temp1.geometry
                    flag = flag - 1

                    if type(temp1) is Oriented:
                        eltstypelist.append('Oriented')
                    elif type(geObject) is Translated:
                        eltstypelist.append('Translated')
                    elif type(geObject) is Scaled:
                        eltstypelist.append('Scaled')

                    eltm4list.append(getTM4Geo(temp1))

                for j in range(len(eltm4list)):
                    if j == 0:
                        ellocaltm4list.append(eltm4list[j])
                    else:
                        ellocaltm4 = eltm4list[j] * eltm4list[j-1].I
                        ellocaltm4list.append(ellocaltm4)

                for j in range(len(ellocaltm4list)):
                    if eltstypelist[j] == 'Oriented':
                        tmo = ellocaltm4list[j]
                    elif eltstypelist[j] == 'Translated':
                        x = ellocaltm4list[j].A[3][0]
                        y = ellocaltm4list[j].A[3][1]
                        z = ellocaltm4list[j].A[3][2]
                    elif eltstypelist[j] == 'Scaled':
                        x = ellocaltm4list[j].A[0][0]
                        y = ellocaltm4list[j].A[1][1]
                        z = ellocaltm4list[j].A[2][2]

            print('ok', metamer_id, len(metamerlist[vertex_id]))
        except KeyError:
            print('ERROR',vertex_id)


def getMat4Shape(material):
    property_dic = {}
    ambient_dic = {}
    specular_dic = {}
    emission_dic = {}

    property_dic['name'] = material.name

    ambient_dic['red'] = material.ambient.red
    ambient_dic['green'] = material.ambient.green
    ambient_dic['blue'] = material.ambient.blue
    property_dic['ambient'] = ambient_dic

    property_dic['diffuse'] = material.diffuse

    specular_dic['red'] = material.specular.red
    specular_dic['green'] = material.specular.green
    specular_dic['blue'] = material.specular.blue
    property_dic['specular'] = specular_dic

    emission_dic['red'] = material.emission.red
    emission_dic['green'] = material.emission.green
    emission_dic['blue'] = material.emission.blue
    property_dic['emission'] = emission_dic

    property_dic['shininess'] = material.shininess
    property_dic['transparency'] = material.transparency

    return property_dic


def getTM4Geo(geoObject):
    """
    compute global transformation matrix for each shape in a metamer
    """
    row1=row2=row3=row4=[]

    if type(geObject) is Oriented:
        vp = geObject.primary
        vs = geObject.secondary
        vt = vp^vs
        row1 = [vp.x, vs.x, vt.x, 0]
        row2 = [vp.y, vs.y, vt.y, 0]
        row3 = [vp.z, vs.z, vt.z, 0]
        row4 = [0, 0, 0, 1]

    elif type(geObject) is Translated:
        vt = geObject.translation
        row1 = [1,0,0,vt[0]]
        row2 = [0,1,0,vt[1]]
        row3 = [0,0,1,vt[2]]
        row4 = [0,0,0,1]

    elif type(geObject) is Scaled:
        vs = geObject.scale
        row1 = [vs[0],0,0,0]
        row2 = [0,vs[1],0,0]
        row3 = [0,0,vs[2],0]
        row4 = [0,0,0,1]

    return np.matrix([row1,row2,row3,row4])


def getPro4Shape(geoObject):
    """
    compute properties for each shape in a metamer
    """
    properties = {}
    property_dic = {}

    if type(geoObject) is Cylinder:
        properties[radius] = geoObject.radius
        properties[length] = geoObject.length

        property_dic['Cylinder'] = properties

    elif type(geoObject) is BezierPatch:
        cpm = geoObject.ctrlPointMatrix
        ctlpts = []
        for rowcount in range(cpm.getRowNb()):
            v4ptlist = cpm.getRow(rowcount)
            for i in range(len(v4ptlist)):
                ctlpts.append(v4ptlist[i].x)
                ctlpts.append(v4ptlist[i].y)
                ctlpts.append(v4ptlist[i].z)
                ctlpts.append(v4ptlist[i].w)

        properties[data] = ctlpts
        properties[dimension] = cpm.getRowNb()
        properties[uCount] = cpm.gettColumnNb()

        property_dic['BezierPatch'] = properties

    elif type(geoObject) is Sphere:
        properties[radius] = geoObject.radius

        property_dic['Sphere'] = properties

    return property_dic


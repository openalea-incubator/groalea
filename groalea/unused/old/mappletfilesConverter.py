# -*- coding: utf-8 -*-
# -*- python -*-
#
#       Topological algorithms to convert MTG into xeg format.
#
#       groalea: GroIMP / OpenAlea Communication framework
#
#       Copyright 2015 Goettingen University - CIRAD - INRIA
#
#       File author(s): Long Qinqin
#
#       File contributor(s): Christophe Pradal
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

offset = 2
edgeid = 0
vtypedic = {'T':'Tree', 'G': 'Growth_unit', 'I': 'Internode', 'M': 'Metamer'}

def mappletfiles_pre(mtgfile, bgeomfile):
    mtg = MTG(mtgfile)
    max_scale = mtg.max_scale()
    vlist = mtg.vertices(max_scale)

    scene = Scene(bgeomfile)
    metamerlist = scene.todict()

    return mtg, vlist, metamerlist


def rootedgraph_pre(mtg):

    rootedgraph = RootedGraph()
    rootedgraph._types = None

    # add initial properties
    rootedgraph.add_vertex_property("name")
    rootedgraph.add_vertex_property("type")
    rootedgraph.add_vertex_property("parameters")
    rootedgraph.add_vertex_property("color")
    rootedgraph.add_vertex_property("geometry")
    rootedgraph.add_vertex_property("transform")
    rootedgraph.add_edge_property("edge_type")

    # add root--> need multiscale modification
  
    #rootedgraph.root = 1
    
    max_scale_root = mtg.component_roots_at_scale_iter(mtg.root, scale=mtg.max_scale()).next()
    #rootedgraph.root = max_scale_root * 10**offset
    
    # set root value
    rootedgraph.root = mtg.root

    # add root to graph
    if rootedgraph.root not in rootedgraph:
        rootedgraph.add_vertex(rootedgraph.root)
	
    return rootedgraph, max_scale_root


def addEdge2RootedGraph(rootedgraph, esrc, edest, etype):

    global edgeid

    edgeid = edgeid + 1
    rootedgraph.edge_property("edge_type")[edgeid] = etype
    edge = (esrc, edest) 
    rootedgraph.add_edge(edge, edgeid)


def uppermetamerLevelConvert(mtg, rootedgraph):

    global edgeid
    
    max_scale = mtg.max_scale()

    # add upper metamer level vertices and set properties
	# the metamer level vertices are added for edge adding
    for scale in range(max_scale):
        for vid in mtg.vertices(scale + 1):
            sid = vid * 10**offset
            if sid not in rootedgraph:
                rootedgraph.add_vertex(sid)

            vcomplex = mtg.complex(vid)
            scomplex = vcomplex * 10**offset
            addEdge2RootedGraph(rootedgraph, scomplex, sid, "/")
            # '<' edge in max scale (metamer scale) will be process in metamer level convert
            if scale != max_scale - 1:
                vparent = mtg.parent(vid)
                
                if vparent != None:
                    sparent = vparent * 10**offset
                    etype = mtg.edge_type(vid)
                    addEdge2RootedGraph(rootedgraph, sparent, sid, etype)
			

            # set name
            label = mtg[vid]["label"]
            rootedgraph.vertex_property("name")[sid] = label

            if scale - 1 == max_scale:
                print "sid :", sid, "label : ", label

            # set type
            rootedgraph.vertex_property("type")[sid] = vtypedic[label[0]]

            # set parameters
            parameters = mtg.get_vertex_property(vid)
            for p in  ['edge_type', 'index', 'label', '_line']:
                if p in parameters:
                    del parameters[p]

            rootedgraph.vertex_property("parameters")[sid] = parameters

    
    #add edges
    #for scale in range(max_scale):
        #for vid in mtg.vertices(scale + 1):
            #vcomplex = mtg.complex(vid)
            #addEdge2RootedGraph(rootedgraph, vcomplex, vid, '/')
            # '<' edge in max scale (metamer scale) will be process in metamer level convert
            #if scale != max_scale - 1:
                #vparent = mtg.parent(vid)
                #etype = mtg.edge_type(vid)
                #if vparent != None:
                    #addEdge2RootedGraph(rootedgraph, vparent, vid, etype)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

def convert(mtgfile, bgeomfile):
    global edgeid
    mtg, vlist, metamerlist = mappletfiles_pre(mtgfile, bgeomfile)
    rootedgraph, max_scale_root = rootedgraph_pre(mtg)

    uppermetamerLevelConvert(mtg, rootedgraph)

    for vid in vlist:
        #if vlist.index(vid) == 1:
            #break
        metamer = getmetamer(vid, metamerlist)
        parentvid = mtg.parent(vid)
        if parentvid == None:
            parentmetamer = None
        else:
            parentmetamer = getmetamer(parentvid, metamerlist)    
        sid, edge_type_list_2children_metamers, children_sid_list =  metamerLevelConvert(vid, mtg, rootedgraph)
        subMetamerLevelConvert(max_scale_root, edge_type_list_2children_metamers, children_sid_list, sid, metamer, parentmetamer, rootedgraph)

    return rootedgraph  


def getmetamer(vid, metamerlist):
    metamer_id = Feature(vid, "id")
    metamer = metamerlist[metamer_id]	
    return metamer


def metamerLevelConvert(vid, mtg, rootedgraph):

    global edgeid

    max_scale = mtg.max_scale()
    edges_at_maxscale = list(mtg.edges(max_scale))
    # set new sid using offset
    sid = vid * 10**offset 
    if sid not in rootedgraph:
        rootedgraph.add_vertex(sid)

    #edgeid = 0
    children_sid_list = []
    edge_type_list = []

    # connect root to the first metamer 
    #if mtg.parent(vid) == None:
        #edgeid = edgeid + 1
        #edge_type = '/'
        #rootedgraph.edge_property("edge_type")[edgeid] = edge_type
        #bran_edge = (rootedgraph.root, sid)         
        #rootedgraph.add_edge(bran_edge, edgeid)
    
    # only use [:] form, the edge_at_maxscale is removeable
    for edge in edges_at_maxscale[:]:
        if edge[0] == vid:
            edgeid = edgeid + 1
            edge_type = mtg.EdgeType(edge[0], edge[1])
            rootedgraph.edge_property("edge_type")[edgeid] = edge_type
            # set new metamer id to each edge 
            child_sid = edge[1] * 10**offset
            # if condition may not necessary
            if child_sid not in rootedgraph:
                rootedgraph.add_vertex(child_sid)

            new_edge = (sid, child_sid) 
            rootedgraph.add_edge(new_edge, edgeid)
            edge_type_list.append(edge_type)            
            children_sid_list.append(child_sid)
            edges_at_maxscale.remove(edge)

    # set name
    label = mtg[vid]["label"]
    rootedgraph.vertex_property("name")[sid] = label
    print "sid :", sid, "label : ", label


    # set type
    rootedgraph.vertex_property("type")[sid] = vtypedic[label[0]]

    # set parameters
    parameters = mtg.get_vertex_property(vid)
    for p in  ['edge_type', 'index', 'label', '_line']:
        if p in parameters:
            del parameters[p]

    rootedgraph.vertex_property("parameters")[sid] = parameters

    return sid, edge_type_list, children_sid_list



def subMetamerLevelConvert(max_scale_root, edge_type_list_2children_metamers, children_sid_list, sid, metamer, parentmetamer, rootedgraph):

    geo_lists, colors = getMetamerGeolists(metamer)
    if parentmetamer == None:
        parent_geo_lists = None
    else:
        parent_geo_lists = getMetamerGeolists(parentmetamer)[0]

    if parent_geo_lists == None:
        parent_adjacent_trans_geo_list = None
    else:
        pglen = len(parent_geo_lists) 
        if (pglen == 1) or (pglen == 3) or (pglen == 4):
            parent_adjacent_trans_geo_list = parent_geo_lists[0][1:]
        else:
            # flower does not have children
            raise(Exception)

    trans_geo_lists = []
    shape_geo_pro_list = []
    i = 1
    for geo_list in geo_lists:        
        for geo in geo_list:
            if (sid + i) not in rootedgraph:
                rootedgraph.add_vertex((sid + i))
                # ? try for keyerror 304
                rootedgraph.vertex_property("name")[sid + i] = str(sid + i)
            i = i + 1
        shape_geo = geo_list[0]
        shape_geo_pro_dic = getPro4Shapegeo(shape_geo)
        shape_geo_pro_list.append([shape_geo, shape_geo_pro_dic])
        trans_geo_list = geo_list[1:]
        trans_geo_lists.append(trans_geo_list)       
      
    trans_geo_composite_localmatrix_list = getLocalTM4transgeo(trans_geo_lists, parent_adjacent_trans_geo_list)
	
    metamer_shape_num = len(geo_lists)

    slen = len(shape_geo_pro_list)
    for i in range(slen):
        trans_geo_composite_localmatrix = trans_geo_composite_localmatrix_list[i]
        shape_geo_pro = shape_geo_pro_list[i]
        color = colors[i]
        trans_geo_list = trans_geo_lists[i]

        preshape_geo_sum = 0
        if i != 0:
            for j in range(i):
                preshape_geo_sum = preshape_geo_sum + len(geo_lists[j])

        geo_list_index = i
        addStructure4SubMetamerLevel(max_scale_root, edge_type_list_2children_metamers, children_sid_list, shape_geo_pro, geo_list_index, trans_geo_list, color, trans_geo_composite_localmatrix, sid, metamer_shape_num, preshape_geo_sum, rootedgraph)



def getMetamerGeolists(metamer):
    
    mlen = len(metamer)
    geo_lists = []
    colors = []

    for i in range(mlen):
        shape = metamer[i]
        material = shape.appearance
        shape_color = getColor4Shape(material)
        colors.append(shape_color)
        temp_geo = shape.geometry
        geo_list = [temp_geo]

        # adjust transformation order
        while isinstance(temp_geo, Transformed):
            temp_geo = temp_geo.geometry
            geo_list.append(temp_geo)

        r_geo_list = list(reversed(geo_list))
        geo_lists.append(r_geo_list)

    return geo_lists, colors



def addStructure4SubMetamerLevel(max_scale_root, edge_type_list_2children_metamers, children_sid_list, shape_geo_pro, geo_list_index, trans_geo_list, color, trans_geo_composite_localmatrix, sid, metamer_shape_num, preshape_geo_sum, rootedgraph):

    global edgeid

    # add an '+' edge from graph root to the first geometry object of the first metamer <-- not necessary, the decomposition relation is transitive 
    '''
    if sid == max_scale_root * 10**offset:
    
        idedgeTo = sid + 1            

        bran_edge = (rootedgraph.root, idedgeTo)
        
        if bran_edge not in rootedgraph._edges.values():
            edgeid = edgeid + 1
            edge_type = '+'
            rootedgraph.edge_property("edge_type")[edgeid] = edge_type
            rootedgraph.add_edge(bran_edge, edgeid)
    '''


    # firstly, add node (and its properties) for shape, and edges (outgoing edges (+ or <) and an incoming edge (/))
    shapegeo_id = sid + preshape_geo_sum + len(trans_geo_list) + 1
    if shapegeo_id not in rootedgraph:
        rootedgraph.add_vertex(shapegeo_id)


 
    if type(shape_geo_pro[0]) is BezierPatch:
        rootedgraph.vertex_property("type")[shapegeo_id] = 'BezierSurface'
    else:
        rootedgraph.vertex_property("type")[shapegeo_id] = type(shape_geo_pro[0]).__name__
     
    rootedgraph.vertex_property("parameters")[shapegeo_id] = shape_geo_pro[1]
    rootedgraph.vertex_property("geometry")[shapegeo_id] = shape_geo_pro[0] 


    # set color to shape geometry object
    rootedgraph.vertex_property("color")[shapegeo_id] = Color3(color)


    # add inter-scale edge for shape geometry object
    edgeid = edgeid + 1
    edge_type = "/"
    rootedgraph.edge_property("edge_type")[edgeid] = edge_type
    decom_edge = (sid, shapegeo_id) 
    rootedgraph.add_edge(decom_edge, edgeid)

    
    # add edges from the frist shape
    if geo_list_index == 0:
        # add successive edges (inter-metamer edges) from shape geometry object of current metamer to the first (transformation) geometry object of children metamers
        if (metamer_shape_num != 18) and (len(edge_type_list_2children_metamers) != 0):
            for i in range(len(edge_type_list_2children_metamers)):
                edgeid = edgeid + 1
                edge_type = edge_type_list_2children_metamers[i]
                rootedgraph.edge_property("edge_type")[edgeid] = edge_type
                children_metamer_sid = children_sid_list[i]
                first_children_metamer_component_id = children_metamer_sid + 1

                if first_children_metamer_component_id not in rootedgraph:
                    rootedgraph.add_vertex(first_children_metamer_component_id)
                    # ? try for keyerror 401 - just for one metamer case test
                    rootedgraph.vertex_property("name")[first_children_metamer_component_id] = str(first_children_metamer_component_id)
                    rootedgraph.vertex_property("type")[first_children_metamer_component_id] = ''
               
                succ_edge = (shapegeo_id, first_children_metamer_component_id) 
                rootedgraph.add_edge(succ_edge, edgeid) 

        # process for the every frist cylinder without any transformation
        #if len(trans_geo_list) == 0:
            #return

        # add branch edges (metamer internal edges) from shape geometry to transformation geometry of branching connected shape, for non-Internode metamer
        if metamer_shape_num != 1:
            edgeid = edgeid + 1
            edge_type = '+'
            rootedgraph.edge_property("edge_type")[edgeid] = edge_type
            bran_edge = (shapegeo_id, shapegeo_id + 1) 
            rootedgraph.add_edge(bran_edge, edgeid)

            if metamer_shape_num == 4:
                edgeid = edgeid + 1
                edge_type = '+'
                rootedgraph.edge_property("edge_type")[edgeid] = edge_type
                bran_edge = (shapegeo_id, shapegeo_id + 4 + 1) 
                rootedgraph.add_edge(bran_edge, edgeid)

            elif metamer_shape_num == 18:
                for i in range(9):
                    edgeid = edgeid + 1
                    edge_type = '+'
                    rootedgraph.edge_property("edge_type")[edgeid] = edge_type
                    mf_step = (i+1) * 3 + 1
                    bran_edge = (shapegeo_id, shapegeo_id + mf_step) 
                    rootedgraph.add_edge(bran_edge, edgeid)

                for i in range(5):
                    edgeid = edgeid + 1
                    edge_type = '+'
                    rootedgraph.edge_property("edge_type")[edgeid] = edge_type
                    pt_step = i * 5 + 1 + 10 * 3
                    bran_edge = (shapegeo_id, shapegeo_id + pt_step) 
                    rootedgraph.add_edge(bran_edge, edgeid)

                edgeid = edgeid + 1
                edge_type = '+'
                rootedgraph.edge_property("edge_type")[edgeid] = edge_type
                bran_edge = (shapegeo_id, shapegeo_id + 1 + 5 * 5 + 10 * 3) 
                rootedgraph.add_edge(bran_edge, edgeid)

        # process for the every frist cylinder without any transformation
        if len(trans_geo_list) == 0:
            return
                                  
    elif geo_list_index == 1:
        if metamer_shape_num == 3: 
            edgeid = edgeid + 1
            edge_type = '<'
            rootedgraph.edge_property("edge_type")[edgeid] = edge_type
            succ_edge = (shapegeo_id, shapegeo_id + 1) 
            rootedgraph.add_edge(succ_edge, edgeid)

    elif geo_list_index == 2:
        if metamer_shape_num == 4: 
            edgeid = edgeid + 1
            edge_type = '<'
            rootedgraph.edge_property("edge_type")[edgeid] = edge_type
            succ_edge = (shapegeo_id, shapegeo_id + 1) 
            rootedgraph.add_edge(succ_edge, edgeid)

    elif geo_list_index == 16:
        if metamer_shape_num == 18: 
            edgeid = edgeid + 1
            edge_type = '<'
            rootedgraph.edge_property("edge_type")[edgeid] = edge_type
            succ_edge = (shapegeo_id, shapegeo_id + 1) 
            rootedgraph.add_edge(succ_edge, edgeid)      
                                     

    # secondly add node (and its properties) for transformations of shape, and edges (a < outgoing edge and a / incoming edge)
    for i in range(len(trans_geo_list)):
		
        temp_composite_localmatrix = np.matrix(trans_geo_composite_localmatrix)

        if type(trans_geo_list[i]) is Oriented:
            temp_composite_localmatrix.A[0,3] = temp_composite_localmatrix.A[1,3] = temp_composite_localmatrix.A[2,3] = 0
            templist = temp_composite_localmatrix.transpose().tolist()
            localm = Matrix4(templist[0], templist[1], templist[2], templist[3])
            mlst = localm.data()
            lmstr = serializeList2string(mlst)
            para = {'transform':lmstr}
            trans_type = "Null"

        elif type(trans_geo_list[i]) is Translated:
            #localm = np.identity.matrix(np.identity(4))
            #localm.A[0,3] = temp_composite_localmatrix.A[0,3] 
            #localm.A[1,3] = temp_composite_localmatrix.A[1,3] 
            #localm.A[2,3] = temp_composite_localmatrix.A[2,3]
            #para = {'transform':localm}
            translateX = str(temp_composite_localmatrix.A[0,3]) 
            translateY = str(temp_composite_localmatrix.A[1,3]) 
            translateZ = str(temp_composite_localmatrix.A[2,3])
            para = {'translateX':translateX, 'translateY':translateY, 'translateZ':translateZ}
            trans_type = "Translate"

        elif type(trans_geo_list[i]) is Scaled:          
            localm = np.matrix(np.identity(4))
            #para = {'transform':localm}
            para = {'scaleX':'1', 'scaleY':'1', 'scaleZ':'1'}
            trans_type = "Scale"

        transgeo_id = sid + preshape_geo_sum + i + 1
        #rootedgraph.add_vertex(transgeo_id)
        rootedgraph.vertex_property("type")[transgeo_id] = trans_type
        rootedgraph.vertex_property("parameters")[transgeo_id] = para
        rootedgraph.vertex_property("geometry")[transgeo_id] = trans_geo_list[i]

        edgeid = edgeid + 1
        edge_type = "/"
        rootedgraph.edge_property("edge_type")[edgeid] = edge_type
        decom_edge = (sid, transgeo_id) 
        rootedgraph.add_edge(decom_edge, edgeid)

        edgeid = edgeid + 1
        edge_type = '<'
        rootedgraph.edge_property("edge_type")[edgeid] = edge_type

        if i < (len(trans_geo_list) - 1):
            #if (i == 0) and (geo_list_index == 0):
            succ_edge = (transgeo_id, transgeo_id + 1) 
        else:
            succ_edge = (transgeo_id, shapegeo_id)
 
        rootedgraph.add_edge(succ_edge, edgeid)
     
    

		

def getLocalTM4transgeo(trans_geo_lists, parent_adjacent_trans_geo_list):

    if parent_adjacent_trans_geo_list == None:
        parent_adjacent_trans_geo_product = np.matrix(np.identity(4))
    else:
        parent_adjacent_trans_geo_product = getShapeTransProduct(parent_adjacent_trans_geo_list)

    trans_geo_composite_localmatrix_list = []
    tlen = len(trans_geo_lists)

    for i in range(tlen):      
        trans_geo_product = getShapeTransProduct(trans_geo_lists[i])
        if i == 0:
            trans_geo_composite_localmatrix = trans_geo_product * parent_adjacent_trans_geo_product.I
        else:
            pre_adjacent_trans_geo_product = getShapeTransProduct(trans_geo_lists[i-1])
            trans_geo_composite_localmatrix = trans_geo_product * pre_adjacent_trans_geo_product.I

        trans_geo_composite_localmatrix_list.append(trans_geo_composite_localmatrix)

    return trans_geo_composite_localmatrix_list

'''    
    for trans_geo_list in trans_geo_lists:
        trans_geo_product = getShapeTransProduct(trans_geo_list)
        if trans_geo_lists.index(trans_geo_list) == 0:
            trans_geo_composite_localmatrix = trans_geo_product * parent_adjacent_trans_geo_product.I
        else:
            index_pre_adjacent_trans_geo_list = trans_geo_lists.index(trans_geo_list) - 1
            pre_adjacent_trans_geo_product = getShapeTransProduct(trans_geo_lists[index_pre_adjacent_trans_geo_list])
            trans_geo_composite_localmatrix = trans_geo_product * pre_adjacent_trans_geo_product.I

        trans_geo_composite_localmatrix_list.append(trans_geo_composite_localmatrix)
'''


def getShapeTransProduct(transgeolist):

    tma4 = np.identity(4)
    tm4 = np.matrix(tma4)
    for transgeo in transgeolist:
        transgeo_tm4 = getTM4Transgeo(transgeo)
        tm4 = transgeo_tm4 * tm4
        
    return tm4     

def getColor4Shape(material):
    
    r = material.ambient.red
    g = material.ambient.green
    b = material.ambient.blue

    return (r, g, b)

            
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


def getTM4Transgeo(transgeo):
    """
    compute global transformation matrix for each shape in a metamer
    """
    row1=row2=row3=row4=[]

    if type(transgeo) is Oriented:
        vp = transgeo.primary
        vs = transgeo.secondary
        vt = vp^vs
        row1 = [vp.x, vs.x, vt.x, 0]
        row2 = [vp.y, vs.y, vt.y, 0]
        row3 = [vp.z, vs.z, vt.z, 0]
        row4 = [0, 0, 0, 1]

    elif type(transgeo) is Translated:
        vt = transgeo.translation
        row1 = [1,0,0,vt[0]]
        row2 = [0,1,0,vt[1]]
        row3 = [0,0,1,vt[2]]
        row4 = [0,0,0,1]

    elif type(transgeo) is Scaled:
        vs = transgeo.scale
        row1 = [vs[0],0,0,0]
        row2 = [0,vs[1],0,0]
        row3 = [0,0,vs[2],0]
        row4 = [0,0,0,1]

    return np.matrix([row1,row2,row3,row4])


def getPro4Shapegeo(shapegeo):
    """
    compute properties for a shape in a metamer
    """
    property_dic = {}

    if type(shapegeo) is Cylinder:
        property_dic['radius'] = str(shapegeo.radius)
        property_dic['length'] = str(shapegeo.height)       

    elif type(shapegeo) is BezierPatch:
        cpm = shapegeo.ctrlPointMatrix
        ctlplst = []

        for rowcount in range(cpm.getRowNb()):
            row = cpm.getRow(rowcount)
            for p in row:
                for num in p:
                    ctlplst.append(num)

        ctlpstr = serializeList2string(ctlplst)

        property_dic['data'] = ctlpstr
        property_dic['dimension'] = str(len(cpm.getRow(0)[0]))
        property_dic['uCount'] = str(shapegeo.udegree + 1)

    elif type(shapegeo) is Sphere:
        property_dic['radius'] = str(shapegeo.radius)
        
    return property_dic


def serializeList2string(lst):

    lstr= ""
    lnum = len(lst)
    for i in range(lnum):
        if i == lnum - 1:
            lstr= lstr+ str(lst[i])
        else:  
            lstr= lstr+ str(lst[i]) + ","
            
    return lstr

def getedgeid(edge, rg):
    eid = None
    for i, e in rg._edges.iteritems():
        if e == edge:
            eid = i
    return eid



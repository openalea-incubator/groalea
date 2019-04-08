# -*- coding: utf-8 -*-
# -*- python -*-
#
#       Topological & geometric algorithms to convert a combination of 
#		MAppleT based MTG and Scene into a rootedgraph.
#
#       groalea: GroIMP / OpenAlea Communication framework
#
#       Copyright 2015 Goettingen University - CIRAD - INRIA
#
#       File author(s): Long Qinqin
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       OpenAlea WebSite : http://openalea.gforge.inria.fr
#
###############################################################################


from .topology import RootedGraph
from openalea.plantgl.all import *
from openalea.mtg.aml import *
import numpy as np
import re
import threading

mtg = None
max_scale = None
g_scale_num = None
offset = 2
edgeid = 0
done=None
# It's better to use the class name + "unit" as the upper and metamer scale node type 
#vtypedic = {'T':'Tree', 'G': 'GrowthUnit', 'I': 'Internode', 'M': 'Metamer'}
#geotypes = ['ShadedNull', 'Translate', 'Scale', 'Cylinder', 'BezierPatch', 'Sphere']
#pv2ftrdic = {}

def inputs_pre(mtg_object, scene_object=None):

    global mtg, max_scale

    mtg = mtg_object
    max_scale = mtg.max_scale()
    ms_vlist = mtg.vertices(max_scale)

    if scene_object == None:
        print "scene is none!!!"
        metamerlist = []
    else:
        scene = scene_object
        metamerlist = scene.todict()

    return ms_vlist, metamerlist




def addMTGProperty(mtg, propertyName):
    
    """
    Add a property to all vertices of the mtg object
    """
	
    mtg.add_property(propertyName)

    light_dic = {}
    for vid in mtg.vertices():
        light_dic[vid] = 0

    ps = mtg.properties()
    ps[propertyName] = light_dic

    return mtg


def rootedgraph_pre():

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
    
    # set root value
    rootedgraph.root = mtg.root

    # add root to graph
    if rootedgraph.root not in rootedgraph:
        rootedgraph.add_vertex(rootedgraph.root)

    rootedgraph._types = {}
	
    return rootedgraph


def addEdge(esrc, edest, etype, rootedgraph):

    global edgeid

    edgeid = edgeid + 1
    rootedgraph.edge_property("edge_type")[edgeid] = etype
    edge = (esrc, edest) 
    rootedgraph.add_edge(edge, edgeid)


def setVetexProperties(vid, sid, rootedgraph):

    try:
        # set name
        label = mtg[vid]["label"]
        rootedgraph.vertex_property("name")[sid] = label #+ "." + str(sid)
    except KeyError:
        #pass
        label = " " #mtg[vid]["name"]
        rootedgraph.vertex_property("name")[sid] = label #+ "." + str(sid)
        


    # set type
    class_name = mtg.class_name(vid)
    rootedgraph.vertex_property("type")[sid] = "MtgVertex" #class_name + "_" + "unit" #vtypedic[label[0]]

    # set parameters
    parameters = mtg.get_vertex_property(vid)
    for p in  ['edge_type']: #, 'index', 'label', '_line']:
        if p in parameters:
            del parameters[p]

    # process for mtg format error
    if sid == 1*10**offset and parameters['observation'] == '0.0000\r' :
        parameters['observation'] = 0.0000

    rootedgraph.vertex_property("parameters")[sid] = parameters


def uppermetamerLevelConvert(rootedgraph, scale_num):

    for scale in range(scale_num):
        for vid in mtg.vertices(scale):
            sid = vid * 10**offset
            if sid not in rootedgraph:
                rootedgraph.add_vertex(sid)

            vcomplex = mtg.complex(vid)
            if vcomplex != None:
                setVetexProperties(vid, sid, rootedgraph)
                scomplex = vcomplex * 10**offset
                addEdge(scomplex, sid, "/", rootedgraph)

            # for "<" and "+" edge
            # just get dic for the max scale (metamer scale)
            vparent = mtg.parent(vid)
            if vparent != None:
                sparent = vparent * 10**offset
                # edge_type() gets type of the edge (parent to vid)
                etype = mtg.edge_type(vid)
                addEdge(sparent, sid, etype, rootedgraph)
                
    ve2pdic = {}            
    for vid in mtg.vertices(max_scale):
        sid = vid * 10**offset
        vparent = mtg.parent(vid)
        if vparent != None:
            sparent = vparent * 10**offset
            # edge_type() gets type of the edge (parent to vid)
            etype = mtg.edge_type(vid)
            ve2pdic[vid] = (sid, vparent, etype)
        else:
            ve2pdic[vid] = (sid, vparent, None)

    return ve2pdic                
            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

def mpt_MtgAndScene2rootedgraph(mtg_object, scene_object=None, scale_num=1):
    global g_scale_num
    
    ms_vlist, metamerlist = inputs_pre(mtg_object, scene_object)
    rootedgraph = rootedgraph_pre()
    scale_num = mtg.nb_scales()
    ve2pdic = uppermetamerLevelConvert(rootedgraph, scale_num)
    g_scale_num = scale_num
    #mtg.display()

    #return rootedgraph

    if len(metamerlist) == 0:
        return rootedgraph

    for vid in ms_vlist:
        th = threading.Thread(target=func_metamer_convert, args=(vid, metamerlist, ve2pdic, rootedgraph,))
        th.start()
        th.join()

    return rootedgraph

def func_metamer_convert(vid, metamerlist, ve2pdic, rootedgraph):

    #if ms_vlist.index(vid) == 11:
        #break
    metamer = getmetamer(vid, metamerlist)
    parentvid = mtg.parent(vid)


    if parentvid == None:
        parentmetamer = None
    else:
        parentmetamer = getmetamer(parentvid, metamerlist)    
    #sid, edge_type_list_2children_metamers, children_sid_list =  metamerLevelConvert(vid, rootedgraph)
    sid = ve2pdic[vid][0]
    vparent = ve2pdic[vid][1]
    e2p_type = ve2pdic[vid][2]

    #subMetamerLevelConvert(edge_type_list_2children_metamers, children_sid_list, sid, metamer, parentmetamer, rootedgraph)
    subMetamerLevelConvert(sid, vparent, e2p_type, metamer, parentmetamer, rootedgraph)


def getmetamer(vid, metamerlist):

    property_dic = mtg.get_vertex_property(vid)
    metamer_id = property_dic['id']
    if type(metamer_id) is str:
        metamer_id = int(metamer_id)
    #metamer_id = Feature(vid, "id")
    metamer = metamerlist[metamer_id]

    return metamer


def subMetamerLevelConvert(sid, vparent, e2p_type, metamer, parentmetamer, rootedgraph):
    
    global done

    geo_lists, colors = getMetamerGeolists(metamer)
    if parentmetamer == None:
        parent_geo_lists = None
    else:
        parent_geo_lists = getMetamerGeolists(parentmetamer)[0]

    if parent_geo_lists == None:
        parent_adjacent_trans_geo_list = None
        parent_adjacent_shape_id = None
    else:
        pglen = len(parent_geo_lists)
        parent_adjacent_shape_id = vparent * 10**offset + len(parent_geo_lists[0]) 
        if (pglen == 1) or (pglen == 3) or (pglen == 4):
            parent_adjacent_trans_geo_list = parent_geo_lists[0][1:]
        else:
            # flower does not have children
            raise(Exception)

    trans_geo_lists = []
    shape_geo_pro_list = []
    #done=None
    i = 1
    for geo_list in geo_lists:        
        for geo in geo_list:
            if (sid + i) not in rootedgraph:
                rootedgraph.add_vertex(sid + i)
                #if sid != rootedgraph.root * 10**2
            if g_scale_num != 1:
                addEdge(sid, sid+i, "/", rootedgraph)
            elif done == None:       
                addEdge(rootedgraph.root, sid+1, "<", rootedgraph)
                #global done 
                done = True
            # ? try for keyerror 304
            rootedgraph.vertex_property("name")[sid + i] = "" #"SM" + "." + str(sid + i)
            i = i + 1
        shape_geo = geo_list[0]
        shape_geo_pro_dic = getPro4Shapegeo(shape_geo)
        shape_geo_pro_list.append([shape_geo, shape_geo_pro_dic])
        trans_geo_list = geo_list[1:]
        trans_geo_lists.append(trans_geo_list)       
      
    trans_geo_composite_localmatrix_list = getLocalTM4transgeo(sid, trans_geo_lists, parent_adjacent_trans_geo_list)
	
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
        addStructure4SubMetamerLevel(shape_geo_pro, geo_list_index, trans_geo_list, color, trans_geo_composite_localmatrix, sid, e2p_type, parent_adjacent_shape_id, metamer_shape_num, preshape_geo_sum, rootedgraph)


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



def addStructure4SubMetamerLevel(shape_geo_pro, geo_list_index, trans_geo_list, color, trans_geo_composite_localmatrix, sid, e2p_type, parent_adjacent_shape_id, metamer_shape_num, preshape_geo_sum, rootedgraph):

    global edgeid

    # firstly, add node (and its properties) for shape, and edges (outgoing edges (+ or <) and an incoming edge (/))
    shapegeo_id = sid + preshape_geo_sum + len(trans_geo_list) + 1
    #if shapegeo_id not in rootedgraph:
        #rootedgraph.add_vertex(shapegeo_id)
 
    if type(shape_geo_pro[0]) is BezierPatch:
        rootedgraph.vertex_property("type")[shapegeo_id] = 'BezierSurface'
    else:
        rootedgraph.vertex_property("type")[shapegeo_id] = type(shape_geo_pro[0]).__name__
     
    rootedgraph.vertex_property("parameters")[shapegeo_id] = shape_geo_pro[1]
    rootedgraph.vertex_property("geometry")[shapegeo_id] = shape_geo_pro[0] 

    # not just for a shape with no trans geometry object, now, color is set to shape geometry object for all shapes 
    #if len(trans_geo_list) == 0:
        # if there is no trans geometry object for a shape, then set color to shape geometry object
    rootedgraph.vertex_property("color")[shapegeo_id] = Color3(color)
    
    # add edges from the frist shape of metamer
    if geo_list_index == 0:

        # add branch edges (metamer internal edges) from shape geometry to transformation geometry of branching connected shape, for non-Internode metamer
        if metamer_shape_num != 1:
            addEdge(shapegeo_id, shapegeo_id + 1, '+', rootedgraph)

            if metamer_shape_num == 4:                
                addEdge(shapegeo_id, shapegeo_id + 4 + 1, '+', rootedgraph)

            elif metamer_shape_num == 18:
                for i in range(9):
                    mf_step = (i+1) * 3 + 1
                    addEdge(shapegeo_id, shapegeo_id + mf_step, '+', rootedgraph)

                for i in range(5):
                    pt_step = i * 5 + 1 + 10 * 3
                    addEdge(shapegeo_id, shapegeo_id + pt_step, '+', rootedgraph)

                addEdge(shapegeo_id, shapegeo_id + 1 + 5 * 5 + 10 * 3, '+', rootedgraph)

        # process for the frist cylinder of the tree that have no transformation
        if len(trans_geo_list) == 0:
            return
                                  
    elif geo_list_index == 1:
        if metamer_shape_num == 3: 
            addEdge(shapegeo_id, shapegeo_id + 1, '<', rootedgraph)

    elif geo_list_index == 2:
        if metamer_shape_num == 4: 
            addEdge(shapegeo_id, shapegeo_id + 1, '<', rootedgraph)

    elif geo_list_index == 16:
        if metamer_shape_num == 18: 
            addEdge(shapegeo_id, shapegeo_id + 1, '<', rootedgraph)                                           

    # secondly add node (and its properties) for transformations of shape, and edges (a < outgoing edge and a / incoming edge)
    for i in range(len(trans_geo_list)):
        # add edges (inter-metamer edges, "+" or "<" type) from shape geometry object of parent metamer to the first (transformation) geometry object of current metamer
        if i == 0 and geo_list_index == 0:
            if parent_adjacent_shape_id != None:
                addEdge(parent_adjacent_shape_id, sid+1, e2p_type, rootedgraph)                

        temp_composite_localmatrix = np.matrix(trans_geo_composite_localmatrix)

        transgeo_id = sid + preshape_geo_sum + i + 1

        if type(trans_geo_list[i]) is Oriented:

            temp_composite_localmatrix.A[0,3] = temp_composite_localmatrix.A[1,3] = temp_composite_localmatrix.A[2,3] = 0
            templist = temp_composite_localmatrix.transpose().tolist()
            localm = Matrix4(templist[0], templist[1], templist[2], templist[3])
            mlst = localm.data()
            lmstr = serializeList2string(mlst)
            para = {'transform':localm}
            trans_type = "ShadedNull" 

        elif type(trans_geo_list[i]) is EulerRotated:

            temp_composite_localmatrix.A[0,3] = temp_composite_localmatrix.A[1,3] = temp_composite_localmatrix.A[2,3] = 0
            templist = temp_composite_localmatrix.transpose().tolist()
            localm = Matrix4(templist[0], templist[1], templist[2], templist[3])
            mlst = localm.data()
            lmstr = serializeList2string(mlst)
            para = {'transform':localm}
            trans_type = "ShadedNull"


        elif type(trans_geo_list[i]) is Translated:        
            translateX = translateY = translateZ = str(0)
            para = {'translateX':translateX, 'translateY':translateY, 'translateZ':translateZ}
            trans_type = "Translate"

        elif type(trans_geo_list[i]) is Scaled:          
            localm = np.matrix(np.identity(4))
            # scale has been taken into local rotation (transform matrix of ShadedNull)
            para = {'scaleX':'1', 'scaleY':'1', 'scaleZ':'1'}
            trans_type = "Scale"

        #transgeo_id = sid + preshape_geo_sum + i + 1
        rootedgraph.vertex_property("type")[transgeo_id] = trans_type
        rootedgraph.vertex_property("parameters")[transgeo_id] = para
        rootedgraph.vertex_property("geometry")[transgeo_id] = trans_geo_list[i]
        
        if i < (len(trans_geo_list) - 1): 
            addEdge(transgeo_id, transgeo_id + 1, "<", rootedgraph)
        else:
            addEdge(transgeo_id, shapegeo_id, "<", rootedgraph)



def getLocalTM4transgeo(sid, trans_geo_lists, parent_adjacent_trans_geo_list):

    if parent_adjacent_trans_geo_list == None:
        parent_adjacent_trans_geo_product = np.matrix(np.identity(4))
    else:
        parent_adjacent_trans_geo_product = getShapeTransProduct(parent_adjacent_trans_geo_list)

    trans_geo_composite_localmatrix_list = []
    tlen = len(trans_geo_lists)

    for i in range(tlen):      
        trans_geo_product = getShapeTransProduct(trans_geo_lists[i])

        if i == 0: 
            trans_geo_composite_localmatrix = parent_adjacent_trans_geo_product.I * trans_geo_product      
        else:
            if tlen == 3:
                pre_adjacent_trans_geo_product = getShapeTransProduct(trans_geo_lists[i-1])                           
                trans_geo_composite_localmatrix = pre_adjacent_trans_geo_product.I * trans_geo_product
            elif tlen == 4:
                if i == 2:
                    pre_adjacent_trans_geo_product = getShapeTransProduct(trans_geo_lists[0])                           
                    trans_geo_composite_localmatrix = pre_adjacent_trans_geo_product.I * trans_geo_product
                else:
                    pre_adjacent_trans_geo_product = getShapeTransProduct(trans_geo_lists[i-1])                           
                    trans_geo_composite_localmatrix = pre_adjacent_trans_geo_product.I * trans_geo_product
            elif tlen == 18:
                if i < 17:
                    pre_adjacent_trans_geo_product = getShapeTransProduct(trans_geo_lists[0])                           
                    trans_geo_composite_localmatrix = pre_adjacent_trans_geo_product.I * trans_geo_product
                else:
                    pre_adjacent_trans_geo_product = getShapeTransProduct(trans_geo_lists[i-1])                           
                    trans_geo_composite_localmatrix = pre_adjacent_trans_geo_product.I * trans_geo_product              

        trans_geo_composite_localmatrix_list.append(trans_geo_composite_localmatrix) 

    return trans_geo_composite_localmatrix_list




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



def getTM4Transgeo(transgeo):
    """
    get transformation matrix for each transformation
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

    elif type(transgeo) is EulerRotated:
        azimuth = transgeo.azimuth
        elevation = transgeo.elevation
        roll = transgeo.roll
        cr = np.cos(roll); sr = np.sin(roll); ce = np.cos(elevation)
        se = np.sin(elevation); ca = np.cos(azimuth); sa = np.sin(azimuth)
        row1 = [ca*ce, ca*se*sr - sa*cr, ca*se*cr + sa*sr, 0]
        row2 = [sa*ce, ca*cr + sa*se*sr, sa*se*cr - ca*sr, 0]
        row3 = [-se, ce*sr, ce*cr, 0]
        row4 = [0, 0, 0, 1]

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



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


from .topology import RootedGraph
from openalea.plantgl.all import *
from openalea.mtg.aml import *
import numpy as np
import re
from collections import defaultdict

mtg = None
max_scale = None
g_scale_num = None
offset = 2
edgeid = 0
done=None
vtypedic = {'T':'Tree', 'G': 'GrowthUnit', 'I': 'Internode', 'M': 'Metamer'}
geotypes = ['ShadedNull', 'Translate', 'Scale', 'PTranslate', 'Cylinder', 'BezierPatch', 'Sphere']
#pv2ftrdic = {}

def mappletfiles_pre(mtgfile, bgeomfile=None):

    global mtg, max_scale

    mtg = MTG(mtgfile)
    max_scale = mtg.max_scale()
    ms_vlist = mtg.vertices(max_scale)

    if bgeomfile == None:
        metamerlist = []
    else:
        scene = Scene(bgeomfile)
        metamerlist = scene.todict()

    return ms_vlist, metamerlist


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

    # set name with sid, which allow sid to be restored
    label = mtg[vid]["label"]
    rootedgraph.vertex_property("name")[sid] = label + "." + str(sid)

    # set type
    rootedgraph.vertex_property("type")[sid] = vtypedic[label[0]]

    # set parameters
    parameters = mtg.get_vertex_property(vid)
    for p in  ['edge_type', 'index', 'label', '_line']:
        if p in parameters:
            del parameters[p]

    # process for mtg format error
    if sid == 100 and parameters['observation'] == '0.0000\r' :
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
            #ve2pdic = {}
            vparent = mtg.parent(vid)
            if vparent != None:
                sparent = vparent * 10**offset
                # edge_type() gets type of the edge (parent to vid)
                etype = mtg.edge_type(vid)
                addEdge(sparent, sid, etype, rootedgraph)
                #ve2pdic[vid] = (sid, vparent, etype)
            #else:
                #ve2pdic[vid] = (sid, vparent, None)
                
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


def adjustToGroIMP(scale_num, rootedgraph):
    
    """
    old version, does not allow multi tree case, and remove_edge does not remove vertex
    Adjustment to suite GroIMP's graph syntax: 
    In MTG, edge type from higher scale nodes to lower scale nodes are always decomposition.
    In GroIMP graph, the graph root is Node.0, it connects to the RGGRoot by branch edge
        RGGRoot corresponds to the MTG root. To allow the geometry object in RGGRoot to be rendered,
        RGGRoot must be connected to first node of each scale by branch edge.  
    
    """
    for scale in range(scale_num):
        # get all the first vertex of each scale (without root)
        if scale == 0:
            continue
        vInScale = mtg.vertices(scale)
        fv = vInScale[0]
        # how to find the first vertex if there are many tree and vertices in each tree are not topologically connected?
        # currently the designed topology matches its geometry, geometrically connected == topologically connected => zero translation 
        sfv = fv * 10 ** offset
        sfv = findFirstVertexInScale(sfv, scale, vInScale, rootedgraph)
    
        
        # for the first vertex that is not in the 2nd scale, there is no edge from root to it, so add a new branch edge from root to each
        #if fv != 1:
        if scale != 1:
            addEdge(rootedgraph.root, sfv, "+", rootedgraph)
            # for sub-metamer scale, add root to the first sub-vertex
            if scale == max_scale:
                addEdge(rootedgraph.root, sfv+1, "+", rootedgraph)
        # for the first vertex that is in the 2nd scale, there is already a decomposition edge from root to it, so change its type to branch
        else:
            edgedic = rootedgraph._edges
            for eid in edgedic.keys():
                if edgedic[eid] == (rootedgraph.root, sfv):
                    if rootedgraph.edge_property("edge_type")[eid]== "/":
                        rootedgraph.edge_property("edge_type")[eid]= "+"

    # there is more than 1 plant, (decomposition) edges from root to the non-frist plant need to removed
    if scale_num >1 :
        fs_vs = mtg.vertices(1)
        fs_vs_without_fv = fs_vs[1:]
        
        if len(fs_vs_without_fv) != 0:
            for v in fs_vs_without_fv:
                edgedic = rootedgraph._edges
                for eid in edgedic.keys():
                    if edgedic[eid] == (rootedgraph.root, v):
                        if rootedgraph.edge_property("edge_type")[eid]== "/":
                            #rootedgraph.edge_property("edge_type")[eid]= "+"
                            rootedgraph.remove_edge(eid)


def adjustToGroIMP_new(pv2fvids, max_vid, metamerlist, rootedgraph):

    """
    Adjustment to suite GroIMP's graph syntax: 
        1. To allow the geometry node to be plotted, the edge from graph root to node in a geometry node should not be in decomposition type.
        2. To allow multi-Tree in GroIMP a tree has to be put into a topology branch. 
            A translation node has to be put between the graph root and the first node at each scale of a tree.
    Doing 2 will allows 1. So Just implment for 2 will be enough.

    =====
    pvlist = mtg.vertices(1)
    sidlst_dic = defaultdict(list)
    for pv in pvlist:
        spv = vid2sid(pv) 
        sidlst_dic[spv] = findsvlistFromSVComplex(spv, rootedgraph)

    
    fsidlst_dic = defaultdict(list)
    # process for each plant
    for scompv in sidlst_dic.keys():
        
        #temp_svlist = []
        sidlst = sidlst_dic[scompv]
        edgedic = rootedgraph._edges

        for scale in range(2, scale_num+1):

            # get first vertex of each plant at each scale
            for eid in edgedic.keys():
                if not (edgedic[eid][0] in sidlist and edgedic[eid][1] in sidlist):
                    del edgedic[eid]

            fsid = edgedic.values()[0][0]
            fsid = findFirstVertexInScalePerPlant(fsid, edgedic, rootedgraph)
            fsidlst_dic[scompv].append(fsid) 
            
            # update the decomposed node set to lower scale interactively
            #for sid in sidlst:              
                #temp_svlist = temp_svlist + findsvlistFromSVComplex(sid, rootedgraph)
            #sidlist_dic[scompv] = temp_svlist

    for scompv in sidlst_dic.keys():
        fsid_mscale = fsidlst_dic[scompv][-2]
        fsids_smscale = findsvlistFromSVComplex(fsid_mscale, rootedgraph)
        for fsid_smscale


    
    pvids = mtg.vertices(1)
    #pv2lvs = defaultdict(list)
    pv2fvids = defaultdict(list)
    
    for scale in range(2, scale_num):
        vids = mtg.vertices(scale)

        for pvid in pvids:
            vids_sp = []
            for vid in vids:
                tmpvid = vid
                for i in range(1, scale):
                    vidcomplex = mtg.complex(tmpvid)
                    tmpvid = vidcomplex
                print "tempvid :", tmpvid
                if pvid == tmpvid:
                    vids_sp.append(vid)
            print "pvid, vids_sp : ", pvid, vids_sp
            
            for vid_sp in vids_sp:
                if (mtg.parent(vid_sp) == None) or (mtg.parent(vid_sp) not in vids_sp):
                    pv2fvids[pvid].append(vid_sp)

    print "pvids = ", pvids
    print "pv2fvids : ", pv2fvids

    """
    i = 0
    for pvid in pv2fvids.keys():

        #the decomposition edge from root node to each plant node will not be removed
        #edgedic = rootedgraph._edges
        #for eid in edgedic.keys():
            #if edgedic[eid] == (rootedgraph.root, vid2sid(pvid)) and rootedgraph.edge_property("edge_type")[eid]== "/":
                #rootedgraph.remove_edge(eid)
        

        fvidm = pv2fvids[pvid][-1]
        metamer = getmetamer(fvidm, metamerlist)
        geo_lists, colors = getMetamerGeolists(metamer)

        tm = np.matrix(np.identity(4))
        # get the geometry objects of the first shape
        for geo in geo_lists[0]:
            if type(geo) is Translated:
                tm = getTM4Transgeo(geo) * tm

 
        tx = str(tm.A[0,3]) 
        ty = str(tm.A[1,3]) 
        tz = str(tm.A[2,3])
        #t3 = pv2ftrdic[pvid]

        #para = {'translateX':t3[0], 'translateY':t3[1], 'translateZ':t3[2]}
        para = {'translateX':tx, 'translateY':ty, 'translateZ':tz}
        rootedgraph._types["PTranslate"] = ["Translate"]
        trans_type = "PTranslate"
        
        i = i + 1
        geosid = vid2sid(max_vid + i)
        rootedgraph.add_vertex(geosid)
        rootedgraph.vertex_property("name")[geosid] = "PTranslate" + "." + str(geosid)
        rootedgraph.vertex_property("type")[geosid] = trans_type
        rootedgraph.vertex_property("parameters")[geosid] = para
        rootedgraph.vertex_property("geometry")[geosid] = geo
        addEdge(rootedgraph.root, geosid, "+", rootedgraph)
        
        fvids = pv2fvids[pvid]
        # plant vertex need to be connected from PTranslate as well
        fsids = [vid2sid(pvid)]
        for fvid in fvids:
            fsids.append(vid2sid(fvid))
        #append frist node of a plant at submetamer scale
        fsids.append(fsids[-1]+1)
        print "pvid, fsids : ", pvid, fsids
        for fsid in fsids:
            addEdge(geosid, fsid, "<", rootedgraph)

    return rootedgraph

   
def getPv2fvidsDic(scale_num):

    pvids = mtg.vertices(1)
    #pv2lvs = defaultdict(list)
    pv2fvids = defaultdict(list)
    
    for scale in range(2, scale_num):
        vids = mtg.vertices(scale)

        for pvid in pvids:
            vids_sp = []
            for vid in vids:
                tmpvid = vid
                for i in range(1, scale):
                    vidcomplex = mtg.complex(tmpvid)
                    tmpvid = vidcomplex
                print "tempvid :", tmpvid
                if pvid == tmpvid:
                    vids_sp.append(vid)
            print "pvid, vids_sp : ", pvid, vids_sp
            
            for vid_sp in vids_sp:
                if (mtg.parent(vid_sp) == None) or (mtg.parent(vid_sp) not in vids_sp):
                    pv2fvids[pvid].append(vid_sp)

    return pv2fvids     



def findsvlistFromSVComplex(sid, rootedgraph):

    svlSubScale = []
    edgedic = rootedgraph._edges
    for eid in edgedic.keys():
        if edgedic[eid][0] == sid and rootedgraph.edge_property("edge_type")[eid]== "/": 
            svlSubScale.append(edgedic[eid][1])

    return svlSubScale


def findFirstVertexInScalePerPlant(fsid, edgedic, rootedgraph):
    """
     recursively get the first/root node in a set of node
    """

    for eid in edgedic.keys():
        if edgedic[eid][1] == fsid and rootedgraph.edge_property("edge_type")[eid] != "/":
            fsid = edgedic[eid][0]
            findFirstVertexInScale(fsid, edgedic, rootedgraph)

    return fsid



def addTypeGraph(max_vid, rootedgraph):

    #tsid = vid2sid(max_vid + 1)
    tid = rootedgraph.add_vertex()
    rootedgraph.vertex_property("type")[tid] = 'TypeRoot'
    rootedgraph.vertex_property("name")[tid] = 'TypeRoot' + "." + str(tid)
    rootedgraph.vertex_property("parameters")[tid] = {}
    addEdge(rootedgraph.root, tid, "/", rootedgraph)

    #ssid = vid2sid(max_vid + 2)
    sid = rootedgraph.add_vertex()
    rootedgraph.vertex_property("type")[sid] = 'SRoot'
    rootedgraph.vertex_property("name")[sid] = 'SRoot' + "." + str(sid)
    rootedgraph.vertex_property("parameters")[sid] = {}
    addEdge(rootedgraph.root, sid, "/", rootedgraph)

    for tp in vtypedic.values():
        #tsid = tsid + 1
        tvid = rootedgraph.add_vertex()
        rootedgraph.vertex_property("type")[tvid] = tp
        rootedgraph.vertex_property("name")[tvid] = tp + "." + str(tvid)
        rootedgraph.vertex_property("parameters")[tvid] = {}
        addEdge(tid, tvid, "/", rootedgraph)

        #ssid = ssid + 1
        svid = rootedgraph.add_vertex()
        rootedgraph.vertex_property("type")[svid] = 'S' + tp
        #rootedgraph._types['S' + tp] =  ["ScaleClass"]
        rootedgraph.vertex_property("name")[svid] = 'S' + tp + "." + str(svid)
        rootedgraph.vertex_property("parameters")[svid] = {}
        addEdge(sid, svid, "/", rootedgraph)

        addEdge(svid, tvid, "+", rootedgraph)

    
    smvid = rootedgraph.add_vertex()
    rootedgraph.vertex_property("type")[smvid] = 'S' + 'Sub' + vtypedic['M']
    #rootedgraph._types['S' + 'Sub' + vtypedic['M']] =  ["ScaleClass"]
    rootedgraph.vertex_property("name")[smvid] = 'S' + 'sub' + vtypedic['M'] + "." + str(smvid)
    rootedgraph.vertex_property("parameters")[smvid] = {}
    addEdge(svid, smvid, "/", rootedgraph)

    #msid = tvid
    geotp_vlist = []

    for geotype in geotypes:

        geovid = rootedgraph.add_vertex()
        geotp_vlist.append(geovid)
        if geotype == "BezierPatch":
            geotype = "BezierSurface"
        rootedgraph.vertex_property("type")[geovid] = geotype
        rootedgraph.vertex_property("name")[geovid] = geotype + "." + str(geovid)
        rootedgraph.vertex_property("parameters")[geovid] = {}
        addEdge(tvid, geovid, "/", rootedgraph)
        addEdge(smvid, geovid, "+", rootedgraph)

    for v in geotp_vlist:
        temp_list = list(geotp_vlist)
        temp_list.remove(v)

        for vrest in temp_list:
            addEdge(v, vrest, "+", rootedgraph)
            addEdge(v, vrest, "<", rootedgraph)


    #rootedgraph._types["TypeRoot"] = rootedgraph._types["SRoot"] = ["Null"]

    #return rootedgraph
    

def vid2sid(vid):
    return vid * 10**offset

 
def findFirstVertexInScale(fv, scale, vInScale, rootedgraph):

    edgedic = rootedgraph._edges
    for eid in edgedic.keys():
        if edgedic[eid][1] == fv and rootedgraph.edge_property("edge_type")[eid] != "/":
            fv = edgedic[eid][0]
            findFirstVertexInScale(fv, scale, vInScale, rootedgraph)

    return fv
            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

def convert(mtgfile, bgeomfile=None, scale_num=1):
    global g_scale_num
    
    ms_vlist, metamerlist = mappletfiles_pre(mtgfile, bgeomfile)
    rootedgraph = rootedgraph_pre()
    scale_num = mtg.nb_scales()
    ve2pdic = uppermetamerLevelConvert(rootedgraph, scale_num)
    g_scale_num = scale_num
    #mtg.display()

    #return rootedgraph
    
    pv2fvids = getPv2fvidsDic(scale_num)

    if len(metamerlist) == 0:
        return rootedgraph

    # get the max number of vid and compute sid for node in Type graph from max_vid+1
    max_vid = -1

    for vid in ms_vlist:
        if max_vid < vid:
            max_vid = vid
        #if ms_vlist.index(vid) == 11:
            #break
        metamer = getmetamer(vid, metamerlist)
        parentvid = mtg.parent(vid)
        print "vid, parentvid", vid, parentvid
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

    rg = adjustToGroIMP_new(pv2fvids, max_vid, metamerlist, rootedgraph)

    # type graph has to be added after adjusttoGroIMP
    addTypeGraph(max_vid+1, rg)
    return rootedgraph  


def getmetamer(vid, metamerlist):
    metamer_id = Feature(vid, "id")
    metamer = metamerlist[metamer_id]	
    return metamer

"""
def metamerLevelConvert(vid, rootedgraph):

    global edgeid

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

    setVetexProperties(vid, sid, rootedgraph)

    return sid, edge_type_list, children_sid_list
"""


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
        parent_adjacent_shape_id = vparent * 10**2 + len(parent_geo_lists[0]) 
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
            rootedgraph.vertex_property("name")[sid + i] = "SM" + "." + str(sid + i)
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

    # add inter-scale edge for shape geometry object
    #addEdge(sid, shapegeo_id, "/", rootedgraph)
    
    # add edges from the frist shape of metamer
    if geo_list_index == 0:
        # add edges (inter-metamer edges, "+" or "<" type) from shape geometry object of current metamer to the first (transformation) geometry object of children metamers
        """
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
        """

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

            # process the mapplet file's error (2nd metamer's stem cylinder has been rotated for 180)
            #if transgeo_id == 401 and temp_composite_localmatrix.A[0,0] == -1 and temp_composite_localmatrix.A[1,1] == -1:
                #temp_composite_localmatrix.A[0,0] = temp_composite_localmatrix.A[1,1] = 1

            temp_composite_localmatrix.A[0,3] = temp_composite_localmatrix.A[1,3] = temp_composite_localmatrix.A[2,3] = 0
            templist = temp_composite_localmatrix.transpose().tolist()
            localm = Matrix4(templist[0], templist[1], templist[2], templist[3])
            mlst = localm.data()
            lmstr = serializeList2string(mlst)
            #norm_color = (float(color[0])/float(255), float(color[1])/float(255), float(color[2])/float(255))
            # not set color here anymore, instead, set color to shape geometry object to allow the color conditional upscaling
            #rootedgraph.vertex_property("color")[transgeo_id] = Color3(color)
            para = {'transform':localm}
            trans_type = "ShadedNull" 


        elif type(trans_geo_list[i]) is Translated:
            #fmvid4ps = []
            #for pvid in pv2fvids.keys():
                #fvids = pv2fvids[pvid]
                #fmvid = fvids[-1]
                #vid = sid/(10**offset)
                #if vid == fmvid:

                    #tX = str(temp_composite_localmatrix.A[0,3]) 
                    #tY = str(temp_composite_localmatrix.A[1,3]) 
                    #tZ = str(temp_composite_localmatrix.A[2,3])
                    #pv2ftrdic[pvid] = (tX, tY, tZ)
                
           
            translateX = translateY = translateZ = str(0)
            #if (geo_list_index == 2 and metamer_shape_num == 4) or (geo_list_index == 16 and metamer_shape_num == 18):
                #para = {'translateX':translateX, 'translateY':translateY, 'translateZ':translateZ}
            #else:
                # translate is not necessary (set to 0)
            #para = {'translateX':'0', 'translateY':'0', 'translateZ':'0'}
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

        #addEdge(sid, transgeo_id, "/", rootedgraph)
        #addEdge(rootedgraph.root, sid, "/", rootedgraph)
        
        if i < (len(trans_geo_list) - 1): 
            addEdge(transgeo_id, transgeo_id + 1, "<", rootedgraph)
        else:
            addEdge(transgeo_id, shapegeo_id, "<", rootedgraph)
                    
        """
        edgeid = edgeid + 1
        edge_type = '<'
        rootedgraph.edge_property("edge_type")[edgeid] = edge_type

        if i < (len(trans_geo_list) - 1):
            succ_edge = (transgeo_id, transgeo_id + 1) 
        else:
            succ_edge = (transgeo_id, shapegeo_id)
 
        rootedgraph.add_edge(succ_edge, edgeid)
        """
		

def getLocalTM4transgeo_stem_unitm(sid, trans_geo_lists, parent_adjacent_trans_geo_list):

    if parent_adjacent_trans_geo_list == None:
        parent_adjacent_trans_geo_product = np.matrix(np.identity(4))
    else:
        parent_adjacent_trans_geo_product = getShapeTransProduct(parent_adjacent_trans_geo_list, 0)

    trans_geo_composite_localmatrix_list = []
    tlen = len(trans_geo_lists)

    for i in range(tlen):      
        trans_geo_product = getShapeTransProduct(trans_geo_lists[i], i)

        if i == 0:        
            trans_geo_composite_localmatrix = trans_geo_product * parent_adjacent_trans_geo_product.I
        else:
            pre_adjacent_trans_geo_product = getShapeTransProduct(trans_geo_lists[i-1], i-1)                           
            trans_geo_composite_localmatrix = trans_geo_product * pre_adjacent_trans_geo_product.I

        trans_geo_composite_localmatrix_list.append(trans_geo_composite_localmatrix)

    return trans_geo_composite_localmatrix_list





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





def fixMapplet2ndStemTans(trans_geo_product):

    if trans_geo_product.A[0,0] < 0 and trans_geo_product.A[1,1] < 0 :
        trans_geo_product.A[0,0] = 0 - trans_geo_product.A[0,0]
        trans_geo_product.A[1,1] = 0 - trans_geo_product.A[1,1]

    return trans_geo_product   


def getShapeTransProduct_stem_unitm(transgeolist, trans_geo_lists_index):

    tma4 = np.identity(4)
    tm4 = np.matrix(tma4)
    for transgeo in transgeolist:
        transgeo_tm4 = getTM4Transgeo(transgeo)
        tm4 = transgeo_tm4 * tm4

    if trans_geo_lists_index == 0:
        tm4 = fixMapplet2ndStemTans(tm4)
        
    return tm4 



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



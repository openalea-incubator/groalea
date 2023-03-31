
from collections import defaultdict
from .mptfs2rg import (addEdge, getmetamer, getMetamerGeolists, getTM4Transgeo,
                        mtg, offset, max_scale, vtypedic, geotypes)

##############################################################################


def adjustToGroIMP_old(scale_num, rootedgraph):
    
    """
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



def addTypeGraph(max_vid, rootedgraph):

    tsid = vid2sid(max_vid + 1)
    rootedgraph.add_vertex(tsid)
    rootedgraph.vertex_property("type")[tsid] = 'TypeRoot'
    addEdge(rootedgraph.root, tsid, "/", rootedgraph)

    ssid = vid2sid(max_vid + 2)
    rootedgraph.add_vertex(ssid)
    rootedgraph.vertex_property("type")[ssid] = 'SRoot'
    addEdge(rootedgraph.root, ssid, "/", rootedgraph)

    for tp in vtypedic.values():
        tsid = tsid + 1
        rootedgraph.add_vertex(tsid)
        rootedgraph.vertex_property("type")[tsid] = tp
        addEdge(tsid - 1, tsid, "/", rootedgraph)

        ssid = ssid + 1
        rootedgraph.add_vertex(ssid)
        rootedgraph.vertex_property("type")[ssid] = 'S' + tp
        addEdge(ssid - 1, ssid, "/", rootedgraph)

        addEdge(ssid, tsid, "+", rootedgraph)

    ssid = ssid + 1
    rootedgraph.add_vertex(ssid)
    rootedgraph.vertex_property("type")[ssid] = 'S' + 'sub' + vtypedic['M']
    addEdge(ssid - 1, ssid, "/", rootedgraph)

    msid = tsid
    geotp_vlist = []

    for geotype in geotypes:
        tsid = tsid + 1
        rootedgraph.add_vertex(tsid)
        geotp_vlist.append(tsid)
        rootedgraph.vertex_property("type")[tsid] = tp
        addEdge(msid, tsid, "/", rootedgraph)
        addEdge(ssid, tsid, "+", rootedgraph)

    for v in geotype_vlist:
        temp_list = list(geotype_vlist)
        temp_list.remove(v)

        for vrest in temp_list:
            addEdge(v, vrest, "+", rootedgraph)
            addEdge(v, vrest, "<", rootedgraph)

    
    

def vid2sid(vid):
    return vid * 10**offset

 
def findFirstVertexInScale(fv, scale, vInScale, rootedgraph):

    edgedic = rootedgraph._edges
    for eid in edgedic.keys():
        if edgedic[eid][1] == fv and rootedgraph.edge_property("edge_type")[eid] != "/":
            fv = edgedic[eid][0]
            findFirstVertexInScale(fv, scale, vInScale, rootedgraph)

    return fv



#####################################################################################
#adjustment for full rootedgraph (with submetamer scale) , including removing type graph and adjusment to mtg graph style
#TODO: create a class

def idRestore(rootedgraph):
    



def removeTypeGraph(graph):

    tps = list(vtypedic.values())
    stypes = tps + ["Sub" + tps[-1]]
    stypes = ["Root"] + stypes

    for i in range(len(stypes)):
        stypes[i] = "S" + stypes[i]

    gtypes = ["TypeRoot"] + tps

    vertices = graph._vertieces

    vids = vertices.keys()
    
    for vid in vids:
        vtype = graph.vertex_property('type')[vid]
        if (vtype in geotypes):
            edgedic = graph._edges

            for eid in edgedic.keys():
                esrc = edgedic[eid][0]
                edst = edgedic[eid][1]
                srcType = graph.vertex_property('type')[esrc]
                
                if srcType == stypes[-1] and edst == vid:
                    graph.remove_vertex(vid)


    for vid in vids:
        vtype = graph.vertex_property('type')[vid]
        if (vtype in stypes) or (vtype in gtypes):
            graph.remove_vertex(vid)


def adjustFromGroIMP(rootedgraph):
    
    edgedic = rootedgraph._edges

    del_sids = []

    for eid in edgedic.keys():

        if edgedic[eid][0] == rootedgraph.root and rootedgraph.edge_property("edge_type")[eid]== "+":
            del_sids.append(edgedic[eid][1])
            for eeid in edgedic.keys():

                if edgedic[eid][1] == edgedic[eeid][0] and rootedgraph.edge_property("edge_type")[eeid]== "<":

                    xsid = edgedic[eeid][1]
                    xtype = rootedgraph.vertex_property("type")[xsid]             

                    if xtype == "Tree":
                        translate3_para = rootedgraph.vertex_property("parameters")[edgedic[eid][1]]
                        
                        for eeeid in edgedic.keys():
                            xxsid = edgedic[eeeid][1]
                            xxtype = rootedgraph.vertex_property("type")[xxsid]
                            if edgedic[eid][1] == edgedic[eeeid][0] and rootedgraph.edge_property("edge_type")[eeeid]== "<" and xxtype == "Metamer":
             

                                for aeid in edgedic.keys():
                                    smsid = edgedic[aeid][1]
                                    smtype = rootedgraph.vertex_property("type")[smsid]
                                    if xxsid == edgedic[aeid][0] and rootedgraph.edge_property("edge_type")[aeid]== "/" and smtype == "Translate":
                                        # here, as we assume that for the first metamer of each plant at mtg's finest scale, only has one associated Translation object
                                        # accordingly, code here only support "one association" case 
                                        rootedgraph.vertex_property("parameters")[smsid] = translate3_para


    for del_sid in del_sids:    
        rootedgraph.remove_vertex(del_sid)

    if "PTranslate" in rootedgraph._types.keys():
        del rootedgraph._types["PTranslate"]

    return rootedgraph

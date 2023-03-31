from copy import deepcopy

from openalea.container.traversal.graph import breadth_first_search
from openalea.mtg import MTG, fat_mtg, traversal


#####################################################################################
#TODO: create a class for mtg creating from adjusted rootedgraph

def adjustmentToMtg(rg):
    """
    delete sub-metamer scale and set the sid of each remained node to original vid
    """
    rootedgraph = deepcopy(rg)
    sids = rootedgraph._vertices.keys()
    # for error caused by that root has no name property
    sids.remove(rootedgraph.root)
    for sid in sids:
        if rootedgraph.vertex_property("name")[sid].split(".")[0] == "SM":
            rootedgraph.remove_vertex(sid)

    # set the sid of each remained node to original vid
    mtg_sids = rootedgraph._vertices.keys()
    mtg_sids_edgedic = rootedgraph._edges
    for mtg_sid in mtg_sids:
        mtg_vid = mtg_sid/ 10**2
        # for error caused by root == 0
        if mtg_vid != mtg_sid:
            rootedgraph._vertices[mtg_vid] = rootedgraph._vertices[mtg_sid]
            del rootedgraph._vertices[mtg_sid]

    # set also the edge (for source and destination vetex) sid to vid
    for mtg_eid in mtg_sids_edgedic.keys():
        srcsid = mtg_sids_edgedic[mtg_eid][0]
        dstsid = mtg_sids_edgedic[mtg_eid][1]
        mtg_sids_edgedic[mtg_eid] = (srcsid/10**2, dstsid/10**2)

    return rootedgraph


#######################################################################################


def spanning_mtg(graph):
    """ Extract an MTG from the GroIMP graph.

    Parameters
    ----------
        - graph is a RootedGraph created from the GroIMP Graph

    TODO:
        - compress vertices to only usefull vertices (geometry)
        - compress edges to spanning mtg
    """
    # Manage a mapping between the graph and the mtg
    graph2mtg = {}
    g = graph
    edge_type = g.edge_property('edge_type')

    mtg = MTG()

    mtg.root = graph.root

    # Check if the graph contains decomposition (/) edges
    if not is_multiscale(graph):
        pass

    # Compute the scale from each vertex.
    # Select one decomposition path if there are several in the GroIMP graph
    _scales = scales(g)

    # Set the internal scale information to the MTG
    mtg._scale = _scales

    # Set the edge_type for each vertex (<, +)
    _edge_type = _build_edge_type(g, mtg)

    # Compute the tree information at all scales
    _children_and_parent(g, mtg)

    # Compute the complex (upscale) and components (downscale) for each vertex
    _complex_and_components(g, mtg)

    # Extract all the vertex properties.
    _vertex_properties(g, mtg)

    # Compute missing links to have constant time access (O(1)) to neighbourhood
    fat_mtg(mtg)
    #print "scales :", mtg._scale

    return mtg


def is_multiscale(graph):
    """ Check if the graph has decomposition edges
    """
    g = graph
    edge_type = g.edge_property('edge_type')

    labels = set(edge_label for edge_label in edge_type.itervalues() if edge_label not in ('<', '+'))
    if labels:
        return True
    else:
        return False


def scales(g):
    """ Compute the scale of each vertex inside the graph.
    One solution: If we have several edges of decomposition, we follow the deeper.
    The root is a real node or a false one?
    TODO: specification
    """

    edge_type = g.edge_property("edge_type").copy()

    root = g.root

    has_decomposition = [eid for eid in g.out_edges(root) if edge_type.get(eid) == '/']

    scale0 = 0
    _scale = {root : scale0}
    if not has_decomposition:
        empty_vertex = True
        for name in g.vertex_property_names():
            if g.root in g.vertex_property(name):
                empty_vertex = False
                break
        if empty_vertex:
            root_edge = g.out_edges(root).next()
            edge_type[root_edge] = '/'


    for v in breadth_first_search(g, root):
        parents = list(g.in_neighbors(v))
        edges = list(g.in_edges(v))

        if not parents:
            continue

        decompositions = [eid for eid in edges if edge_type.get(eid) == '/']
        if decompositions:
            eid_max = None
            scale_max = 0
            for eid in decompositions:
                s = _scale[g.source(eid)]
                if s > scale_max:
                    scale_max = s
                    eid_max = eid

            scale = scale_max + 1
        else:
            scale = max(_scale.get(pid,0) for pid in parents)
        _scale[v] = scale

    return _scale

def _build_edge_type(g, mtg):
    edge_type = g.edge_property("edge_type")
    root = g.root

    vertex_edge_type = {}

    for eid, et in edge_type.iteritems():
        source, target = g.source(eid), g.target(eid)
        if source != root and et in ('<', '+'):
            vertex_edge_type[target] = et

    mtg.properties()['edge_type'] = vertex_edge_type
    return vertex_edge_type


def _children_and_parent(g, mtg):
    """
    """
    edge_type = g.edge_property("edge_type")

    parents = {}
    children = {}

    # TODO : filter the edges that belong to the spanning MTG

    for eid, et in edge_type.iteritems():
        source, target = g.source(eid), g.target(eid)
        if source == g.root:
            continue
        if et in ('<', '+'):
            parents[target] = source
            children.setdefault(source,[]).append(target)

    # reorder the children to have < edges at the end of the children
    vet = vertex_edge_type = mtg.properties()['edge_type']
    reorder = {}
    for p, cids in children.iteritems():
        cids = list(cids)
        ets = [vet[cid] for cid in cids]
        nb_less = ets.count('<')
        if nb_less > 1:
            print 'ERROR: %d has more than one successor (%d)' % (p, nb_less)
        elif nb_less == 1:
            n = len(ets)
            index = ets.index('<')
            if index != n - 1:
                vid = cids[index]
                del cids[index]
                cids.append(vid)
                reorder[p] = cids
    #if reorder:
    # print 'REORDER: ', reorder
    children.update(reorder)

    # print parents, children

    parents[g.root] = None

    # Add rooted vertices
    for vid in mtg._scale:
        if vid not in parents:
            parents[vid] = None

    mtg._parent = parents
    mtg._children = children
    #print parents, children


def _complex_and_components(g, mtg):

    edge_type = g.edge_property("edge_type")
    root = g.root
    max_scale = mtg.max_scale()
    scales = mtg._scale

    #print "scales :", scales

    complex = {}
    components = {}
    for eid, et in edge_type.iteritems():
        source, target = g.source(eid), g.target(eid)
        if (source == root) or (et == '/'):
            complex[target] = source
            components.setdefault(source, []).append(target)

            #assert scales[source] == scales[target] - 1
    # Sort all components such that only those that do not have their parent in the same complex are given

    mtg._complex = complex

    # all the components have to be sorted in the pre_order order (parent before children)
    minimum_components = {}
    for vid, components_ids in components.iteritems():
        new_comp = []
        for cid in components_ids:
            pid = mtg.parent(cid)
            if pid not in components_ids:
                new_comp.append(cid)
        minimum_components[vid] = new_comp

    components = minimum_components
    mtg._components = components

def _vertex_properties(g, mtg):
    vp = g._vertex_property
    mtg.properties().update(vp)


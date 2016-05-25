# -*- coding: utf-8 -*-
# -*- python -*-
#
#       Topological algorithms for computing the MTG from the GroIMP graph.
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
from openalea.core.graph.property_graph import PropertyGraph
from openalea.container.traversal.graph import breadth_first_search
from openalea.mtg import MTG, fat_mtg


class RootedGraph(PropertyGraph):
    """ A general graph with a root vertex. """

    def _set_root(self, root):
        self._root = root

    def _get_root(self):
        return self._root

    root = property(_get_root, _set_root)


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

    print "scales :", mtg._scale
    # Extract all the vertex properties.
    _vertex_properties(g, mtg)

    # Compute missing links to have constant time access (O(1)) to neighbourhood
    fat_mtg(mtg)
    print "scales :", mtg._scale

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
    if reorder:
        print 'REORDER: ', reorder
    children.update(reorder)

    print parents, children

    parents[g.root] = None

    mtg._parent = parents
    mtg._children = children
    print parents, children


def _complex_and_components(g, mtg):

    edge_type = g.edge_property("edge_type")
    root = g.root
    max_scale = mtg.max_scale()
    scales = mtg._scale

    print "scales :", scales

    complex = {}
    components = {}
    for eid, et in edge_type.iteritems():
        source, target = g.source(eid), g.target(eid)
        if (source == root) or (et == '/'):
            complex[target] = source
            components.setdefault(source, []).append(target)

            #assert scales[source] == scales[target] - 1

    print complex, components
    mtg._complex = complex
    mtg._components = components

def _vertex_properties(g, mtg):
    vp = g._vertex_property
    mtg.properties().update(vp)



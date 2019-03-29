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
import copy

from openalea.core.graph.property_graph import PropertyGraph
import openalea.plantgl.all as pgl

from .geometry import (TurtleState, FunctionalGeometry, rgb_color,
                       is_matrix, transform4, frame,
                       orientation, project3Dto2D, determinant, no_interior,
                       grotation, directionalTropism, orthogonalTropism, adjust_lu)

from .topology import RootedGraph

from .mappletfilesConverterCopy3 import (vtypedic, geotypes)

Vector3 = pgl.Vector3
Vector4 = pgl.Vector4
Color4Array = pgl.Color4Array

#FUNCTIONAL = -10



####################################################################################
# create scene graph from a rooted graph
# node in rooted graph has been converted by node type correspondances (see dispatch2)

    def setFinalGeometry(self):
        # traverse the graph
        g = self._graph
        #if self._scene:
            #return self._scene
        #else:
            #self._scene = pgl.Scene()

        self.visited = set()

        self._graph.add_vertex_property('final_geometry')
        #final_geometry = g.vertex_property("final_geometry")

        transfos = [pgl.Matrix4()]

        self.traverse2(g.root)
        #self._scene.merge(pgl.Scene(final_geometry.values()))
        #return self._scene



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
            local_turtle = local_turtles.get(v, TurtleState())
            global_turtle = ts.combine(local_turtle)
            return global_turtle

        for v in breadth_first_search(g, vid):
            pid = parent(v)

            if pid == v and v != g.root:
                print "ERRRORRRR"
                print v
                continue
            # print "v",v
            # print "parent(v)", parent(v)
            # print "transfos", transfos
            #m = transfos[parent(v)]
            m = transfos.get(pid)

            if not m:
                m=pgl.Matrix4()

            # CPL
            ts = turtles.get(pid, TurtleState())
            gt = global_turtle = update_turtle(v, ts)
            local_t = transform.get(v)
            if local_t == self.FUNCTIONAL:
                # Get the functional shape to compute the transformation and geometry
                # with the turtle state
                local_t = self.f_shape(v, global_turtle)


            # print "every m : ", m
            # Transform the current shape with the stack of transfos m from the root.
            # Store the result in the graph.
            self._local2global(v, m, global_turtle.color)



            # print "local_t : ", local_t
            if local_t == -1:
                m = adjust_lu(m)
            elif local_t == -2:
                #RV and RG
                local_m = grotation(m, gt.tropism)
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
                # print m
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











##################################################################








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


    
# wrong details, and not take into account of multi-plant case
def adjustFromGroIMP_old(graph):
    eid_max = -1;
    edgedic = graph._edges

    for eid in edgedic.keys():

        if eid_max < eid:
            eid_max = eid

        if edgedic[eid][0] == graph.root:
            fv = edgedic[eid][1]
            vs = []
            for eidd in edgedic.keys():
                if fv == edgedic[eidd][1]:
                    vs.append(eidd)

            if len(vs) != 1:
                graph.remove_edge(eid)
            else:
                graph.edge_property("edge_type")[eid] = "/"
                # add decomposition edge from root to other plant vertex if there are more than one plant
                pvs = [fv]
                cvs = []
                findChildren(pvs, cvs, edgedic, graph) 
                if len(cvs) != 0:
                    new_eid = eid_max
                    for cv in cvs:
                        new_eid =  new_eid + 1
                        graph.edge_property("edge_type")[new_eid] = "/"
                        edge = (graph.root, cv) 
                        graph.add_edge(edge, new_eid)

    return graph


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


# wrong idea
def findChildren(pvs, cvs, edgedic, graph):
    oneGenChv = []

    for pv in pvs:
        for eeid in edgedic.keys():      
            if pv == edgedic[eeid][0] and graph.edge_property("edge_type")[eeid] != "/":
                oneGenChv.append(edgedic[eeid][1])

    if len(oneGenChv) !=0:
        cvs = cvs + oneGenChv
        findChildren(oneGenChv, cvs, edgedic, graph)

    return cvs





def adjustmentToMtg_old(rootedgraph):
    """
    delete sub-metamer scale and set the sid of each remained node to original vid
    """
    # delete nodes in sub-metamer scale
    sids = rootedgraph._vertices.keys()
    edgedic = rootedgraph._edges
    for sid in sids:
        eids = []
        for eid in edgedic.keys(): 
            if edgedic[eid][0] == sid:
                eids.append(eid)
        has_decomp_flag = False
        for eid in eids:
            if rootedgraph.edge_property("edge_type")[eid] == "/":
                has_decomp_flag = True
        if has_decomp_flag == False:
            rootedgraph.remove_vertex(sid)

    # set the sid of each remained node to original vid
    mtg_sids = rootedgraph._vertices.keys()
    mtg_sids_edgedic = rootedgraph._edges
    for mtg_sid in mtg_sids:
        mtg_vid = mtg_sid/ 10**2
        rootedgraph._vertices[mtg_vid] = rootedgraph._vertices[mtg_sid]
        del rootedgraph._vertices[mtg_sid]

    # set also the edge (for source and destination vetex) sid to vid
    for mtg_eid in mtg_sids_edgedic.keys():
        srcsid = mtg_sids_edgedic[mtg_eid][0]
        dstsid = mtg_sids_edgedic[mtg_eid][1]
        mtg_sids_edgedic[mtg_eid] = (srcsid/10**2, dstsid/10**2)

    return rootedgraph

def adjustmentToMtg(rootedgraph):
    """
    delete sub-metamer scale and set the sid of each remained node to original vid
    """
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





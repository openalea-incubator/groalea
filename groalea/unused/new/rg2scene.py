
from copy import deepcopy

import openalea.plantgl.all as pgl
from openalea.container.traversal.graph import breadth_first_search

from .utils import (TurtleState, 
                    transform4, grotation, directionalTropism, orthogonalTropism, adjust_lu)



########################################################################################

# move scene graph part out from the parser class, 
#TODO create a class for scene graph creating from adjusted rootedgraph 
#(fristly get the finest scale graph, then get final geometry, then merge)

def scenegraph(rootedgraph):
    # traverse the graph
    g = deepcopy(rootedgraph)
    g = getSMScale(g)
    scene = pgl.Scene()

    g.add_vertex_property('final_geometry')
    final_geometry = g.vertex_property("final_geometry")

    traverse2(g.root, g)
    scene.merge(pgl.Scene(final_geometry.values()))
    return scene

def getSMScale(rootedgraph):
    g = rootedgraph
    sids = g._vertices.keys().remove(g.root)
    edgedic = g._edges
    for sid in sids:
        if g.vertex_property("name")[sid].split(".")[0] == "SM":
            eset_src = g._vertices[sid][0]
            if len(eset_src) == 1 and g.edge_property("edge_type")[list[eset_src][0]] == "/":
                new_edge = (g.root, sid)
                new_eid = g.add_edge((new_edge)
                g.edge_property("edge_type")[new_eid] = "<"

    for sid in sids:
        if g.vertex_property("name")[sid].split(".")[0] != "SM":
            g.remove_vertex(sid)
    
    return g
    


def traverse2(vid, rootedgraph):
    g = rootedgraph

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
        if local_t == FUNCTIONAL:
            # Get the functional shape to compute the transformation and geometry
            # with the turtle state
            local_t = f_shape(v, global_turtle, g)


        # print "every m : ", m
        # Transform the current shape with the stack of transfos m from the root.
        # Store the result in the graph.
        local2global(v, m, global_turtle.color, g)



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



def f_shape(vid, t, rootedgraph):
    g = rootedgraph
    geometry = g.vertex_property("geometry")
    transform = g.vertex_property("transform")

    shape = geometry.get(vid)
    geom, transfo = shape(t)

    geometry[vid] = geom
    transform[vid] = transfo
    return transfo


def local2global(vid, matrix, color, rootedgraph):
    g = rootedgraph
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

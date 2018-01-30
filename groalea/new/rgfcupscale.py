###################################################################################

def upscaling4Light(rootedgraph):
    """
    aggregate light interception value from submetamer scale (0-many blades) to metamer scale (1 vertex) 
    using color to detact BezierSurface typed blades 
    """
    sids = rootedgraph._vertices.keys()
    sids.remove(rootedgraph.root)
    edgedic = rootedgraph._edges
    for sid in sids:  
        if rootedgraph.vertex_property("type")[sid] == "BezierSurface":
            rgb_color = rootedgraph.vertex_property("color")[sid] 
            if isGreen(rgb_color):
                print " BezierSurface node sid == ", sid
                for eid in edgedic.keys(): 
                    if edgedic[eid][1] == sid and rootedgraph.edge_property("edge_type")[eid]== "/":
                        msid = edgedic[eid][0]
                        rootedgraph.vertex_property("lightInterception")[mid] += rootedgraph.vertex_property("lightInterception")[sid]

    return rootedgraph

def isGreen(rgb_color):
    r=rgb_color.red
    g=rgb_color.green
    b=rgb_color.blue
    if (r*1.5<=g and b*1.5<=g and g!=0):
        return True


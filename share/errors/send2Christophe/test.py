from openalea.plantgl.all import *
from openalea.mtg.plantframe import color

from openalea import groalea
from openalea.groalea.mappletConverter import convert
from openalea.groalea.graphio import (graph2xml, xml2graph,
                                      xmlFile2graph, getSceneXEG,
                                      getMTGRootedGraph)
from openalea.groalea.topology import spanning_mtg

from openalea.deploy import shared_data
from path import Path
import time

data_dir = Path('.')
with_fruit, sphere, with_flower, small = data_dir.glob('*.xeg')
fns = with_fruit, sphere, with_flower, small



# test1

# scenes = []
# for filename in fns:
#     graph, scene = xmlFile2graph(filename, True)
#     xeg_graph = getSceneXEG(graph)
#     _g, scene = xml2graph(xeg_graph)
#     scenes.append(scene)

#     graph, scene = xmlFile2graph(filename, True)
#     _graph = getMTGRootedGraph(graph)
#     g = spanning_mtg(_graph)

#     print g.display()
#     #g.display()

def xeg2mtg(xeg_fn):
    """
    """
    filename = xeg_fn

    graph, scene = xmlFile2graph(filename, False)
    #xeg_graph = getSceneXEG(graph)
    #_g, scene = xml2graph(xeg_graph)

    graph, _scene = xmlFile2graph(filename, True)
    #_graph = getMTGRootedGraph(graph)
    g = spanning_mtg(graph)

    geometries = scene.todict()
    gids = g.property('lstring_id')

    geoms = dict([(v, geometries.get(int(gid))) for v, gid in gids.iteritems()])

    g.properties()['geometry']= geoms

    pconvert(g, float)

    # Visu
    light = g.property('interceptedLightAmount')
    surface = g.property('leaf_area')

    lps = dict([(k, v*0.575/surface[k]) for k, v in light.iteritems() if surface.get(k,0.)])
    g.properties()['light_per_surface'] = lps

    g.properties()['photosynthesis'] = dict([(k, photosynthesis(v)) for k, v in lps.iteritems()])

    color.colormap(g,'photosynthesis', lognorm=False)
    #color.colormap(g,'interceptedLightAmount')
    color.plot3d(g)

    return g

def pconvert(g, _type):
    names = ['interceptedLightAmount',
             'star_pgl',
             'ta_pgl',
             'leaf_area',
             'YY',
             'XX',
             'radius',
             'ZZ',
             'sa_pgl',
             'length',
             'fruit',
             'TopDia']

    properties = g.properties()
    for pname in names:
        if pname not in properties:
            continue
        prop = properties[pname]
        properties[pname] = dict([(k, _type(v)) for k, v in prop.iteritems()])


def photosynthesis(par, alpha=0.03464, Amax= 16.1, Rd = 0.72690):
    x = par
    y = ((alpha * x) / (1+ ((alpha**2) * (x**2) / (Amax**2)) **(.5))) - Rd

    return y






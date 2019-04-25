from openalea.plantgl.all import *
from openalea.mtg.plantframe import color

from groalea.mappletConverter import mpt_MtgAndScene2rootedgraph
from groalea.graphio import (graph2xml, xml2graph,
                             xmlFile2graph, getSceneXEG,
                             getMTGRootedGraph, xeg2MtgAndScene,
                             produceXEGfile)
from groalea.topology import spanning_mtg

from openalea.deploy import shared_data
from path import Path
import time

data_dir = Path('.')
fns = data_dir.glob('*.xeg')
#fns = with_fruit, sphere, with_flower, small
f1, f2 = fns

"""
# TODO:
# 1. recuperer les fruits (vid) et les spheres
# 2. fonction modifiant le rayon d ela sphere en fonction de la photosynthese

fruit: indicate fruit biomass
"""


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

def xeg2mtg(xeg_fn, plot=False):
    """
    """
    filename = xeg_fn

    _file = open(filename,'r')
    g_str = _file.read()
    _file.close()

    g, scene = xeg2MtgAndScene(g_str)
    #graph, scene = xmlFile2graph(filename, False)
    #xeg_graph = getSceneXEG(graph)
    #_g, scene = xml2graph(xeg_graph)

    #graph, _scene = xmlFile2graph(filename, True)
    #_graph = getMTGRootedGraph(graph)
    #g = spanning_mtg(_graph)


    geometries = scene.todict()
    gids = g.property('lstring_id')

    geoms = dict([(v, geometries.get(int(gid))) for v, gid in gids.iteritems()])

    g.properties()['geometry']= geoms

    convert_props(g, float, float_names)
    convert_props(g, int, int_names)

    # Visu
    light = g.property('interceptedLightAmount')
    surface = g.property('leaf_area')

    lps = dict([(k, v*0.575/surface[k]) for k, v in light.iteritems() if surface.get(k,0.)])
    g.properties()['light_per_surface'] = lps

    g.properties()['photosynthesis'] = dict([(k, photosynthesis(v)) for k, v in lps.iteritems()])

    if plot:
        color.colormap(g,'photosynthesis', lognorm=False)
        #color.colormap(g,'interceptedLightAmount')
        color.plot3d(g)

    return g

float_names = ['interceptedLightAmount',
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
int_names = ['id', 'unit_id', 'lstring_id', 'branch_id']

def convert_props(g, _type, pnames):
    names = pnames
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

def fruit_growth(g, NRJ=8.):
    """
    Integrate photosynthesis over the GU (I and the Descendants).
    Multiply by leaf_surface on each of the components

    """

    fruits = g.property('fruit_biomass')
    photos = g.property('photosynthesis')
    surfaces = g.property('leaf_area')

    fids = get_fruits(g)
    inflos = fids

    def fruit_mass(v_fruit):
        v_inflo = g.complex(v_fruit)
        descendants = g.Descendants(v_inflo)
        NRJ = sum(photos.get(c,0.) * surfaces.get(c,0.) for d in descendants for c in g.components(d))
        NRJ = NRJ if NRJ >=0 else 0.
        fruit_id = g.components(v_inflo)[-1]
        fruits[fruit_id]= fruits.get(fruit_id,0.) + energy2biomass(NRJ)

        #print ('NRJ :', NRJ, v_inflo, fruit_id)

    map(fruit_mass, inflos)

    max_biomass = max(fruits.itervalues())

    max_NRJ = 8.
    max_radius = 0.3 + 0.3 * (NRJ/max_NRJ) # 6 cm
    fruit_radius = g.properties()['fruit_radius'] = {}

    geometry = g.property('geometry')

    for vid in fids:
        # get the actual fruit radius
        geoms = geometry[vid]

        radius = 0.
        fruit_shape = None
        for sh in geoms:
            if get_final_shape(sh, 'Sphere'):
                scaled_sh = get_scaled(sh)
                radius = scaled_sh.scale[0]
                fruit_shape = scaled_sh
                break

        #radius = 0.3
        biomass = fruits[vid]
        # Uodate the radius
        new_radius = radius + (biomass / max_biomass) * (max_radius - radius)

        # modify the geometry
        fruit_shape.scale = [new_radius, new_radius, new_radius]
        print(vid, new_radius)
        fruit_radius[vid] = new_radius


def energy2biomass(nrj):
    "TODO: Transform energy to biomass."
    return nrj


def extract_scene(g):
    geoms = g.property('geometry')
    scene = Scene([g for lg in geoms.itervalues()for g in lg])
    return scene

def mtg2xeg(g, scene, xeg_fn):

    rooted_graph = mpt_MtgAndScene2rootedgraph(g, scene)
    xml_g = graph2xml(rooted_graph)

    dname = xeg_fn.dirname()
    name = xeg_fn.namebase
    ext = xeg_fn.ext

    files = Path('result').glob('%s_*.xeg'%name)
    def num(fn):
        name = fn.namebase
        _id=name.split('_')[-1]
        return int(_id)

    l = name.split('_')
    if files:
        new_id = max(num(f) for f in files)

        l[-1] = str(new_id+1)
        name = ('_').join(l)
    else:
        name = name + '_1'


    fn = dname/'result'/name+ext
    produceXEGfile(xml_g, fn)

    return fn


def getgeom(sh):
    if hasattr(sh,'geometry'):
        return getgeom(sh.geometry)
    else:
        return sh

def sType(geom):
    return type(geom).__name__

def get_final_shape(shape, _type='Sphere'):
    geom = getgeom(shape)
    if sType(geom) == _type:
        return geom
    else:
        return None

def get_scaled(shape):
    geom = None
    _geom = shape
    while hasattr(_geom,'geometry'):
        _geom = _geom.geometry
        if sType(_geom) == 'Scaled':
            geom = _geom
    return geom


def get_fruits(g):
    ids =g.property('id')
    fruit = g.property('fruit')

    fruit_ids = [vid for vid, f in fruit.iteritems() if f != 0.]
    return fruit_ids


def run(fn):
    g = xeg2mtg(fn)
    fruit_growth(g, NRJ=5.)
    scene = extract_scene(g)

    #fruits_sphere = detect_fruits(g, scene)
    del g.properties()['geometry']
    fn1 = mtg2xeg(g, scene, fn)
    return g, scene

if __name__ == '__main__':
    #g = xeg2mtg(f4)
    #fruit_growth(g)
    #scene = extract_scene(g)
    #f41 = mtg2xeg(g, scene, f4)
    pass
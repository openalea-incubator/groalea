from openalea.plantgl.all import *
from openalea.deploy.shared_data import shared_data
from openalea.mtg import *

import openalea.groalea
from openalea.groalea.graphio import *
from openalea.groalea import convert_mtg

data = shared_data(openalea.groalea)/'..'/'mapplet'

def read_mtg():
    mtg_file = data/'mapplet.mtg'
    geom_file = data/'mapplet.bgeom'
    g = MTG(mtg_file)
    scene = Scene()
    scene.read(str(geom_file.abspath()))

    return g, scene

def display_turtle(g, vid):
    """ Print the turtle command of a MTG vertex. """

    pid = g.parent(vid)
    geometries = g.property('geometry')

    if pid is None:


def test1():
    g, scene = read_mtg()
    max_scale = g.max_scale()

    geom_ids = g.property('id')

    geom_shapes = scene.todict()

    # add geometry object to MTG
    geometries = g.properties()['geometry'] = {}
    color = g.properties()['color'] = {}

    for vid, gid in geom_ids.items():
        # select only geometries at max_scale
        if g.scale(vid) != max_scale:
            continue

        shape = geom_ids[gid]
        geometries[vid] = shapes[0]

        display_turtle(g, vid)

    # We have several shapes for one vertex. Do we add automatically a new scale in the MTG?

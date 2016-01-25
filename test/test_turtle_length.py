# Author: C. Pradal
# Licence : GPL
import openalea.groalea
from openalea.groalea.graphio import *
from openalea.plantgl.all import *
from openalea.deploy.shared_data import shared_data

from math import fabs

data = None
def setup():
    global data
    data = shared_data(openalea.groalea)
    print 'Data ', data


def teardown():
    pass

def geometries(g):
    return g.vertex_property('final_geometry')

def color3s(g):
    return g.vertex_property('color')

def test_length():
    data = shared_data(openalea.groalea)
    f1 = data/'length'/'L_XEG_r.xml'
    f2 = data/'length'/'L_XEG_w.xml'
    parser = Parser()
    g1, scene1 = parser.parse(f1)
    parser = Parser()
    g2, scene2 = parser.parse(f2)


    assert len(g1) == len(g2)
    assert fabs(volume(scene1) - volume(scene2)) < 1e-5, 'Not the same volume %f'%fabs(volume(scene1) - volume(scene2))


def test_DMul_branch():
    data = shared_data(openalea.groalea)
    f1 = data/'DMul_Branch'/'DMul_Branch_3Run_Interface_XEG.xml'

    parser = Parser()
    g1, scene1 = parser.parse(f1)	

    sh1 = geometries(g1)
    assert fabs(volume(sh1[25]) - volume(sh1[13])) < 1e-5, 'Not the same volume %f'%fabs(volume(sh1[25]) - volume(sh1[13]))

def test_LMul():
    data = shared_data(openalea.groalea)
    f1 = data/'LMul'/'LMul_2Run_Interface_XEG.xml'

    parser = Parser()
    g1, scene1 = parser.parse(f1)	

    sh1 = geometries(g1)
    assert fabs(volume(sh1[9]) - volume(sh1[11])) < 1e-5, 'Not the same volume %f'%fabs(volume(sh1[25]) - volume(sh1[13]))

def test_LMul_Branch():
    data = shared_data(openalea.groalea)
    f1 = data/'LMul_Branch'/'LMul_Branch_3Run_Interface_XEG.xml'

    parser = Parser()
    g1, scene1 = parser.parse(f1)	

    sh1 = geometries(g1)
    assert fabs(volume(sh1[14]) - volume(sh1[21])) < 1e-5, 'Not the same volume %f'%fabs(volume(sh1[25]) - volume(sh1[13]))

def test_p():
    data = shared_data(openalea.groalea)
    f1 = data/'P'/'P_3Run_Interface_XEG.xml'

    parser = Parser()
    g1, scene1 = parser.parse(f1)	

    c3s1 = color3s(g1)
    for i in [20, 21, 10, 18]:
        assert c3s1[i] != c3s1[4]


def test_p():
    data = shared_data(openalea.groalea)
    f1 = data/'L'/'L.xml'

    parser = Parser()
    g1, scene1 = parser.parse(f1)	

    c3s1 = color3s(g1)
    for i in [20, 21, 10, 18]:
        assert c3s1[i] != c3s1[4]



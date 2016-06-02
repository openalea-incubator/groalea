# Author: C. Pradal
# Licence : GPL

from openalea.plantgl.all import *
from openalea.deploy.shared_data import shared_data

import openalea.groalea
from openalea.groalea.graphio import *
from openalea.groalea import topology as topo

data = None


def setup():
    global data
    data = shared_data(openalea.groalea)
    print 'Data ', data


def teardown():
    pass


def test_length():
    data = shared_data(openalea.groalea)
    f = data / 'length' / 'L_XEG_r.xml'
    parser = Parser()
    g, scene = parser.parse(f)

    mtg = topo.spanning_mtg(g)
    return mtg, g, scene


def test_DMul_branch():
    data = shared_data(openalea.groalea)
    f = data/'DMul_Branch'/'DMul_Branch_3Run_Interface_XEG.xml'

    parser = Parser()
    g, scene = parser.parse(f)

    mtg = topo.spanning_mtg(g)
    return mtg, g, scene


def test_LMul():
    data = shared_data(openalea.groalea)
    f = data/'LMul'/'LMul_2Run_Interface_XEG.xml'

    parser = Parser()
    g, scene = parser.parse(f)

    mtg = topo.spanning_mtg(g)
    return mtg, g, scene


def test_LMul_Branch():
    data = shared_data(openalea.groalea)
    f = data/'LMul_Branch'/'LMul_Branch_3Run_Interface_XEG.xml'

    parser = Parser()
    g, scene = parser.parse(f)

    mtg = topo.spanning_mtg(g)
    return mtg, g, scene

def test_p():
    data = shared_data(openalea.groalea)
    f = data/'P'/'P_3Run_Interface_XEG.xml'

    parser = Parser()
    g, scene = parser.parse(f)

    mtg = topo.spanning_mtg(g)
    return mtg, g, scene

def test_msc2():
    f = 'ex_msc2.xeg'
    parser = Parser()
    g, scene = parser.parse(f)

    mtg = topo.spanning_mtg(g)

    # assert mtg.scales() == [0,1,2]
    return mtg, g, scene



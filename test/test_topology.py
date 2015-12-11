# Author: C. Pradal
# Licence : GPL
import openalea.groalea
from openalea.groalea.graphio import *
from openalea.plantgl.all import *
from openalea.deploy.shared_data import shared_data


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
    return g, scene


def test_DMul_branch():
    data = shared_data(openalea.groalea)
    f = data/'DMul_Branch'/'DMul_Branch_3Run_Interface_XEG.xml'

    parser = Parser()
    g, scene = parser.parse(f)
    return g, scene


def test_LMul():
    data = shared_data(openalea.groalea)
    f = data/'LMul'/'LMul_2Run_Interface_XEG.xml'

    parser = Parser()
    g, scene = parser.parse(f)
    return g, scene


def test_LMul_Branch():
    data = shared_data(openalea.groalea)
    f = data/'LMul_Branch'/'LMul_Branch_3Run_Interface_XEG.xml'

    parser = Parser()
    g, scene = parser.parse(f)
    return g, scene

def test_p():
    data = shared_data(openalea.groalea)
    f = data/'P'/'P_3Run_Interface_XEG.xml'

    parser = Parser()
    g, scene = parser.parse(f)
    return g, scene

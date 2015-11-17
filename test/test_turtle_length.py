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


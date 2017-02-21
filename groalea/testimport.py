from .testimportdef import (globaltestdef, y)

def test(rg):
    rg.add_vertex(10)
    from pprint import pprint
    pprint(vars(rg))


def testglobal():
    print globaltestdef()

def testvar():
    print y+8

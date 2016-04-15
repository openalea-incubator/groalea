# -*- coding: utf-8 -*-
# -*- python -*-
#
#       Topological algorithms to convert MTG into xeg format.
#
#       groalea: GroIMP / OpenAlea Communication framework
#
#       Copyright 2015 Goettingen University - CIRAD - INRIA
#
#       File author(s): Christophe Pradal
#
#       File contributor(s):
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       OpenAlea WebSite : http://openalea.gforge.inria.fr
#
###############################################################################

"""

"""
from .topology import RootedGraph



def mtg2graph(g):
    """ Convert an MTG into a Rooted graph.

    Just topology
    author: Christophe Pradal

    Add _types for types

    """
    max_scale = g.max_scale

    # one scale
    graph = RootedGraph()
    graph._types = None

    mtg2graph = {}

    # for vertices:
    """
        pname = g.vertex_property('name') label
        ptype = g.vertex_property('type') class_type
        properties = g.vertex_property('parameters') properties

        geometry or transformation
    """

    # edges
    # edge_type = g.edge_property('edge_type')


def geometry2turtle(geometry):
    """ Code from translation, oriented to xeg turtle commands.

    Examples:
      translation (x,y,z) rotation cylinder=> x-xp, y-yp, z-zp (xp, yp, zp for parent coordinate)
      Direction and length

      2 oriented => RU
      RU

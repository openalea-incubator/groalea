# -*- coding: utf-8 -*-
# -*- python -*-
#
#       Topological algorithms for computing the MTG from the GroIMP graph.
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
from openalea.core.graph.property_graph import PropertyGraph


class RootedGraph(PropertyGraph):
    """ A general graph with a root vertex. """

    def _set_root(self, root):
        self._root = root

    def _get_root(self):
        return self._root

    root = property(_get_root, _set_root)


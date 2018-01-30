from openalea.core.graph.property_graph import PropertyGraph

x = 0
y = 1
class RootedGraph(PropertyGraph):
    """ A general graph with a root vertex. """

    def _set_root(self, root):
        self._root = root

    def _get_root(self):
        return self._root

    root = property(_get_root, _set_root)


def getrg():
    return RootedGraph()

def globaltestdef():
    global x
    x = x + 1
    return x


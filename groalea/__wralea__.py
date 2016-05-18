
# This file has been generated at Thu Aug 14 18:24:24 2008

from openalea.core import *


__name__ = 'groalea'

__editable__ = True
__description__ = 'OpenAlea / GroIMP communication.'
__license__ = 'CECILL-C'
__url__ = 'http://openalea.gforge.inria.fr'
__alias__ = ['groimp']
__version__ = '0.0.1'
__authors__ = 'C. Pradal'
__institutes__ = 'INRIA/CIRAD/University of Cottbus'
__icon__ = 'plant1.png'


__all__ = ['graphio_xml2graph', 'client_simulation', 'client_connexion', 'graphio_graph2xml']



graphio_xml2graph = Factory(name='xml2graph',
                description='Import a graph from an xml graph.',
                category='codec',
                nodemodule='openalea.groalea.graphio',
                nodeclass='xml2graph',
                inputs=({'interface': ITextStr, 'name': 'xml_graph', 'value': '', 'desc': 'Graph description in xml with 3D information.'},),
                outputs=({'name': 'graph', 'desc': 'graph object with properties.'}, {'name': 'scene', 'desc': 'PlantGL scene graph.'}),
                widgetmodule=None,
                widgetclass=None,
                )




client_simulation = Factory(name='simulation',
                description='Send XL code and a graph to GroIMPi server for simulation.',
                category='web,data',
                nodemodule='openalea.groalea.client',
                nodeclass='simulation',
                inputs=({'interface': ITextStr, 'name': 'xl_code', 'value': '', 'desc': 'XL code defining the simulation'}, {'interface': ITextStr, 'name': 'graph', 'desc': 'Optional xml graph str used as Axiom.'}, {'interface': IStr, 'name': 'command', 'desc': 'The command to be executed.'}, {'name': 'connexion', 'desc': 'HTTP Connexion to GroIMP.'}),
                outputs=({'interface': ITextStr, 'name': 'graph'},),
                widgetmodule=None,
                widgetclass=None,
                )




client_connexion = Factory(name='http connexion',
                description='Connexion to the GroIMP server',
                category='web,data',
                nodemodule='openalea.groalea.client',
                nodeclass='connexion',
                inputs=({'interface': IStr, 'name': 'host', 'value': 'localhost', 'desc': 'http address of the GroIMP server.'}, {'interface': IStr, 'name': 'port', 'value': '58090', 'desc': 'port used for the connexion with the server'}),
                outputs=({'name': 'connexion'},),
                widgetmodule=None,
                widgetclass=None,
                )




graphio_graph2xml = Factory(name='graph2xml',
                description='Export a graph to an xml string.',
                category='codec',
                nodemodule='openalea.groalea.graphio',
                nodeclass='graph2xml',
                inputs=({'name': 'graph', 'desc': 'OpenAlea graph object.'},),
                outputs=({'name': 'xml_graph', 'desc': 'graph object with properties.'},),
                widgetmodule=None,
                widgetclass=None,
                )





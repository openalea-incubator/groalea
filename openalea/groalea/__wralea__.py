
# This file has been generated at Wed Jan 24 17:44:25 2018

from openalea.core import *


__name__ = 'groalea'

__editable__ = True
__description__ = 'OpenAlea / GroIMP communication.'
__license__ = 'CECILL-C'
__url__ = 'http://openalea.gforge.inria.fr'
__alias__ = ['groimp']
__version__ = '0.0.1'
__authors__ = 'C. Pradal and Long Qinqin'
__institutes__ = 'INRIA/CIRAD/University of Cottbus'
__icon__ = 'plant1.png'


__all__ = ['mappletConverter_convert', 'graphio_graph2xml', 'mappletConverter_addMTGProperty', 'graphio_getMTGRootedGraph', 'graphio_xml2graph', 'client_simulation', 'graphio_getSceneXEG', 'client_connexion', 'topology_spanning_mtg', 'graphio_produceXEGfile', 'graphio_produceMTGContentfile', 'graphio_produceMTGDisplayfile']



mappletConverter_convert = Factory(name='convert',
                authors='C. Pradal and Long Qinqin (wralea authors)',
                description='Add a property to all vertices of the mtg object.',
                category='codec',
                nodemodule='openalea.groalea.mappletConverter',
                nodeclass='convert',
                inputs=({'name': 'mtg_object', 'desc': 'OpenAlea MTG object.'}, {'name': 'scene_object', 'desc': 'OpenAlea PlantGL scene object.'}, {'name': 'scale_num', 'desc': 'OpenAlea PlantGL scene object.'}),
                outputs=({'name': 'rootedgraph', 'desc': 'OpenAlea RootedGraph object with both MTG scales and a geometric scale (at finest scale).'},),
                widgetmodule=None,
                widgetclass=None,
               )




graphio_graph2xml = Factory(name='graph2xml',
                authors='C. Pradal and Long Qinqin (wralea authors)',
                description='Export a graph to an xml string.',
                category='codec',
                nodemodule='openalea.groalea.graphio',
                nodeclass='graph2xml',
                inputs=({'name': 'graph', 'desc': 'OpenAlea graph object.'},),
                outputs=({'name': 'xml_graph', 'desc': 'graph object with properties.'},),
                widgetmodule=None,
                widgetclass=None,
               )




mappletConverter_addMTGProperty = Factory(name='addMTGProperty',
                authors='C. Pradal and Long Qinqin (wralea authors)',
                description='Add a property to all vertices of the mtg object.',
                category='codec',
                nodemodule='openalea.groalea.mappletConverter',
                nodeclass='addMTGProperty',
                inputs=({'name': 'mtg', 'desc': 'OpenAlea MTG object.'}, {'name': 'propertyName', 'desc': 'the string of new property name.'}),
                outputs=({'name': 'mtg', 'desc': 'OpenAlea MTG object with added property.'},),
                widgetmodule=None,
                widgetclass=None,
               )




graphio_getMTGRootedGraph = Factory(name='getMTGRootedGraph',
                authors='C. Pradal and Long Qinqin (wralea authors)',
                description='remove the geometric scale (at finest scale) from the input rooted graph.',
                category='codec',
                nodemodule='openalea.groalea.graphio',
                nodeclass='getMTGRootedGraph',
                inputs=({'name': 'rootedgraph', 'desc': 'OpenAlea RootedGraph object with both MTG scales and a geometric scale (at finest scale).'},),
                outputs=({'name': 'rootedgraph', 'desc': 'OpenAlea RootedGraph object with only the MTG scales.'},),
                widgetmodule=None,
                widgetclass=None,
               )




graphio_xml2graph = Factory(name='xml2graph',
                authors='C. Pradal and Long Qinqin (wralea authors)',
                description='Import a graph from an xml graph.',
                category='codec',
                nodemodule='openalea.groalea.graphio',
                nodeclass='xml2graph',
                inputs=({'interface': ITextStr, 'name': 'xml_graph', 'value': '', 'desc': 'Graph description in xml with 3D information.'}, {'name': 'onlyTopology', 'desc': ' Boolean value indicating if xeg will be converted to pure topology (true).'}),
                outputs=({'name': 'graph', 'desc': 'graph object with properties.'}, {'name': 'scene', 'desc': 'PlantGL scene graph.'}),
                widgetmodule=None,
                widgetclass=None,
               )




client_simulation = Factory(name='simulation',
                authors='C. Pradal and Long Qinqin (wralea authors)',
                description='Send XL code and a graph to GroIMPi server for simulation.',
                category='web,data',
                nodemodule='openalea.groalea.client',
                nodeclass='simulation',
                inputs=({'interface': ITextStr, 'name': 'xl_code', 'value': '', 'desc': 'XL code defining the simulation'}, {'interface': ITextStr, 'name': 'graph', 'desc': 'Optional xml graph str used as Axiom.'}, {'interface': IStr, 'name': 'command', 'desc': 'The command to be executed.'}, {'name': 'connexion', 'desc': 'HTTP Connexion to GroIMP.'}),
                outputs=({'interface': ITextStr, 'name': 'graph'},),
                widgetmodule=None,
                widgetclass=None,
               )




graphio_getSceneXEG = Factory(name='getSceneXEG',
                authors='C. Pradal and Long Qinqin (wralea authors)',
                description='get the finest/gometric scale XEG from rooted graph.',
                category='codec',
                nodemodule='openalea.groalea.graphio',
                nodeclass='getSceneXEG',
                inputs=({'name': 'rootedgraph', 'desc': 'OpenAlea RootedGraph object.'},),
                outputs=({'name': 'single_scale_xeg', 'desc': 'graph object with properties.'},),
                widgetmodule=None,
                widgetclass=None,
               )




client_connexion = Factory(name='http connexion',
                authors='C. Pradal and Long Qinqin (wralea authors)',
                description='Connexion to the GroIMP server',
                category='web,data',
                nodemodule='openalea.groalea.client',
                nodeclass='connexion',
                inputs=({'interface': IStr, 'name': 'host', 'value': 'localhost', 'desc': 'http address of the GroIMP server.'}, {'interface': IStr, 'name': 'port', 'value': '58090', 'desc': 'port used for the connexion with the server'}),
                outputs=({'name': 'connexion'},),
                widgetmodule=None,
                widgetclass=None,
               )




topology_spanning_mtg = Factory(name='spanning_mtg',
                authors='C. Pradal and Long Qinqin (wralea authors)',
                description='get MTG object from RootedGraph object.',
                category='codec',
                nodemodule='openalea.groalea.topology',
                nodeclass='spanning_mtg',
                inputs=({'name': 'graph', 'desc': 'OpenAlea Rooted Graph object.'},),
                outputs=({'name': 'mtg', 'desc': 'OpenAlea MTG object.'},),
                widgetmodule=None,
                widgetclass=None,
               )




graphio_produceXEGfile = Factory(name='produceXEGfile',
                authors='C. Pradal and Long Qinqin (wralea authors)',
                description='produce .xeg file taking string of xeg object as content.',
                category='codec',
                nodemodule='openalea.groalea.graphio',
                nodeclass='produceXEGfile',
                inputs=({'name': 'xeg_object', 'desc': 'XEG object.'}, {'name': 'xeg_file_abname', 'desc': 'absolute name of wanted XEG file.'}),
                outputs=({'interface': None, 'name': 'out'},),
                widgetmodule=None,
                widgetclass=None,
               )




graphio_produceMTGContentfile = Factory(name='produceMTGContentfile',
                authors='C. Pradal and Long Qinqin (wralea authors)',
                description='produce text file taking the content of mtg object as its content.',
                category='codec',
                nodemodule='openalea.groalea.graphio',
                nodeclass='produceMTGContentfile',
                inputs=({'name': 'mtg_object', 'desc': 'OpenAlea MTG object.'}, {'name': 'mtg_file_abname', 'desc': 'absolute name of wanted MTG content file.'}),
                outputs=({'interface': None, 'name': 'out'},),
                widgetmodule=None,
                widgetclass=None,
               )



graphio_produceMTGDisplayfile = Factory(name='produceMTGDisplayfile',
                authors='C. Pradal and Long Qinqin (wralea authors)',
                description='produce text file taking the display of mtg object as its content.',
                category='codec',
                nodemodule='openalea.groalea.graphio',
                nodeclass='produceMTGDisplayfile',
                inputs=({'name': 'mtg_object', 'desc': 'OpenAlea MTG object.'}, {'name': 'mtg_file_abname', 'desc': 'absolute name of wanted MTG display file.'}),
                outputs=({'interface': None, 'name': 'out'},),
                widgetmodule=None,
                widgetclass=None,
               )





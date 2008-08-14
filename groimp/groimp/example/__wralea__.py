
# This file has been generated at Thu Aug 14 18:33:33 2008

from openalea.core import *


__name__ = 'groimp.example'

__editable__ = True
__description__ = 'OpenAlea / GroIMP communication.'
__license__ = 'CECILL-C'
__url__ = 'http://openalea.gforge.inria.fr'
__alias__ = []
__version__ = '0.0.1'
__authors__ = 'C. Pradal'
__institutes__ = 'INRIA/CIRAD/University of Cottbus'
__icon__ = ''
 

__all__ = ['_89494736', '_89494800', '_47288272']


_89494736 = DataFactory(name='code1.xl', 
                    description='XL code used in the 1st example.', 
                    editors=None,
                    includes=None,
                    )


_89494800 = DataFactory(name='graph1.xml', 
                    description='XML graph description', 
                    editors=None,
                    includes=None,
                    )



_47288272 = CompositeNodeFactory(name='1. Simple Tree', 
                             description='', 
                             category='Unclassified',
                             doc='',
                             inputs=[],
                             outputs=[],
                             elt_factory={  2: ('groimp', 'http connexion'),
   3: ('groimp', 'simulation'),
   4: ('groimp.example', 'code1.xl'),
   5: ('groimp.example', 'graph1.xml'),
   7: ('groimp', 'xml2graph'),
   10: ('vplants.plantgl.visualization', 'plot3D'),
   11: ('openalea.file', 'read'),
   12: ('openalea.file', 'read')},
                             elt_connections={  9791800: (12, 0, 3, 1),
   9791812: (5, 0, 12, 0),
   9791824: (2, 0, 3, 3),
   9791860: (11, 0, 3, 0),
   9791872: (3, 0, 7, 0),
   9791884: (7, 1, 10, 0),
   9791896: (4, 0, 11, 0)},
                             elt_data={  2: {  'block': False,
         'caption': 'http connexion',
         'hide': True,
         'lazy': True,
         'port_hide_changed': set([]),
         'posx': 437.37309968370568,
         'posy': 86.529180695847373,
         'priority': 0,
         'user_application': None},
   3: {  'block': False,
         'caption': 'simulation',
         'hide': True,
         'lazy': True,
         'port_hide_changed': set([]),
         'posx': 340.94543924089379,
         'posy': 161.6878379757168,
         'priority': 0,
         'user_application': None},
   4: {  'block': False,
         'caption': 'code1.xl',
         'hide': True,
         'lazy': True,
         'port_hide_changed': set([2]),
         'posx': 229.56841138659291,
         'posy': 48.464442403836358,
         'priority': 0,
         'user_application': None},
   5: {  'block': False,
         'caption': 'graph1.xml',
         'hide': True,
         'lazy': True,
         'port_hide_changed': set([2]),
         'posx': 326.49729619426603,
         'posy': 44.638302214059792,
         'priority': 0,
         'user_application': None},
   7: {  'block': False,
         'caption': 'xml2graph',
         'hide': True,
         'lazy': True,
         'port_hide_changed': set([]),
         'posx': 340.79953576165707,
         'posy': 211.33251709009289,
         'priority': 0,
         'user_application': None},
   10: {  'block': False,
          'caption': 'plot3D',
          'hide': True,
          'lazy': True,
          'port_hide_changed': set([]),
          'posx': 372.29071013161922,
          'posy': 261.12947658402203,
          'priority': 0,
          'user_application': None},
   11: {  'block': False,
          'caption': 'read',
          'hide': True,
          'lazy': False,
          'port_hide_changed': set([]),
          'posx': 239.77145189266395,
          'posy': 99.479644934190333,
          'priority': 0,
          'user_application': None},
   12: {  'block': False,
          'caption': 'read',
          'hide': True,
          'lazy': False,
          'port_hide_changed': set([]),
          'posx': 348.17875726966628,
          'posy': 110.95806550352,
          'priority': 0,
          'user_application': None},
   '__in__': {  'block': False,
                'caption': 'In',
                'hide': True,
                'lazy': True,
                'port_hide_changed': set([]),
                'posx': 20.0,
                'posy': 5.0,
                'priority': 0,
                'user_application': None},
   '__out__': {  'block': False,
                 'caption': 'Out',
                 'hide': True,
                 'lazy': True,
                 'port_hide_changed': set([]),
                 'posx': 20.0,
                 'posy': 250.0,
                 'priority': 0,
                 'user_application': None}},
                             elt_value={  2: [(0, "'localhost'"), (1, "'4711'")],
   3: [(2, "''")],
   4: [(0, 'PackageData(groimp.example, code1.xl)'), (1, 'None'), (2, 'None')],
   5: [  (0, 'PackageData(groimp.example, graph1.xml)'),
         (1, 'None'),
         (2, 'None')],
   7: [],
   10: [],
   11: [],
   12: [],
   '__in__': [],
   '__out__': []},
                             lazy=True,
                             )





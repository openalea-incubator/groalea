#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-


import os, sys
from setuptools import setup
pj = os.path.join


# Setup script

name = 'groimp'
namespace = 'openalea'
pkg_name = 'openalea.groimp'

version= '0.0.2'

description= 'GroIMP' 
long_description= ''' '''

author= 'Reinhard, Hoover, Christophe Pradal'
author_email= ''
url= ''

license= 'GPL' 

# Main setup
setup(
    name="groimp",
    version=version,
    description=description,
    long_description=long_description,
    author=author,
    author_email=author_email,
    url=url,
    license=license,
    
    namespace_packages = ["openalea"],
    create_namespaces = True,

    py_modules = [],
    # pure python  packages
    packages= [pkg_name],
    # python packages directory
    package_dir= {pkg_name : 'groimp'},

    include_package_data = True,
    package_data = {'' : ['*.png']},

    # Add package platform libraries if any
    zip_safe = False,

    # Scripts
    entry_points = { 'wralea': [ 'groimp = openalea.groimp',] },
 
    # Dependencies
    setup_requires = ['openalea.deploy'],
    install_requires = [],
    dependency_links = ['http://openalea.gforge.inria.fr/pi'],
   )



    

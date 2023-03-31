#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

import os
from setuptools import setup, find_packages
pj = os.path.join

# Setup script

name = 'groalea'
namespace = 'openalea'
pkg_name = 'openalea.groalea'

version = '1.1.0'

description = 'GroIMP / OpenAlea interface'
long_description = ''' '''

author = 'Reinhard Hoover, Long QinQin, Christophe Pradal'
author_email = 'christophe pradal cirad fr'
url = ''

license = 'GPL'

# Main setup
setup(
    name="openalea.groalea",
    version=version,
    description=description,
    long_description=long_description,
    author=author,
    author_email=author_email,
    url=url,
    license=license,

    #namespace_packages=["openalea"],
    #create_namespaces=True,

    # pure python  packages
    packages=find_packages('src'),
    # python packages directory
    package_dir={'': 'src'},

    include_package_data=True,
    package_data={'groalea_data': ['*.png']},

    # Add package platform libraries if any
    zip_safe=False,

    # Scripts
    entry_points={'wralea': ['groalea = openalea.groalea']},

    # Dependencies
    setup_requires=['openalea.deploy'],
    install_requires=[],
)

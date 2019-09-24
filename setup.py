#!/usr/bin/env python

from setuptools import setup, find_packages

import versioneer

setup(
    name='eugl',
    description='Modules that deal with sensor and data quality characterisation.',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    url='https://github.com/OpenDataCubePipelines/eugl',
    author='The wagl authors',
    author_email='earth.observation@ga.gov.au',
    maintainer='wagl developers',
    packages=find_packages(),
    install_requires=[
        'click',
        'click_datetime',
        'numpy',
        'rasterio',
        'rios',
        'python-fmask',
        'wagl',
    ],
    package_data={'eugl.gqa': ['data/*.csv']},
    dependency_links=[
        'hg+https://bitbucket.org/chchrsc/rios/get/rios-1.4.5.zip#egg=rios-1.4.5',
        'hg+https://bitbucket.org/chchrsc/python-fmask/get/python-fmask-0.4.5.zip#egg=python-fmask-0.4.5'
        'git+https://github.com/GeoscienceAustralia/wagl@develop#egg=wagl',
    ]
)

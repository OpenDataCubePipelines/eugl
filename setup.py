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
        'git+https://github.com/ubarsc/rios@rios-1.4.10#egg=rios-1.4.10',
        'git+https://github.com/ubarsc/python-fmask@pythonfmask-0.5.4#egg=python-fmask-0.5.4',
        'git+https://github.com/GeoscienceAustralia/wagl@develop#egg=wagl',
    ]
)

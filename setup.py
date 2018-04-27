#!/usr/bin/env python

from setuptools import setup

setup(name='eugl',
      version='0.0.2',
      description='Modules that deal with sensor and data quality characterisation.',
      packages=['eugl'],
      install_requires=[
          'click',
          'click_datetime',
          'numpy',
          'rasterio',
          'rios',
          'python-fmask'
      ],
      dependency_links=[
          'hg+https://bitbucket.org/chchrsc/rios@1.4.4/#egg=rios-1.4.4',
          'hg+https://bitbucket.org/chchrsc/python-fmask@0.4.5#egg=python-fmask-0.4.5'
      ]
      )

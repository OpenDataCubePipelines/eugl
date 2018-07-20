#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='eugl',
      version='0.0.3',
      description='Modules that deal with sensor and data quality characterisation.',
      packages=find_packages(),
      install_requires=[
          'eodatasets',
          'click',
          'click_datetime',
          'numpy',
          'rasterio',
          'rios',
          'python-fmask'
      ],
      dependency_links=[
          'hg+https://bitbucket.org/chchrsc/rios/get/rios-1.4.5.zip#egg=rios-1.4.5',
          'hg+https://bitbucket.org/chchrsc/python-fmask/get/python-fmask-0.4.5.zip#egg=python-fmask-0.4.5'
      ]
      )

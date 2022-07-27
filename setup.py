#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name="eugl",
    description="Modules that deal with sensor and data quality characterisation.",
    use_scm_version=True,
    url="https://github.com/OpenDataCubePipelines/eugl",
    author="The wagl authors",
    author_email="earth.observation@ga.gov.au",
    maintainer="wagl developers",
    packages=find_packages(),
    install_requires=[
        "click",
        "click_datetime",
        "numpy",
        "rasterio",
        "rios",
        "python-fmask==0.5.7",
        "s2cloudless==1.5.0",
        "sentinelhub==3.4.2",
        "wagl",
        "importlib-metadata;python_version<'3.8'",
    ],
    setup_requires=["setuptools_scm"],
    package_data={"eugl.gqa": ["data/*.csv"]},
    dependency_links=[
        "git+https://github.com/ubarsc/rios@rios-1.4.10#egg=rios-1.4.10",
        "git+https://github.com/ubarsc/python-fmask@pythonfmask-0.5.7#egg=python-fmask-0.5.7",  # noqa: E501
        "git+https://github.com/GeoscienceAustralia/wagl@develop#egg=wagl",
    ],
)

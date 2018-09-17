#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Functions in this file represent work to be refactored.

from __future__ import print_function
import datetime
import logging
import math
import os
from os.path import join as pjoin, splitext, basename
import re
import subprocess

import yaml
import pandas
import rasterio
from rasterio.warp import Resampling
from eugl.version import __version__


def _rounded(d):
    return round(float(d), 2)

def _populate_nan_residuals():
    empty_points = {'x': pandas.np.nan,
                    'y': pandas.np.nan,
                    'xy': pandas.np.nan}

    residuals = {'mean': empty_points.copy(),
                 'stddev': empty_points.copy(),
                 'iterative_mean': empty_points.copy(),
                 'iterative_stddev': empty_points.copy(),
                 'abs_iterative_mean': empty_points.copy(),
                 'abs': empty_points.copy(),
                 'cep90': pandas.np.nan}

    return residuals

def _gls_version(ref_fname):
    # TODO a more appropriate method of version detection and/or population of metadata
    if 'GLS2000_GCP_SCENE' in ref_fname:
        gls_version = 'GLS_v1'
    else:
        gls_version = 'GQA_v3'

    return gls_version

def _write_gqa_yaml(out_fname, data):
    _LOG.debug('Writing result yaml: %s', out_fname)
    with open(out_fname, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False, indent=4)


# TODO replace with structlog (same as wagl and tesp)
_LOG = logging.getLogger(__name__)

# Post SLC-OFF date
SLC_OFF = datetime.date(2003, 6, 1)

# TODO only work with the latest naming convention provided in the MTL file
# TODO replace the quick and dirty BAND_MAP that accounts for different sensors
#      different naming conventions (even of the same satellite)
BAND_MAP = {
    'LE7': {
        'LS5': {
            '1': 'B1',
            '2': 'B2',
            '3': 'B3',
            '4': 'B4',
            '5': 'B5',
            '6': 'B6_VCID_1',
            '7': 'B7',
        },
        'LS7': {
            '1': 'B1',
            '2': 'B2',
            '3': 'B3',
            '4': 'B4',
            '5': 'B5',
            '61': 'B6_VCID_1',
            '62': 'B6_VCID_2',
            '7': 'B7',
        },
        'LS8': {
            '1': 'B1',
            '2': 'B1',
            '3': 'B2',
            '4': 'B3',
            '5': 'B4',
            '6': 'B5',
            '7': 'B7',
            '10': 'B6_VCID_1',
            '11': 'B6_VCID_1',
        },
        'S2A': {
            '1': 'B1',
            '2': 'B1',
            '3': 'B2',
            '4': 'B3',
            '5': 'B3',
            '6': 'B3',
            '7': 'B3',
            '8': 'B4',
            '8A': 'B4',
            '11': 'B5',
            '12': 'B7',
        },
        'S2B': {
            '1': 'B1',
            '2': 'B1',
            '3': 'B2',
            '4': 'B3',
            '5': 'B3',
            '6': 'B3',
            '7': 'B3',
            '8': 'B4',
            '8A': 'B4',
            '11': 'B5',
            '12': 'B7',
        },
    },
    'LT5': {
        'LS5': {
            '1': 'B1',
            '2': 'B2',
            '3': 'B3',
            '4': 'B4',
            '5': 'B5',
            '6': 'B6',
            '7': 'B7',
        },
        'LS7': {
            '1': 'B1',
            '2': 'B2',
            '3': 'B3',
            '4': 'B4',
            '5': 'B5',
            '61': 'B6_VCID_1',
            '62': 'B6_VCID_2',
            '7': 'B7',
        },
        'LS8': {
            '1': 'B1',
            '2': 'B1',
            '3': 'B2',
            '4': 'B3',
            '5': 'B4',
            '6': 'B5',
            '7': 'B7',
            '10': 'B6_VCID_1',
            '11': 'B6_VCID_1',
        },
        'S2A': {
            '1': 'B1',
            '2': 'B1',
            '3': 'B2',
            '4': 'B3',
            '5': 'B3',
            '6': 'B3',
            '7': 'B3',
            '8': 'B4',
            '8A': 'B4',
            '11': 'B5',
            '12': 'B7',
        },
        'S2B': {
            '1': 'B1',
            '2': 'B1',
            '3': 'B2',
            '4': 'B3',
            '5': 'B3',
            '6': 'B3',
            '7': 'B3',
            '8': 'B4',
            '8A': 'B4',
            '11': 'B5',
            '12': 'B7',
        },
    },
    'LC8': {
        'LS5': {
            '1': 'B2',
            '2': 'B3',
            '3': 'B4',
            '4': 'B5',
            '5': 'B6',
            '6': 'B10',
            '7': 'B7',
        },
        'LS7': {
            '1': 'B2',
            '2': 'B3',
            '3': 'B4',
            '4': 'B5',
            '5': 'B6',
            '61': 'B10',
            '62': 'B10',
            '7': 'B7',
        },
        'LS8': {
            '1': 'B1',
            '2': 'B2',
            '3': 'B3',
            '4': 'B4',
            '5': 'B5',
            '6': 'B6',
            '7': 'B7',
            '10': 'B10',
            '11': 'B11',
        },
        'S2A': {
            '1': 'B1',
            '2': 'B2',
            '3': 'B3',
            '4': 'B4',
            '5': 'B4',
            '6': 'B4',
            '7': 'B4',
            '8': 'B5',
            '8A': 'B5',
            '11': 'B6',
            '12': 'B7',
        },
        'S2B': {
            '1': 'B1',
            '2': 'B2',
            '3': 'B3',
            '4': 'B4',
            '5': 'B4',
            '6': 'B4',
            '7': 'B4',
            '8': 'B5',
            '8A': 'B5',
            '11': 'B6',
            '12': 'B7',
        },
    },
    'LO8': {
        'LS5': {
            '1': 'B2',
            '2': 'B3',
            '3': 'B4',
            '4': 'B5',
            '5': 'B6',
            '6': 'B10',
            '7': 'B7',
        },
        'LS7': {
            '1': 'B2',
            '2': 'B3',
            '3': 'B4',
            '4': 'B5',
            '5': 'B6',
            '61': 'B10',
            '62': 'B10',
            '7': 'B7',
        },
        'LS8': {
            '1': 'B1',
            '2': 'B2',
            '3': 'B3',
            '4': 'B4',
            '5': 'B5',
            '6': 'B6',
            '7': 'B7',
            '10': 'B10',
            '11': 'B11',
        },
        'S2A': {
            '1': 'B1',
            '2': 'B2',
            '3': 'B3',
            '4': 'B4',
            '5': 'B4',
            '6': 'B4',
            '7': 'B4',
            '8': 'B5',
            '8A': 'B5',
            '11': 'B6',
            '12': 'B7',
        },
        'S2B': {
            '1': 'B1',
            '2': 'B2',
            '3': 'B3',
            '4': 'B4',
            '5': 'B4',
            '6': 'B4',
            '7': 'B4',
            '8': 'B5',
            '8A': 'B5',
            '11': 'B6',
            '12': 'B7',
        },
    },
}


OLD_BAND_MAP = {
    'LS5': {
        '1': '10',
        '2': '20',
        '3': '30',
        '4': '40',
        '5': '50',
        '6': '61',
        '7': '70',
    },
    'LS7': {
        '1': '10',
        '2': '20',
        '3': '30',
        '4': '40',
        '5': '50',
        '61': '61',
        '62': '62',
        '7': '70',
    },
    'LS8': {
        '1': '10',
        '2': '10',
        '3': '20',
        '4': '30',
        '5': '40',
        '6': '50',
        '7': '70',
        '10': '61',
        '11': '61',
    },
    'S2A': {
        '1': '10',
        '2': '10',
        '3': '20',
        '4': '30',
        '5': '30',
        '6': '30',
        '7': '30',
        '8': '40',
        '8A': '40',
        '11': '50',
        '12': '70',
    },
    'S2B': {
        '1': '10',
        '2': '10',
        '3': '20',
        '4': '30',
        '5': '30',
        '6': '30',
        '7': '30',
        '8': '40',
        '8A': '40',
        '11': '50',
        '12': '70',
    },
}


def _clean_name(s):
    """
    >>> _clean_name("Residual x ")
    'residual_x'
    """
    return str(s).strip().lower().replace(' ', '_')


def reproject(source_fname, reference_fname, out_fname,
              resampling=Resampling.bilinear):
    """
    Reproject an image.

    :param source_fname:
        A `string` representing the filepath name of the source image.

    :param reference_fname:
        A `string` representing the filepath name of the reference image.

    :param out_fname:
        A `string` representing the filepath name of the output image.

    :param resampling:
        The resampling method to use during image re-projection.
        Defaults to `bilinear`.
        See rasterio.warp.Resampling for options.

    :notes:
        Just a wrapper for command line GDAL, as the initial testing
        of in-memory vs GDAL command line, failed.
        TODO re-evaluate using a more recent version of rasterio
    """

    with rasterio.open(reference_fname) as ds:
        crs = ds.crs.wkt
        res = ds.res

    # extract the resampling string identifier
    resampling_method = {0: 'near', 1: 'bilinear', 2: 'cubic'}
    resampling = resampling_method[resampling]

    cmd = ['gdalwarp',
           '-r', resampling,
           '-t_srs', '{}'.format(crs),
           '-srcnodata', '0',
           '-dstnodata', '0',
           '-tr', '{}'.format(res[0]), '{}'.format(res[1]),
           '-tap', '-tap',
           source_fname,
           out_fname]

    _LOG.info('calling gdalwarp:\n {}'.format(cmd))
    subprocess.check_call(cmd)


def get_reference_data(acquisition, base_reference_dir):
    """
    Finds the reference image file to be used for the
    gverify comparison.

    :param acquisition:
        An instance of a `gaip.Acquisition`.

    :param base_reference_dir:
        A `string` representing the directory path for the
        reference imagery to be used in the gverify process.
    """

    tag = acquisition.tag.lower()
    band_num = acquisition.band_id
    dt = acquisition.acquisition_datetime

    # TODO sensor agnostic acquisition grid/tile id; i.e path/row properties no longer exist
    path = '{0:0=3d}'.format(int(acquisition.path))
    row = '{0:0=3d}'.format(int(acquisition.row))
    ref_dir = pjoin(pjoin(base_reference_dir, path), row)

    # we have to consider multiple matches (different dates for a given band)
    exts = ['.tif', '.tiff']
    ref_imgs = os.listdir(pjoin(pjoin(base_reference_dir, path), row))
    ref_imgs = [f for f in ref_imgs if splitext(f)[1].lower() in exts]

    # initialise a dataframe
    df = pandas.DataFrame(columns=["ref_fname", "date"])

    ptrn = ("(?P<sat>[A-Z, 0-9]{3})(?P<pr>[0-9]{6})(?P<date>[0-9]{7})"
            "(?P<stuff>\w+?_)(?P<band>\w+)")
    match = re.match(ptrn, ref_imgs[0])
    if match is not None:
        # we have a hit and can assume that we are dealing with ref imagery
        # containing multiple sensors
        # i.e. LE70900812002182ASA00_B2.TIF, LC80910812013291LGN00_B10.TIF

        for fname in ref_imgs:
            match = re.match(ptrn, fname)
            ref_tag = match.group('sat')
            ref_band = match.group('band')
            bnum = BAND_MAP[ref_tag][tag][band_num]
            ref_date = datetime.datetime.strptime(match.group('date'), '%Y%j')
            if ref_band == bnum:
                df = df.append({"ref_fname": fname, "date": ref_date},
                               ignore_index=True)

        subs = df.loc[(df['date'] - dt).abs().argmin()]
        reference_fname = subs['ref_fname']
        ref_date = subs['date']
    else:
        # we are dealing with the old reference imagery
        # i.e. p090r081_7dt20020701_z56_30.tif, p090r081_7dk20020701_z56_61.tif
        ptrn = ("p(?P<path>[0-9]{3})r(?P<row>[0-9]{3})"
                "(?P<junk>_[A-Za-z, 0-9]{3})(?P<date>[0-9]{8})"
                "_z(?P<zone>[0-9]{2})_(?P<band>[0-9]{2})")

        for fname in ref_imgs:
            match = re.match(ptrn, fname)
            ref_band = match.group('band')
            ref_date = datetime.datetime.strptime(match.group('date'),
                                                  '%Y%m%d')
            bnum = OLD_BAND_MAP[tag][band_num]
            if ref_band == bnum:
                df = df.append({"ref_fname": fname, "date": ref_date},
                               ignore_index=True)

        subs = df.loc[(df['date'] - dt).abs().argmin()]
        reference_fname = subs['ref_fname']
        ref_date = subs['date']

    return pjoin(ref_dir, reference_fname), ref_date.strftime('%Y%m%d')

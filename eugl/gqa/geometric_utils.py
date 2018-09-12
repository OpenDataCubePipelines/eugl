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

def _point(x, y, xy):
    return {'x': _rounded(x), 'y': _rounded(y), 'xy': _rounded(xy)}

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

def _write_failure_yaml(out_fname, granule, msg, ref_source=None,
                        ref_source_path=None, ref_date=None, gverify_version=None):
    """
    We'll cater for future tasks by passing through reference image details,
    if we decide to write a yaml for when a gverify execution fails
    for some reason.
    """
    _LOG.info('Writing yaml with failure message: %s', out_fname)
    repo_path = 'https://github.com/OpenDataCubePipelines/eugl.git'
    data = {}
    data['software_version'] = __version__
    data['software_repository'] = repo_path
    data['error_message'] = msg
    data['granule'] = granule
    data['ref_source'] = ref_source
    data['ref_source_path'] = ref_source_path
    data['ref_date'] = ref_date
    data['final_gcp_count'] = 0
    data['residual'] = _populate_nan_residuals()
    data['gverify_version'] = gverify_version
    _write_gqa_yaml(out_fname, data)


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


def call_gverify(source_image, reference_image, out_path, pyramid_levels=5,
                 thread_count=4, null_value=0, correlation_coefficient=0.75,
                 resampling=Resampling.bilinear, chip_size=33, grid_size=66,
                 gverify_binary='', fix_qa_location_file=None):
    """
    Executes the gverify program.
    """
    resampling_method = {0: 'NN', 1: 'BI', 2: 'CI'}
    resampling = resampling_method[resampling]

    gverify_cmd = [
        gverify_binary,
        '-b', reference_image,
        '-m', source_image,
        '-w', out_path,
        '-l', out_path,
        '-o', out_path,
        '-p', str(pyramid_levels),
        '-n', str(thread_count),
        '-nv', str(null_value),
        '-c', str(correlation_coefficient),
        '-r', resampling,
        '-t FIXED_LOCATION ',
        '-cs', str(chip_size),
        '-t_file', fix_qa_location_file,
        '-g', str(grid_size)
    ]

    # TODO; pass through gverify lib path, gdal_data path, geotiff_csv path
    GVERIFY_LIB_PATH = ''

    # TODO: GDAL_DATA and GEOTIFF_CSV values

    # We call bash explicitly because we're using bash syntax ("shell=True" could be anything)
    wrapped_cmd = (
        'bash', '-c', (
            'unset LD_LIBRARY_PATH; '
            'export LD_LIBRARY_PATH={lib_path}:$LD_LIBRARY_PATH; '
            'export GDAL_DATA={<include gdal data path>}; '
            'export GEOTIFF_CSV={<include gdal epsg path>}; '
            '{gverify_cmd}'.format(
                lib_path=GVERIFY_LIB_PATH,
                gverify_cmd=' '.join(gverify_cmd)
            )
        )
    )
    print("Calling gverify: %r", wrapped_cmd)
    subprocess.check_call(wrapped_cmd)


def gverify(acquisition, source_fname, reference_fname, reference_date,
            output_path, pyramid_levels, thread_count, null_value,
            correlation_coefficient, chip_size, grid_size,
            gverify_output_format, renamed_format, gverify_binary,
            fix_qa_location_file):
    """
    Sets up the required arguments necessary to run gverify.
    Acts as a wrapper for the luigi workflow to interface with gverify.
    """
    # Resampling method
    resampling = Resampling.bilinear

    # Acquisitions info
    src_date = acquisition.acquisition_datetime.isoformat().replace('-', '')
    band = acquisition.band_id

    if ((acquisition.tag.lower() == 'ls7') and
            (acquisition.acquisition_datetime >= SLC_OFF)):
        resampling = Resampling.nearest

    # execute gverify
    call_gverify(source_fname, reference_fname, output_path,
                 pyramid_levels=pyramid_levels,
                 thread_count=thread_count, null_value=null_value,
                 correlation_coefficient=correlation_coefficient,
                 resampling=resampling, chip_size=chip_size,
                 grid_size=grid_size, gverify_binary=gverify_binary,
                 fix_qa_location_file=fix_qa_location_file)

    # rename the files (aids in the validation process)
    for fmt in gverify_output_format:
        # Remove the image- or image_ portion of the filename
        new_fmt = fmt[fmt.find('gverify'):]

        out_fname = renamed_format.format(acquisition_date=src_date,
                                          reference_date=reference_date,
                                          band=band, fmt=new_fmt)
        fmt = pjoin(output_path, fmt)
        out_fname = pjoin(output_path, out_fname)
        _LOG.info('Renaming %r to %r', fmt, out_fname)
        os.rename(fmt, out_fname)


def calculate_gqa(results_file, out_fname, ref_fname, scene_id, r=0.75,
                  stddev=3, iterations=1, resolution=(25, 25)):
    """
    Calculate the Geometric Quality Assessment.
    Loads and queries the results created by gverify.

    :param results_file:
        A string containing the full file path name to the *.res file
        containing the output from gverify.

    :param out_fname:
        A string containing the full file path name that will contain
        the results from the Geometric Quality Assessment.

    :param ref_fname:
        A string containing the full file path name that will contain
        the reference image used for the gverify process.

    :param scene_id:
        A string containing the ID of the acquisition.

    :param r:
        The correlation coefficient to filter the results by.
        Default is 0.75.

    :param stddev:
        The standard deviation to filter the results by.
        Default is 3.

    :param iterations:
        The number of iterations on which to recursively filter the
        results and recalculate the mean and standard deviation.
        Default is 1.

    :param resolution:
        A `tuple` containing the (x_res, y_res) of the reference data.

    :return:
        None. The output is written directly to disk.
    """
    # date of reference imagery
    ptrn = "(?P<src_date>[0-9]{8})_(?P<ref_date>[0-9]{7,9})_(?P<stuff>\w*)"
    match = re.match(ptrn, basename(results_file))
    ref_date = match.group('ref_date')
    ref_date = datetime.datetime.strptime(ref_date, '%Y%m%d')

    # TODO ref_source_path could be multiple files if a mosaic was used
    # initialise the dict to hold the output values
    # TODO include gverify version
    out_values = {
        'software_version': get_version(),
        'software_repository': 'https://github.com/GeoscienceAustralia/gqa.git',
        'scene_id': scene_id,
        'ref_source': _gls_version(ref_fname),
        'ref_source_path': ref_fname,
        'ref_date': ref_date.date()}

    # Read the 'Residual histogram' section of the file
    rh = pandas.read_csv(results_file, sep=r'\s*', skiprows=6,
                         names=['Color', 'Residual'], header=None, nrows=5,
                         engine='python')

    out_values['colors'] = {_clean_name(i): _rounded(rh[rh.Color == i].Residual.values[0])
                            for i in rh.Color.values}

    # Read the 'Total residuals' section of the file
    tr = pandas.read_csv(results_file, sep=r'\=', skiprows=3,
                         names=['Residual_XY', 'Residual'], header=None,
                         nrows=2, engine='python')

    column_names = ['Point_ID', 'Chip', 'Line', 'Sample', 'Correlation',
                    'Y_Residual', 'X_Residual', 'Outlier']

    # read the tabulated data returned by gverify
    try:
        df = pandas.read_csv(results_file, sep=r'\s*', skiprows=19,
                             names=column_names, header=None, engine='python')
    except StopIteration:
        # empty table for the gverify results file
        msg = "No GCP's were found."
        _LOG.error("Gverify results file contains no tabulated data; %s", results_file)
        _LOG.info('Defaulting to NaN for the residual values.')

        # a copy of empty points is used to force yaml not to use an alias
        out_values['final_gcp_count'] = 0
        out_values['residual'] = _populate_nan_residuals()
        out_values['error_message'] = msg

        _write_gqa_yaml(out_fname, out_values)
        return

    # Query the data to exclude low values of r and any outliers
    subset = df[(df.Correlation > r) & (df.Outlier == 1)]

    # Convert the data to a pixel unit
    xres, yres = resolution
    subset.X_Residual = subset.X_Residual / xres
    subset.Y_Residual = subset.Y_Residual / yres

    # Calculate the mean value for both X & Y residuals
    original_mean_x = subset.X_Residual.mean()
    original_mean_y = subset.Y_Residual.mean()

    # Calculate the sample standard deviation for both X & Y residuals
    original_stddev_x = subset.X_Residual.std(ddof=1)
    original_stddev_y = subset.Y_Residual.std(ddof=1)

    # Compute new values to refine the selection
    mean_x = original_mean_x
    mean_y = original_mean_y
    stddev_x = original_stddev_x
    stddev_y = original_stddev_y
    for i in range(iterations):
        # Look for any residuals
        subset = subset[(abs(subset.X_Residual - mean_x) <
                         (stddev * stddev_x)) &
                        (abs(subset.Y_Residual - mean_y) <
                         (stddev * stddev_y))]

        # Re-calculate the mean and standard deviation for both X & Y residuals
        mean_x = subset.X_Residual.mean()
        mean_y = subset.Y_Residual.mean()
        mean_xy = math.sqrt(mean_x**2 + mean_y**2)
        stddev_x = subset.X_Residual.std(ddof=1)
        stddev_y = subset.Y_Residual.std(ddof=1)
        stddev_xy = math.sqrt(stddev_x**2 + stddev_y**2)

        # Calculate abs mean value for both X & Y residuals
        abs_mean_x = abs(subset.X_Residual).mean()
        abs_mean_y = abs(subset.Y_Residual).mean()
        abs_mean_xy = math.sqrt(abs_mean_x**2 + abs_mean_y**2)

    # Calculate the Circular Error Probable 90 (CEP90)
    # Formulae taken from:
    # http://calval.cr.usgs.gov/JACIE_files/JACIE04/files/1Ross16.pdf
    delta_r = (subset.X_Residual**2 + subset.Y_Residual**2)**0.5
    cep90 = delta_r.quantile(0.9)

    original_mean_xy = math.sqrt(original_mean_x**2 + original_mean_y**2)
    original_stddev_xy = math.sqrt(original_stddev_x**2 + original_stddev_y**2)
    abs_ = {_clean_name(i).split('_')[-1]: _rounded(tr[tr.Residual_XY == i].Residual.values[0])
            for i in tr.Residual_XY.values}
    abs_['xy'] = _rounded(math.sqrt(abs_['x']**2 + abs_['y']**2))

    # final gcp count
    out_values['final_gcp_count'] = int(subset.shape[0])

    # gqa residual results
    out_values['residual'] = {
        'mean': _point(original_mean_x, original_mean_y, original_mean_xy),
        'stddev': _point(original_stddev_x, original_stddev_y, original_stddev_xy),
        'iterative_mean': _point(mean_x, mean_y, mean_xy),
        'iterative_stddev': _point(stddev_x, stddev_y, stddev_xy),
        'abs_iterative_mean': _point(abs_mean_x, abs_mean_y, abs_mean_xy),
        'abs': abs_,
        'cep90': _rounded(cep90)
        }

    out_values['error_message'] = 'no errors'

    # Output the results to disk
    _write_gqa_yaml(out_fname, out_values)


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

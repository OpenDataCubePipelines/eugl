#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GQA Workflow
-------------

Workflow settings can be configured in `luigi.cfg` file.
"""

# pylint: disable=missing-docstring,no-init,too-many-function-args
# pylint: disable=too-many-locals

from __future__ import print_function

import math
import re
import logging
import os
from os.path import join as pjoin, dirname, basename, exists, isdir, abspath
import glob
import shutil
from datetime import datetime, timezone
from collections import Counter, namedtuple
from functools import partial
from itertools import chain

from pkg_resources import resource_filename
from structlog import wrap_logger
from structlog.processors import JSONRenderer

import luigi
import pandas
import rasterio
from rasterio.warp import Resampling
from pathlib import Path
import fiona
import h5py
import yaml

from wagl.data import write_img
from wagl.acquisition import acquisitions
from wagl.constants import BandType
from wagl.singlefile_workflow import DataStandardisation
from eugl.fmask import run_command, CommandError
from eugl.acquisition_info import acquisition_info
from eugl.metadata import get_gqa_metadata

from eugl.gqa.geometric_utils import (
    get_reference_data, reproject,
    _write_gqa_yaml, _populate_nan_residuals,
    _gls_version, _clean_name, _rounded,
    BAND_MAP, OLD_BAND_MAP, SLC_OFF
)

from eodatasets.metadata.gqa import populate_from_gqa
from eodatasets.serialise import read_yaml_metadata
from eodatasets.serialise import write_yaml_metadata
from eodatasets.verify import PackageChecksum

ERROR_LOGGER = wrap_logger(logging.getLogger('errors'),
                           processors=[JSONRenderer(indent=1, sort_keys=True)])

write_yaml = partial(yaml.safe_dump, default_flow_style=False, indent=4)

_LOG = wrap_logger(logging.getLogger(__name__),
                   processors=[JSONRenderer(indent=1, sort_keys=True)])

# TODO variable change; scene id is landsat specific, level1 is more generic
# TODO remove refs to l1t (landsat specific)


class GverifyTask(luigi.Task):

    # Imagery arguments
    level1 = luigi.Parameter()
    acq_parser_hint = luigi.OptionalParameter(default='')
    granule = luigi.Parameter()
    workdir = luigi.Parameter()

    # Gverify arguments
    executable = luigi.Parameter()
    ld_library_path = luigi.Parameter()
    gdal_data = luigi.Parameter()
    pyramid_levels = luigi.Parameter()
    geotiff_csv = luigi.Parameter()
    thread_count = luigi.Parameter()
    null_value = luigi.Parameter()
    chip_size = luigi.Parameter()
    grid_size = luigi.Parameter()
    root_fix_qa_location = luigi.Parameter()
    correlation_coefficient = luigi.FloatParameter()
    timeout = luigi.IntParameter(default=300)

    # Gverify Argument preparation
    landsat_scenes_shapefile = luigi.Parameter()
    ocean_tile_list = luigi.DictParameter()
    root_fix_qa_location = luigi.Parameter()
    reference_directory = luigi.Parameter()
    backup_reference_directory = luigi.Parameter()

    _args_file = 'gverify_run.yaml'
    _gverify_results = 'image-gverify.res'

    def requires(self):
        return [DataStandardisation(self.level1, self.workdir, self.granule)]

    def output(self):
        workdir = pjoin(self.workdir, 'gverify')

        return {
            'runtime_args': luigi.LocalTarget(pjoin(workdir, self._args_file)),
            'results': luigi.LocalTarget(pjoin(workdir, self._gverify_results))
        }

    def exists(self):
        return all(os.path.isfile(_f) for _f in self.output().values())

    def run(self):

        # Subdirectory in the task workdir
        workdir = pjoin(self.workdir, 'gverify')

        if not exists(workdir):
            os.makedirs(workdir)

        # Get acquisition metadata, limit it to executing granule
        container = (
            acquisitions(self.level1, self.acq_parser_hint)
            .get_granule(self.granule, container=True)
        )

        acq_info = acquisition_info(container, self.granule)

        # Initialise output variables for error case
        error_msg = ''
        ref_date = ''
        ref_source_path = ''
        reference_resolution = ''

        try:
            # retrieve a set of matching landsat scenes
            # lookup is based on polygon for Sentinel-2
            landsat_scenes = acq_info.intersecting_landsat_scenes(self.landsat_scenes_shapefile)

            def fixed_extra_parameters():
                points_txt = pjoin(workdir, 'points.txt')
                collect_gcp(self.root_fix_qa_location, landsat_scenes, points_txt)
                return ['-t', 'FIXED_LOCATION', '-t_file', points_txt]

            if acq_info.is_land_tile(self.ocean_tile_list):
                location = acq_info.land_band()
                # for sentinel-2 land tiles we prefer grid points
                # rather than GCPs
                if acq_info.preferred_gverify_method == 'grid':
                    extra = ['-g', self.grid_size]
                else:
                    extra = fixed_extra_parameters()
            else:
                # for sea tiles we always pick GCPs
                location = acq_info.ocean_band()
                extra = fixed_extra_parameters()

            # Extract the source band from the results archive
            with h5py.File(self.input()[0].path, 'r') as h5:
                band_id = h5[location].attrs['band_id']
                source_band = pjoin(workdir, 'source.tif')
                source_image = h5[location][:]
                source_image[source_image == -999] = 0
                write_img(source_image, source_band, geobox=acq_info.geobox, nodata=0,
                          options={'compression': 'deflate', 'zlevel': 1})

            # returns a reference image from one of ls5/7/8
            #  the gqa band id will differ depending on if the source image is 5/7/8
            reference_imagery = get_reference_imagery(
                landsat_scenes, acq_info.timestamp, band_id, acq_info.tag,
                [self.reference_directory, self.backup_reference_directory]
            )

            ref_date = get_reference_date(
                basename(reference_imagery[0].filename), band_id, acq_info.tag
            )
            ref_source_path = reference_imagery[0].filename

            # reference resolution is required for the gqa calculation
            reference_resolution = [abs(x) for x in most_common(reference_imagery).resolution]

            vrt_file = pjoin(workdir, 'reference.vrt')
            build_vrt(reference_imagery, vrt_file, workdir)

            self._run_gverify(vrt_file, source_band, outdir=workdir, extra=extra,
                              resampling=acq_info.preferred_resampling_method)
        except (ValueError, FileNotFoundError, CommandError) as ve:
            error_msg = str(ve)
            ERROR_LOGGER.error('gverify was not executed because:\n {}'.format(error_msg))
        finally:
            # Write out runtime data to be processed by the gqa task
            run_args = {
                'executable': self.executable,
                'ref_resolution': reference_resolution,
                'ref_date': (ref_date.isoformat() if ref_date else ''),
                'ref_source_path': str(ref_source_path),
                'granule': str(self.granule),
                'error_msg': str(error_msg)
            }
            with self.output()['runtime_args'].open('w') as fd:
                write_yaml(run_args, fd)
            # if gverify failed to product the .res file writ out a blank one
            if not exists(self.output()['results'].path):
                with self.output()['results'].open('w') as fd:
                    pass

    def _run_gverify(self, reference, source, outdir, extra=None,
                     resampling=Resampling.bilinear):

        resampling_method = {Resampling.nearest: 'NN', Resampling.bilinear: 'BI', Resampling.cubic: 'CI'}
        extra = extra or []  # Default to empty list

        wrapper = [f'export LD_LIBRARY_PATH={self.ld_library_path}:$LD_LIBRARY_PATH; ',
                   f'export GDAL_DATA={self.gdal_data}; ',
                   f'export GEOTIFF_CSV={self.geotiff_csv}; ']

        gverify = [self.executable,
                   '-b', reference,
                   '-m', source,
                   '-w', outdir,
                   '-l', outdir,
                   '-o', outdir,
                   '-p', str(self.pyramid_levels),
                   '-n', str(self.thread_count),
                   '-nv', str(self.null_value),
                   '-c', str(self.correlation_coefficient),
                   '-r', resampling_method[resampling],
                   '-cs', str(self.chip_size)]

        cmd = [' '.join(chain(wrapper, gverify, extra))]

        _LOG.debug('calling gverify {}'.format(' '.join(cmd)))
        run_command(cmd, outdir, timeout=self.timeout, command_name='gverify')


class GQATask(luigi.Task):
    """
    Calculate Geometric Quality Assessment for a granule.
    """

    level1 = luigi.Parameter()
    acq_parser_hint = luigi.OptionalParameter(default='')
    granule = luigi.Parameter()
    workdir = luigi.Parameter()
    output_yaml = luigi.Parameter()
    cleanup = luigi.Parameter()

    # GQA Algorithm parameters
    correlation_coefficient = luigi.FloatParameter()
    iterations = luigi.IntParameter()
    standard_deviations = luigi.FloatParameter()

    def requires(self):
        return GverifyTask(
            level1=self.level1,
            granule=self.granule,
            acq_parser_hint=self.acq_parser_hint,
            correlation_coefficient=self.correlation_coefficient,
            workdir=self.workdir
        )

    def output(self):
        output_yaml = pjoin(
            self.workdir,
            str(self.output_yaml).format(granule=self.granule)
        )
        return luigi.LocalTarget(output_yaml)

    def run(self):
        temp_yaml = pjoin(
            self.workdir,
            'gverify',
            str(self.output_yaml).format(granule=self.granule)
        )

        res = {}

        # Read gverify arguments from yaml
        with self.input()['runtime_args'].open('r') as _md:
            gverify_args = yaml.load(_md)

        try:
            if 'error_msg' not in gverify_args or gverify_args['error_msg'] == '':  # Gverify successfully ran
                rh, tr, df = parse_gverify(self.input()['results'].path)
                res = calculate_gqa(df, tr, gverify_args['ref_resolution'], self.standard_deviations,
                                    self.iterations, self.correlation_coefficient)

                # Add color residual values to the results
                res['colors'] = {
                    _clean_name(i): _rounded(rh[rh.Color == i].Residual.values[0])
                    for i in rh.Color.values
                }
            else:
                _LOG.debug('Writing NaNs for residuals; gverify failed to run')
                res = {
                    'final_gcp_count': 0,
                    'residual': _populate_nan_residuals(),
                    'error_message': gverify_args['error_msg']
                }

        except (StopIteration, FileNotFoundError) as _:
            ERROR_LOGGER.error(
                "Gverify results file contains no tabulated data; {}".format(
                    self.input()['results'].path)
            )

            _LOG.debug('Defaulting to NaN for the residual values.')
            res = {
                'final_gcp_count': 0,
                'residual': _populate_nan_residuals(),
                'error_message': "No GCP's were found"
            }

        finally:
            metadata = get_gqa_metadata(gverify_args['executable'])
            metadata['ref_source_path'] = gverify_args['ref_source_path']
            metadata['ref_source'] = (
                _gls_version(metadata['ref_source_path'])
                if metadata['ref_source_path'] else ''
            )  # if ref_source_path is non-empty calculate version
            metadata['ref_date'] = gverify_args['ref_date']
            metadata['granule'] = gverify_args['granule']
            _write_gqa_yaml(temp_yaml, {**metadata, **res})

        self.output().makedirs()
        # copy temp to output final location
        shutil.copy(temp_yaml, self.output().path)

        if int(self.cleanup):
            _cleanup_workspace(pjoin(self.workdir, 'gverify'))


def collect_gcp(fix_location, landsat_scenes, result_file):
    """ Concatenates gcps from multiple scenes """
    with open(result_file, 'w') as dest:
        for scene in landsat_scenes:
            path = '{0:0=3d}'.format(scene['path'])
            row = '{0:0=3d}'.format(scene['row'])
            _LOG.debug('collecting GCPs from {} {}'.format(path, row))
            scene_gcp_file = pjoin(fix_location, path, row, 'points.txt')
            with open(scene_gcp_file) as src:
                for l in src:
                    dest.write(l)


def parse_gverify(res_filepath):
    """ Read from the image-gverify.res output from gverify """
    # I want a comment on what rh stands for
    rh = pandas.read_csv(res_filepath, sep=r'\s*', skiprows=6,
                         names=['Color', 'Residual'], header=None, nrows=5,
                         engine='python')

    # Something talking about tr / Residual XY
    tr = pandas.read_csv(res_filepath, sep=r'\=', skiprows=3,
                         names=['Residual_XY', 'Residual'], header=None,
                         nrows=2, engine='python')

    column_names = [
        'Point_ID', 'Chip', 'Line', 'Sample', 'Map_X', 'Map_Y',
        'Correlation', 'Y_Residual', 'X_Residual', 'Outlier'
    ]

    df = pandas.read_csv(res_filepath, sep=r'\s*', skiprows=22,
                         names=column_names, header=None, engine='python')

    return (rh, tr, df)


def calculate_gqa(df, tr, resolution, stddev=1.0, iterations=1, correl=0.75):
    # Query the data to exclude low values of correl and any outliers
    subset = df[(df.Correlation > correl) & (df.Outlier == 1)]

    # Convert the data to a pixel unit
    xres, yres = resolution
    subset.X_Residual = subset.X_Residual / xres
    subset.Y_Residual = subset.Y_Residual / yres

    def calculate_stats(data):
        # Calculate the mean value for both X & Y residuals
        mean = dict(x=data.X_Residual.mean(), y=data.Y_Residual.mean())

        # Calculate the sample standard deviation for both X & Y residuals
        stddev = dict(x=data.X_Residual.std(ddof=1), y=data.Y_Residual.std(ddof=1))

        mean['xy'] = math.sqrt(mean['x'] ** 2 + mean['y'] ** 2)
        stddev['xy'] = math.sqrt(stddev['x'] ** 2 + stddev['y'] ** 2)
        return {'mean': mean, 'stddev': stddev}

    original = calculate_stats(subset)
    current = dict(**original)  # create a copy

    # Compute new values to refine the selection
    for _ in range(iterations):
        # Look for any residuals
        subset = subset[
            (abs(subset.X_Residual - current['mean']['x']) < (stddev * current['stddev']['x'])) &
            (abs(subset.Y_Residual - current['mean']['y']) < (stddev * current['stddev']['y']))]

        # Re-calculate the mean and standard deviation for both X & Y residuals
        current = calculate_stats(subset)

    # Calculate the Circular Error Probable 90 (CEP90)
    # Formulae taken from:
    # http://calval.cr.usgs.gov/JACIE_files/JACIE04/files/1Ross16.pdf
    delta_r = (subset.X_Residual ** 2 + subset.Y_Residual ** 2) ** 0.5
    cep90 = delta_r.quantile(0.9)

    abs_ = {_clean_name(i).split('_')[-1]: tr[tr.Residual_XY == i].Residual.values[0]
            for i in tr.Residual_XY.values}
    abs_['xy'] = math.sqrt(abs_['x']**2 + abs_['y']**2)

    abs_mean = dict(x=abs(subset.X_Residual).mean(),
                    y=abs(subset.Y_Residual).mean())
    abs_mean['xy'] = math.sqrt(abs_mean['x'] ** 2 + abs_mean['y'] ** 2)

    def _point(stat):
        return {key: _rounded(value) for key, value in stat.items()}

    return {
        'final_gcp_count': int(subset.shape[0]),
        'error_message': 'no errors',
        'residual': {
            'mean': _point(original['mean']),
            'stddev': _point(original['stddev']),
            'iterative_mean': _point(current['mean']),
            'iterative_stddev': _point(current['stddev']),
            'abs_iterative_mean': _point(abs_mean),
            'abs': _point(abs_),
            'cep90': _rounded(cep90)
        }
    }


def most_common(sequence):
    result, _ = Counter(sequence).most_common(1)[0]
    return result


class CSR(namedtuple('CSRBase', ['filename', 'crs', 'resolution'])):
    """
    Do two images have the same coordinate system and resolution?
    """
    @classmethod
    def from_file(cls, filename):
        with rasterio.open(filename) as fl:
            return cls(filename, fl.crs, fl.res)

    def __eq__(self, other):
        if not isinstance(other, CSR):
            return False
        return self.crs == other.crs and self.resolution == other.resolution

    def __hash__(self):
        return hash((self.crs.data['init'], self.resolution))


def build_vrt(reference_images, out_file, work_dir):
    temp_directory = pjoin(work_dir, 'reprojected_references')
    if not exists(temp_directory):
        os.makedirs(temp_directory)

    common_csr = most_common(reference_images)
    _LOG.debug("GQA: chosen CRS {}".format(common_csr))

    def reprojected_images():
        for image in reference_images:
            if image == common_csr:
                yield image
            else:
                src_file = image.filename
                ref_file = common_csr.filename
                out_file = pjoin(temp_directory, basename(src_file))
                reproject(src_file, ref_file, out_file)
                yield CSR.from_file(out_file)

    reprojected = [abspath(image.filename) for image in reprojected_images()]
    command = ['gdalbuildvrt', '-srcnodata', '0', '-vrtnodata', '0', out_file] + reprojected
    run_command(command, work_dir)


def get_reference_imagery(path_rows, timestamp, band_id, sat_id, reference_directories):
    australian = [
        entry
        for entry in path_rows
        if 87 <= entry['path'] <= 116 and 67 <= entry['row'] <= 91
    ]

    if australian == []:
        raise ValueError("No Australian path row found")

    def find_references(entry, directories):
        path = '{0:0=3d}'.format(entry['path'])
        row = '{0:0=3d}'.format(entry['row'])

        if directories == []:
            return []

        first, *rest = directories
        folder = pjoin(first, path, row)

        # A closest match, or set of reference images is considered in situations
        #  where temporal variance (for example with sand dunes) introduces
        #  errors into the GQA assessment.
        # This can be determined by examining the error of a stack of images in
        #  pairwise comparison against a stack compared against a master image.

        # If a folder exists for the pathrow find the closest match
        #  otherwise iterate through the directory list.
        if isdir(folder):
            return closest_match(folder, timestamp, band_id, sat_id)

        return find_references(entry, rest)

    result = [reference
              for entry in australian
              for reference in find_references(entry, reference_directories)]

    if not result:
        raise ValueError(f"No reference found for {path_rows}")

    return [CSR.from_file(image) for image in result]


def get_reference_date(filename, band_id, sat_id):
    """get_reference_date: extracts date from reference filename

    :param filename: GQA reference image
    :param band_id: band id for the observed band
    :param sat_id: satellite id for the acquisition
    """

    matches = re.match(
        '(?P<sat>[A-Z0-9]{3})(?P<pathrow>[0-9]{6})'
        '(?P<year_doy>[0-9]{7})[^_]+_(?P<band>\\w+)', filename
    )

    # Primary reference set use Julian date
    if matches and matches.group('band') == BAND_MAP[matches.group('sat')][sat_id][band_id]:
        return (
            datetime.strptime(matches.group('year_doy'), '%Y%j')
            .replace(tzinfo=timezone.utc)
        )

    # Back up set use YYYY-MM-DD format
    matches = re.match(
        'p(?P<path>[0-9]{3})r(?P<row>[0-9]{3}).{4}(?P<yyyymmdd>[0-9]{8})'
        '_z(?P<zone>[0-9]{2})_(?P<band>[0-9]{2})', filename
    )

    if matches and matches.group('band') == OLD_BAND_MAP[sat_id][band_id]:
        return (
            datetime.strptime(matches.group('yyyymmdd'), '%Y%m%d')
            .replace(tzinfo=timezone.utc)
        )

    return None


def closest_match(folder, timestamp, band_id, sat_id):
    """
    Returns the reference observation closest to the observation being
        evaluated
    """
    # copied from geometric_utils.get_reference_data

    # We can't filter for band_ids here because it depends on the
    #  platform for the reference image
    filenames = [
        name
        for name in os.listdir(folder)
        if re.match(r'.*\.tiff?$', name, re.IGNORECASE)
    ]

    if not filenames:
        return []

    df = pandas.DataFrame(columns=["filename", "diff"])
    for filename in filenames:
        date = get_reference_date(filename, band_id, sat_id)
        if date is None:
            continue

        diff = abs(date - timestamp).total_seconds()

        df = df.append({"filename": filename, "diff": diff}, ignore_index=True)

    closest = df.loc[df['diff'].idxmin()]
    return [pjoin(folder, closest['filename'])]


# TODO path/row are no longer properties of acquisition as they're landsat
#      specific. Need alternate method of finding correct reference directory
def _can_process(l1t_path, granule):
    _LOG.debug('Checking L1T: {}'.format(str(l1t_path)))
    acqs = acquisitions(l1t_path).get_all_acquisitions(granule)
    landsat_path = int(acqs[0].path)
    landsat_row = int(acqs[0].row)

    # TODO
    # the path/row exclusion logic is not long-term viable and the prototype
    # for S2 will follow a similar exclusion logic, but use MGRS tiles instead.
    # A geometry exclusion is probably better suited in going forward with
    # multi-sensor/platform support

    # Is it an Australian scene? That's all we support at the moment.
    # (numbers specified by Lan-Wei.)
    msg = 'Not an Australian {} ({}): {}'
    if not (87 <= landsat_path <= 116):
        msg = msg.format('path', landsat_path, basename(l1t_path))
        _LOG.info(msg)
        return False, msg
    if not (67 <= landsat_row <= 91):
        msg = msg.format('row', landsat_row, basename(l1t_path))
        _LOG.info(msg)
        return False, msg

    # Do we have a reference dir available to compute GQA?
    ref_dir, msg = get_acq_reference_directory(acqs[0])
    if not ref_dir:
        return ref_dir, msg

    return True, None


def _cleanup_workspace(out_path):
    _LOG.debug('Cleaning up working directory: {}'.format(out_path))
    shutil.rmtree(out_path)


def get_acquisition(l1t_path, granule):
    # Get the acquisitions, metadata, and filter by wavelength
    acqs = acquisitions(l1t_path).get_all_acquisitions(granule=granule)

    # TODO include MGRS id logic
    # TODO improve path/row or MGRS id decision logic

    # check if the path/row is identified as a reef scene
    path = acqs[0].path
    row = acqs[0].row
    df = pandas.read_csv(REEF_PR)
    reef_scene = ((df.Path == path) & (df.Row == row)).any()

    # Get the wavelengths to filter the acquisitions
    # TODO parse min/max as args not config
    if reef_scene:
        min_lambda = CONFIG.getfloat('work', 'reef_min_lambda')
        max_lambda = CONFIG.getfloat('work', 'reef_max_lambda')
    else:
        min_lambda = CONFIG.getfloat('work', 'min_lambda')
        max_lambda = CONFIG.getfloat('work', 'max_lambda')

    # only accept a single wavelength (for now...)
    acq = [acq for acq in acqs if (acq.band_type == BandType.REFLECTIVE and
                                   min_lambda < acq.wavelength[1] <= max_lambda)]

    return acq[0]


def get_acq_reference_directory(acq):
    # TODO a sensor agnostic method of retrieving the reference imagery
    # currently all our reference imagery is from Landsat
    scene_name = basename(dirname(acq.dir_name))
    landsat_path = int(acq.path)
    landsat_row = int(acq.row)
    return get_reference_directory(scene_name, landsat_path, landsat_row)


def get_reference_directory(scene_name, landsat_path, landsat_row):
    # TODO sensor agnostic approach
    #      cfg params to be parsed through as functional params
    path = '{0:0=3d}'.format(landsat_path)
    row = '{0:0=3d}'.format(landsat_row)
    # TODO parse config items as params
    ref_dir = CONFIG.get('work', 'reference_directory')
    bckup_ref_dir = CONFIG.get('work', 'backup_reference')
    reference_dir = pjoin(ref_dir, path, row)
    if not isdir(reference_dir):
        _LOG.info('No reference directory ({}) for {}'.format(reference_dir, scene_name))
        _LOG.info('Trying backup reference directory: ({})'.format(bckup_ref_dir))
        ref_dir = bckup_ref_dir
        reference_dir = pjoin(ref_dir, path, row)
        if not isdir(reference_dir):
            _LOG.info('No backup reference directory ({}) for {}'.format(
                reference_dir, scene_name))
            msg = "No reference or backup reference imagery available."
            return False, msg
    return ref_dir, None


class CreateGQADirs(luigi.Task):
    """
    Create the output directory.
    """

    #: :type: str
    out_path = luigi.Parameter()

    def requires(self):
        return []

    def output(self):
        out_path = self.out_path
        return luigi.LocalTarget(out_path)

    def run(self):
        out_path = self.out_path
        if not exists(out_path):
            os.makedirs(out_path)


# TODO assess whether specific identification of tasks is required anymore
#      it might be more suitable to have a sequential set of steps.
#      Independent tasks were initially used in the early days for debug
#      but now should be safer to remove them and have a singular gqa task
#      for a given granule
#      also aids when analysing the task database generated by luigi
class ReprojectImage(luigi.Task):
    """
    Reproject the source image to match the reference image.
    """

    #: :type: str
    l1t_path = luigi.Parameter()
    #: :type: str
    out_path = luigi.Parameter()

    def requires(self):
        return [CreateGQADirs(self.out_path)]

    def output(self):
        out_path = self.out_path
        # TODO define the config items as a luigi parameter
        base = pjoin(out_path, CONFIG.get('work', 'reproject_output_format'))
        acq = get_acquisition(self.l1t_path)
        out_fname = pjoin(out_path, base.format(band=acq.band_num))

        return luigi.LocalTarget(out_fname)

    def run(self):
        out_path = self.out_path
        base = pjoin(out_path, CONFIG.get('work', 'reproject_output_format'))
        acq = get_acquisition(self.l1t_path)
        out_fname = pjoin(out_path, base.format(band=acq.band_num))

        # Resampling method
        if ((acq.tag.lower() == 'ls7') and
                (acq.acquisition_datetime >= SLC_OFF)):
            resampling = Resampling.nearest
        else:
            resampling = Resampling.bilinear

        # get the reference image
        ref_dir, _ = get_acq_reference_directory(acq)
        ref_fname, _ = get_reference_data(acq, ref_dir)

        # warp
        src_fname = pjoin(acq.dir_name, acq.file_name)
        reproject(src_fname, ref_fname, out_fname, resampling)


# TODO import this task into tesp
# TODO enable updating dataset in-place
class UpdateSource(luigi.Task):

    """
    For each scene that is processed, insert the new results back
    into the source datasets metadata.

    Items for processing:

    * Update the ga-metadata.yaml
    * Update the package.sha1 checksum file
    * Copy across the new *gverify.log gqa-results.yaml files
    * Backup the original files, by moving them to the
      gqa output directory
    """

    #: :type: str
    l1t_path = luigi.Parameter()
    #: :type: str
    out_path = luigi.Parameter()

    def requires(self):
        return [GQATask(self.l1t_path, self.out_path)]

    def output(self):
        out_path = self.out_path
        out_fname = pjoin(out_path, 'Level1-Updated.txt')
        return luigi.LocalTarget(out_fname)

    def run(self):
        yaml_fname = CONFIG.get('gqa', 'gqa_output_format')
        new_yaml_file = pjoin(self.out_path, yaml_fname)
        new_log_file = glob.glob(pjoin(self.out_path, '*gverify.log'))[0]

        gqa_path = Path(new_yaml_file)

        l1t_dir = pjoin(self.l1t_path, 'additional')
        original_gqa_yaml = pjoin(l1t_dir, yaml_fname)
        original_gverify_log = glob.glob(pjoin(l1t_dir, '*gverify.log'))

        # check for the existance of a gqa yaml, and gverify log
        # then backup as required
        bckup = '.backup'
        if exists(original_gqa_yaml):
            out_fname = pjoin(self.out_path, yaml_fname + bckup)
            shutil.move(original_gqa_yaml, out_fname)

        if len(original_gverify_log) != 0:
            gverify_log = original_gverify_log[0]
            out_fname = pjoin(self.out_path, basename(gverify_log + bckup))
            shutil.move(gverify_log, out_fname)

        # copy the new files into the level-1 directory
        shutil.copy(new_yaml_file, l1t_dir)
        shutil.copy(new_log_file, l1t_dir)

        # backup the ga-metadata.yaml & package.sha1 files
        original_metadata_fname = pjoin(self.l1t_path, 'ga-metadata.yaml')
        md = read_yaml_metadata(original_metadata_fname)
        md = populate_from_gqa(md, gqa_path)
        out_fname = pjoin(self.out_path, 'ga-metadata.yaml' + bckup)
        shutil.move(original_metadata_fname, out_fname)
        write_yaml_metadata(md, original_metadata_fname)

        original_checksum_fname = pjoin(self.l1t_path, 'package.sha1')
        out_fname = pjoin(self.out_path, 'package.sha1' + bckup)
        shutil.move(original_checksum_fname, out_fname)

        # output the new checksum
        checksum = PackageChecksum()
        l1t_path = Path(self.l1t_path)
        tree = l1t_path.rglob('*')
        for item in tree:
            if item.is_dir() or item.suffix == '.IMD':
                continue
            checksum.add_file(item)

        checksum.write(original_checksum_fname)

        with self.output().open('w') as src:
            src.write('Original level-1 backed up and updated in-place.')


class GQA(luigi.WrapperTask):
    # this is a convenient entry point
    # to process a list of level1 datasets

    level1_list = luigi.Parameter()
    workdir = luigi.Parameter()
    acq_parser_hint = luigi.Parameter(default=None)

    # TODO enable updating dataset in-place
    # update_source = luigi.BoolParameter()

    def requires(self):
        def tasks(level1_list):
            # TODO check with Lan-Wei regarding multi-granule vs single-granule
            #      gqa operation.
            #      Below demo's submit all granules as single granule gqa operation
            #      (same as wagl)
            for level1 in level1_list:
                container = acquisitions(level1, self.acq_parser_hint)
                for granule in container.granules:
                    # TODO enable updating dataset in-place
                    # if update_source:
                    #     yield UpdateSource(level1, work_root)
                    # else:
                    yield GQATask(level1, granule, self.workdir)

        with open(self.level1_list) as src:
            return list(tasks([level1.strip() for level1 in src]))

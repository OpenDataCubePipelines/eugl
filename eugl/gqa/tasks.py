#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QGA Workflow
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
from os.path import join as pjoin, dirname, basename, exists, isdir, splitext, abspath
import glob
from pkg_resources import resource_filename
import shutil
import traceback
import argparse
from datetime import datetime
from dateutil.tz import tzutc
from collections import Counter, namedtuple
from subprocess import CalledProcessError

import luigi
import osr
import pandas
import rasterio
from rasterio.warp import Resampling
from pathlib import Path
from shapely.geometry import Polygon, shape
import fiona
import h5py
from dateutil.parser import parse as parse_timestamp

from wagl.data import write_img
from wagl.acquisition import acquisitions
from wagl.singlefile_workflow import DataStandardisation
from wagl.constants import BandType
from wagl.geobox import GriddedGeoBox
from eugl.version import get_version
from eugl.fmask import run_command
from eugl.gqa.geometric_utils import BAND_MAP, OLD_BAND_MAP
from eugl.gqa.geometric_utils import SLC_OFF
from eugl.gqa.geometric_utils import get_reference_data
from eugl.gqa.geometric_utils import reproject
from eugl.gqa.geometric_utils import _write_failure_yaml, _write_gqa_yaml
from eugl.gqa.geometric_utils import _gls_version, _clean_name, _rounded
from eodatasets.metadata.gqa import populate_from_gqa
from eodatasets.serialise import read_yaml_metadata
from eodatasets.serialise import write_yaml_metadata
from eodatasets.verify import PackageChecksum

# TODO general luigi task cleanup
#      see wagl.singlefile_workflow or wagl.multifile_workflow or tesp.workflow
#      for better and cleaner examples of luigi construction

# TODO remove these two .csv files from this tree, and find a permanent home for them
# TODO do not forget about the Landsat ocean list
REEF_PR = resource_filename('eugl.gqa', 'ocean_list.csv')

# TODO convert to structlog
_LOG = logging.getLogger(__name__)

# TODO functionally parse the parameters in the cfg at the task level
# rather than find and read the internal cfg.
# Similar to how wagl and tesp do it
CONFIG = luigi.configuration.get_config()

# TODO purge this
# CONFIG.add_config_path(resource_filename('gqa', 'gqa-defaults.luigi.cfg'))

# TODO variable change; scene id is landsat specific, level1 is more generic
# TODO remove refs to l1t (landsat specific)


class GQATask(luigi.Task):
    """
    WIP: GQA for Sentinel-2.
    TODO: Landsat compatibility.
    TODO: Modularity.
    """

    level1 = luigi.Parameter()
    granule = luigi.Parameter()
    workdir = luigi.Parameter()
    ocean_tile_list = luigi.Parameter()
    reference_directory = luigi.Parameter()
    backup_reference = luigi.Parameter()
    landsat_scenes_shapefile = luigi.Parameter()
    output_yaml = luigi.Parameter()
    land_band = luigi.Parameter()
    ocean_band = luigi.Parameter()
    cleanup = luigi.Parameter()

    # TODO this is unmaintainable, obviously gverify needs to be a separate task
    gverify_binary = luigi.Parameter()
    gverify_ld_library_path = luigi.Parameter()
    gverify_gdal_data = luigi.Parameter()
    gverify_pyramid_levels = luigi.Parameter()
    gverify_geotiff_csv = luigi.Parameter()
    gverify_formats = luigi.Parameter()
    gverify_thread_count = luigi.Parameter()
    gverify_null_value = luigi.Parameter()
    gverify_correlation_coefficient = luigi.Parameter()
    gverify_chip_size = luigi.Parameter()
    gverify_grid_size = luigi.Parameter()
    gverify_root_fix_qa_location = luigi.Parameter()
    gverify_iterations = luigi.Parameter()
    gverify_standard_deviations = luigi.Parameter()

    def requires(self):
        return [DataStandardisation(self.level1, self.workdir, self.granule)]

    def output(self):
        output_yaml = pjoin(self.workdir, self.output_yaml.format(granule=self.granule))
        return luigi.LocalTarget(output_yaml)

    def run(self):
        temp_directory = pjoin(self.workdir, 'work')
        if not exists(temp_directory):
            os.makedirs(temp_directory)

        temp_yaml = pjoin(temp_directory, self.output_yaml.format(granule=self.granule))

        try:
            land = is_land_tile(self.granule, self.ocean_tile_list)
            if land:
                location = "{}/{}".format(self.granule, self.land_band)
            else:
                location = "{}/{}".format(self.granule, self.ocean_band)

            h5 = h5py.File(self.input()[0].path, 'r')
            geobox = GriddedGeoBox.from_dataset(h5[location])

            landsat_scenes = intersecting_landsat_scenes(geobox_to_polygon(geobox),
                                                         self.landsat_scenes_shapefile)
            timestamp = acquisition_timestamp(h5, self.granule)
            band_id = h5[location].attrs['band_id']
            # TODO landsat sat_id
            sat_id = 's2'
            references = reference_imagery(landsat_scenes, timestamp, band_id, sat_id,
                                           [self.reference_directory, self.backup_reference])

            _LOG.debug("granule %s found reference images %s",
                       self.granule, [ref.filename for ref in references])
            vrt_file = pjoin(temp_directory, 'reference.vrt')
            build_vrt(references, vrt_file, temp_directory)

            source_band = pjoin(temp_directory, 'source.tif')
            source_image = h5[location][:]
            source_image[source_image == -999] = 0
            write_img(source_image, source_band, geobox=geobox, nodata=0,
                      options={'compression': 'deflate', 'zlevel': 1})

            if land:
                extra = ['-g', self.gverify_grid_size]
                cmd = gverify_cmd(self, vrt_file, source_band, temp_directory, extra=extra)
                _LOG.debug('calling gverify %s', ' '.join(cmd))
                run_command(cmd, temp_directory)
            else:
                # create a set of fix-points from landsat path-row
                points_txt = pjoin(temp_directory, 'points.txt')
                collect_gcp(self.gverify_root_fix_qa_location, landsat_scenes, points_txt)

                extra = ['-t', 'FIXED_LOCATION', '-t_file', points_txt]
                cmd = gverify_cmd(self, vrt_file, source_band, temp_directory, extra=extra)
                _LOG.debug('calling gverify %s', ' '.join(cmd))
                run_command(cmd, temp_directory)

            _LOG.debug('finished gverify on %s', self.granule)
            parse_gqa(self, temp_yaml, references, band_id, sat_id, temp_directory)

        except (ValueError, FileNotFoundError, CalledProcessError) as ve:
            # failed because GQA cannot be calculated
            _write_failure_yaml(temp_yaml, self.granule, str(ve))
            with open(pjoin(temp_directory, 'gverify.log'), 'w') as src:
                src.write('gverify was not executed because:\n')
                src.write(str(ve))

        self.output().makedirs()
        shutil.copy(temp_yaml, self.output().path)

        for temp_log in glob.glob(pjoin(temp_directory, '*gverify.log')):
            shutil.copy(temp_log, pjoin(self.workdir, basename(temp_log)))
            break

        if int(self.cleanup):
            _cleanup_workspace(temp_directory)


def collect_gcp(fix_location, landsat_scenes, points_txt):
    with open(points_txt, 'w') as fl:
        for scene in landsat_scenes:
            path = '{0:0=3d}'.format(scene['path'])
            row = '{0:0=3d}'.format(scene['row'])
            _LOG.debug('collecting GPCs from %s %s', path, row)
            this_points_txt = pjoin(fix_location, path, row, 'points.txt')
            with open(this_points_txt) as fl2:
                for l in fl2:
                    _LOG.debug('GQA: {} says {}'.format(this_points_txt, l))
                    fl.write(l)


def parse_gqa(task, output_yaml, reference_images, band_id, sat_id, work_dir):
    granule = task.granule
    formats = task.gverify_formats.split(',')
    result_file = pjoin(work_dir, [f for f in formats if f.endswith('.res')][0])
    resolution = [abs(x) for x in most_common(reference_images).resolution]
    first_ref = reference_images[0].filename

    repo_path = 'https://github.com/OpenDataCubePipelines/eugl.git'

    ref_date = find_reference_date(basename(first_ref), band_id, sat_id)
    _LOG.debug('ref_date for {} is {}'.format(first_ref, ref_date))
    result = dict(software_version=get_version(),
                  software_repository=repo_path,
                  granule=granule,
                  ref_source=_gls_version(first_ref),
                  ref_date=ref_date)

    rh = pandas.read_csv(result_file, sep=r'\s*', skiprows=6,
                         names=['Color', 'Residual'], header=None, nrows=5,
                         engine='python')

    result['colors'] = {_clean_name(i): _rounded(rh[rh.Color == i].Residual.values[0])
                            for i in rh.Color.values}

    tr = pandas.read_csv(result_file, sep=r'\=', skiprows=3,
                         names=['Residual_XY', 'Residual'], header=None,
                         nrows=2, engine='python')

    column_names = ['Point_ID', 'Chip', 'Line', 'Sample', 'Map_X', 'Map_Y', 'Correlation',
                    'Y_Residual', 'X_Residual', 'Outlier']

    try:
        df = pandas.read_csv(result_file, sep=r'\s*', skiprows=22,
                             names=column_names, header=None, engine='python')

        _LOG.debug('calculating GQA for %s', granule)
        gqa = calculate_gqa(task, df, tr, resolution)
        result = {**result, **gqa}

        _write_gqa_yaml(output_yaml, result)
        _LOG.debug('finished writing GQA for %s', granule)

    except StopIteration:
        # empty table for the gverify results file
        msg = "No GCP's were found."
        _LOG.error("Gverify results file contains no tabulated data; %s", results_file)
        _LOG.info('Defaulting to NaN for the residual values.')

        # a copy of empty points is used to force yaml not to use an alias
        result['final_gcp_count'] = 0
        result['residual'] = _populate_nan_residuals()
        result['error_message'] = msg

        _write_gqa_yaml(output_yaml, result)


def calculate_gqa(task, df, tr, resolution):
    stddev = float(task.gverify_standard_deviations)
    correl = float(task.gverify_correlation_coefficient)
    iterations = int(task.gverify_iterations)

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
    current = dict(original)

    # Compute new values to refine the selection
    for i in range(iterations):
        # Look for any residuals
        subset = subset[(abs(subset.X_Residual - current['mean']['x']) < (stddev * current['stddev']['x'])) &
                        (abs(subset.Y_Residual - current['mean']['y']) < (stddev * current['stddev']['y']))]

        # Re-calculate the mean and standard deviation for both X & Y residuals
        current = calculate_stats(subset)

        # Calculate abs mean value for both X & Y residuals

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


def gverify_cmd(task, reference, source, work_dir,
                extra=None, resampling=Resampling.bilinear):

    resampling_method = {0: 'NN', 1: 'BI', 2: 'CI'}
    if extra is None:
        extra = []

    wrapper = [ f'export LD_LIBRARY_PATH={task.gverify_ld_library_path}:$LD_LIBRARY_PATH; ',
                f'export GDAL_DATA={task.gverify_gdal_data}; ',
                f'export GEOTIFF_CSV={task.gverify_geotiff_csv}; ' ]

    gverify = [ task.gverify_binary,
                '-b', reference,
                '-m', source,
                '-w', work_dir,
                '-l', work_dir,
                '-o', work_dir,
                '-p', str(task.gverify_pyramid_levels),
                '-n', str(task.gverify_thread_count),
                '-nv', str(task.gverify_null_value),
                '-c', str(task.gverify_correlation_coefficient),
                '-r', resampling_method[resampling],
                '-cs', str(task.gverify_chip_size) ]

    return ['bash', '-c',
            "'{}'".format(' '.join(wrapper + gverify + extra))]


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


def acquisition_timestamp(h5_file, granule):
    result = h5_file[f'{granule}/ATMOSPHERIC-INPUTS'].attrs['acquisition-datetime']
    return parse_timestamp(result)


def is_land_tile(granule, ocean_tile_list):
    tile_id = granule.split('_')[-2][1:]

    with open(ocean_tile_list) as fl:
        for line in fl:
            if tile_id == line.strip():
                return False

    return True


def geobox_to_polygon(geobox):
    return Polygon([geobox.ul_lonlat, geobox.ur_lonlat,
                    geobox.lr_lonlat, geobox.ll_lonlat])


def intersecting_landsat_scenes(dataset_polygon, landsat_scenes_shapefile):
    landsat_scenes = fiona.open(landsat_scenes_shapefile)

    def path_row(properties):
        return dict(path=int(properties['PATH']), row=int(properties['ROW']))

    return [path_row(scene['properties'])
            for scene in landsat_scenes
            if shape(scene['geometry']).intersects(dataset_polygon)]


def reference_imagery(path_rows, timestamp, band_id, sat_id, reference_directories):
    australian = [entry
                  for entry in path_rows
                  if 87 <= entry['path'] <= 116 and 67 <= entry['row'] <= 91]

    if australian == []:
        raise ValueError("No Australian path row found")

    def find_references(entry, directories):
        path = '{0:0=3d}'.format(entry['path'])
        row = '{0:0=3d}'.format(entry['row'])

        if directories == []:
            return []

        first, *rest = directories
        folder = pjoin(first, path, row)
        if isdir(folder):
            return closest_match(folder, timestamp, band_id, sat_id)
        return find_references(entry, rest)

    result = [reference
              for entry in australian
              for reference in find_references(entry, reference_directories)]

    if result == []:
        raise ValueError(f"No reference found for {path_rows}")

    return [CSR.from_file(image) for image in result]


def find_reference_date(filename, band_id, sat_id):
    pattern1 = re.compile("(?P<sat>[A-Z, 0-9]{3})(?P<pr>[0-9]{6})(?P<date>[0-9]{7})"
                          "(?P<stuff>\\w+?_)(?P<band>\\w+)")
    pattern2 = re.compile("p(?P<path>[0-9]{3})r(?P<row>[0-9]{3})(?P<junk>_[A-Za-z, 0-9]{3})"
                          "(?P<date>[0-9]{8})_z(?P<zone>[0-9]{2})_(?P<band>[0-9]{2})")

    match1 = pattern1.match(filename)
    match2 = pattern2.match(filename)

    if match1 is not None:
        if match1.group('band') == BAND_MAP[match1.group('sat')][sat_id][band_id]:
            return datetime.strptime(match1.group('date'), '%Y%j')

    if match2 is not None:
        if match2.group('band') == OLD_BAND_MAP[sat_id][band_id]:
            return datetime.strptime(match2.group('date'), '%Y%m%d')

    return None


def closest_match(folder, timestamp, band_id, sat_id):
    # copied from geometric_utils.get_reference_data

    filenames = [name
                 for name in os.listdir(folder)
                 if splitext(name)[1].lower() in ['.tif', '.tiff']]

    if filenames == []:
        return []

    df = pandas.DataFrame(columns=["filename", "diff"])

    for filename in filenames:
        date = find_reference_date(filename, band_id, sat_id)
        if date is None:
            continue

        diff = abs(date.replace(tzinfo=tzutc()) - timestamp).total_seconds()
        df = df.append({"filename": filename, "diff": diff}, ignore_index=True)

    closest = df.loc[df['diff'].argmin()]
    return [pjoin(folder, closest['filename'])]


# TODO path/row are no longer properties of acquisition as they're landsat
#      specific. Need alternate method of finding correct reference directory
def _can_process(l1t_path, granule):
    _LOG.debug('Checking L1T: %r', l1t_path)
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
    _LOG.debug('Cleaning up working directory: %r', out_path)
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
        _LOG.info('No reference directory (%r) for %r',
                  reference_dir, scene_name)
        _LOG.info('Trying backup reference directory: (%r)', bckup_ref_dir)
        ref_dir = bckup_ref_dir
        reference_dir = pjoin(ref_dir, path, row)
        if not isdir(reference_dir):
            _LOG.info('No backup reference directory (%r) for %r',
                      reference_dir, scene_name)
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


# TODO potentially not need if better fleshed out
class RunGverify(luigi.Task):
    """
    Run the Gverify tool.
    """
    #: :type: str
    l1t_path = luigi.Parameter()
    #: :type: str
    out_path = luigi.Parameter()

    allow_failure = luigi.BoolParameter(default=True)

    resources = {
        'gverify': 1
    }

    def requires(self):
        # Get the acquisition & geobox
        acq = get_acquisition(self.l1t_path)
        geobox = acq.gridded_geo_box()

        # get the reference image
        ref_dir, _ = get_acq_reference_directory(acq)
        ref_fname, _ = get_reference_data(acq, ref_dir)

        with rasterio.open(ref_fname) as ds:
            ref_res = ds.res
            ref_crs = osr.SpatialReference()
            ref_crs.ImportFromWkt(ds.crs.wkt)

        same_crs = geobox.crs.IsSame(ref_crs)
        same_pix = ((abs(ref_res[0]) == abs(geobox.pixelsize[0])) and
                    (abs(ref_res[1]) == abs(geobox.pixelsize[1])))

        # we reproject if either the crs or the pixel resolutions
        # between the source and reference imagery are different
        if same_crs and same_pix:
            return [CreateGQADirs(self.out_path)]
        else:
            return [ReprojectImage(self.l1t_path, self.out_path)]

    def output(self):
        return luigi.LocalTarget(pjoin(self.out_path, 'run.txt'))

    def on_failure(self, exception):
        msg = exception.message
        traceback_string = traceback.format_exc()
        _LOG.info(msg)
        _LOG.info(traceback_string)
        return_code = exception.returncode

        # get the reference image
        acq = get_acquisition(self.l1t_path)
        scene_id = basename(dirname(acq.dir_name))
        ref_dir, _ = get_acq_reference_directory(acq)
        ref_fname, ref_date = get_reference_data(acq, ref_dir)

        # TODO replace scene_id with level1 (more generic)
        report = [u'Status: fail\n',
                  u'gverify error code: {}'.format(return_code),
                  u'scene_id: {}'.format(scene_id),
                  u'reference_filename: {}'.format(ref_fname),
                  u'reference_acquisition_date: {}'.format(ref_date),
                  msg]
        with self.output().open('w') as src:
            src.writelines(report)

        if not self.allow_failure:
            return traceback_string
        else:
            pass

    def on_success(self):
        report = [u'Status: success\n',
                  u'No errors reported']
        with self.output().open('w') as src:
            src.writelines(report)

    def run(self):
        out_path = self.out_path
        # TODO define the config items as a luigi parameter
        input_img_format = CONFIG.get('work', 'reproject_output_format')

        # Gverify parameters
        # TODO define the config items as a luigi parameter
        output_format = CONFIG.get('gverify', 'formats').split(',')
        renamed_format = CONFIG.get('gverify', 'renamed_format')
        pyramid_levels = CONFIG.getint('gverify', 'pyramid_levels')
        thread_count = CONFIG.getint('gverify', 'thread_count')
        null_value = CONFIG.getint('gverify', 'null_value')
        cor_coef = CONFIG.getfloat('gverify', 'correlation_coefficient')
        cs = CONFIG.getint('gverify', 'chip_size')
        gs = CONFIG.getint('gverify', 'grid_size')
        root_fix_qa = CONFIG.get('gverify', 'root_fix_qa_location')
        gverify_binary = CONFIG.get('gverify', 'binary')

        # get the acquisition
        # TODO fix path/row logic
        acq = get_acquisition(self.l1t_path)
        path = '{0:0=3d}'.format(acq.path)
        row = '{0:0=3d}'.format(acq.row)

        # file path name to the fix_QA points file
        fix_qa_location_file = pjoin(root_fix_qa, path, row, 'points.txt')

        # get the reference image
        ref_dir, _ = get_acq_reference_directory(acq)
        ref_fname, ref_date = get_reference_data(acq, ref_dir)

        # check if we have needed to reproject the source image or not
        band = acq.band_num
        src_fname = pjoin(out_path, input_img_format.format(band=band))
        if not exists(src_fname):
            src_fname = pjoin(acq.dir_name, acq.file_name)

        # TODO: * execute as luigi.ExternalTask (might simplify the code)
        gverify(acq, src_fname, ref_fname, ref_date, out_path, pyramid_levels,
                thread_count, null_value, cor_coef, cs, gs, output_format,
                renamed_format, gverify_binary, fix_qa_location_file)


# TODO potentially not need if better fleshed out
class CalculateGQA(luigi.Task):
    """
    Interpret the results that are output from the Gverfiy program
    and calcualte the GQA.
    """
    #: :type: str
    l1t_path = luigi.Parameter()
    #: :type: str
    out_path = luigi.Parameter()

    def requires(self):
        return [RunGverify(self.l1t_path, self.out_path)]

    def output(self):
        out_path = self.out_path
        # TODO define the config items as a luigi parameter
        out_fname = pjoin(out_path, CONFIG.get('gqa', 'gqa_output_format'))
        return luigi.LocalTarget(out_fname)

    def run(self):
        could_be_gverified, msg = self._check_was_successful()
        acq = get_acquisition(self.l1t_path)
        scene_id = basename(dirname(acq.dir_name))
        if not could_be_gverified:
            _write_failure_yaml(self.output().path, scene_id, msg)
            return

        out_path = self.out_path
        src_date = acq.scene_center_date.isoformat().replace('-', '')
        band = acq.band_num

        # Config params
        # TODO define the config items as a luigi parameter
        stddev = CONFIG.getfloat('gqa', 'standard_deviations')
        cor_coef = CONFIG.getfloat('gqa', 'correlation_coefficient')
        formats = CONFIG.get('gverify', 'formats').split(',')
        results_fmt = CONFIG.get('gverify', 'renamed_format')

        # get the reference image
        ref_dir, _ = get_acq_reference_directory(acq)
        ref_fname, ref_date = get_reference_data(acq, ref_dir)

        # output resolution
        with rasterio.open(ref_fname) as ds:
            res = (abs(ds.res[0]), abs(ds.res[1]))

        # get the gverify results filename
        fmt = [f for f in formats if '.res' in f][0]
        fmt = fmt[fmt.find('gverify'):]
        results_fname = results_fmt.format(acquisition_date=src_date,
                                           reference_date=ref_date,
                                           band=band, fmt=fmt)
        results_fname = pjoin(out_path, results_fname)

        # calculate the gqa figures
        out_fname = pjoin(out_path, CONFIG.get('gqa', 'gqa_output_format'))
        _LOG.info('Calculating results file %r -> %r', results_fname,
                  out_fname)
        calculate_gqa(results_fname, out_fname, ref_fname, scene_id,
                      r=cor_coef, stddev=stddev, iterations=1, resolution=res)

    def _check_was_successful(self):
        with self.input()[0].open('r') as f:
            report = f.readlines()

        was_successful = str(report[0].split(':')[1]).strip() == 'success'
        msg = str(report[1])
        return was_successful, msg


# TODO potentially not need if better fleshed out
class NotProcessedTask(luigi.Task):

    """
    A separate task for scenes that couldn't be run through
    the gverify process.
    """

    #: :type: str
    l1t_path = luigi.Parameter()
    #: :type: str
    out_path = luigi.Parameter()
    #: :type: str
    msg = luigi.Parameter()

    def requires(self):
        return []

    def output(self):
        out_path = self.out_path
        # TODO define the config items as a luigi parameter
        out_fname = pjoin(out_path, CONFIG.get('gqa', 'gqa_output_format'))
        return luigi.LocalTarget(out_fname)

    def run(self):
        self.output().makedirs()
        scene_id = basename(self.l1t_path)
        _write_failure_yaml(self.output().path, scene_id, self.msg)
        report = ['The gverify program was not executed because:\n',
                  self.msg]
        with open(pjoin(self.out_path, 'gverify.log'), 'w') as src:
            src.writelines(report)


# TODO delete this once GQATask is done
class GqaTask(luigi.Task):

    """
    Issues a RunGverify task or NotProcessedTask based on whether
    the scene is within the Australian path/row list, and has
    reference imagery available.
    """

    level1 = luigi.Parameter()
    granule = luigi.Parameter()
    workdir = luigi.Parameter()

    def requires(self):
        out_path = self.out_path + '-work'
        process, msg = _can_process(self.level1)
        if process:
            return [CalculateGQA(self.level1, out_path)]
        else:
            return [NotProcessedTask(self.level1, out_path, msg)]

    def output(self):
        out_path = self.out_path
        # TODO define the config items as a luigi parameter
        out_fname = pjoin(out_path, CONFIG.get('gqa', 'gqa_output_format'))
        return luigi.LocalTarget(out_fname)

    def run(self):
        work_directory = self.out_path + '-work'
        # TODO define the config items as a luigi parameter
        yaml_fname = CONFIG.get('gqa', 'gqa_output_format')
        yaml_file = pjoin(work_directory, yaml_fname)
        log_file = glob.glob(pjoin(work_directory, '*gverify.log'))[0]
        log_fname = basename(log_file)

        self.output().makedirs()
        shutil.copy(yaml_file, pjoin(self.out_path, yaml_fname))
        shutil.copy(log_file, pjoin(self.out_path, log_fname))

        # cleanup the workspace
        if CONFIG.getboolean('work', 'cleanup'):
            _cleanup_workspace(work_directory)


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


if __name__ == '__main__':
    luigi.run()

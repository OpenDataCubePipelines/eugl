#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QGA Workflow
-------------

Workflow settings can be configured in `gqa.cfg` file.

"""
# pylint: disable=missing-docstring,no-init,too-many-function-args
# pylint: disable=too-many-locals

from __future__ import print_function

import logging
import os
from os.path import join as pjoin, dirname, basename, exists, isdir
import glob
from pkg_resources import resource_filename
import shutil
import traceback
import argparse

import luigi
import osr
import pandas
import rasterio
from rasterio.warp import Resampling
from pathlib import Path

from wagl.acquisition import acquisitions
from wagl.constants import BandType
from gqa.geometric_utils import SLC_OFF
from gqa.geometric_utils import calculate_gqa
from gqa.geometric_utils import get_reference_data
from gqa.geometric_utils import gverify
from gqa.geometric_utils import reproject
from gqa.geometric_utils import _write_failure_yaml
from eodatasets.metadata.gqa import populate_from_gqa
from eodatasets.serialise import read_yaml_metadata
from eodatasets.serialise import write_yaml_metadata
from eodatasets.verify import PackageChecksum

# TODO general luigi task cleanup
#      see wagl.singlefile_workflow or wagl.multifile_workflow or tesp.workflow
#      for better and cleaner examples of luigi construction

# TODO include the S2 ocean list (quick implementation. Long term solution would be geometry related)
REEF_PR = resource_filename('gqa', 'ocean_list.csv')

# TODO convert to structlog
_LOG = logging.getLogger(__name__)

# TODO functionally parse the parameters in the cfg at the task level
# rather than find and read the internal cfg.
# Similar to how wagl and tesp do it
CONFIG = luigi.configuration.get_config()
CONFIG.add_config_path(resource_filename('gqa', 'gqa-defaults.luigi.cfg'))

# TODO variable change; scene id is landsat specific, level1 is more generic
# TODO remove refs to l1t (landsat specific)


# TODO path/row are no longer properties of acquisition as they're landsat
#      specific. Need alternate method of finding correct reference directory
def _can_process(l1t_path, granule=granule):
    _LOG.debug('Checking L1T: %r', l1t_path)
    acqs = acquisitions(l1t_path).get_all_acquisitions(granule=granule)
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


# TODO import this task into tesp
class GqaTask(luigi.Task):

    """
    Issues a RunGverify task or NotProcessedTask based on whether
    the scene is within the Australian path/row list, and has
    reference imagery available.
    """

    # TODO remove refs to l1t (landsat specific)
    #: :type: str
    l1t_path = luigi.Parameter()
    #: :type: str
    out_path = luigi.Parameter()

    def requires(self):
        out_path = self.out_path + '-work'
        process, msg = _can_process(self.l1t_path)
        if process:
            return [CalculateGQA(self.l1t_path, out_path)]
        else:
            return [NotProcessedTask(self.l1t_path, out_path, msg)]

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
        return [GqaTask(self.l1t_path, self.out_path)]

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

    level1_list = luigi.Parameter()
    workdir = luigi.Parameter()
    update_source = luigi.BoolParameter()

    def requires(self):
        with open(self.level1_list) as src:
            level1_list = [level1.strip() for level1 in src.readlines()]

        tasks = []
        # TODO check with lan-wei regarding multi-granule vs single-granule
        #      gqa operation.
        #      Below demo's submit all granules as single granule gqa operation
        #      (same as wagl)
        for level1 in level1_list:
            work_root = pjoin(self.workdir, '{}.GQA'.format(basename(level1)))
            container = acquisitions(level1, self.acq_parser_hint)
            for granule in container.granules:
            if update_source:
                tasks.append(UpdateSource(level1, work_root))
            else:
                tasks.append(GqaTask(level1, work_root))


if __name__ == '__main__':
    luigi.run()

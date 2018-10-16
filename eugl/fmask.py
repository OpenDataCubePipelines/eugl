# coding=utf-8
"""
Execution method for FMask - http://pythonfmask.org - (cloud, cloud shadow, water and
snow/ice classification) code supporting Sentinel-2 Level 1 C SAFE format zip archives hosted by the
Australian Copernicus Data Hub - http://www.copernicus.gov.au/ - for direct (zip) read access
by datacube.
"""
from __future__ import absolute_import
import os
from os.path import join as pjoin, abspath, basename, dirname, exists
import subprocess
import tempfile
import logging

from pathlib import Path
import click

from wagl.acquisition import acquisitions
from wagl.constants import BandType

_LOG = logging.getLogger(__name__)

os.environ["CPL_ZIP_ENCODING"] = "UTF-8"

# NOTE
# This module was quickly put together to achieve the deadlines
# and have an operation version of Fmask working for both S2 and Landsat.
# See TODO below

# TODO
# rework this entire module to be more dynamic for better sensor support
# potentially use the module and pass in the require vars rather
# than a command line call.


class CommandError(RuntimeError):
    """
    Custom class to capture subprocess call errors
    """
    pass


def run_command(command, work_dir, timeout=None):
    """
    A simple utility to execute a subprocess command.
    Raises a CalledProcessError for backwards compatibility
    """
    _proc = subprocess.Popen(
        ' '.join(command),
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        shell=True,
        cwd=str(work_dir)
    )

    timed_out = False

    try:
        _proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _proc.kill()
        timed_out = True

    stdout, stderr = _proc.communicate()
    if _proc.returncode != 0:
        _LOG.error(stderr.decode('utf-8'))
        _LOG.info(stdout.decode('utf-8'))

        if timed_out:
            raise CommandError('"%s" timed out' % (command))
        else:
            raise CommandError('"%s" failed with return code: %s' % (command, str(_proc.returncode)))
    else:
        _LOG.debug(stdout.decode('utf-8'))


def _landsat_fmask(acquisition, out_fname, work_dir):
    """
    Fmask algorithm for Landsat.
    """
    acquisition_path = Path(acquisition.pathname)
    if ".tar" in str(acquisition_path):
        tmp_dir = Path(work_dir) / 'fmask_imagery'
        if not tmp_dir.is_dir():
            tmp_dir.mkdir()
        cmd = ['tar', 'zxvf', str(acquisition_path)]
        run_command(cmd, tmp_dir)

        acquisition_path = tmp_dir

    container = acquisitions(str(acquisition_path))
    # [-1] index Avoids panchromatic band
    acqs = sorted(
        container.get_acquisitions(group=container.groups[-1], only_supported_bands=False),
        key=lambda a: a.band_id
    )

    # internal output filenames
    ref_fname = pjoin(work_dir, 'reflective.img')
    thm_fname = pjoin(work_dir, 'thermal.img')
    angles_fname = pjoin(work_dir, 'angles.img')
    mask_fname = pjoin(work_dir, 'saturation-mask.img')
    toa_fname = pjoin(work_dir, 'toa-reflectance.img')

    reflective_bands = [acq.uri for acq in acqs if acq.band_type is BandType.REFLECTIVE]
    thermal_bands = [acq.uri for acq in acqs if acq.band_type is BandType.THERMAL]

    # copy the mtl to the work space
    mtl_fname = str(list(acquisition_path.rglob('*_MTL.txt'))[0])

    cmd = ['gdal_merge.py', '-separate', '-of', 'HFA', '-co', 'COMPRESSED=YES',
           '-o', ref_fname, *reflective_bands]
    run_command(cmd, acquisition_path)

    # angles
    cmd = ['fmask_usgsLandsatMakeAnglesImage.py', '-m', mtl_fname,
           '-t', ref_fname, '-o', angles_fname]
    run_command(cmd, work_dir)

    # saturation
    cmd = ['fmask_usgsLandsatSaturationMask.py', '-i', ref_fname,
           '-m', mtl_fname, '-o', mask_fname]
    run_command(cmd, work_dir)

    # toa
    cmd = ['fmask_usgsLandsatTOA.py', '-i', ref_fname, '-m', mtl_fname,
           '-z', angles_fname, '-o', toa_fname]
    run_command(cmd, work_dir)


    cmd = ['gdal_merge.py', '-separate', '-of', 'HFA', '-co', 'COMPRESSED=YES',
           '-o', thm_fname, *thermal_bands]
    run_command(cmd, acquisition_path)

    cmd = ['fmask_usgsLandsatStacked.py', '-t', thm_fname, '-a', toa_fname,
           '-m', mtl_fname, '-z', angles_fname, '-s', mask_fname,
           '-o', out_fname]
    run_command(cmd, work_dir)


def _sentinel2_fmask(dataset_path, container, granule, out_fname, work_dir):
    """
    Fmask algorithm for Sentinel-2.
    """
    # filenames
    vrt_fname = pjoin(work_dir, "reflective.vrt")
    angles_fname = pjoin(work_dir, ".angles.img")

    acqs = []
    for grp in container.groups:
        acqs.extend(container.get_acquisitions(grp, granule, False))

    band_ids = [acq.band_id for acq in acqs]
    required_ids = [str(i) for i in range(1, 13)]
    required_ids.insert(8, '8A')

    acq = container.get_acquisitions(granule=granule)[0]

    # zipfile extraction
    xml_out_fname = pjoin(work_dir, Path(acq.granule_xml).name)
    if ".zip" in acq.uri:
        cmd = ['unzip', '-p', dataset_path, acq.granule_xml, '>',
               xml_out_fname]
        run_command(cmd, work_dir)

    # vrt creation
    cmd = ["gdalbuildvrt", "-resolution", "user", "-tr", "20", "20",
           "-separate", "-overwrite", vrt_fname]
    for band_id in required_ids:
        acq = acqs[band_ids.index(band_id)]
        if ".zip" in acq.uri:
            cmd.append(acq.uri.replace('zip:', '/vsizip/').replace('!', ''))
        else:
            cmd.append(acq.uri)

    run_command(cmd, work_dir)

    # angles generation
    if ".zip" in acq.uri:
        cmd = ["fmask_sentinel2makeAnglesImage.py", "-i", xml_out_fname,
               "-o", angles_fname]
    else:
        cmd = ["fmask_sentinel2makeAnglesImage.py", "-i", acq.granule_xml,
               "-o", angles_fname]

    run_command(cmd, work_dir)

    # run fmask
    cmd = ["fmask_sentinel2Stacked.py", "-a", vrt_fname, "-z", angles_fname,
           "-o", out_fname]
    run_command(cmd, work_dir)


def fmask(dataset_path, granule, out_fname, outdir, acq_parser_hint=None):
    """
    Execute the fmask process.
    """
    container = acquisitions(dataset_path, acq_parser_hint)
    with tempfile.TemporaryDirectory(dir=outdir,
                                     prefix='pythonfmask-') as tmpdir:
        acq = container.get_acquisitions(None, granule, False)[0]

        if 'SENTINEL' in acq.platform_id:
            _sentinel2_fmask(dataset_path, container, granule, out_fname,
                             tmpdir)
        elif 'LANDSAT' in acq.platform_id:
            _landsat_fmask(acq, out_fname, tmpdir)
        else:
            msg = "Sensor not supported"
            raise Exception(msg)


def fmask_cogtif(fname, out_fname):
    """
    Convert the standard fmask output to a cloud optimised geotif.
    """
    command = ["gdal_translate",
               "-of",
               "GTiff",
               "-co",
               "COMPRESS=DEFLATE",
               "-co",
               "ZLEVEL=4",
               "-co",
               "PREDICTOR=2",
               "-co",
               "COPY_SRC_OVERVIEWS=YES",
               fname,
               out_fname]

    run_command(command, dirname(fname))

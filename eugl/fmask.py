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
import signal
import tempfile
import logging

from pathlib import Path
import click

from wagl.acquisition import acquisitions
from wagl.constants import BandType

from eugl.metadata import fmask_metadata

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


def run_command(command, work_dir, timeout=None, command_name=None):
    """
    A simple utility to execute a subprocess command.
    Raises a CalledProcessError for backwards compatibility
    """
    _proc = subprocess.Popen(
        ' '.join(command),
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        preexec_fn=os.setsid,
        shell=True,
        cwd=str(work_dir)
    )

    timed_out = False

    try:
        stdout, stderr = _proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # see https://stackoverflow.com/questions/36952245/subprocess-timeout-failure
        os.killpg(os.getpgid(_proc.pid), signal.SIGTERM)
        stdout, stderr = _proc.communicate()
        timed_out = True

    if _proc.returncode != 0:
        _LOG.error(stderr.decode('utf-8'))
        _LOG.info(stdout.decode('utf-8'))

        if command_name is None:
            command_name = str(command)

        if timed_out:
            raise CommandError('"%s" timed out' % (command_name))
        else:
            raise CommandError('"%s" failed with return code: %s' % (command_name, str(_proc.returncode)))
    else:
        _LOG.debug(stdout.decode('utf-8'))


def _landsat_fmask(acquisition, out_fname, work_dir, cloud_buffer_distance,
                   cloud_shadow_buffer_distance):
    """
    Fmask algorithm for Landsat.
    """
    acquisition_path = Path(acquisition.pathname)
    if ".tar" in str(acquisition_path):
        tmp_dir = Path(work_dir) / 'fmask_imagery'
        if not tmp_dir.is_dir():
            tmp_dir.mkdir()
        cmd = ['tar', 'zxvf' if acquisition_path.suffix == '.gz' else 'xvf', str(acquisition_path)]
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

    if not thermal_bands:
        raise NotImplementedError(
            "python-fmask requires thermal bands to process landsat imagery"
        )

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
           '-o', out_fname,
           '--cloudbufferdistance', str(cloud_buffer_distance),
           '--shadowbufferdistance', str(cloud_shadow_buffer_distance)]
    run_command(cmd, work_dir)


def _sentinel2_fmask(dataset_path, container, granule, out_fname, work_dir,
                     cloud_buffer_distance, cloud_shadow_buffer_distance,
                     parallax_test):
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
           "-o", out_fname,
           "--cloudbufferdistance", str(cloud_buffer_distance),
           "--shadowbufferdistance", str(cloud_shadow_buffer_distance)]

    if parallax_test:
        cmd.append("--parallaxtest")

    run_command(cmd, work_dir)


def fmask(dataset_path, granule, out_fname, metadata_out_fname, workdir,
          acq_parser_hint=None, cloud_buffer_distance=150.0,
          cloud_shadow_buffer_distance=300.0, parallax_test=False):
    """
    Execute the fmask process.

    :param dataset_path:
        A str containing the full file pathname to the dataset.
        The dataset can be either a directory or a file, and
        interpretable by wagl.acquisitions.
    :type dataset_path: str

    :param granule:
        A str containing the granule name. This will is used to
        selectively process a given granule.
    :type granule: str

    :param out_fname:
        A fully qualified name to a file that will contain the
        result of the Fmask algorithm.
    :type out_fname: str

    :param metadata_out_fname:
        A fully qualified name to a file that will contain the
        metadata from the fmask process.
    :type metadata_out_fname: str

    :param workdir:
        A fully qualified name to a directory that can be
        used as scratch space for fmask processing.
    :type workdir: str

    :param acq_parser_hint:
        A hinting helper for the acquisitions parser. Default is None.

    :param cloud_buffer_distance:
        Distance (in metres) to buffer final cloud objects. Default
        is 150m.
    :type cloud_buffer_distance: float

    :param cloud_shadow_buffer_distance:
        Distance (in metres) to buffer final cloud shadow objects.
        Default is 300m.
    :type cloud_shadow_buffer_distance: float

    :param parallax_test:
        A bool of whether to turn on the parallax displacement test
        from Frantz (2018). Default is False.
        Setting this parameter to True has no effect for Landsat
        scenes.
    :type parallax_test: bool
    """
    container = acquisitions(dataset_path, acq_parser_hint)
    with tempfile.TemporaryDirectory(dir=workdir,
                                     prefix='pythonfmask-') as tmpdir:
        acq = container.get_acquisitions(None, granule, False)[0]

        if 'SENTINEL' in acq.platform_id:
            _sentinel2_fmask(dataset_path, container, granule, out_fname,
                             tmpdir, cloud_buffer_distance,
                             cloud_shadow_buffer_distance,
                             parallax_test)
        elif 'LANDSAT' in acq.platform_id:
            _landsat_fmask(acq, out_fname, tmpdir, cloud_buffer_distance,
                           cloud_shadow_buffer_distance)
        else:
            msg = "Sensor not supported"
            raise Exception(msg)

        # metadata
        fmask_metadata(out_fname, metadata_out_fname, cloud_buffer_distance,
                       cloud_shadow_buffer_distance, parallax_test)


def fmask_cogtif(fname, out_fname, platform):
    """
    Convert the standard fmask output to a cloud optimised geotif.
    """

    with tempfile.TemporaryDirectory(dir=dirname(fname),
                                     prefix='cogtif-') as tmpdir:

        # set the platform specific options for gdal function

        # setting the fmask's overview block size depending on the specific sensor.
        # Current, only USGS dataset are tiled at 512 x 512 for standardizing
        # Level 2 ARD products. Sentinel-2 tile size are inherited from the
        # L1C products and its overview's blocksize are default value of GDAL's
        # overview block size of 128 x 128

        # TODO Standardizing the Sentinel-2's overview tile size with external inputs

        if platform == "LANDSAT":
            options = {'compress': 'deflate',
                       'zlevel': 4,
                       'blockxsize': 512,
                       'blockysize': 512}

            config_options = {'GDAL_TIFF_OVR_BLOCKSIZE': options['blockxsize']}
        else:
            options = {'compress': 'deflate',
                       'zlevel': 4}

            config_options = None

        # clean all previous overviews
        command = ["gdaladdo",
                   "-clean",
                   fname]
        run_command(command, tmpdir)

        # build new overviews/pyramids consistent with NBAR/NBART products
        # the overviews are built with 'mode' re-sampling method
        cmd = ['gdaladdo',
               '-r',
               'mode',
               fname,
               '2',
               '4',
               '8',
               '16',
               '32']
        run_command(cmd, tmpdir)

        # create the cogtif
        command = ["gdal_translate",
                   "-of",
                   "GTiff",
                   "-co",
                   "TILED=YES",
                   "-co",
                   "PREDICTOR=2",
                   "-co",
                   "COPY_SRC_OVERVIEWS=YES"]

        for key, value in options.items():
            command.extend(['-co', '{}={}'.format(key, value)])

        if config_options:
            for key, value in config_options.items():
                command.extend(['--config', '{}'.format(key), '{}'.format(value)])

        command.extend([fname, out_fname])

        run_command(command, dirname(fname))

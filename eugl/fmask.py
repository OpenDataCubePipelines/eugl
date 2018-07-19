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
from subprocess import check_call
import tempfile
from pathlib import Path
import click

from wagl.acquisition import acquisitions

os.environ["CPL_ZIP_ENCODING"] = "UTF-8"

# NOTE
# This module was quickly put together to achieve the deadlines
# and have an operation version of Fmask working for both S2 and Landsat.
# See TODO below

# TODO
# rework this entire module to be more dynamic for better sensor support
# potentially use the module and pass in the require vars rather
# than a command line call.


def run_command(command, work_dir):
    """
    A simple utility to execute a subprocess command.
    """
    check_call(' '.join(command), shell=True, cwd=str(work_dir))


def _fmask_landsat(acquisition, out_fname, work_dir):
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

    # wild cards for the reflective bands
    reflective_wcards = {'LANDSAT_5': 'L*_B[1,2,3,4,5,7].TIF',
                         'LANDSAT_7': 'L*_B[1,2,3,4,5,7].TIF',
                         'LANDSAT_8': 'L*_B[1-7,9].TIF'}

    # wild cards for the thermal bands
    thermal_wcards = {'LANDSAT_5': 'L*_B6.TIF',
                      'LANDSAT_7': 'L*_B6_VCID_?.TIF',
                      'LANDSAT_8': 'L*_B1[0,1].TIF'}

    # internal output filenames
    ref_fname = pjoin(work_dir, 'reflective.img')
    thm_fname = pjoin(work_dir, 'thermal.img')
    angles_fname = pjoin(work_dir, 'angles.img')
    mask_fname = pjoin(work_dir, 'saturation-mask.img')
    toa_fname = pjoin(work_dir, 'toa-reflectance.img')

    reflective_bands = [str(path) for path in acquisition_path.rglob(
                        reflective_wcards[acquisition.platform_id])]
    
    thermal_bands = [str(path) for path in acquisition_path.rglob(
                     thermal_wcards[acquisition.platform_id])]

    # reflective image stack
    cmd = ['gdal_merge.py', '-separate', '-of', 'HFA', '-co', 'COMPRESSED=YES',
           '-o', ref_fname, *reflective_bands]
    run_command(cmd, acquisition_path)

    # thermal band(s)
    cmd = ['gdal_merge.py', '-separate', '-of', 'HFA', '-co', 'COMPRESSED=YES',
           '-o', thm_fname, *thermal_bands]
    run_command(cmd, acquisition_path)

    # copy the mtl to the work space
    mtl_fname = str(list(acquisition_path.rglob('*_MTL.txt'))[0])

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

    # fmask
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
            _fmask_landsat(acq, out_fname, tmpdir)
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

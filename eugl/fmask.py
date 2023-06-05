# coding=utf-8
"""
Execution method for FMask - http://pythonfmask.org - (cloud, cloud shadow, water and
snow/ice classification) code supporting Sentinel-2 Level 1 C SAFE format zip archives
hosted by the Australian Copernicus Data Hub - http://www.copernicus.gov.au/ - for
direct (zip) read access by datacube.
"""
from __future__ import absolute_import

import logging
import os
import signal
import subprocess
import tarfile
import tempfile
import zipfile
from os.path import join as pjoin, dirname
from pathlib import Path

from eugl.metadata import fmask_metadata, grab_offset_dict
from wagl.acquisition import acquisitions, Acquisition
from wagl.constants import BandType

import rasterio.path

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


def url_to_gdal(url: str):
    """
    Convert a rio-like URL into gdal-compatible vsi paths.

    fmask tooling uses gdal, not rio, so we need to do the same conversion.


    >>> rio_url = 'tar:///tmp/LC08_L1GT_109080_20210601_20210608_02_T2.tar!/LC08_L1GT_109080_20210601_20210608_02_T2_B1.TIF'
    >>> url_to_gdal(rio_url)
    '/vsitar//tmp/LC08_L1GT_109080_20210601_20210608_02_T2.tar/LC08_L1GT_109080_20210601_20210608_02_T2_B1.TIF'
    """
    # rio is considering removing this, so it's confined here to one place.
    return rasterio.path.parse_path(url).as_vsi()


def run_command(command, work_dir, timeout=None, command_name=None):
    """
    A simple utility to execute a subprocess command.
    Raises a CalledProcessError for backwards compatibility
    """
    _proc = subprocess.Popen(
        " ".join(command),
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        preexec_fn=os.setsid,
        shell=True,
        cwd=str(work_dir),
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
        _LOG.error(stderr.decode("utf-8"))
        _LOG.info(stdout.decode("utf-8"))

        if command_name is None:
            command_name = str(command)

        if timed_out:
            raise CommandError('"%s" timed out' % (command_name))
        else:
            raise CommandError(
                '"%s" failed with return code: %s' % (command_name, str(_proc.returncode))
            )
    else:
        _LOG.debug(stdout.decode("utf-8"))


def extract_mtl(archive_path: Path, output_folder: Path) -> Path:
    """
    Find and extract the MTL from an archive.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    mtl_files = []
    if archive_path.is_dir():
        for mtl in archive_path.rglob("*_MTL.txt"):
            mtl_files.append(mtl.name)
            (output_folder / mtl.name).write_bytes(mtl.read_bytes())

    elif archive_path.suffix in [".tar", ".gz", ".tgz", ".bz2"]:
        with tarfile.open(archive_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith("_MTL.txt"):
                    mtl_files.append(member.name)
                    tar.extract(member, output_folder, set_attrs=False)

    elif archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zipf:
            for member in zipf.namelist():
                if member.endswith("_MTL.txt"):
                    mtl_files.append(member)
                    zipf.extract(member, output_folder)

    else:
        raise ValueError(
            "Invalid archive format. Only .tar, .tar.gz, .zip are supported."
        )

    if len(mtl_files) == 0:
        raise ValueError("No _MTL.txt file found in the archive.")
    elif len(mtl_files) > 1:
        raise ValueError("Multiple _MTL.txt files found in the archive. ")

    return output_folder / mtl_files[0]


def _landsat_fmask(
    acquisition: Acquisition,
    out_fname: str,
    work_dir: str,
    cloud_buffer_distance: float,
    cloud_shadow_buffer_distance: float,
):
    """
    Fmask algorithm for Landsat.
    """
    acquisition_path = Path(acquisition.pathname)

    mtl_fname = extract_mtl(
        acquisition_path, Path(work_dir) / "fmask_imagery2"
    ).as_posix()

    container = acquisitions(str(acquisition_path))
    # [-1] index Avoids panchromatic band
    acqs = sorted(
        container.get_acquisitions(
            group=container.groups[-1], only_supported_bands=False
        ),
        key=lambda a: a.band_id,
    )

    # internal output filenames
    ref_fname = pjoin(work_dir, "reflective.img")
    thm_fname = pjoin(work_dir, "thermal.img")
    angles_fname = pjoin(work_dir, "angles.img")
    mask_fname = pjoin(work_dir, "saturation-mask.img")
    toa_fname = pjoin(work_dir, "toa-reflectance.img")

    reflective_bands = [
        url_to_gdal(acq.uri) for acq in acqs if acq.band_type is BandType.REFLECTIVE
    ]
    thermal_bands = [
        url_to_gdal(acq.uri) for acq in acqs if acq.band_type is BandType.THERMAL
    ]

    if not thermal_bands:
        raise NotImplementedError(
            "python-fmask requires thermal bands to process landsat imagery"
        )

    cmd = [
        "gdal_merge.py",
        "-separate",
        "-of",
        "HFA",
        "-co",
        "COMPRESSED=YES",
        "-o",
        ref_fname,
        *reflective_bands,
    ]
    run_command(cmd, work_dir)

    # angles
    cmd = [
        "fmask_usgsLandsatMakeAnglesImage.py",
        "-m",
        mtl_fname,
        "-t",
        ref_fname,
        "-o",
        angles_fname,
    ]
    run_command(cmd, work_dir)

    # saturation
    cmd = [
        "fmask_usgsLandsatSaturationMask.py",
        "-i",
        ref_fname,
        "-m",
        mtl_fname,
        "-o",
        mask_fname,
    ]
    run_command(cmd, work_dir)

    # toa
    cmd = [
        "fmask_usgsLandsatTOA.py",
        "-i",
        ref_fname,
        "-m",
        mtl_fname,
        "-z",
        angles_fname,
        "-o",
        toa_fname,
    ]
    run_command(cmd, work_dir)

    cmd = [
        "gdal_merge.py",
        "-separate",
        "-of",
        "HFA",
        "-co",
        "COMPRESSED=YES",
        "-o",
        thm_fname,
        *thermal_bands,
    ]
    run_command(cmd, work_dir)

    cmd = [
        "fmask_usgsLandsatStacked.py",
        "-t",
        thm_fname,
        "-a",
        toa_fname,
        "-m",
        mtl_fname,
        "-z",
        angles_fname,
        "-s",
        mask_fname,
        "-o",
        out_fname,
        "--cloudbufferdistance",
        str(cloud_buffer_distance),
        "--shadowbufferdistance",
        str(cloud_shadow_buffer_distance),
    ]
    run_command(cmd, work_dir)


def _sentinel2_fmask(
    dataset_path,
    container,
    granule,
    out_fname,
    work_dir,
    cloud_buffer_distance,
    cloud_shadow_buffer_distance,
    parallax_test,
):
    """
    Fmask algorithm for Sentinel-2.
    """
    # temp_vrt_fname: save vrt with offest values in metadata
    temp_vrt_fname = pjoin(work_dir, "reflective.tmp.vrt")
    # vrt_fname: save vrt with applied offest values by gdal_translate
    vrt_fname = pjoin(work_dir, "reflective.vrt")
    angles_fname = pjoin(work_dir, ".angles.img")

    acqs = []
    for grp in container.groups:
        acqs.extend(container.get_acquisitions(grp, granule, False))

    band_ids = [acq.band_id for acq in acqs]
    required_ids = [str(i) for i in range(1, 13)]
    required_ids.insert(8, "8A")

    acq = container.get_acquisitions(granule=granule)[0]

    # zipfile extraction
    xml_out_fname = pjoin(work_dir, Path(acq.granule_xml).name)
    if ".zip" in acq.uri:
        cmd = ["unzip", "-p", dataset_path, acq.granule_xml, ">", xml_out_fname]
        run_command(cmd, work_dir)

    offsets = grab_offset_dict(dataset_path)

    # vrt creation
    cmd = [
        "gdalbuildvrt",
        "-resolution",
        "user",
        "-tr",
        "20",
        "20",
        "-separate",
        "-overwrite",
        temp_vrt_fname,
    ]

    # when we use band to create VRT, also pass its offset values
    # to offset_values list, which will be used in gdal_edit later
    offset_values = []

    for band_id in required_ids:
        acq = acqs[band_ids.index(band_id)]

        # if we process data before 2021-Nov, we will have an emtry
        # offset_dict because there is no offset values in metadata.xml.
        # ideally, we should create a default offset_dict
        # with same keys but all values are 0, but S2 band_ids
        # is hard to use a simple loop to create.
        if band_id in offsets:
            offset_values.append(offsets[band_id])
        else:
            offset_values.append(0)

        if ".zip" in acq.uri:
            jp2_path = acq.uri.replace("zip:", "/vsizip/").replace("!", "")
            cmd.append(jp2_path)

        else:
            cmd.append(acq.uri)

    run_command(cmd, work_dir)

    # use this CLI to attach offset values to VRT as metadata info
    cmd = [
        "gdal_edit.py",
        "-ro",
        "-offset",
        " ".join([str(e) for e in offset_values]),
        temp_vrt_fname,
    ]

    run_command(cmd, work_dir)

    # use this CLI to apply offset metadata for the bands
    cmd = ["gdal_translate", temp_vrt_fname, vrt_fname, "-unscale"]

    run_command(cmd, work_dir)

    # angles generation
    if ".zip" in acq.uri:
        cmd = [
            "fmask_sentinel2makeAnglesImage.py",
            "-i",
            xml_out_fname,
            "-o",
            angles_fname,
        ]
    else:
        cmd = [
            "fmask_sentinel2makeAnglesImage.py",
            "-i",
            acq.granule_xml,
            "-o",
            angles_fname,
        ]
    run_command(cmd, work_dir)

    # run fmask
    cmd = [
        "fmask_sentinel2Stacked.py",
        "-a",
        vrt_fname,
        "-z",
        angles_fname,
        "-o",
        out_fname,
        "--cloudbufferdistance",
        str(cloud_buffer_distance),
        "--shadowbufferdistance",
        str(cloud_shadow_buffer_distance),
    ]

    if parallax_test:
        cmd.append("--parallaxtest")

    run_command(cmd, work_dir)


def fmask(
    dataset_path: str,
    granule: str,
    out_fname: str,
    metadata_out_fname: str,
    workdir: str,
    acq_parser_hint=None,
    cloud_buffer_distance: float = 150.0,
    cloud_shadow_buffer_distance: float = 300.0,
    parallax_test: bool = False,
):
    """
    Execute the fmask process.

    :param dataset_path:
        A str containing the full file pathname to the dataset.
        The dataset can be either a directory or a file, and
        interpretable by wagl.acquisitions.

    :param granule:
        A str containing the granule name. This will is used to
        selectively process a given granule.

    :param out_fname:
        A fully qualified name to a file that will contain the
        result of the Fmask algorithm.

    :param metadata_out_fname:
        A fully qualified name to a file that will contain the
        metadata from the fmask process.

    :param workdir:
        A fully qualified name to a directory that can be
        used as scratch space for fmask processing.

    :param acq_parser_hint:
        A hinting helper for the acquisitions parser. Default is None.

    :param cloud_buffer_distance:
        Distance (in metres) to buffer final cloud objects. Default
        is 150m.

    :param cloud_shadow_buffer_distance:
        Distance (in metres) to buffer final cloud shadow objects.
        Default is 300m.

    :param parallax_test:
        A bool of whether to turn on the parallax displacement test
        from Frantz (2018). Default is False.
        Setting this parameter to True has no effect for Landsat
        scenes.
    """
    container = acquisitions(dataset_path, acq_parser_hint)
    with tempfile.TemporaryDirectory(dir=workdir, prefix="pythonfmask-") as tmpdir:
        acq = container.get_acquisitions(None, granule, False)[0]

        if "SENTINEL" in acq.platform_id:
            _sentinel2_fmask(
                dataset_path,
                container,
                granule,
                out_fname,
                tmpdir,
                cloud_buffer_distance,
                cloud_shadow_buffer_distance,
                parallax_test,
            )
        elif "LANDSAT" in acq.platform_id:
            _landsat_fmask(
                acq,
                out_fname,
                tmpdir,
                cloud_buffer_distance,
                cloud_shadow_buffer_distance,
            )
        else:
            msg = "Sensor not supported"
            raise Exception(msg)

        # metadata
        fmask_metadata(
            out_fname,
            metadata_out_fname,
            cloud_buffer_distance,
            cloud_shadow_buffer_distance,
            parallax_test,
        )


def fmask_cogtif(fname, out_fname, platform):
    """
    Convert the standard fmask output to a cloud optimised geotif.
    """

    with tempfile.TemporaryDirectory(dir=dirname(fname), prefix="cogtif-") as tmpdir:
        # set the platform specific options for gdal function

        # setting the fmask's overview block size depending on the specific sensor.
        # Current, only USGS dataset are tiled at 512 x 512 for standardizing
        # Level 2 ARD products. Sentinel-2 tile size are inherited from the
        # L1C products and its overview's blocksize are default value of GDAL's
        # overview block size of 128 x 128

        # TODO Standardizing the Sentinel-2's overview tile size with external inputs

        if platform == "LANDSAT":
            options = {
                "compress": "deflate",
                "zlevel": 4,
                "blockxsize": 512,
                "blockysize": 512,
            }

            config_options = {"GDAL_TIFF_OVR_BLOCKSIZE": options["blockxsize"]}
        else:
            options = {"compress": "deflate", "zlevel": 4}

            config_options = None

        # clean all previous overviews
        command = ["gdaladdo", "-clean", fname]
        run_command(command, tmpdir)

        # build new overviews/pyramids consistent with NBAR/NBART products
        # the overviews are built with 'mode' re-sampling method
        cmd = ["gdaladdo", "-r", "mode", fname, "2", "4", "8", "16", "32"]
        run_command(cmd, tmpdir)

        # create the cogtif
        command = [
            "gdal_translate",
            "-of",
            "GTiff",
            "-co",
            "TILED=YES",
            "-co",
            "PREDICTOR=2",
            "-co",
            "COPY_SRC_OVERVIEWS=YES",
        ]

        for key, value in options.items():
            command.extend(["-co", "{}={}".format(key, value)])

        if config_options:
            for key, value in config_options.items():
                command.extend(["--config", "{}".format(key), "{}".format(value)])

        command.extend([fname, out_fname])

        run_command(command, dirname(fname))

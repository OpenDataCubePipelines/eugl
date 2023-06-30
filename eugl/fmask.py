"""
Execution method for FMask - http://pythonfmask.org - (cloud, cloud shadow, water and
snow/ice classification) code supporting Sentinel-2 Level 1 C SAFE format zip archives
hosted by the Australian Copernicus Data Hub - http://www.copernicus.gov.au/ - for
direct (zip) read access by datacube.
"""

import shlex

import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from os.path import join as pjoin, dirname
from typing import Dict

import zipfile
import tarfile
import fnmatch
from pathlib import Path, PurePosixPath
from typing import Optional

import rasterio.path
from fmask import config
from fmask import landsatTOA
from fmask import landsatangles
from fmask import saturationcheck
from fmask.cmdline.sentinel2Stacked import checkAnglesFile
from fmask.cmdline.sentinel2makeAnglesImage import makeAngles
from fmask import fmask as fmask_algorithm

from rios import fileinfo

from eugl.metadata import fmask_metadata, grab_offset_dict
from wagl.acquisition import acquisitions, Acquisition, AcquisitionsContainer
from wagl.acquisition.sentinel import Sentinel2Acquisition
from wagl.constants import BandType

REQUIRED_S2_BAND_IDS = (
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "8A",
    "9",
    "10",
    "11",
    "12",
)


FMASK_S2_BAND_MAPPINGS = {
    config.BAND_BLUE: "B02",
    config.BAND_GREEN: "B03",
    config.BAND_RED: "B04",
    config.BAND_NIR: "B08",
    config.BAND_SWIR1: "B11",
    config.BAND_SWIR2: "B12",
    config.BAND_CIRRUS: "B10",
    config.BAND_S2CDI_NIR8A: "B08A",
    config.BAND_S2CDI_NIR7: "B07",
    config.BAND_WATERVAPOUR: "B09",
}

_LOG = logging.getLogger(__name__)

os.environ["CPL_ZIP_ENCODING"] = "UTF-8"


class CommandError(RuntimeError):
    """
    Custom class to capture subprocess call errors
    """

    pass


def uri_to_gdal(url: str):
    """
    Convert a rio-like URL into gdal-compatible vsi paths.

    fmask tooling uses gdal, not rio, so we need to do the same conversion.


    >>> url = (
    ...     'tar:///tmp/LC08_L1GT_109080_20210601_20210608_02_T2.tar!'
    ...     '/LC08_L1GT_109080_20210601_20210608_02_T2_B1.TIF'
    ... )
    >>> uri_to_gdal(url)
    '/vsitar//tmp/LC08_L1GT_109080_20210601_20210608_02_T2.tar/LC08_L1GT_109080_20210601_20210608_02_T2_B1.TIF'
    >>> # Local paths are returned as-is
    >>> uri_to_gdal('/tmp/example-local-path.tif')
    '/tmp/example-local-path.tif'
    """
    # rio is considering removing this, so it's confined here to one place.
    # Some old wagl code did a much simpler but incomplete method:
    # >>> jp2_path = acq.uri.replace("zip:", "/vsizip/").replace("!", "")
    return rasterio.path.parse_path(url).as_vsi()


def run_command(command, work_dir, timeout=None, command_name=None, allow_shell=False):
    """
    A simple utility to execute a subprocess command.
    Raises a CalledProcessError for backwards compatibility
    """

    def to_simple_str(s):
        if isinstance(s, bytes):
            return s.decode("utf-8")
        elif isinstance(s, Path):
            return s.as_posix()
        return str(s)

    command_ = [to_simple_str(o) for o in command]

    printable_command = " ".join([shlex.quote(o) for o in command_])
    _LOG.debug("Running command: %s", printable_command)
    _proc = subprocess.Popen(
        command_,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        preexec_fn=os.setsid,
        shell=allow_shell,
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

    command_name = command_name or printable_command
    if _proc.returncode != 0:
        _LOG.error("stderr: %s", stderr.decode("utf-8"))
        _LOG.info("stdout: %s", stdout.decode("utf-8"))

        if timed_out:
            raise CommandError(f"{command_name!r} timed out (timeout={timeout}")
        else:
            raise CommandError(
                f"{command_name!r} failed with return code: {str(_proc.returncode)}"
            )
    else:
        _LOG.debug("Command %s had output: %s", command_name, stdout.decode("utf-8"))


def _landsat_fmask(
    acquisition: Acquisition,
    output_path: Path,
    work_dir: Path,
    cloud_buffer_distance: float,
    cloud_shadow_buffer_distance: float,
):
    """
    Fmask algorithm for Landsat.
    """
    acquisition_path = Path(acquisition.pathname)

    tmp_dir = work_dir / "fmask_imagery"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mtl_path = _extract_mtl(acquisition_path, tmp_dir)

    container = acquisitions(str(acquisition_path))
    # [-1] index Avoids panchromatic band
    acqs = sorted(
        container.get_acquisitions(
            group=container.groups[-1], only_supported_bands=False
        ),
        key=lambda a: a.band_id,
    )

    # internal output filenames
    reflective_path = work_dir / "reflective.img"
    thermal_path = work_dir / "thermal.img"
    angles_path = work_dir / "angles.img"
    sat_mask_path = work_dir / "saturation-mask.img"
    toa_reflectance_path = work_dir / "toa-reflectance.img"

    reflective_bands = [
        uri_to_gdal(acq.uri) for acq in acqs if acq.band_type is BandType.REFLECTIVE
    ]
    thermal_bands = [
        uri_to_gdal(acq.uri) for acq in acqs if acq.band_type is BandType.THERMAL
    ]

    if not thermal_bands:
        raise NotImplementedError(
            "python-fmask requires thermal bands to process landsat imagery"
        )

    cmd = [
        "gdal_merge.py",
        "-q",
        "-separate",
        "-of",
        "HFA",
        "-co",
        "COMPRESSED=YES",
        "-o",
        reflective_path,
        *reflective_bands,
    ]
    run_command(cmd, work_dir)

    mtl_info = config.readMTLFile(str(mtl_path))

    img_info = fileinfo.ImageInfo(str(reflective_path))
    nadir_line = landsatangles.findNadirLine(
        landsatangles.findImgCorners(str(reflective_path), img_info)
    )

    landsatangles.makeAnglesImage(
        str(reflective_path),
        str(angles_path),
        nadir_line,
        landsatangles.sunAnglesForExtent(img_info, mtl_info),
        landsatangles.satAzLeftRight(nadir_line),
        img_info,
    )

    landsat_number = mtl_info["SPACECRAFT_ID"][-1]
    if landsat_number in ("4", "5", "7"):
        sensor = config.FMASK_LANDSAT47
    elif landsat_number in ("8", "9"):
        sensor = config.FMASK_LANDSATOLI
    else:
        raise SystemExit(f"Unsupported Landsat sensor: {landsat_number!r}")

    # needed so the saturation function knows which
    # bands are visible etc.
    fmask_config = config.FmaskConfig(sensor)

    saturationcheck.makeSaturationMask(
        fmask_config, str(reflective_path), str(sat_mask_path)
    )
    landsatTOA.makeTOAReflectance(
        str(reflective_path), str(mtl_path), str(angles_path), str(toa_reflectance_path)
    )

    run_command(
        [
            "gdal_merge.py",
            "-q",
            "-separate",
            "-of",
            "HFA",
            "-co",
            "COMPRESSED=YES",
            "-o",
            thermal_path,
            *thermal_bands,
        ],
        work_dir,
    )

    # 1040nm thermal band should always be the first (or only) band in a
    # stack of Landsat thermal bands
    thermal_info = config.readThermalInfoFromLandsatMTL(str(mtl_path))

    angles_info = config.AnglesFileInfo(
        str(angles_path), 3, str(angles_path), 2, str(angles_path), 1, str(angles_path), 0
    )

    fmask_filenames = config.FmaskFilenames()
    fmask_filenames.setTOAReflectanceFile(str(toa_reflectance_path))
    fmask_filenames.setThermalFile(str(thermal_path))
    fmask_filenames.setOutputCloudMaskFile(str(output_path))
    fmask_filenames.setSaturationMask(str(sat_mask_path))

    fmask_config = config.FmaskConfig(sensor)
    fmask_config.setThermalInfo(thermal_info)
    fmask_config.setAnglesInfo(angles_info)
    fmask_config.setKeepIntermediates(False)
    fmask_config.setVerbose(False)
    fmask_config.setTempDir(tmp_dir)

    # TODO: Assuming the defaults are the same in the API as in CMD app?
    # fmask_config.setMinCloudSize(0)
    # fmask_config.setEqn17CloudProbThresh
    # fmask_config.setEqn20NirSnowThresh
    # fmask_config.setEqn20GreenSnowThresh

    # Work out a suitable buffer size, in pixels, dependent on the resolution
    # of the input TOA image
    toa_img_info = fileinfo.ImageInfo(str(toa_reflectance_path))
    fmask_config.setCloudBufferSize(int(cloud_buffer_distance / toa_img_info.xRes))
    fmask_config.setShadowBufferSize(
        int(cloud_shadow_buffer_distance / toa_img_info.xRes)
    )

    from fmask import fmask as fmask_algorithm

    _LOG.debug("Setup complete. Triggering fmask.")
    fmask_algorithm.doFmask(fmask_filenames, fmask_config)
    # TODO: Clean up thermal/angles/saturation/toa ?


def _extract_mtl(archive_path: Path, out_dir: Path) -> Path:
    """
    Find and extract the MTL file from a dataset.

    It will be placed in the given output directory, and the resulting path is returned.
    """
    with FileArchive(archive_path) as archive:
        mtls = [Path(file) for file in archive.files if file.endswith("_MTL.txt")]
        if len(mtls) != 1:
            raise RuntimeError(f"Expected one MTL file, found {len(mtls)}: {mtls}")
        mtl = mtls[0]
        mtl_out_path = out_dir / mtl.name
        archive.extract_file(mtl.as_posix(), mtl_out_path)
        return mtl_out_path


def _sentinel2_fmask(
    dataset_path: Path,
    container: AcquisitionsContainer,
    granule_name: str,
    output_path: Path,
    work_dir: Path,
    cloud_buffer_distance: float,
    cloud_shadow_buffer_distance: float,
    parallax_test: bool,
):
    """
    Fmask algorithm for Sentinel-2.
    """

    # TODO: Recheck non-zip file support.

    work_dir = Path(work_dir)

    acqs = []
    for grp in container.groups:
        acqs.extend(container.get_acquisitions(grp, granule_name, False))

    band_ids = [acq.band_id for acq in acqs]
    acq: Sentinel2Acquisition = container.get_acquisitions(granule=granule_name)[0]

    # Pull out the important metadata files.
    granule_xml_file = work_dir / Path(acq.granule_xml).name
    top_level_xml = work_dir / "MTD_MSIL1C.xml"
    with FileArchive(dataset_path) as archive:
        archive.extract_file(
            file_pattern=acq.granule_xml, destination_path=granule_xml_file
        )
        archive.extract_file(
            # There should be exactly one xml file in the top-level folder.
            # It's often called MTD_MSIL1C.xml, but there are several variations.
            file_pattern=pjoin(
                _base_folder_from_granule_xml(acq.granule_xml),
                "*.xml",
            ),
            exclude_name="INSPIRE.xml",
            destination_path=top_level_xml,
        )

    offsets = grab_offset_dict(dataset_path)

    # vrt creation
    # When we use bands to create a VRT, also embed its offset values in the metadata.
    band_uris = []
    band_offset_values = []

    for band_id in REQUIRED_S2_BAND_IDS:
        acq = acqs[band_ids.index(band_id)]
        assert isinstance(acq, Sentinel2Acquisition)

        # if we process data before 2021-Nov, we will have an emtry
        # offset_dict because there is no offset values in metadata.xml.
        # ideally, we should create a default offset_dict
        # with same keys but all values are 0, but S2 band_ids
        # is hard to use a simple loop to create.
        if band_id in offsets:
            band_offset_values.append(offsets[band_id])
        else:
            band_offset_values.append(0)

        band_uris.append(uri_to_gdal(acq.uri))

    reflective_vrt_img = work_dir / "reflective.vrt"
    run_command(
        [
            "gdalbuildvrt",
            "-resolution",
            "user",
            "-tr",
            "20",
            "20",
            "-separate",
            "-overwrite",
            reflective_vrt_img,
            *band_uris,
        ],
        work_dir,
    )

    fmask_config = config.FmaskConfig(config.FMASK_SENTINEL2)
    fmask_does_scaling = hasattr(fmask_config, "setTOARefOffsetDict")

    # If this version of fmask doesn't support scaling, do it ourselves.
    if not fmask_does_scaling:
        reflective_tmp_vrt = work_dir / "reflective.tmp.vrt"
        reflective_vrt_img.rename(reflective_tmp_vrt)

        # Now attach offset values to VRT as metadata info
        run_command(
            [
                "gdal_edit.py",
                "-ro",
                "-offset",
                *[str(e) for e in band_offset_values],
                reflective_tmp_vrt,
            ],
            work_dir,
        )

        # Apply offset metadata for the bands
        run_command(
            ["gdal_translate", reflective_tmp_vrt, reflective_vrt_img, "-unscale"],
            work_dir,
        )

    from fmask import sen2meta

    # Angles generation
    angles_img = work_dir / "angles.img"
    makeAngles(str(granule_xml_file), str(angles_img))

    # Run fmask

    angles_file = checkAnglesFile(str(angles_img), str(reflective_vrt_img))
    fmask_filenames = config.FmaskFilenames()
    fmask_filenames.setTOAReflectanceFile(str(reflective_vrt_img))
    fmask_filenames.setOutputCloudMaskFile(str(output_path))

    fmask_config.setAnglesInfo(
        config.AnglesFileInfo(
            angles_file, 3, angles_file, 2, angles_file, 1, angles_file, 0
        )
    )

    fmask_config.setTempDir(work_dir)

    if fmask_does_scaling:
        top_metadata = sen2meta.Sen2ZipfileMeta(xmlfilename=top_level_xml)
        fmask_config.setTOARefScaling(top_metadata.scaleVal)
        fmask_config.setTOARefOffsetDict(create_s2_band_offset_translation(top_metadata))

    fmask_config.setSen2displacementTest(parallax_test)

    # Work out a suitable buffer size, in pixels, dependent on the
    # resolution of the input TOA image
    toa_img_info = fileinfo.ImageInfo(str(reflective_vrt_img))
    fmask_config.setCloudBufferSize(int(cloud_buffer_distance / toa_img_info.xRes))
    fmask_config.setShadowBufferSize(
        int(cloud_shadow_buffer_distance / toa_img_info.xRes)
    )

    fmask_algorithm.doFmask(fmask_filenames, fmask_config)

    # TODO: Remove intermediates? toa, angles files


class FileArchive:
    """A simple abstraction over zip/tar files, or directories."""

    def __init__(self, archive_path: Path):
        self.archive_path = archive_path

    def __enter__(self):
        # Our "archive" can be a directory, a zip file, or a tar file.
        if self.archive_path.is_dir():
            self.open_file = lambda p: open(self.archive_path / p, "rb")
            self.files = [
                p.relative_to(self.archive_path).as_posix()
                for p in self.archive_path.rglob("*")
                if p.is_file()
            ]
        elif self.archive_path.suffix in [".zip"]:
            self._archive = zipfile.ZipFile(self.archive_path, "r")
            self.open_file = self._archive.open
            self.files = self._archive.namelist()
        elif self.archive_path.suffix in [".tar", ".tar.gz", ".tar.bz2", ".tgz"]:
            self._archive = tarfile.open(self.archive_path, "r")
            self.open_file = self._archive.extractfile
            self.files = self._archive.getnames()
        else:
            raise ValueError(f"Unsupported file type: {self.archive_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "_archive"):
            self._archive.close()

    def extract_file(
        self,
        file_pattern: str,
        destination_path: Path,
        exclude_name: Optional[str] = None,
    ):
        """
        Extract one file from an archive.

        The given file_pattern may be a filename glob.

        (If it matches multiple files, an error is raised.)

        The matched file is written to the given destination_path.
        """
        # Split the file pattern into directory and pattern
        directory, pattern = (
            PurePosixPath(file_pattern).parent,
            PurePosixPath(file_pattern).name,
        )

        # Find files that match the pattern within the specified directory
        matched_files = [
            member
            for member in self.files
            if PurePosixPath(member).parent == directory
            and fnmatch.fnmatch(PurePosixPath(member).name, pattern)
        ]

        if exclude_name:
            matched_files = [f for f in matched_files if exclude_name not in f]

        if len(matched_files) == 0:
            raise FileNotFoundError(
                f"No files match the pattern {file_pattern} in {self.archive_path}"
            )
        elif len(matched_files) > 1:
            raise ValueError(
                f"Multiple files match the pattern {file_pattern} in {self.archive_path}"
            )

        # Extract the file
        matched_file = matched_files[0]
        with self.open_file(matched_file) as archive_file_ref, destination_path.open(
            "wb"
        ) as f:
            f.write(archive_file_ref.read())


def _base_folder_from_granule_xml(granule_xml):
    """
    >>> g = (
    ...     "S2B_MSIL1C_20230306T002059_N0509_R116_T55GCP_20230306T014524.SAFE"
    ...     "/GRANULE/L1C_T55GCP_A031316_20230306T002318/MTD_TL.xml"
    ... )
    >>> _base_folder_from_granule_xml(g)
    'S2B_MSIL1C_20230306T002059_N0509_R116_T55GCP_20230306T014524.SAFE'
    """
    return Path(granule_xml).parent.parent.parent


def create_s2_band_offset_translation(
    s2metadata,
) -> Dict[int, int]:
    """
    Using S2 metadata, create a translation dict from fmask's
    band id to the offset value.

    <Fmask band id:int> -> <offset value:int>

    These are the RADIO_ADD_OFFSET fields in metadata:

        <RADIO_ADD_OFFSET band_id="0">-1000</RADIO_ADD_OFFSET>

    This type doesn't exist on older fmask, so we can't use a usual imported type def.

    :type s2metadata: fmask.sen2meta.Sen2ZipfileMeta
    """
    return {
        fmask_band_i: s2metadata.offsetValDict[FMASK_S2_BAND_MAPPINGS[fmask_band_i]]
        for fmask_band_i in FMASK_S2_BAND_MAPPINGS
    }


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
    clean_up_working_files: bool = True,
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
        A hinting helper for the acquisitions' parser. Default is None.

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
    :param clean_up_working_files:
        Whether to clean up the intermediate fmask files.
        Default is True.
    """

    dataset_path = Path(dataset_path)
    out_fname = Path(out_fname)
    metadata_out_fname = Path(metadata_out_fname)

    container = acquisitions(str(dataset_path), acq_parser_hint)
    acq = container.get_acquisitions(None, granule, False)[0]

    tmp_dir = Path(tempfile.mkdtemp(prefix="tmp-work-", dir=workdir))
    try:
        if "SENTINEL" in acq.platform_id:
            _sentinel2_fmask(
                dataset_path,
                container,
                granule,
                out_fname,
                tmp_dir,
                cloud_buffer_distance,
                cloud_shadow_buffer_distance,
                parallax_test,
            )
        elif "LANDSAT" in acq.platform_id:
            _landsat_fmask(
                acq,
                out_fname,
                tmp_dir,
                cloud_buffer_distance,
                cloud_shadow_buffer_distance,
            )
        else:
            raise ValueError(f"Sensor {acq.platform_id!r} not supported")
    finally:
        if clean_up_working_files:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            print(f"Keeping work directory {tmp_dir!r}", file=sys.stderr)

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
            command.extend(["-co", f"{key}={value}"])

        if config_options:
            for key, value in config_options.items():
                command.extend(["--config", f"{key}", f"{value}"])

        command.extend([fname, out_fname])

        run_command(command, dirname(fname))

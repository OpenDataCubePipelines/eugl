from pathlib import Path

try:
    from importlib.metadata import distribution
except ImportError:
    # Running on pre-3.8 Python; use importlib-metadata package
    from importlib_metadata import distribution

import yaml
import rasterio
from idl_functions import histogram
from wagl.acquisition import xml_via_safe
import zipfile
import re


# TODO: Fix update to merge the dictionaries


def _get_eugl_metadata():
    dist = distribution("eugl")
    return {
        "software_versions": {
            "eugl": {"version": dist.version, "repo_url": dist.metadata.get("Home-page")}
        }
    }


def _get_fmask_metadata():
    base_info = _get_eugl_metadata()
    dist = distribution("python-fmask")
    base_info["software_versions"]["fmask"] = {
        "version": dist.version,
        "repo_url": dist.metadata.get("Home-page"),
    }

    return base_info


def _get_s2cloudless_metadata():
    base_info = _get_eugl_metadata()
    dist = distribution("s2cloudless")
    base_info["software_versions"]["s2cloudless"] = {
        "version": dist.version,
        "repo_url": dist.metadata.get("Home-page"),
    }

    return base_info


def get_gqa_metadata(gverify_executable):
    """get_gqa_metadata: provides processing metadata for gqa_processing

    :param gverify_executable: GQA version is determined from executable
    :returns metadata dictionary:
    """

    gverify_version = gverify_executable.split("_")[-1]
    base_info = _get_eugl_metadata()
    base_info["software_versions"]["gverify"] = {"version": gverify_version}

    return base_info


def _gls_version(ref_fname):
    # TODO a more appropriate method of version detection and/or population of metadata
    if "GLS2000_GCP_SCENE" in ref_fname:
        gls_version = "GLS_v1"
    else:
        gls_version = "GQA_v3"

    return gls_version


def fmask_metadata(
    fmask_img_path: Path,
    output_metadata_path: Path,
    cloud_buffer_distance: float = 150.0,
    cloud_shadow_buffer_distance: float = 300.0,
    parallax_test: bool = False,
):
    """
    Produce a yaml metadata document.

    :param fmask_img_path:
        A fully qualified name to the file containing the output
        from the import Fmask algorithm.

    :param output_metadata_path:
        A fully qualified name to a file that will contain the
        metadata.

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

    :return:
        None.  Metadata is written directly to disk.
    :rtype: None
    """
    with rasterio.open(fmask_img_path) as ds:
        hist = histogram(ds.read(1), minv=0, maxv=5)["histogram"]

    # base info (software versions)
    base_info = _get_fmask_metadata()

    # Classification schema
    # 0 -> Invalid
    # 1 -> Clear
    # 2 -> Cloud
    # 3 -> Cloud Shadow
    # 4 -> Snow
    # 5 -> Water

    # info will be based on the valid pixels only (exclude 0)
    # scaled probability density function
    pdf = hist[1:] / hist[1:].sum() * 100

    md = {
        "parameters": {
            "cloud_buffer_distance_metres": cloud_buffer_distance,
            "cloud_shadow_buffer_distance_metres": cloud_shadow_buffer_distance,
            "frantz_parallax_sentinel_2": parallax_test,
        },
        "percent_class_distribution": {
            "clear": float(pdf[0]),
            "cloud": float(pdf[1]),
            "cloud_shadow": float(pdf[2]),
            "snow": float(pdf[3]),
            "water": float(pdf[4]),
        },
    }

    for key, value in base_info.items():
        md[key] = value

    with output_metadata_path.open("w") as src:
        yaml.safe_dump(md, src, default_flow_style=False, indent=4)


def s2cloudless_metadata(
    prob_out_fname,
    mask_out_fname,
    metadata_out_fname,
    threshold,
    average_over,
    dilation_size,
):
    """
    Produce a yaml metadata document.

    :param prob_out_fname:
        A fully qualified name to the file containing the output
        from the probability layer from the s2cloudless algorithm.
    :type fname: str

    :param mask_out_fname:
        A fully qualified name to the file containing the output
        from the mask layer from the s2cloudless algorithm.
    :type fname: str

    :param metadata_out_fname:
        A fully qualified name to a file that will contain the
        metadata.
    :type metadata_out_fname: str

    :param threshold:
    :param average_over:
    :param dilation_size:

    :return:
        None.  Metadata is written directly to disk.
    :rtype: None
    """
    with rasterio.open(mask_out_fname) as ds:
        hist = histogram(ds.read(1), minv=0, maxv=2)["histogram"]

    # base info (software versions)
    base_info = _get_s2cloudless_metadata()

    # info will be based on the valid pixels only (exclude 0)
    # scaled probability density function
    pdf = hist[1:] / hist[1:].sum() * 100

    md = {
        "parameters": {
            "threshold": threshold,
            "average_over": average_over,
            "dilation_size": dilation_size,
        },
        "percent_class_distribution": {
            "clear": float(pdf[0]),
            "cloud": float(pdf[1]),
        },
    }

    for key, value in base_info.items():
        md[key] = value

    with open(metadata_out_fname, "w") as src:
        yaml.safe_dump(md, src, default_flow_style=False, indent=4)


def grab_offset_dict(dataset_path):
    """grab_offset_dict: get the offset values from zipped XML metadata file

    :param dataset_path: S2 dataset (zip file path)
    :returns metadata dictionary: {band_id: offset_value}
    """

    try:
        archive = zipfile.ZipFile(dataset_path)
    except IsADirectoryError:
        # not a .zip archive
        # in the NRT pipeline, the offsets have already been applied
        return {}

    xml_root = xml_via_safe(archive, dataset_path)

    # ESA image ids
    esa_ids = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
        "TCI",
    ]

    # ESA L1C upgrade introducing scaling/offset
    search_term = (
        "./*/Product_Image_Characteristics/Radiometric_Offset_List/RADIO_ADD_OFFSET"
    )

    return {
        re.sub(r"B[0]?", "", esa_ids[int(x.attrib["band_id"])]): int(x.text)
        for x in xml_root.findall(search_term)
    }

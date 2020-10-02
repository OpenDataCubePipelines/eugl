try:
    from importlib.metadata import distribution
except ImportError:
    # Running on pre-3.8 Python; use importlib-metadata package
    from importlib_metadata import distribution

import yaml
import rasterio
from idl_functions import histogram


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
    fname,
    out_fname,
    cloud_buffer_distance=150.0,
    cloud_shadow_buffer_distance=300.0,
    parallax_test=False,
):
    """
    Produce a yaml metadata document.

    :param fname:
        A fully qualified name to the file containing the output
        from the import Fmask algorithm.
    :type fname: str

    :param out_fname:
        A fully qualified name to a file that will contain the
        metadata.
    :type out_fname: str

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

    :return:
        None.  Metadata is written directly to disk.
    :rtype: None
    """
    with rasterio.open(fname) as ds:
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

    with open(out_fname, "w") as src:
        yaml.safe_dump(md, src, default_flow_style=False, indent=4)

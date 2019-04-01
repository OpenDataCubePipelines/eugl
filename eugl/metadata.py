import rasterio
import fmask
from idl_functions import histogram

from eugl.version import __version__, REPO_URL

FMASK_REPO_URL = 'https://bitbucket.org/chchrsc/python-fmask'

# TODO: Fix update to merge the dictionaries


def _get_eugl_metadata():
    return {
        'software_versions': {
            'eugl': {
                'version': __version__,
                'repo_url': REPO_URL,
            }
        }
    }


def _get_fmask_metadata():
    base_info = _get_eugl_metadata()
    base_info['software_versions']['fmask'] = {
        'version': fmask.__version__,
        'repo_url': FMASK_REPO_URL
    }

    return base_info


def get_gqa_metadata(gverify_executable):
    """get_gqa_metadata: provides processing metadata for gqa_processing

    :param gverify_executable: GQA version is determined from executable
    :returns metadata dictionary:
    """

    gverify_version = gverify_executable.split('_')[-1]
    base_info = _get_eugl_metadata()
    base_info['software_versions']['gverify'] = {
        'version': gverify_version
    }

    return base_info


def _gls_version(ref_fname):
    # TODO a more appropriate method of version detection and/or population of metadata
    if 'GLS2000_GCP_SCENE' in ref_fname:
        gls_version = 'GLS_v1'
    else:
        gls_version = 'GQA_v3'

    return gls_version


def fmask_metadata(fname, out_fname, cloud_buffer_distance=150.0,
                   cloud_shadow_buffer_distance=300.0, parallax_test=False):
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
        hist = histogram(ds.read(1), minv=0, maxv=5)['histogram']

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
        "Parameters": {
            "Cloud Buffer Distance (metres)": cloud_buffer_distance,
            "Cloud Shadow Buffer Distance (metres)": cloud_shadow_buffer_distance,
            "Sentinel-2 Parallax (Frantz 2018)": parallax_test
        },
        "Class Distribution (%)": {
            "Clear": pdf[0],
            "Cloud": pdf[1],
            "Cloud Shadow": pdf[2],
            "Snow": pdf[3],
            "Water": pdf[4]
        }
    }

    for key, value in base_info.items():
        md[key] = value

    with open(out_fname, 'w') as src:
        yaml.safe_dump(md, src, default_flow_style=False, indent=4) 

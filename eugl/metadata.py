import fmask

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


def get_fmask_metadata():
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

"""
Acquisition info provides additional attributes to the Acquisition class
defined in the wagl package required for quality assessment.
"""

from wagl.acquisition.sentinel import Sentinel2Acquisition

from wagl.acquisition.landsat import (
    Landsat5Acquisition,
    Landsat7Acquisition,
    Landsat8Acquisition
)

from wagl.constants import GroupName, DatasetName


def get_land_ocean_bands(container, granule_id=None, product='NBAR'):
    """get_landocean_bands: Returns the h5py keys for the land/ocean bands

    :param container: container for the imagery
    :param product: which product to reference
    """

    acq = container.get_all_acquisitions()[0]
    ds_fmt = DatasetName.REFLECTANCE_FMT.value

    if not granule_id and len(container.granules) == 1:
        granule_id = container.granules[0]

    # TODO Need a better way to resolve resolution group for output bands
    if issubclass(acq.__class__, Sentinel2Acquisition):
        return {
            'land_band': (
                "{}/RES-GROUP-1/{}/".format(granule_id, GroupName.STANDARD_GROUP.value) +
                ds_fmt.format(product=product, band_name='BAND-11')
            ),
            'ocean_band': (
                "{}/RES-GROUP-1/{}/".format(granule_id, GroupName.STANDARD_GROUP.value) +
                ds_fmt.format(product=product, band_name='BAND-2')
            ),
        }
    elif issubclass(acq.__class__, Landsat5Acquisition):
        raise NotImplementedError
    elif issubclass(acq.__class__, Landsat7Acquisition):
        raise NotImplementedError
    elif issubclass(acq.__class__, Landsat8Acquisition):
        raise NotImplementedError

    raise RuntimeError("Unknown acquisition type")

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


def get_landocean_bands(container, product='NBAR'):
    """get_landocean_bands: Returns the h5py keys for the land/ocean bands

    :param container: container for the imagery
    :param product: which product to reference
    """

    # TODO Need a better way to resolve resolution group for output bands
    if isinstance(container, Sentinel2Acquisition):
        return {
            'land_band': (
                f"RES-GROUP-1/{GroupName.STANDARD_GROUP}/"
                "{DatasetName.REFLECTANCE_FMT}".format(
                    product=product, band_name='BAND-11'
                )
            ),
            'sea_band': (
                f"RES-GROUP-0/{GroupName.STANDARD_GROUP}/"
                f"{DatasetName.REFLECTANCE_FMT}".format(
                    product=product, band_name='BAND-2'
                )
            ),
        }
    elif isinstance(container, Landsat5Acquisition):
        raise NotImplementedError
    elif isinstance(container, Landsat7Acquisition):
        raise NotImplementedError
    elif isinstance(container, Landsat8Acquisition):
        raise NotImplementedError

    raise RuntimeError("Unknown acquisition type")

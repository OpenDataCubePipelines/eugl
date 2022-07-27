"""
Acquisition info provides additional attributes to the Acquisition class
defined in the wagl package required for quality assessment.
"""
from datetime import timezone

import fiona
from shapely.geometry import Polygon, shape
from rasterio.warp import Resampling

from wagl.acquisition.sentinel import Sentinel2Acquisition

from wagl.acquisition.landsat import (
    Landsat5Acquisition,
    Landsat7Acquisition,
    Landsat8Acquisition,
    Landsat9Acquisition,
)

from wagl.constants import GroupName, DatasetName
from eugl.gqa.geometric_utils import SLC_OFF

DS_FMT = DatasetName.REFLECTANCE_FMT.value

# TODO Need a better way to resolve resolution group for output bands?


class AcquisitionInfo:
    def __init__(self, container, granule, sample_acq):
        self.container = container
        self.granule = granule
        self.sample_acq = sample_acq

    @property
    def geobox(self):
        return self.sample_acq.gridded_geo_box()

    @property
    def timestamp(self):
        return self.sample_acq.acquisition_datetime.replace(tzinfo=timezone.utc)

    @property
    def tag(self):
        return self.sample_acq.tag

    @property
    def preferred_resampling_method(self):
        return Resampling.bilinear


class LandsatAcquisitionInfo(AcquisitionInfo):
    @property
    def path(self):
        return int(self.granule[3:6])

    @property
    def row(self):
        return int(self.granule[6:9])

    def is_land_tile(self, ocean_tile_list):
        path_row = "{},{}".format(self.path, self.row)

        with open(ocean_tile_list["Landsat"]) as fl:
            for line in fl:
                if path_row == line.strip():
                    return False

        return True

    def intersecting_landsat_scenes(self, landsat_scenes_shapefile):
        return [dict(path=self.path, row=self.row)]

    @property
    def preferred_gverify_method(self):
        return "fixed"


class Landsat5AcquisitionInfo(LandsatAcquisitionInfo):
    def land_band(self, product="NBAR"):
        return "{}/RES-GROUP-0/{}/{}".format(
            self.granule,
            GroupName.STANDARD_GROUP.value,
            DS_FMT.format(product=product, band_name="BAND-5"),
        )

    def ocean_band(self, product="NBAR"):
        return "{}/RES-GROUP-0/{}/{}".format(
            self.granule,
            GroupName.STANDARD_GROUP.value,
            DS_FMT.format(product=product, band_name="BAND-1"),
        )


class Landsat7AcquisitionInfo(LandsatAcquisitionInfo):
    def land_band(self, product="NBAR"):
        return "{}/RES-GROUP-1/{}/{}".format(
            self.granule,
            GroupName.STANDARD_GROUP.value,
            DS_FMT.format(product=product, band_name="BAND-5"),
        )

    def ocean_band(self, product="NBAR"):
        return "{}/RES-GROUP-1/{}/{}".format(
            self.granule,
            GroupName.STANDARD_GROUP.value,
            DS_FMT.format(product=product, band_name="BAND-1"),
        )

    @property
    def preferred_resampling_method(self):
        if self.timestamp >= SLC_OFF.replace(tzinfo=timezone.utc):
            return Resampling.nearest

        return Resampling.bilinear


class Landsat8AcquisitionInfo(LandsatAcquisitionInfo):
    def land_band(self, product="NBAR"):
        return "{}/RES-GROUP-1/{}/{}".format(
            self.granule,
            GroupName.STANDARD_GROUP.value,
            DS_FMT.format(product=product, band_name="BAND-6"),
        )

    def ocean_band(self, product="NBAR"):
        return "{}/RES-GROUP-1/{}/{}".format(
            self.granule,
            GroupName.STANDARD_GROUP.value,
            DS_FMT.format(product=product, band_name="BAND-2"),
        )


class Landsat9AcquisitionInfo(LandsatAcquisitionInfo):
    def land_band(self, product="NBAR"):
        return "{}/RES-GROUP-1/{}/{}".format(
            self.granule,
            GroupName.STANDARD_GROUP.value,
            DS_FMT.format(product=product, band_name="BAND-6"),
        )

    def ocean_band(self, product="NBAR"):
        return "{}/RES-GROUP-1/{}/{}".format(
            self.granule,
            GroupName.STANDARD_GROUP.value,
            DS_FMT.format(product=product, band_name="BAND-2"),
        )


class Sentinel2AcquisitionInfo(AcquisitionInfo):
    def land_band(self, product="NBAR"):
        return "{}/RES-GROUP-1/{}/{}".format(
            self.granule,
            GroupName.STANDARD_GROUP.value,
            DS_FMT.format(product=product, band_name="BAND-11"),
        )

    def ocean_band(self, product="NBAR"):
        return "{}/RES-GROUP-0/{}/{}".format(
            self.granule,
            GroupName.STANDARD_GROUP.value,
            DS_FMT.format(product=product, band_name="BAND-2"),
        )

    @property
    def tile_id(self):
        return self.granule.split("_")[-2][1:]

    def is_land_tile(self, ocean_tile_list):
        with open(ocean_tile_list["Sentinel-2"]) as fl:
            for line in fl:
                if self.tile_id == line.strip():
                    return False

        return True

    def intersecting_landsat_scenes(self, landsat_scenes_shapefile):
        landsat_scenes = fiona.open(landsat_scenes_shapefile)

        def path_row(properties):
            return dict(path=int(properties["PATH"]), row=int(properties["ROW"]))

        geobox = self.geobox
        polygon = Polygon(
            [geobox.ul_lonlat, geobox.ur_lonlat, geobox.lr_lonlat, geobox.ll_lonlat]
        )

        return [
            path_row(scene["properties"])
            for scene in landsat_scenes
            if shape(scene["geometry"]).intersects(polygon)
        ]

    @property
    def preferred_gverify_method(self):
        return "grid"


def acquisition_info(container, granule=None):
    if granule is None:
        if len(container.granules) == 1:
            granule = container.granules[0]
        else:
            raise ValueError("granule not specified for a multi-granule container")

    acqs, group = container.get_highest_resolution(granule)
    acq = acqs[0]

    if isinstance(acq, Sentinel2Acquisition):
        return Sentinel2AcquisitionInfo(container, granule, acq)
    elif isinstance(acq, Landsat5Acquisition):
        return Landsat5AcquisitionInfo(container, granule, acq)
    elif isinstance(acq, Landsat7Acquisition):
        return Landsat7AcquisitionInfo(container, granule, acq)
    elif isinstance(acq, Landsat8Acquisition):
        return Landsat8AcquisitionInfo(container, granule, acq)
    elif isinstance(acq, Landsat9Acquisition):
        return Landsat9AcquisitionInfo(container, granule, acq)
    else:
        raise ValueError("Unknown acquisition type")

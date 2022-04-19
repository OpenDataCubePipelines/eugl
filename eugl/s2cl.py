from os.path import join as pjoin

from s2cloudless import S2PixelCloudDetector
from rasterio.warp import reproject as rio_reproject
from rasterio.enums import Resampling
from rasterio.crs import CRS
import rasterio
import numpy as np

from wagl.acquisition import acquisitions
from eugl.metadata import s2cloudless_metadata

S2CL_BANDS = ["BAND-1", "BAND-2", "BAND-4", "BAND-5", "BAND-8",
              "BAND-8A", "BAND-9", "BAND-10", "BAND-11", "BAND-12"]


THRESHOLD = 0.4
AVERAGE_OVER = 4
DILATION_SIZE = 2


def s2cloudless_array(band_data,
                      threshold=THRESHOLD, average_over=AVERAGE_OVER, dilation_size=DILATION_SIZE):
    """
    :param band_data: array of shape (1, y, x, band_count=10) of DN values between 0.0 and 1.0
    :return: dict of arrays of shape (1, y, x) with entries 'cloud_prob' and 'cloud_mask'
    """
    cloud_detector = S2PixelCloudDetector(
        threshold=threshold, average_over=average_over, dilation_size=dilation_size
    )

    cloud_prob = cloud_detector.get_cloud_probability_maps(band_data)
    cloud_mask = cloud_detector.get_mask_from_prob(cloud_prob).astype(rasterio.uint8)

    noncontiguous = np.any(band_data == 0, axis=3)
    cloud_prob = np.where(~noncontiguous, cloud_prob, np.nan)
    cloud_mask = np.where(~noncontiguous, cloud_mask + 1, 0)

    return {'cloud_prob': cloud_prob, 'cloud_mask': cloud_mask}


def s2cloudless_container(container, granule=None,
                          threshold=THRESHOLD, average_over=AVERAGE_OVER, dilation_size=DILATION_SIZE):
    """
    :param container: wagl acquisition container
    :return: cloud_prob and cloud_mask with georeference info
    """
    bands = {f'BAND-{band.band_id}': band for band in get_all_acquisitions(container, granule=granule)}
    # reproject every band to band-1 (with resolution 60m x 60m)
    band_1 = bands[S2CL_BANDS[0]]

    def band_crs(band):
        return CRS.from_string(band.gridded_geo_box().crs.ExportToProj4())

    def band_transform(band):
        return band.gridded_geo_box().transform

    def reproject(band):
        result = np.empty(band_1.tile_size)

        rio_reproject(band.data(),
                      result,
                      src_transform=band_transform(band),
                      dst_transform=band_transform(band_1),
                      src_nodata=band.no_data,
                      dst_nodata=band_1.no_data,
                      src_crs=band_crs(band),
                      dst_crs=band_crs(band_1),
                      resampling=Resampling.bilinear)

        return result

    band_data = np.stack([reproject(bands[band_name])[np.newaxis, ...] for band_name in S2CL_BANDS],
                         axis=3) / 10000.0

    result = s2cloudless_array(band_data, threshold=threshold, average_over=average_over, dilation_size=dilation_size)

    return {**result, 'crs': band_crs(band_1), 'transform': band_transform(band_1)}


def s2cloudless_processing(
    dataset_path,
    granule,
    prob_out_fname,
    mask_out_fname,
    metadata_out_fname,
    workdir,
    acq_parser_hint=None,
    threshold=THRESHOLD,
    average_over=AVERAGE_OVER,
    dilation_size=DILATION_SIZE
):
    """
    Execute the s2cloudless process.

    :param dataset_path:
        A str containing the full file pathname to the dataset.
        The dataset can be either a directory or a file, and
        interpretable by wagl.acquisitions.
    :type dataset_path: str

    :param granule:
        A str containing the granule name. This will is used to
        selectively process a given granule.
    :type granule: str

    :param prob_out_fname:
        A fully qualified name to a file that will contain the
        cloud probability layer of the s2cloudless algorithm.
    :type out_fname: str

    :param mask_out_fname:
        A fully qualified name to a file that will contain the
        cloud mask layer of the s2cloudless algorithm.
    :type out_fname: str

    :param metadata_out_fname:
        A fully qualified name to a file that will contain the
        metadata from the s2cloudless process.
    :type metadata_out_fname: str

    :param workdir:
        A fully qualified name to a directory that can be
        used as scratch space for s2cloudless processing.
    :type workdir: str

    :param acq_parser_hint:
        A hinting helper for the acquisitions parser. Default is None.

    :param threshold:
    :param average_over:
    :param dilation_size:
    """
    container = acquisitions(dataset_path, acq_parser_hint)

    result = s2cloudless_container(container,
                                   threshold=threshold, average_over=average_over, dilation_size=dilation_size)

    cloud_prob = result['cloud_prob']

    export_kwargs = dict(
        mode="w",
        driver="GTiff",
        compress="deflate",
        height=cloud_prob.shape[1],
        width=cloud_prob.shape[2],
        count=1,
        crs=result['crs'],
        transform=result['transform'],
    )

    with rasterio.open(
        pjoin(workdir, prob_out_fname),
        dtype=cloud_prob.dtype,
        nodata=np.nan,
        **export_kwargs,
    ) as output:
        output.write(result['cloud_prob'])

    with rasterio.open(
        pjoin(workdir, mask_out_fname),
        dtype=rasterio.uint8,
        nodata=0,
        **export_kwargs
    ) as output:

        output.write(result['cloud_mask'])

    s2cloudless_metadata(
      pjoin(workdir, prob_out_fname),
      pjoin(workdir, mask_out_fname),
      pjoin(workdir, metadata_out_fname),
      threshold,
      average_over,
      dilation_size,
    )


def get_all_acquisitions(container, granule=None):
    """
    Get all acquisitions (including band 9 and 10).
    """
    acquisitions = []
    for group in container.groups:
        acqs = container.get_acquisitions(group, granule, only_supported_bands=False)
        if acqs:
            acquisitions.extend(acqs)

    return acquisitions

from s2cloudless import S2PixelCloudDetector
from rasterio.warp import reproject as rio_reproject
from rasterio.enums import Resampling
from rasterio.crs import CRS
import rasterio
import numpy as np

from wagl.acquisition import acquisitions
from .metadata import _get_s2cloudless_metadata

L1 = '/g/data/up71/projects/ARD_implementation_validation/ARD_fieldwork_L1C/S2B_MSIL1C_20180724T000239_N0206_R030_T55HGB_20180724T011606.zip'

S2CL_BANDS = ["BAND-1", "BAND-2", "BAND-4", "BAND-5", "BAND-8",
              "BAND-8A", "BAND-9", "BAND-10", "BAND-11", "BAND-12"]

def get_all_acquisitions(container, granule=None):
        acquisitions = []
        for group in container.groups:
            acqs = container.get_acquisitions(group, granule, only_supported_bands=False)
            if acqs:
                acquisitions.extend(acqs)

        return acquisitions

def s2cloudless(container, threshold=0.4, average_over=4, dilation_size=2):
    bands = {f'BAND-{band.band_id}': band for band in get_all_acquisitions(container)}
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

    cloud_detector = S2PixelCloudDetector(
        threshold=threshold, average_over=average_over, dilation_size=dilation_size
    )

    cloud_prob = cloud_detector.get_cloud_probability_maps(band_data)
    cloud_mask = cloud_detector.get_mask_from_prob(cloud_prob).astype(rasterio.uint8)

    noncontiguous = np.any(band_data == 0, axis=3)
    cloud_prob = np.where(~noncontiguous, cloud_prob, np.nan)
    cloud_mask = np.where(~noncontiguous, cloud_mask + 1, 0)

    export_kwargs = dict(
        mode="w",
        driver="GTiff",
        compress="deflate",
        height=cloud_prob.shape[1],
        width=cloud_prob.shape[2],
        count=1,
        crs=band_crs(band_1),
        transform=band_transform(band_1),
    )

    with rasterio.open(
        f"output_cloud_prob.tif",
        dtype=cloud_prob.dtype,
        nodata=np.nan,
        **export_kwargs,
    ) as output:
        output.write(cloud_prob)

    with rasterio.open(
        f"output_cloud_mask.tif",
        dtype=rasterio.uint8,
        nodata=0,
        **export_kwargs
    ) as output:
        output.write(cloud_mask)

    print(_get_s2cloudless_metadata())

s2cloudless(acquisitions(L1))


# def s2cloudless_safe(dataset, threshold=0.4, average_over=4, dilation_size=2):
#     """
#     A simple wrapper function that loads and reprojects
#     SAFE-format Sentinel-2 Level 1C data into a single
#     numpy array, then uses this to run the s2cloudless
#     classifier.
# 
#     This generates cloud probability layer and a cloud
#     mask GeoTIFF outputs.
#     """
# 
#     import rasterio
#     import numpy as np
#     from s2cloudless import S2PixelCloudDetector
# 
#     print(f"Processing {dataset.id}")
#     band_list = []
# 
#     # Load 60 m band as template
#     with rasterio.open(dataset.measurements["B01"]["path"]) as band:
#         print("    B01")
#         B01_data = band.read()
#         dst_transform = band.transform
#         dst_crs = band.crs
#         band_list.append(B01_data)
# 
#     # Reproject subsequent bands into template grid
#     for band_name in ["B02", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12"]:
#         print(f"    {band_name}")
#         with rasterio.open(dataset.measurements[band_name]["path"]) as src:
#             band_data = src.read()
#             dst_array = np.empty_like(B01_data)
#             rasterio.warp.reproject(
#                 band_data,
#                 dst_array,
#                 src_transform=src.transform,
#                 dst_transform=dst_transform,
#                 src_crs=src.crs,
#                 dst_crs=dst_crs,
#                 resampling=rasterio.warp.Resampling.bilinear,
#             )
#             band_list.append(dst_array)
# 
#     # Stack outputs into single (time, y, x, bands) array,
#     # and scale digital numbers between 0.0 and 1.0
#     # TODO: use correct band offsets/scalings to
#     # account for ESA update
#     bands = np.stack(band_list, axis=3) / 10000.0
# 
#     # Apply s2cloudless cloud detection
#     print("    Computing cloud probabilities and masks")
#     cloud_detector = S2PixelCloudDetector(
#         threshold=threshold, average_over=average_over, dilation_size=dilation_size
#     )
#     cloud_prob = cloud_detector.get_cloud_probability_maps(bands)
#     cloud_mask = cloud_detector.get_mask_from_prob(cloud_prob).astype(rasterio.uint8)
# 
#     # Apply contiguity mask to identify invalid results as any pixel
#     # missing data in any band. For cloud_mask, set nodata to 0,
#     # no clouds to 1 and clouds to 2 to match Fmask
#     noncontiguous = np.any(bands == 0, axis=3)
#     cloud_prob = np.where(~noncontiguous, cloud_prob, np.nan)
#     cloud_mask = np.where(~noncontiguous, cloud_mask + 1, 0)
# 
#     # Set up export kwargs
#     print("    Exporting to file")
#     export_kwargs = dict(
#         mode="w",
#         driver="GTiff",
#         compress="deflate",
#         height=cloud_prob.shape[1],
#         width=cloud_prob.shape[2],
#         count=1,
#         crs=dst_crs,
#         transform=dst_transform,
#     )
# 
#     # Export cloud probability
#     with rasterio.open(
#         f"{dataset.id}_cloud_prob.tif",
#         dtype=cloud_prob.dtype,
#         nodata=np.nan,
#         **export_kwargs,
#     ) as output:
#         output.write(cloud_prob)
# 
#     # Export cloud mask
#     with rasterio.open(
#         f"{dataset.id}_cloud_mask.tif",
#         dtype=rasterio.uint8,
#         nodata=0,
#         **export_kwargs
#     ) as output:
#         output.write(cloud_mask)
# 
#     return cloud_prob, cloud_mask

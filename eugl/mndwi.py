import h5py
import yaml
import numexpr
import numpy as np

from rasterio.warp import Resampling

from wagl.geobox import GriddedGeoBox
from wagl.tiling import generate_tiles
from wagl.data import reproject_array_to_array
from wagl.hdf5 import find, attach_image_attributes

from os.path import split as psplit


def compute_mndwi(green_im, swir_im):
    """
    Computes the MNDWI = (green_im - swir_im) / (green_im + swir_im)

    Parameters
    ----------
    green_im : numpy.ndarray
        The green band as a 2-dimensional numpy array
    swir_im : numpy.ndarray
        The SWIR band as a 2-dimensional numpy array

    Returns
    -------
    mndwi_im : numpy.ndarray
        The MNDWI as a 2-dimensional numpy array (np.float32)

    """

    mndwi_im = numexpr.evaluate("(green_im - swir_im) / (green_im + swir_im)")

    return mndwi_im


def get_mndwi_bands(granule, platform_id, product, paths):
    """
    Get the sensor name and the group paths to
    the green and swir bands used in the MNDWI
    calculation.

    Parameters
    ----------
    granule : str
        granule name e.g. LC81130822020034LGN00

    platform_id : str
        Platform name, e.g. LANDSAT_8, SENTINEL_2A, SENTINEL_2B

    product : str
        name of reflectance product from which the relevant
        MNDWI bands are selected from

    paths : list
        list of paths to datasets contained in the wagl h5 file

    Returns
    -------
    green_path : str
        path to green band used in the MNDWI calculation

    swir_path : str
        path to swir band used in the MNDWI calculation
    """
    if (platform_id == "LANDSAT_7") or (platform_id == "LANDSAT_5"):
        green_bname = "BAND-2"
        swir_bname = "BAND-5"

    if platform_id == "LANDSAT_8":
        green_bname = "BAND-3"
        swir_bname = "BAND-6"

    if (platform_id == "SENTINEL_2A") or (platform_id == "SENTINEL_2B"):
        green_bname = "BAND-3"
        swir_bname = "BAND-11"

    # search for the green and swir band
    green_path = [f for f in paths if (product in f and green_bname in f)]
    swir_path = [f for f in paths if (product in f and swir_bname in f)]

    if not green_path:
        raise Exception(f'could not find "{product}"{green_bname} in {granule}')

    if not swir_path:
        raise Exception(f'could not find "{product}"{swir_bname} in {granule}')

    return green_path[0], swir_path[0]


def mndwi(wagl_h5_file, granule, out_fname):
    """
    Computes the mndwi for a given granule in a wagl h5 file.

    Parameters
    ----------
    wagl_h5_file : str
        wagl-water-atcor generated h5 file

    granule : str
        Group path of the granule within the h5 file

    out_fname : str
        Output filename of the h5 file
    """

    # specify the reflectance products to use in generating mndwi
    products = ["LMBADJ"]

    # specify the resampling approach for the SWIR band
    resample_approach = Resampling.bilinear

    h5_fid = h5py.File(out_fname, "w")

    # find the granule index in the wagl_h5_file
    fid = h5py.File(wagl_h5_file, "r")
    granule_fid = fid[granule]
    paths = find(granule_fid, "IMAGE")

    # get platform name
    md = yaml.load(fid[granule + "/METADATA/CURRENT"][()], Loader=yaml.FullLoader)
    platform_id = md["source_datasets"]["platform_id"]

    # store mndwi-based products into a group
    mndwi_grp = h5_fid.create_group("mndwi")

    for i, prod in enumerate(products):

        # search the h5 groups & get paths to the green and swir bands
        green_path, swir_path = get_mndwi_bands(granule, platform_id, prod, paths)

        green_ds = granule_fid[green_path]
        chunks = green_ds.chunks
        nRows, nCols = green_ds.shape
        geobox = GriddedGeoBox.from_dataset(green_ds)
        nodata = green_ds.attrs["no_data_value"]

        # create output h5 attributes
        desc = "MNDWI derived with {0} and {1} ({2} reflectances)".format(
            psplit(green_path)[-1], psplit(swir_path)[-1], prod,
        )

        attrs = {
            "crs_wkt": geobox.crs.ExportToWkt(),
            "geotransform": geobox.transform.to_gdal(),
            "no_data_value": nodata,
            "granule": granule,
            "description": desc,
            "platform": platform_id,
            "spatial_resolution": abs(geobox.transform.a),
        }

        if platform_id.startswith("SENTINEL_2"):
            # we need to upscale the swir band
            swir_ds = granule_fid[swir_path]
            swir_im = reproject_array_to_array(
                src_img=swir_ds[:],
                src_geobox=GriddedGeoBox.from_dataset(swir_ds),
                dst_geobox=geobox,
                src_nodata=swir_ds.attrs["no_data_value"],
                dst_nodata=nodata,
                resampling=resample_approach,
            )
            attrs["SWIR_resampling_method"] = resample_approach.name

        else:
            swir_im = granule_fid[swir_path][:]

        # ------------------------- #
        #  Compute mndwi via tiles  #
        #   and save tiles to h5    #
        # ------------------------- #
        tiles = generate_tiles(
            samples=nRows, lines=nCols, xtile=chunks[1], ytile=chunks[0]
        )

        # create mndwi dataset
        mndwi_ds = mndwi_grp.create_dataset(
            f"mndwi_image_{prod}",
            shape=(nRows, nCols),
            dtype="float32",
            compression="lzf",
            chunks=chunks,
            shuffle=True,
        )

        for tile in tiles:
            green_tile = green_ds[tile]
            swir_tile = swir_im[tile]
            mndwi_tile = compute_mndwi(green_tile, swir_tile)

            # perform masking
            mask = (
                (green_tile == nodata)
                | (swir_tile == nodata)
                | (~np.isfinite(mndwi_tile))
            )
            mndwi_tile[mask] = nodata

            mndwi_ds[tile] = mndwi_tile

        # add attrs to dataset
        attach_image_attributes(mndwi_ds, attrs)

    fid.close()
    h5_fid.close()

#!/usr/bin/env python3

import h5py
import yaml
import numexpr
import numpy as np

from rasterio.warp import Resampling

from wagl.hdf5 import find
from wagl.geobox import GriddedGeoBox
from wagl.tiling import generate_tiles
from wagl.data import reproject_array_to_array

# from funcs_data import reproject_array_to_array

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


def get_RGB_bands(platform_id, product, paths):
    """
    Get the RGB band paths from the H5 file.

    platform_id : str
        Platform name, e.g. LANDSAT_8, SENTINEL_2A, SENTINEL_2B

    product : str
        name of reflectance product from which the relevant
        MNDWI bands are selected from

    paths : list
        list of paths to datasets contained in the wagl h5 file

    Returns
    -------
    red_path : str
        path to red band

    green_path : str
        path to green band

    blue_path : str
        path to blue band
    """

    # for sentinel-2, landsat-8 and WV2
    red_bname = "BAND-4"
    green_bname = "BAND-3"
    blue_bname = "BAND-2"

    if (platform_id == "LANDSAT_7") or (platform_id == "LANDSAT_5"):
        red_bname = "BAND-3"
        green_bname = "BAND-2"
        blue_bname = "BAND-1"

    # search for the RGB bands
    red_path = [f for f in paths if (product in f and red_bname in f)]
    green_path = [f for f in paths if (product in f and green_bname in f)]
    blue_path = [f for f in paths if (product in f and blue_bname in f)]

    if not red_path:
        raise Exception(f'could not find "{product}"{red_bname}')

    if not green_path:
        raise Exception(f'could not find "{product}"{green_bname}')

    if not blue_path:
        raise Exception(f'could not find "{product}"{blue_bname}')

    return red_path[0], green_path[0], blue_path[0]


def seadas_style_scaling(band_im, sfactor):
    """
    Perform a (NASA-OBPG) SeaDAS style scaling on a single band.
    When applied to the red, green and blue bands this scaling
    generates very impressive RGB's

    Parameters
    ----------
    band_im : numpy.ndarray
        reflectance image at either the  band [nRows, nCols]

    sfactor : float or int
        The multiplicative factor used to convert to reflectances

    Returns
    -------
    scaled_im : numpy.ndarray [dtype = np.uint8]
        Scaled and enhanced band with dimensions of [nRows, nCols].

    Raises
    ------
        * Exception if len(rgb_ix) != 3

    """

    # specify coefficient used in transformation:
    c1 = 0.091935692
    c2 = 0.61788
    c3 = 10.0
    c4 = -0.015

    scaled_im = c1 + c2 * np.arctan(c3 * (band_im / sfactor + c4))
    scaled_im[scaled_im < 0] = 0
    scaled_im[scaled_im > 1] = 1
    return np.array(255 * scaled_im, order="C", dtype=np.uint8)


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
    # products = ["LMBADJ"]
    products = ["LAMBERTIAN", "LMBSKYG", "LMBADJ"]

    h5_fid = h5py.File(out_fname, "w")

    # find the granule index in the wagl_h5_file
    fid = h5py.File(wagl_h5_file, "r")
    grp = fid[granule]
    paths = find(grp, "IMAGE")

    # get platform name
    md = yaml.load(fid[granule + "/METADATA/CURRENT"][()], Loader=yaml.FullLoader)
    platform_id = md["source_datasets"]["platform_id"]

    # Get the 1st (lowerVal_arr) and 99th (upperVal_arr) percentikes
    # for the each of the products. These arrays are very useful
    # for creating MNDWI images, but this code will be removed once
    # testing is complete.
    lowerVal_arr = np.zeros([len(products)], order="C", dtype=np.float32)
    upperVal_arr = np.zeros([len(products)], order="C", dtype=np.float32)

    # store mndwi-based products into a group
    mndwi_grp = h5_fid.create_group("mndwi")

    for i, prod in enumerate(products):

        # search the h5 groups & get paths to the green and swir bands
        green_path, swir_path = get_mndwi_bands(granule, platform_id, prod, paths)

        green_ds = grp[green_path]
        green_im = green_ds[:]
        geobox = GriddedGeoBox.from_dataset(green_ds)
        spatial_res = abs(geobox.transform.a)
        nodata = green_ds.attrs["no_data_value"]

        if platform_id.startswith("SENTINEL_2"):
            # we need to upscale the swir band
            swir_ds = grp[swir_path]
            swir_im = reproject_array_to_array(
                src_img=swir_ds[:],
                src_geobox=GriddedGeoBox.from_dataset(swir_ds),
                dst_geobox=geobox,
                src_nodata=swir_ds.attrs["no_data_value"],
                dst_nodata=nodata,
                resampling=Resampling.bilinear,
            )

        else:
            swir_im = grp[swir_path][:]

        # ------------------------- #
        #  Compute mndwi via tiles  #
        #   and save tiles to h5    #
        # ------------------------- #
        nRows, nCols = green_im.shape
        tiles = generate_tiles(
            samples=nRows, lines=nCols, xtile=green_ds.chunks[1], ytile=green_ds.chunks[0]
        )

        # create mndwi dataset
        mndwi_ds = mndwi_grp.create_dataset(
            f"mndwi_image_{prod}",
            shape=green_im.shape,
            dtype="float32",
            compression="lzf",
            chunks=green_ds.chunks,
            shuffle=True,
        )

        # create h5 attributes
        desc = "MNDWI ({0} m) derived with {1} and {2} ({3} reflectances)".format(
            int(spatial_res), psplit(green_path)[-1], psplit(swir_path)[-1], prod,
        )

        attrs = {
            "crs_wkt": geobox.crs.ExportToWkt(),
            "geotransform": geobox.transform.to_gdal(),
            "no_data_value": nodata,
            "granule": granule,
            "description": desc,
            "n_lines": nRows,
            "n_samples": nCols,
            "platform": platform_id,
            "spatial_resolution": spatial_res,
        }

        # add attrs to dataset
        for key in attrs:
            mndwi_ds.attrs[key] = attrs[key]

        mask = (green_im == nodata) | (swir_im == nodata)
        mndwi_im = np.zeros([nRows, nCols], order="C", dtype=np.float32)
        for tile in tiles:
            mndwi_tile = compute_mndwi(green_im[tile], swir_im[tile])

            # perform masking
            mndwi_tile[~np.isfinite(mndwi_tile)] = nodata
            mndwi_tile[mask[tile]] = nodata

            mndwi_ds[tile] = mndwi_tile
            mndwi_im[tile] = mndwi_tile

        lowerVal, UpperVal = np.percentile(mndwi_im[~mask], (1, 99))

        lowerVal_arr[i] = lowerVal
        upperVal_arr[i] = UpperVal

    h5_fid.create_dataset(
        "mndwi_1st_percentiles",
        data=lowerVal_arr,
        shape=lowerVal_arr.shape,
        dtype="float32",
        compression="lzf",
    )

    h5_fid.create_dataset(
        "mndwi_99th_percentiles",
        data=upperVal_arr,
        shape=upperVal_arr.shape,
        dtype="float32",
        compression="lzf",
    )

    del green_im, swir_im, mndwi_im

    # ----------------------------------- #
    #  Create  an rgb image (uint8) that  #
    #  can directly be  opened from  the  #
    #  MNDWI H5 file. The following code  #
    #  will be deleted after testing      #
    # ----------------------------------- #

    # get the rgb band paths from the first product
    red_path, grn_path, blu_path = get_RGB_bands(platform_id, products[0], paths)
    rgb_paths = [red_path, grn_path, blu_path]
    rgb_names = ["red", "green", "blue"]

    # create a group
    rgb_grp = h5_fid.create_group("seadas_rgb")

    # because no reprojection is necessary, we can load
    # the red & blue bands as tiles to conserve memory
    for i in range(0, len(rgb_names)):

        chunks = grp[rgb_paths[i]].chunks

        rgb_ds = rgb_grp.create_dataset(
            rgb_names[i],
            shape=(nRows, nCols),
            dtype="uint8",
            compression="lzf",
            chunks=chunks,
            shuffle=True,
        )

        # create h5 attributes
        desc = "scaled {0} band ({1} m) derived from {2} {3}".format(
            rgb_names[i], int(spatial_res), products[0], psplit(rgb_paths[i])[-1],
        )
        attrs["description"] = desc

        # add attrs to dataset
        for key in attrs:
            rgb_ds.attrs[key] = attrs[key]

        # For some reason I have to recreate the tiles. Attempts
        # to reuse the tiles from above were unsuccessful.
        tiles = generate_tiles(
            samples=nRows, lines=nCols, xtile=chunks[1], ytile=chunks[0]
        )

        for tile in tiles:
            rgb_ds[tile] = seadas_style_scaling(grp[rgb_paths[i]][tile], 10000.0)

    fid.close()
    h5_fid.close()

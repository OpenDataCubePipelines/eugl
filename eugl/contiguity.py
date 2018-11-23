# coding=utf-8
"""
Execution method for contiguous observations within band stack

example usage:
    contiguity.py <allbands.vrt>
    --output /tmp/
"""
from __future__ import absolute_import
import os
import logging
from pathlib import Path
import rasterio
import numpy as np
import click
from wagl.data import write_img
from wagl.geobox import GriddedGeoBox

os.environ["CPL_ZIP_ENCODING"] = "UTF-8"


def contiguity(fname, output, platform):
    """
    Write a contiguity mask file based on the intersection of valid data pixels across all
    bands from the input file and output to the specified directory
    """
    with rasterio.open(fname) as ds:
        geobox = GriddedGeoBox.from_dataset(ds)
        yblock, xblock = ds.block_shapes[0]
        ones = np.ones((ds.height, ds.width), dtype='uint8')
        for band in ds.indexes:
            ones &= ds.read(band) > 0

    # setting the contiguity's  block size depending on the specific sensor.
    # Currently, only USGS dataset are tiled at 512 x 512 for standardizing
    # Level 2 ARD products. Sentinel-2 tile size are inherited from the
    # L1C products and its overview's blocksize are default value of GDAL's
    # overview block size of 128 x 128

    # TODO Standardizing the Sentinel-2's overview tile size with external inputs

    if platform == "LANDSAT":
        blockxsize = 512
        blockysize = 512
        config_options = {'GDAL_TIFF_OVR_BLOCKSIZE': blockxsize}
    else:
        blockysize = yblock
        blockxsize = xblock
        config_options = None

    options = {'compress': 'deflate',
               'zlevel': 4,
               'blockxsize': blockxsize,
               'blockysize': blockysize}

    write_img(ones, output, cogtif=True, levels=[2, 4, 8, 16, 32],
              geobox=geobox, options=options, config_options=config_options)

    return ones


@click.command(help=__doc__)
@click.option('--output', help="Write contiguity datasets into this directory",
              type=click.Path(exists=False, writable=True, dir_okay=True))
@click.argument('datasets',
                type=click.Path(exists=True, readable=True, writable=False),
                nargs=-1)
def main(output, datasets):
    """
    For input 'vrt' generate Contiguity
    outputs and write to the destination path specified by 'output'
    """
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    for dataset in datasets:
        path = dataset
        stem = Path(path).stem
        out = os.path.join(output, stem)
        contiguity = out+".CONTIGUITY.TIF"
        logging.info("Create contiguity image " + contiguity)
        contiguity(path, contiguity)

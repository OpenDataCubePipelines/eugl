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


def contiguity(fname):
    """
    Write a contiguity mask file based on the intersection of valid data pixels across all
    bands from the input file and returns with the geobox of the source dataset
    """
    with rasterio.open(fname) as ds:
        geobox = GriddedGeoBox.from_dataset(ds)
        yblock, xblock = ds.block_shapes[0]
        ones = np.ones((ds.height, ds.width), dtype="uint8")
        for band in ds.indexes:
            ones &= ds.read(band) > 0

    return ones, geobox


@click.command(help=__doc__)
@click.option(
    "--output",
    help="Write contiguity datasets into this directory",
    type=click.Path(exists=False, writable=True, dir_okay=True),
)
@click.argument(
    "datasets", type=click.Path(exists=True, readable=True, writable=False), nargs=-1
)
@click.option(
    "--platform", help=" Sensor platform where dataset is source from.", default=None
)
def main(output, datasets, platform):
    """
    For input 'vrt' generate Contiguity
    outputs and write to the destination path specified by 'output'
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
    )
    for dataset in datasets:
        path = dataset
        stem = Path(path).stem
        out = os.path.join(output, stem)
        contiguity_img = out + ".CONTIGUITY.TIF"
        logging.info("Create contiguity image %s", contiguity_img)
        contiguity_data, geobox = contiguity(path)

        write_img(
            contiguity_data,
            contiguity_img,
            geobox=geobox,
            options={"compress": "deflate", "zlevel": 4},
            config_options={},
        )

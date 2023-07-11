#!/usr/bin/env python3

import os

import inspect

import sys
from pathlib import Path
from typing import Dict
import traceback
import argparse
import logging


from click import echo, secho

from eodatasets3.prepare import sentinel_l1_prepare

import datacube.utils.uris as dc_uris

# Needed when packaging zip or tar files.
dc_uris.register_scheme("zip", "tar")


# Cut-down integration test datasets. Small and stable but ... not real pixels.
EOD3_TEST_DATASETS = dict(
    ls8_tar="tests/integration/data/LC09_L1TP_112081_20220209_20220209_02_T1.tar",
    ls8_dir="tests/integration/data/LC08_L1GT_089074_20220506_20220512_02_T2",
    s2_zip="tests/integration/data/esa_s2_l1c/S2B_MSIL1C_20201011T000249_N0209_R030_T55HFA_20201011T011446.zip",
    s2_multigran="tests/integration/data/multi-granule/S2A_OPER_PRD_MSIL1C_PDMC_20161213T162432_R088_V20151007T012016_20151007T012016.zip",
    s2_sinergise="tests/integration/data/sinergise_s2_l1c/S2B_MSIL1C_20201011T000249_N0209_R030_T55HFA_20201011T011446",
)


# Real datasets stored on NCI.
NCI_S2_TESTS = dict(
    s2_new_format="/g/data/fj7/Copernicus/Sentinel-2/MSI/L1C/2023/2023-02/35S145E-40S150E/S2B_MSIL1C_20230211T001109_N0509_R073_T55HEB_20230211T012907.zip",
    s2_older_format="/g/data/fj7/Copernicus/Sentinel-2/MSI/L1C/2016/2016-12/25S135E-30S140E/S2A_OPER_PRD_MSIL1C_PDMC_20161204T065230_R002_V20161204T005702_20161204T005702.zip",
    # Some with valid past ARD outputs.
    # Near Canberra
    s2_near_canberra_multigran="/g/data/fj7/Copernicus/Sentinel-2/MSI/L1C/2015/2015-11/35S145E-40S150E/S2A_OPER_PRD_MSIL1C_PDMC_20170207T120115_R030_V20151122T000632_20151122T000632.zip",
    # Near Newcastle
    s2_near_newcastle_multigran="/g/data/fj7/Copernicus/Sentinel-2/MSI/L1C/2016/2016-01/30S150E-35S155E/S2A_OPER_PRD_MSIL1C_PDMC_20160118T230529_R130_V20160117T235915_20160117T235915.zip",
    # Darwin wet season.
    s2_darwin_wet_season="/g/data/fj7/Copernicus/Sentinel-2/MSI/L1C/2018/2018-01/10S130E-15S135E/S2A_MSIL1C_20180130T014031_N0206_R031_T52LFM_20180130T063131.zip",
)
NCI_LS_TESTS = dict(
    ls7="/g/data/da82/AODH/USGS/L1/Landsat/C1/104_076/LE71040762021365/LE07_L1TP_104076_20211231_20220126_01_T1.tar",
    ls8="/g/data/up71/projects/ARD_implementation_validation/ARD_update_for_USGS_C2L1/input/LC08_L1GT_109080_20210601_20210608_02_T2.tar",
    ls9="/g/data/da82/AODH/USGS/L1/Landsat/C2/091_084/LC90910842022276/LC09_L1TP_091084_20221003_20230327_02_T1.tar",
)
NCI_S2_PROBLEM_DATASETS = dict(
    s2_old_failing_scene="/g/data/fj7/Copernicus/Sentinel-2/MSI/L1C/2023/2023-05/40S140E-45S145E/S2B_MSIL1C_20230515T002059_N0509_R116_T55GCP_20230515T014126.zip",
    s2_all_nan_fmask="/g/data/fj7/Copernicus/Sentinel-2/MSI/L1C/2023/2023-03/40S140E-45S145E/S2B_MSIL1C_20230306T002059_N0509_R116_T55GCP_20230306T014524.zip",
)
NCI_FUQIN_S2_TEST_DATASETS = dict(
    # Fuqin's test scenes.
    s2_t54jwl_2021="/g/data/fj7/Copernicus/Sentinel-2/MSI/L1C/2021/2021-12/30S140E-35S145E/S2A_MSIL1C_20211212T003701_N0301_R059_T54JWL_20211212T020702.zip",
    s2_t55hfa_2022="/g/data/fj7/Copernicus/Sentinel-2/MSI/L1C/2022/2022-02/35S145E-40S150E/S2A_MSIL1C_20220218T000241_N0400_R030_T55HFA_20220218T012533.zip",
    s2_t54jwl_2017="/g/data/fj7/Copernicus/Sentinel-2/MSI/L1C/2017/2017-12/30S140E-35S145E/S2B_MSIL1C_20171208T003659_N0206_R059_T54JWL_20171208T034537.zip",
    s2_t55hev_2021="/g/data/fj7/Copernicus/Sentinel-2/MSI/L1C/2021/2021-05/35S145E-40S150E/S2B_MSIL1C_20210522T001109_N0300_R073_T55HEV_20210522T012307.zip",
)

# Combine all test dicts together
NCI_ALL_TEST_DATASETS = dict(
    **NCI_LS_TESTS,
    **NCI_S2_TESTS,
    **NCI_S2_PROBLEM_DATASETS,
    **NCI_FUQIN_S2_TEST_DATASETS,
)


def main(
    datasets: Dict[str, str],
    output_path: Path = Path("fmask_test_outputs"),
    debug: bool = True,
    write_thumb=True,
):
    import eugl.fmask as eugl_fmask

    datasets = {k: Path(p) for k, p in datasets.items()}
    # Fail fast.
    for dataset_name, dataset in datasets.items():
        if not dataset.exists():
            raise ValueError(f"Dataset {dataset} does not exist")

    failures = 0
    for dataset_name, dataset in datasets.items():
        output_dir = output_path / dataset_name
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        from wagl.acquisition import acquisitions

        try:
            optional_args = {}
            if "clean_up_working_files" in inspect.signature(eugl_fmask.fmask).parameters:
                optional_args["clean_up_working_files"] = not debug

            container = acquisitions(str(dataset))
            for granule in container.granules:
                granule = granule or "all"
                secho(f"Running fmask on {dataset_name!r} granule {granule!r}", fg="blue")

                granule_dir = output_dir / granule
                granule_dir.mkdir(exist_ok=True)
                work = granule_dir / "work"
                work.mkdir(exist_ok=True)

                out_fname = granule_dir / "fmask.img"
                metadata_out_fname = granule_dir / "fmask.yaml"

                if not metadata_out_fname.exists():
                    eugl_fmask.fmask(
                        dataset_path=dataset.as_posix(),
                        granule=granule,
                        out_fname=str(out_fname),
                        metadata_out_fname=str(metadata_out_fname),
                        workdir=str(work),
                        cloud_buffer_distance=150.0,
                        cloud_shadow_buffer_distance=300.0,
                        parallax_test=False,
                        **optional_args,
                    )
                    echo(f"Output written to {out_fname.as_posix()!r}")

                if write_thumb:
                    odc_yaml = granule_dir / "dataset.odc-metadata.yaml"
                    if not odc_yaml.exists():
                        if "S2" in dataset.name.upper():
                            sentinel_l1_prepare.prepare_and_write(
                                dataset_location=dataset,
                                output_yaml=odc_yaml,
                                producer="esa.int",
                                embed_location=True,
                            )
                        else:
                            from eodatasets3.prepare import landsat_l1_prepare

                            landsat_l1_prepare.prepare_and_write(
                                ds_path=dataset,
                                output_yaml_path=odc_yaml,
                                producer="usgs.gov",
                                embed_location=True,
                            )
                        echo(f"ODC metadata written to {odc_yaml.as_posix()!r}")

                    thumb_jpg = output_dir / f"{granule}.jpg"
                    if not thumb_jpg.exists():
                        create_thumbnail(odc_yaml, thumb_jpg)
                        echo(f"Thumbnail written to {thumb_jpg.as_posix()!r}")

        except Exception as e:
            secho(f"Error processing {dataset.as_posix()!r}: {str(e)}", fg="red")
            traceback.print_exc()
            failures += 1
            continue

    sys.exit(failures)


def cli():
    parser = argparse.ArgumentParser(
        description="Run fmask alone on a series of datasets. "
        "The output can then be compared to the reference outputs."
    )
    parser.add_argument("datasets", nargs="*", help="Optional dataset paths")
    parser.add_argument(
        "--test-data-nci",
        default=False,
        action="store_true",
        help="Run default test datasets on NCI",
    )
    parser.add_argument(
        "--test-data-eod3",
        default=False,
        action="store_true",
        help="Run default test datasets from eodatasets3 repo",
    )
    parser.add_argument(
        "--out-path", default="fmask_test_outputs", help="Path to store outputs"
    )
    parser.add_argument(
        "--quiet", default=False, action="store_true", help="Set logging to INFO level"
    )
    args = parser.parse_args()

    log_level = logging.INFO if args.quiet else logging.DEBUG
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    import eugl.fmask as eugl_fmask

    eugl_fmask._LOG.setLevel(log_level)

    datasets = {str(k): v for k, v in enumerate(args.datasets)}
    if args.test_data_nci:
        datasets.update(NCI_ALL_TEST_DATASETS)
    if args.test_data_eod3:
        eod3 = Path(
            os.getenv(
                "EODATASETS3_PATH", Path(__file__).resolve().parent.parent / "eo-datasets"
            )
        )
        if not eod3.exists():
            raise ValueError(
                "Please clone eo-datasets3 into the parent directory "
                "of this repository, or set $EODATASETS3_PATH."
            )
        datasets.extend(eod3 / p for p in EOD3_TEST_DATASETS)

    main(datasets, Path(args.out_path), debug=not args.quiet)


def create_thumbnail(
    dataset_path: Path, thumbnail_out: Path, r_g_b=("red", "green", "blue")
):
    """
    Read the given dataset doc, and create a thumbnail for it using the default settings.
    """
    from eodatasets3 import serialise as eod3_serialise
    from eodatasets3.images import FileWrite
    from urllib.parse import urljoin

    d = eod3_serialise.from_path(dataset_path, skip_validation=True)

    location = d.locations[0]

    FileWrite().create_thumbnail(
        tuple(urljoin(location, d.measurements[band].path) for band in r_g_b),
        thumbnail_out,
    )


if __name__ == "__main__":
    cli()

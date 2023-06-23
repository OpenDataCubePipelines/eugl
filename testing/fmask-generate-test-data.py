import os

import inspect

import sys
from pathlib import Path
from typing import List
import traceback
import argparse
import logging


from click import echo, secho

# Cut-down integration test datasets. Small and stable but ... not real pixels.
EOD3_TEST_DATASETS = (
    "tests/integration/data/LC09_L1TP_112081_20220209_20220209_02_T1.tar",
    "tests/integration/data/LC08_L1GT_089074_20220506_20220512_02_T2",
    "tests/integration/data/esa_s2_l1c/S2B_MSIL1C_20201011T000249_N0209_R030_T55HFA_20201011T011446.zip",
    "tests/integration/data/multi-granule/S2A_OPER_PRD_MSIL1C_PDMC_20161213T162432_R088_V20151007T012016_20151007T012016.zip",
    "tests/integration/data/sinergise_s2_l1c/S2B_MSIL1C_20201011T000249_N0209_R030_T55HFA_20201011T011446",
)

# Real datasets stored on NCI.
NCI_TEST_DATASETS = (
    # LS7
    "/g/data/da82/AODH/USGS/L1/Landsat/C1/104_076/LE71040762021365/LE07_L1TP_104076_20211231_20220126_01_T1.tar",
    # LS8
    "/g/data/up71/projects/ARD_implementation_validation/ARD_update_for_USGS_C2L1/input/LC08_L1GT_109080_20210601_20210608_02_T2.tar",
    # LS9
    "/g/data/da82/AODH/USGS/L1/Landsat/C2/091_084/LC90910842022276/LC09_L1TP_091084_20221003_20230327_02_T1.tar",
    # New-format S2 with NaN fmask.
    "/g/data/fj7/Copernicus/Sentinel-2/MSI/L1C/2023/2023-03/40S140E-45S145E/S2B_MSIL1C_20230306T002059_N0509_R116_T55GCP_20230306T014524.zip",
    # New format S2 with valid fmask.
    "/g/data/fj7/Copernicus/Sentinel-2/MSI/L1C/2023/2023-02/35S145E-40S150E/S2B_MSIL1C_20230211T001109_N0509_R073_T55HEB_20230211T012907.zip",
    # Older format S2
    "/g/data/fj7/Copernicus/Sentinel-2/MSI/L1C/2016/2016-12/25S135E-30S140E/S2A_OPER_PRD_MSIL1C_PDMC_20161204T065230_R002_V20161204T005702_20161204T005702.zip",
    # Multi-granule test data. Is there a real one somewhere?
    "/g/data/v10/agdc/jez/eo-datasets/tests/integration/data/multi-granule/S2A_OPER_PRD_MSIL1C_PDMC_20161213T162432_R088_V20151007T012016_20151007T012016.zip",
)


def main(
    datasets: List[str],
    output_path: Path = Path("fmask_test_outputs"),
    debug: bool = True,
):
    import eugl.fmask as eugl_fmask

    datasets = [Path(p) for p in datasets]
    for dataset in datasets:
        if not dataset.exists():
            raise ValueError(f"Dataset {dataset} does not exist")

    failures = 0
    for dataset in datasets:
        output_dir = output_path / Path(dataset).stem
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        out_fname = output_dir / "fmask.img"
        metadata_out_fname = output_dir / "fmask.yaml"

        work = output_dir / "work"
        work.mkdir(exist_ok=True)

        secho(f"Running fmask on {dataset.as_posix()!r}", fg="blue")
        try:
            optional_args = {}
            if "clean_up_working_files" in inspect.signature(eugl_fmask.fmask).parameters:
                optional_args["clean_up_working_files"] = not debug

            eugl_fmask.fmask(
                dataset_path=dataset.as_posix(),
                granule=None,
                out_fname=str(out_fname),
                metadata_out_fname=str(metadata_out_fname),
                workdir=str(work),
                acq_parser_hint=None,
                cloud_buffer_distance=150.0,
                cloud_shadow_buffer_distance=300.0,
                parallax_test=False,
                **optional_args,
            )
            echo(f"Output written to {out_fname.as_posix()!r}")
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

    datasets = list(args.datasets)
    if args.test_data_nci:
        datasets.extend(NCI_TEST_DATASETS)
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


if __name__ == "__main__":
    cli()

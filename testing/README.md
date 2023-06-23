
### eugl fmask testing

Example session to run tests for eugl's fmask test on NCI.

Load your modules (whatever the latest prod module is):

	module use /g/data/v10/public/modules/modulefiles /g/data/v10/private/modules/modulefiles
	module load ard-pipeline/20230325-all

Generate test data using the that code, and our default NCI datasets.

	./fmask-generate-test-data.py --test-data-nci --out-path output-original

Now repeat this with your own dev version of eugl: prepended it to the PYHTHONPATH,
and write the output a different directory.

Example:

    export PYTHONPATH="~/eugl:${PYTHONPATH}"
    ./fmask-generate-test-data.py --test-data-nci --out-path output-dev

Then run a comparison of the two output directories:

	./cmp-fmask-outputs.py output-original output-dev

It will compare both imagery and metadata.

## Raw commands

```bash
❯ ./fmask-generate-test-data.py --help
usage: fmask-generate-test-data.py [-h] [--test-data-nci] [--test-data-eod3] [--out-path OUT_PATH]
                                   [--quiet]
                                   [datasets ...]

Run fmask alone on a series of datasets. The output can then be compared to the reference outputs.

positional arguments:
  datasets             Optional dataset paths

options:
  -h, --help           show this help message and exit
  --test-data-nci      Run default test datasets on NCI
  --test-data-eod3     Run default test datasets from eodatasets3 repo
  --out-path OUT_PATH  Path to store outputs
  --quiet              Set logging to INFO level

```

```bash
❯ ./cmp-fmask-outputs.py --help
usage: cmp-fmask-outputs.py [-h] original_directory new_directory

Compare fmask outputs for differences

positional arguments:
  original_directory  Path to the original directory
  new_directory       Path to the new directory

options:
  -h, --help          show this help message and exit

```

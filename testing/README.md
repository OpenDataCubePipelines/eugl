

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

#!/bin/bash

set -xe

printenv;

[ "$AZUREML_DATAREFERENCE_ocrd" ]

pushd /tmp; python3 -m pip uninstall latent-diffusion -y; popd;
python3 -m pip install -e .;

export LAION_ROOT_DIR="${AZUREML_DATAREFERENCE_ocrd}/Vertical/vdipainter/data"
ls "$LAION_ROOT_DIR";

python3 ldm/data/laion.py --name laion/laion2B-en-aesthetic --mode download;
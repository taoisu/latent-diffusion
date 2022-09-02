#!/bin/bash

set -xe

[ "$AZUREML_DATAREFERENCE_ocrd" ]

printenv;
python3 -m pip install -e . --force;

export LAION_ROOT_DIR="${AZUREML_DATAREFERENCE_ocrd}/Vertical/vdipainter/data"
ls "$LAION_ROOT_DIR";

python3 ldm/data/laion.py --name laion/laion2B-en-aesthetic --mode download;
#!/bin/bash

set -xe

[ "$AZUREML_DATAREFERENCE_ocrd" ]

printenv;
python3 -m pip install -e . --force;

export AVID_ROOT_DIR="${AZUREML_DATAREFERENCE_ocrd}/Vertical/vdipainter/data/avidxchange"
ls "$AVID_ROOT_DIR";

python3 ldm/data/avid.py --mode gen;
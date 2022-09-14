#!/bin/bash

set -xe

printenv;

[ "$AZUREML_DATAREFERENCE_ocrd" ]

pushd /tmp; python3 -m pip uninstall latent-diffusion -y; popd;
python3 -m pip install -e .;

export AVID_ROOT_DIR="${AZUREML_DATAREFERENCE_ocrd}/Vertical/vdipainter/data/avidxchange"
ls "$AVID_ROOT_DIR";

export LOG_DIR="${AZUREML_DATAREFERENCE_ocrd}/Vertical/vdipainter/logs/inpaintavid128"
export TOKENIZERS_PARALLELISM="false"

python3 main.py                                                                 \
    --scale_lr false                                                            \
    --base configs/just-diffusion/txtinpaintavid128_fsdp-fairscale-t5xl.yaml    \
    --accelerator gpu                                                           \
    --devices 0,1,2,3,4,5,6,7                                                   \
    -t                                                                          \
    -l "$LOG_DIR"
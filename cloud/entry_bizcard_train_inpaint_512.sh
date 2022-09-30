#!/bin/bash

set -xe

printenv;

[ "$AZUREML_DATAREFERENCE_ocrd" ]

pushd /tmp; python3 -m pip uninstall latent-diffusion -y; popd;
python3 -m pip install -e .;

export BIZCARD_ROOT_DIR="${AZUREML_DATAREFERENCE_ocrd}/Vertical/vdipainter/data/business_card"
ls "$BIZCARD_ROOT_DIR";

export LOG_DIR="${AZUREML_DATAREFERENCE_ocrd}/Vertical/vdipainter/logs/business_card/super_res"
export TOKENIZERS_PARALLELISM="false"

python3 main.py                                                                 \
    --scale_lr false                                                            \
    --base configs/just-diffusion/bizcard512_fsdp-fairscale.yaml      \
    --accelerator gpu                                                           \
    --devices 0,1,2,3,4,5,6,7                                                   \
    -t                                                                          \
    -l "$LOG_DIR"

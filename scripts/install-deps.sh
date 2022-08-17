#!/bin/bash

set -xe

python3 -m pip install                              \
    sentencepiece datasets pytorch-lightning        \
    fairscale einops transformers kornia numpy      \
    albumentations opencv-python pudb imageio       \
    imageio-ffmpeg omegaconf test-tube streamlit

python3 -m pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu113 --force

sudo apt-get install ffmpeg libsm6 libxext6 -y
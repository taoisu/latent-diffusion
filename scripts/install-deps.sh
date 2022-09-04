#!/bin/bash

set -xe

cd "$(dirname "$0")/.."

python3 -m pip install                              \
    sentencepiece datasets pytorch-lightning        \
    fairscale einops transformers kornia numpy      \
    albumentations opencv-python pudb imageio       \
    imageio-ffmpeg omegaconf test-tube streamlit    \
    fire accelerate

python3 -m pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu113 --force

sudo apt-get install ffmpeg libsm6 libxext6 -y

python3 -m pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip

python3 -m pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers

python3 -m pip install -e .

sudo apt-get install psmisc -y
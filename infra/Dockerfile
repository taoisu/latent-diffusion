FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH

# System Utility
RUN set -xe; \
    apt-get update -y; \
    apt-get install curl wget unzip git sudo -y;

# Python Environment
RUN set -xe; \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Repo Environment
COPY [".", "/latent-diffusion"]

RUN set -xe; \
    bash /latent-diffusion/scripts/install-deps.sh;
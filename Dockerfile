# syntax=docker/dockerfile:1

# Variables used at build time.
## Base CUDA version. See all supported version at https://hub.docker.com/r/nvidia/cuda/tags?page=2&name=-devel-ubuntu
ARG CUDA_VERSION=11.8.0
## Base Ubuntu version.
ARG OS_VERSION=22.04

# Define base image.
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION} AS base

# Dublicate args because of the visibility zone
# https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG CUDA_VERSION
ARG OS_VERSION

## Base TensorRT version.
ARG TRT_VERSION=8.6.1.6
## Base PyTorch version.
ARG TORCH_VERSION=2.1.0
## Base TorchVision version.
ARG TORCHVISION_VERSION=0.16
## Base Timezone
ARG TZ=Europe/Moscow

# Set environment variables.
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive \
    ## Set timezone as it is required by some packages.
    TZ=${TZ} \
    ## CUDA Home, required to find CUDA in some packages.
    CUDA_HOME="/usr/local/cuda" \
    ## Set LD_LIBRARY_PATH for local libs (glog etc.)
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    ## Accelerate compilation flags (use all cores)
    MAKEFLAGS=-j$(nproc)

# Install linux packages
# g++ required to build 'tflite_support' and 'lap' packages, libusb-1.0-0 required for 'tflite_support' package
# openssl and tar due to security updates https://security.snyk.io/vuln/SNYK-UBUNTU1804-OPENSSL-3314796
RUN apt update && \
    apt install \
        --no-install-recommends \
        --yes \
            build-essential \
            cmake \
            ca-certificates \
            git \
            git-lfs \
            zip \
            unzip \
            curl \
            wget \
            htop \
            libgl1 \
            libglib2.0-0 \
            gnupg \
            libusb-1.0-0 \
            openssl \
            tar \
            tzdata \
            python-is-python3 \
            python3.10-dev \
            python3-pip \
            ffmpeg && \
    ## Clean cached files
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    ## Set timezone
    ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone

SHELL ["/bin/bash", "-c"]

# Install TensorRT
## Now only supported for Ubuntu 22.04
## Cannot install via pip because cuda-based errors
RUN v="${TRT_VERSION}-1+cuda${CUDA_VERSION%.*}" distro="ubuntu${OS_VERSION//./}" arch=$(uname -m) && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/${distro}/${arch}/cuda-archive-keyring.gpg && \
    mv cuda-archive-keyring.gpg /usr/share/keyrings/cuda-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/${distro}/${arch}/ /" | \
    tee /etc/apt/sources.list.d/cuda-${distro}-${arch}.list && \
    apt-get update && \
    apt-get install \
        libnvinfer-headers-dev=${v} \
        libnvinfer-dispatch8=${v} \
        libnvinfer-lean8=${v} \
        libnvinfer-dev=${v} \
        libnvinfer-headers-plugin-dev=${v} \
        libnvinfer-lean-dev=${v} \
        libnvinfer-dispatch-dev=${v} \
        libnvinfer-plugin-dev=${v} \
        libnvinfer-vc-plugin-dev=${v} \
        libnvparsers-dev=${v} \
        libnvonnxparsers-dev=${v} \
        libnvinfer8=${v} \
        libnvinfer-plugin8=${v} \
        libnvinfer-vc-plugin8=${v} \
        libnvparsers8=${v} \
        libnvonnxparsers8=${v} && \
    apt-get install \
        python3-libnvinfer=${v} \
        tensorrt-dev=${v} && \
    apt-mark hold tensorrt-dev

# Create working directory
WORKDIR /usr/src/app

# Copy project to /usr/src/app
COPY . .

# install nkb classification package with export support
RUN pip install -e .[optional]

#!/bin/bash

PACKER_SOURCE_NAME="${PACKER_SOURCE_NAME:-NA}"

if [[ ${PACKER_SOURCE_NAME} == *"google"* ]]; then

    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y coccinelle binutils-devel

    echo "Downloading latest GVE Driver"
    git clone  https://github.com/GoogleCloudPlatform/compute-virtual-ethernet-linux.git
    cd compute-virtual-ethernet-linux
    echo "Building GVE Driver"
    ./build_src.sh --target=oot

    echo "installing GVE Driver"
    sudo make -C /lib/modules/`uname -r`/build M=$(pwd)/build modules modules_install
    sudo depmod
    sudo modprobe gve
    echo gve | sudo tee /etc/modules-load.d/gve.conf

fi

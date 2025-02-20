# Use the NVIDIA base image for CUDA
FROM nvcr.io/nvidia/cuda:12.3.2-cudnn9-devel-ubuntu20.04

# Set environment variables
ENV COPPELIASIM_ROOT=${HOME}/code/coppelia_sim
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
ENV QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
ENV CONDA_ALWAYS_YES=true
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="5.0;5.2;5.3;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0+PTX"

# Create necessary directories
RUN mkdir -p ${HOME}/code

# Install dependencies and essential tools
RUN apt-get update && apt-get install -y \
    tzdata sudo curl git vim htop tar bzip2 pigz rsync less mlocate \
    build-essential gdb ca-certificates stress sysstat itop \
    xauth xvfb mesa-utils mesa-utils-extra x11-apps \
    xorg xserver-xorg-core libxv1 x11-xserver-utils libxcb-randr0-dev \
    libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev \
    libavformat-dev libswscale-dev '^libxcb.*-dev' libx11-xcb-dev \
    libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev \
    libxkbcommon-x11-dev libegl1-mesa libarchive-dev libarchive13 \
    && rm -rf /var/lib/apt/lists/*

# Install VirtualGL
RUN TEMP_DIR=$(mktemp -d -p /) && cd $TEMP_DIR && \
    curl -L -o virtualgl.deb https://sourceforge.net/projects/virtualgl/files/3.1/virtualgl_3.1_amd64.deb/download && \
    dpkg -i virtualgl.deb && \
    /opt/VirtualGL/bin/vglserver_config +glx +egl +s +f +t && \
    rm -rf $TEMP_DIR

RUN mkdir ${HOME}/.ssh && chmod -R 700 ${HOME}/.ssh

RUN ssh-keyscan github.com >> ${HOME}/.ssh/known_hosts

RUN curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
RUN export PATH=/opt/conda/bin:${PATH}   

# Install code and dependencies

WORKDIR ${HOME}/code

RUN eval "$(/opt/conda/bin/conda shell.bash hook)" && conda init bash
RUN eval "$(/opt/conda/bin/conda shell.bash hook)" && conda install mamba -c conda-forge
#RUN conda config --set auto_activate_base false


RUN git clone https://github.com/markusgrotz/peract_bimanual.git ${HOME}/code/peract_bimanual


RUN eval  "$(/opt/conda/bin/conda shell.bash hook)" && ${HOME}/code/peract_bimanual/scripts/install_dependencies.sh


# Activate the environment by default
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate rlbench" >> ~/.bashrc


WORKDIR /root/code/peract_bimanual

# Default command
CMD ["/bin/bash"]


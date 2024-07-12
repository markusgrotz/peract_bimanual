#!/bin/bash


# edit this line if you want to install the dependencies to another directory

WORKSPACE_DIR=${HOME}/code
ENVIRONMENT_NAME=rlbench

basedir=$(dirname $0)
basedir=$(readlink -f $basedir)


if ! [ -x "$(command -v curl)" ]; then
    echo "Unable to find curl. installing."    
    sudo apt install curl 
fi

if ! [ -x "$(command -v git)" ]; then
    echo "Unable to find git. installing."    
    sudo apt install git 
fi

if ! [ -x "$(command -v conda)" ]; then
    echo "Unable to find conda"
    exit 1
fi

conda create -n ${ENVIRONMENT_NAME} python=3.8
mamba install -n ${ENVIRONMENT_NAME} pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia   


export COPPELIASIM_ROOT=${WORKSPACE_DIR}/coppelia_sim
mkdir -p $COPPELIASIM_ROOT 

TEMP_DIR=$(mktemp --tmpdir -d coppelia_XXXXXXXXXX)
cd $TEMP_DIR

curl -L -O https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -xvf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

CONDA_PREFIX=$(conda info --envs | grep -e "^${ENVIRONMENT_NAME}\ " | awk '{print $2}')
cat > $CONDA_PREFIX/etc/conda/activate.d/coppelia_sim.sh <<EOF
export COPPELIASIM_ROOT=$COPPELIASIM_ROOT
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=\$COPPELIASIM_ROOT
EOF


cd ${WORKSPACE_DIR}

# YARR
git clone https://github.com/markusgrotz/YARR.git yarr
cd yarr
conda run -n ${ENVIRONMENT_NAME} pip install -e .
cd ..

# Pyep
git clone https://github.com/markusgrotz/PyRep.git pyrep
cd pyrep
conda run -n ${ENVIRONMENT_NAME} pip install -e .
cd ..

# RLBench
git clone https://github.com/markusgrotz/RLBench.git rlbench
cd rlbench
conda run -n ${ENVIRONMENT_NAME} pip install -e .
cd ..

# PerAct Bimanual
cd $basedir/..
conda run -n ${ENVIRONMENT_NAME} pip install -e .
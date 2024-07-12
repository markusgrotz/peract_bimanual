# INSTALLATION

To install the dependencies execute the `scripts/install_dependencies.sh`

```bash
scripts/install_conda.sh # Skip this step if you already have conda installed.
scripts/install_dependencies.sh
```

Please see the [README](README.md) for a quick start instruction.


Alternatively, you can follow the detailed instructions to setup the software from scratch

#### 2. PyRep and Coppelia Simulator

Follow instructions from my [PyRep fork](https://github.com/markusgrotz/PyRep); reproduced here for convenience:

PyRep requires version **4.1** of CoppeliaSim. Download: 
- [Ubuntu 20.04](https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)

Once you have downloaded CoppeliaSim, you can pull PyRep from git:

```bash
cd <install_dir>
git clone https://github.com/markusgrotz/PyRep.git
cd PyRep
```

Add the following to your *~/.bashrc* file: (__NOTE__: the 'EDIT ME' in the first line)

```bash
export COPPELIASIM_ROOT=<EDIT ME>/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.

**Warning**: CoppeliaSim might cause conflicts with ROS workspaces. 

Finally install the python library:

```bash
pip install -e .
```

You should be good to go!
You could try running one of the examples in the *examples/* folder.

#### 3. RLBench

PerAct uses my [RLBench fork](https://github.com/markusgrotz/RLBench/tree/peract). 

```bash
cd <install_dir>
git clone https://github.com/markusgrotz/RLBench.git

cd RLBench
pip install -e .
```

For [running in headless mode](https://github.com/MohitShridhar/RLBench/tree/peract#running-headless), tasks setups, and other issues, please refer to the [official repo](https://github.com/stepjam/RLBench).

#### 4. YARR

PerAct uses my [YARR fork](https://github.com/markusgrotz/YARR/).

```bash
cd <install_dir>
git clone https://github.com/markusgrotz/YARR.git 

cd YARR
pip install -e .
```



#### RVT baseline

pip install git+https://github.com/NVlabs/RVT.git 
pip install -e .





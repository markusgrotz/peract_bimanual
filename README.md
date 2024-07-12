# Perceiver-Actor^2: A Multi-Task Transformer for Bimanual Robotic Manipulation Tasks

This work extends previous work [PerAct](https://peract.github.io) as well as
[RLBench](https://sites.google.com/view/rlbench) for bimanual manipulation
tasks.

The repository and documentation are still work in progress.

For the latest updates, see: [bimanual.github.io](https://bimanual.github.io)


## Installation

### Prerequisites

The code PerAct^2 is built-off the [PerAct](https://peract.github.io) which itself is
built on the [ARM repository](https://github.com/stepjam/ARM) by James et al.
The prerequisites are the same as PerAct or ARM. 

#### 1. Environment


Install miniconda if not already present on the current system.
You can use `scripts/install_conda.sh` for this step:

```bash

sudo apt install curl 

curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh 
./Miniconda3-latest-Linux-x86_64.sh

SHELL_NAME=`basename $SHELL`
eval "$($HOME/miniconda3/bin/conda shell.${SHELL_NAME} hook)"
conda init ${SHELL_NAME}
conda install mamba -c conda-forge
conda config --set auto_activate_base false
```

Next, create the rlbench environment and install the dependencies

```bash
conda create -n rlbench python=3.8
conda activate rlbench
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```


#### 2. Dependencies

You need to setup RBench, PyRep, and YARR. 
You  can use `scripts/install_dependencies.sh` to do so
See [Installation](INSTALLATION.md) for details.

```bash
./scripts/install_dependencies.sh
```



### Pre-Generated Datasets


Please checkout the website for [pre-generated RLBench
demonstrations](https://bimanual.github.io). If you directly use these
datasets, you don't need to run `tools/bimanual_data_generator.py` from
RLBench. Using these datasets will also help reproducibility since each scene
is randomly sampled in `data_generator_bimanual.py`.



## Acknowledgements

This repository uses code from the following open-source projects:

#### ARM 
Original:  [https://github.com/stepjam/ARM](https://github.com/stepjam/ARM)  
License: [ARM License](https://github.com/stepjam/ARM/LICENSE)    
Changes: Data loading was modified for PerAct. Voxelization code was modified for DDP training.

#### PerceiverIO
Original: [https://github.com/lucidrains/perceiver-pytorch](https://github.com/lucidrains/perceiver-pytorch)   
License: [MIT](https://github.com/lucidrains/perceiver-pytorch/blob/main/LICENSE)   
Changes: PerceiverIO adapted for 6-DoF manipulation.

#### ViT
Original: [https://github.com/lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)     
License: [MIT](https://github.com/lucidrains/vit-pytorch/blob/main/LICENSE)   
Changes: ViT adapted for baseline.   

#### LAMB Optimizer
Original: [https://github.com/cybertronai/pytorch-lamb](https://github.com/cybertronai/pytorch-lamb)   
License: [MIT](https://github.com/cybertronai/pytorch-lamb/blob/master/LICENSE)   
Changes: None.

#### OpenAI CLIP
Original: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)  
License: [MIT](https://github.com/openai/CLIP/blob/main/LICENSE)  
Changes: Minor modifications to extract token and sentence features.

Thanks for open-sourcing! 

## Licenses
- [PerAct License (Apache 2.0)](LICENSE) - Perceiver-Actor Transformer
- [ARM License](ARM_LICENSE) - Voxelization and Data Preprocessing 
- [YARR Licence (Apache 2.0)](https://github.com/stepjam/YARR/blob/main/LICENSE)
- [RLBench Licence](https://github.com/stepjam/RLBench/blob/master/LICENSE)
- [PyRep License (MIT)](https://github.com/stepjam/PyRep/blob/master/LICENSE)
- [Perceiver PyTorch License (MIT)](https://github.com/lucidrains/perceiver-pytorch/blob/main/LICENSE)
- [LAMB License (MIT)](https://github.com/cybertronai/pytorch-lamb/blob/master/LICENSE)
- [CLIP License (MIT)](https://github.com/openai/CLIP/blob/main/LICENSE)

## Release Notes

**Update 2024-07-10**

Initial release


## Citations 


**PerAct^2**
```
@misc{grotz2024peract2,
      title={PerAct2: A Perceiver Actor Framework for Bimanual Manipulation Tasks}, 
      author={Markus Grotz and Mohit Shridhar and Tamim Asfour and Dieter Fox},
      year={2024},
      eprint={2407.00278},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2407.00278}, 
}
```

**PerAct**
```
@inproceedings{shridhar2022peract,
  title     = {Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation},
  author    = {Shridhar, Mohit and Manuelli, Lucas and Fox, Dieter},
  booktitle = {Proceedings of the 6th Conference on Robot Learning (CoRL)},
  year      = {2022},
}
```

**C2FARM**
```
@inproceedings{james2022coarse,
  title={Coarse-to-fine q-attention: Efficient learning for visual robotic manipulation via discretisation},
  author={James, Stephen and Wada, Kentaro and Laidlow, Tristan and Davison, Andrew J},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13739--13748},
  year={2022}
}
```

**PerceiverIO**
```
@article{jaegle2021perceiver,
  title={Perceiver io: A general architecture for structured inputs \& outputs},
  author={Jaegle, Andrew and Borgeaud, Sebastian and Alayrac, Jean-Baptiste and Doersch, Carl and Ionescu, Catalin and Ding, David and Koppula, Skanda and Zoran, Daniel and Brock, Andrew and Shelhamer, Evan and others},
  journal={arXiv preprint arXiv:2107.14795},
  year={2021}
}
```

**RLBench**
```
@article{james2020rlbench,
  title={Rlbench: The robot learning benchmark \& learning environment},
  author={James, Stephen and Ma, Zicong and Arrojo, David Rovick and Davison, Andrew J},
  journal={IEEE Robotics and Automation Letters},
  volume={5},
  number={2},
  pages={3019--3026},
  year={2020},
  publisher={IEEE}
}
```

## Questions or Issues?

Please file an issue with the issue tracker.  

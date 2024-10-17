# Perceiver-Actor^2: A Multi-Task Transformer for Bimanual Robotic Manipulation Tasks

[![Code style](https://img.shields.io/badge/code%20style-black-black)](https://black.readthedocs.io/en/stable/)

This work extends previous work [PerAct](https://peract.github.io) as well as
[RLBench](https://sites.google.com/view/rlbench) for bimanual manipulation
tasks.

The repository and documentation are still work in progress.

For the latest updates, see: [bimanual.github.io](https://bimanual.github.io)


## Installation


Please see [Installation](INSTALLATION.md) for further details.

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

You need to setup  [RLBench](https://github.com/markusgrotz/rlbench/), [Pyrep](https://github.com/markusgrotz/Pyrep/), and [YARR](https://github.com/markusgrotz/YARR/).
Please note that due to the bimanual functionallity the main repository does not work.
You can use `scripts/install_dependencies.sh` to do so.
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

### Training


#### Single-GPU Training

To configure and train the model, follow these guidelines:

- **General Parameters**: You can find and modify general parameters in the `conf/config.yaml` file. This file contains overall settings for the training environment, such as the number of cameras or the the tasks to use.

- **Method-Specific Parameters**: For parameters specific to each method, refer to the corresponding files located in the `conf/method` directory. These files define configurations tailored to each method's requirements.



When training adjust the `replay.batch_size` parameter to maximize the utilization of your GPU resources. Increasing this value can improve training efficiency based on the capacity of your available hardware.
You can either modify the config files directly or you can pass parameters directly through the command line when running the training script. This allows for quick adjustments without editing configuration files:

```bash
python train.py replay.batch_size=3 method=BIMANUAL_PERACT
```

In this example, the command sets replay.batch_size to 3 and specifies the use of the BIMANUAL_PERACT method for training.
Another important parameter to specify the tasks is `rlbench.task_name`, which sets the overall task, and `rlbench.tasks`, which is a list of tasks used for training. Note that these can be different for evaluation.
A complete set of tasks is shown below:

```yaml

rlbench:
  task_name: multi
  tasks:
  - coordinated_push_box
  - coordinated_lift_ball
  - dual_push_buttons
  - bimanual_pick_plate
  - coordinated_put_item_in_drawer
  - coordinated_put_bottle_in_fridge
  - handover_item
  - bimanual_pick_laptop
  - bimanual_straighten_rope
  - bimanual_sweep_to_dustpan
  - coordinated_lift_tray
  - handover_item_easy
  - coordinated_take_tray_out_of_oven
```


#### Multi-GPU and Multi-Node Training

This repository supports multi-GPU training and distributed training across multiple nodes using [PyTorch Distributed Data Parallel (DDP)](https://pytorch.org/docs/stable/notes/ddp.html). 
Follow the instructions below to configure and run training across multiple GPUs and nodes.

1. Multi-GPU Training on a Single Node

To train using multiple GPUs on a single node, set the parameter `ddp.num_devices` to the number of GPUs available. For example, if you have 4 GPUs, you can start the training process as follows:

```bash
python train.py replay.batch_size=3 method=BIMANUAL_PERACT ddp.num_devices=4
```

This command will utilize 4 GPUs on the current node for training. Remember to set the `replay.batch_size`, which is per GPU.

2. Multi-Node Training Across Different Nodes

If you want to perform distributed training across multiple nodes, you need to set additional parameters: ddp.master_addr and ddp.master_port. These parameters should be configured as follows:

`ddp.master_addr`: The IP address of the master node (usually the node where the training is initiated).
`ddp.master_port`: A port number to be used for communication across nodes.

Example Command:

```bash
python train.py replay.batch_size=3 method=BIMANUAL_PERACT ddp.num_devices=4 ddp.master_addr=192.168.1.1 ddp.master_port=29500
```

Note: Ensure that all nodes can communicate with each other through the specified IP and port, and that they have the same codebase, data access, and configurations for a successful distributed training run.



### Evaluation


Similar to training you can find general parameters in  `conf/eval.yaml` and method specific parameters in the `conf/method` directory.
For each method, you have to set the execution mode in RLBench. For bimanual agents such as `BIMANUAL_PERACT` or `PERACT_BC` this is:

```yaml
rlbench:
  gripper_mode: 'BimanualDiscrete'
  arm_action_mode: 'BimanualEndEffectorPoseViaPlanning'
  action_mode: 'BimanualMoveArmThenGripper'
```


To generate videos of the current evaluation you can set `cinematic_recorder.enabled` to `True`.
It is recommended during evalution to disable the recorder, i.e. `cinematic_recorder.enabled=False`, as rendering the video increases the total evaluation time.


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


**Update 2024-10-17**

- Update Readme



**Update 2024-07-10**

Initial release


## Citations 


**PerAct^2**
```
@misc{grotz2024peract2,
      title={PerAct2: Benchmarking and Learning for Robotic Bimanual Manipulation Tasks}, 
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

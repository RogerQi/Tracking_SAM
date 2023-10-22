# Prepare env

## Clone codebase

The repo contains submodule. So clone needs to be recursive.

```bash
git clone --recursive https://github.com/RogerQi/Tracking_SAM
```

## Install dependencies

Tested on Anaconda with mamba solver.

**It's known that sometimes default solver behaves different than mamba. So be cautious.**

### General packages

```bash
conda create -y -n tracking_SAM python=3.8 && conda activate tracking_SAM
conda install -c conda-forge gcc=10.3.0 --strict-channel-priority
conda install -c conda-forge cxx-compiler --strict-channel-priority
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 cudatoolkit=11.7 cudatoolkit-dev=11.7 -c pytorch -c conda-forge -c nvidia
pip install opencv-python Pillow tqdm matplotlib
```

### Compile PyTorch Correlation (for efficient VOS inference)

```bash
cd tracking_SAM/third_party/Pytorch-Correlation-extension
python setup.py install
```

### Install SAM

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Install GroundingDINO

```
cd tracking_SAM/third_party/GroundingDINO
pip install .
```

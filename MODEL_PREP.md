# Pre-trained model weights

Create a folder. All the pre-trained weights will be saved here.

```bash
mkdir pretrained_weights
cd pretrained_weights
```

## Getting SAM weights

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Getting GroundingDINO weights

```bash
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

## Getting AOT weights

Get pre-trained AOT VOS weights from [here](https://drive.google.com/file/d/1owPmwV4owd_ll6GuilzklqTyAd0ZvbCu/view?usp=sharing)
and save to `./pretrained_weights`.

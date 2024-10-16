# Tracking_SAM
Language/Clicking grounded SAM + VOS for real-time video object tracking

## Prepare env

See [INSTALL.md](INSTALL.md) to prepare environment.

Follow [MODEL_PREP.md](MODEL_PREP.md) to prepare pre-trained model weight.

## Run main demo

You can run the main tracking_SAM example on the sample data by running:

```bash
python demo.py
```

**NOTE: if you do nothing after running python. It will simply play a video.**

For the tracking to happen, an initial frame needs to be annotated via clicking or language.

### Interactive clicking annotation

Press 'a' on the keyboard to go in an annotator. Inside the annotator, you can use your mouse to left click and add points.
Press 'enter' after a satisfactory mask is generated and the mask tracking will automatically start.

If you are unhappy with your clicks, you can also do `ctrl+r` to reset all annotating progress.

### Language-based detection

Press 'd' on the keyboard to send a pre-defined language query (beat) to the model. GroundingDINO will generate a bbox
for SAM to refine. The mask tracking will automatically start.

## Check intermediate processes

See all the iPython notebooks in the root project directory.

## TODOs

- [x] Add VOS
- [x] Add SAM
- [x] Add VOS+SAM
- [x] Add Clicking
- [x] Add Grounding DINO for languaged-conditioned mask generation
- [ ] Support multiple objects
- [ ] Save memory by loading models only when necessary and offloading when not used
- [ ] Serialize used models to ONNX for deployment (easy access and potential TRT optimization)
- [ ] Switch to FastSAM for faster inference

## Citation

If you find our tool useful in your research, please consider citing VBC (for which TrackingSAM was originally developed)

```
@article{liu2024visual,
    title={Visual Whole-Body Control for Legged Loco-Manipulation},
    author={Liu, Minghuan and Chen, Zixuan and Cheng, Xuxin and Ji, Yandong and Yang, Ruihan and Wang, Xiaolong},
    journal={arXiv preprint arXiv:2403.16967},
    year={2024}
}
```

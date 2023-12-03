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

Press 'a' on the keyboard to go in an annotator. Inside the annotator, you can use your mouse to left click and add points.
Press 'enter' after a satisfactory mask is generated and the mask tracking will automatically start.

If you are unhappy with your clicks, you can also do `ctrl+r` to reset all annotating progress.

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

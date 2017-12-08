# CapsNet for MNIST Classification

This is an example of CapsNet for MNIST.
For the detail, see [Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf), Sara Sabour, Nicholas Frosst, Geoffrey E Hinton, NIPS 2017.


## Training

```
python -u train.py -g 0 --save saved_model --reconstruct
```

## Visualization of Reconstruction

```
python visualize.py -g 0 --load saved_model
```

produces a tiled image of digits reconstructed from capsules.

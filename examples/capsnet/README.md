# Dynamic Routing Between Capsules

Chainer implementation of CapsNet for MNIST.

For the detail, see [Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf), Sara Sabour, Nicholas Frosst, Geoffrey E Hinton, NIPS 2017.

```
python -u train.py -g 0 --save saved_model --reconstruct
```

Test accuracy of a trained model (without reconstruction) reached 99.60%.
The paper does not provide detailed information about initialization and optimization, so the performance might not reach that in the paper. For alleviating those issues, I replaced relu with leaky relu with a very small slope (0.05). The modified model achieved 99.66% (i.e. error rate is 0.34%), as the paper reported.


## Visualization through Reconstruction

```
python visualize.py -g 0 --load saved_model
```

produces some images for analyzing digit capsules.

### Different masks

![vis_all.png](https://raw.githubusercontent.com/soskek/dynamic_routing_between_capsules/upload-imgs/data/vis_imgs/vis_all.png)

The top green images are real images which are given to the model. Blue images in i-th represents reconstructed ones of digit "i".

If an correct digit is selected as a target, the model reconstructs an image well (see the diagonal cells).

If an irrelevant target is selected, the reconstructed image gets spoiled (see "0" and the others in the column leftmost), maybe because of lack of information in its digit capsule. However, reconstruction toward a relevant target is not always spoiled, even if a target is not correct (see "8" and "9" the column rightmost).


### Interpolation of values in digit capsules

Here, I show reconstructed images after linearly tweaking the value in a dimension in the capsule (as well as section 5.1 and figure 4 in the paper). Green images in the center are reconstructed images without perturbation. Note that a dimension has a different factor if the digit capsule differs, because each matrix for reconstructing each digit is unshared.

You can find and enjoy some factors of variation.

![vis_tweaked0.png](https://raw.githubusercontent.com/soskek/dynamic_routing_between_capsules/upload-imgs/data/vis_imgs/vis_tweaked0.png)

![vis_tweaked1.png](https://raw.githubusercontent.com/soskek/dynamic_routing_between_capsules/upload-imgs/data/vis_imgs/vis_tweaked1.png)

![vis_tweaked2.png](https://raw.githubusercontent.com/soskek/dynamic_routing_between_capsules/upload-imgs/data/vis_imgs/vis_tweaked2.png)

![vis_tweaked3.png](https://raw.githubusercontent.com/soskek/dynamic_routing_between_capsules/upload-imgs/data/vis_imgs/vis_tweaked3.png)

![vis_tweaked4.png](https://raw.githubusercontent.com/soskek/dynamic_routing_between_capsules/upload-imgs/data/vis_imgs/vis_tweaked4.png)

![vis_tweaked5.png](https://raw.githubusercontent.com/soskek/dynamic_routing_between_capsules/upload-imgs/data/vis_imgs/vis_tweaked5.png)

![vis_tweaked6.png](https://raw.githubusercontent.com/soskek/dynamic_routing_between_capsules/upload-imgs/data/vis_imgs/vis_tweaked6.png)

![vis_tweaked7.png](https://raw.githubusercontent.com/soskek/dynamic_routing_between_capsules/upload-imgs/data/vis_imgs/vis_tweaked7.png)

![vis_tweaked8.png](https://raw.githubusercontent.com/soskek/dynamic_routing_between_capsules/upload-imgs/data/vis_imgs/vis_tweaked8.png)

![vis_tweaked9.png](https://raw.githubusercontent.com/soskek/dynamic_routing_between_capsules/upload-imgs/data/vis_imgs/vis_tweaked9.png)

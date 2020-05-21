---
layout:     post
title:      Colorization Insight Automatic 2
subtitle:   Automatic Colorization Evolution
date:       2020-05-21
author:     Rasin
header-img: img/post-colorization-insight-5.jpg
catalog: true
tags:
    - Computer Vision
    - Colorization
    - Deep Learning
---

> Header Image: Reddit: [Actress Deborah Kerr -- promo shot for "From Here To Eternity" (1953)](https://www.reddit.com/r/Colorization/comments/gniu0q/actress_deborah_kerr_promo_shot_for_from_here_to/) 

# Evolution

## Automatic Colorization

### Learning Representations for Automatic Colorization

Larsson [^1] has developed a fully automatic image colorization system. The system does not rely on hand-crafted features, is trained end-to-end, and treats color prediction as a histogram estimation task rather than as regression.

#### Method

Given a grayscale image patch \\(x \in \mathcal{x} = [0, 1]^{S\times S}, f\\) predicts the color \\(y \in \mathcal{y}\\) of its center pixel. The patch size \\(S \times S\\) is the receptive field of the colorizer. The output space \\(\mathcal{y}\\) depends on the choice of color parameterization.

Skip-layer connections, which directly link low- and mid-level features to prediction layers, are an architectural addition beneficial for many image-to-image problems. We use the strategy which extracts per-pixel discriptors by reading localized slices of multiple layers and adopt the recently coined *hypercolumn* terminology for such slice.

##### Color Space

We generate training data by converting color images to grayscale according to \\(L = \frac{R+B+G}{3}\\).

For the representation of color predictions, using RGB is overdetermined, as lightness L is already known. We instead consider output color spaces with L (or a closely related quantity) conveniently appearing as a separate pass-through channel:

- HSL: can be thought of as a color cylinder, with angular coordinate H(hue), radical distance S (saturation), and height L (lightness). The values of S and H are unstable at the bot- tom (black) and top (white) of the cylinder. HSV describes a similar color cylinder which is only unstable at the bottom. However, L is no longer one of the channels. We wish to avoid both instabilities and still retain L as a channel. The solution is a color bicone, where chroma (C) takes the place of saturation. Conversion to HSV is given by \\(V=L+\frac{C}{2}, S=\frac{C}{V}\\)
- Lab: is designed to be perceptually linear. The color vector (a, b) defines a Euclidean space where the distance to the origin determines chroma.

##### Loss

A first consideratio is L2 regression in Lab:

$$
L_{reg}(x,y)=||f(x) - y||
$$

However, regression targets do not handle multimodal color distributions well. To address this, we instead predict distributions over a set of color bins, a technique also used in:

$$
L_{hist}(x,y)=D_{KL}(y||f(x))
$$

where *y* describes a histogram over \\(K\\) bins. The ground-truth histogram y is set as the empirical distribution in a rectangular region of size R around the center pixel. Somewhat surprisingly, our experiments see no benefit to predicting smoothed histograms, so we simply set R = 1. For histogram predictions, the last layer of neural network f is always a softmax.

We bin the Lab axes by evenly spaced Gaussian quantiles (µ = 0, σ = 25). They can be encoded separately for a and b (as marginal distributions), in which case our loss becomes the sum of two separate terms. They can also be encoded as a joint distribution over a and b, in which case we let the quantiles form a 2D grid of bins. In our experiments, we set K = 32 for marginal distributions and K = 16 × 16 for joint. We determined these numbers, along with σ, to offer a good compromise of output fidelity and output complexity.

For hue/chroma, we only consider marginal distributions and bin axes uniformly in [0, 1]. Since hue becomes unstable as chroma approaches zero, we add a sample weight to the hue based on the chroma: 

$$
L_{hue/chroma}(x,y) = D_{KL}(y_C||f_C(x)) + \lambnda_H y_C D_{KL}(y_H||f_H(x))
$$

where \\(y_C\\) is the sample pixel's chroma. We set \\(\lambda_H=5\\), roughly the inverse expectation of \\(y_C\\), thus equally weighting hue and chroma.

#### Inference

For the L2 loss, all that remains is to combine each \\(\hat{y}_n\\) with the respective lightness and convert to RGB. With histogram predictions, we consider options for inferring a final color:

- Sample: Draw a sample from the histogram. If done per pixel, this may create high-frequency color changes in areas of high-entropy histograms.
– Mode: Take the \\(\arg\max_k \hat{y}_{n,k}\\) as the color. This can create jarring transitions between colors, and is prone to vote splitting for proximal centroids.
– Median: Compute cumulative sum of \\(\hat{y}_n\\) and use linear interpolation to find the value at the middle bin. Undefined for circular histograms, such as hue. 
– Expectation: Sum over the color bin centroids weighted by the histogram.

For Lab output, we achieve the best qualitative and quantitative results using expectations. For hue/chroma, the best results are achieved by taking the median of the chroma. Many objects can appear both with and without chroma, which means \\(C = 0\\) is a particularly common bin. This mode draws the expectation closer to zero, producing less saturated images. As for hue, since it is circular, we first compute the complex expectation:

$$
z=\mathcal{E}_{H~f_h(x)}[H] \triangleq \frac{1}{K}\sum_k [f_h(x)]_k \e^{i\theta_k},\ \theta_k=2\pi \frac{k + 0.5}{K}
$$

We then set hue to the argument of z remapped to lie in [0, 1).

In cases where the estimate of the chroma is
high and z is close to zero, the instability of the hue can create artifacts. A simple, yet effective, fix is chromatic fading: downweight the chroma if the absolute value of z is too small. We thus re- define the predicted chroma by multiplying it by a factor of \\(\max(\eta − 1|z|, 1)\\). In our experiments, we set η = 0.03 (obtained via cross-validation).

#### Neural network architecture and training

The base network is a fully convolutional version of VGG-16 with two changes: 

1.  the classification layer (fc8) is discarded
2.  the first filter layer (conv1 1) operates on a single intensity channel instead of mean-subtracted RGB.

We extract a hypercolumn descriptor for a pixel by concatenating the features at its spatial location in all layers, from data to conv7 (fc7), resulting in a 12, 417 channel descriptor. We feed this hypercolumn into a fully connected layer with 1024 channels, to which we connect output predictors.

Processing each pixel separately in such manner is quite costly. We instead run an entire image through a single forward pass of VGG-16 and approximate hypercolumns using bilinear interpolation. Extracting batches of only 128 hypercolumn descrip- tors per input image, sampled at random locations, provides sufficient training signal. In the backward pass of stochastic gradient descent, an interpolated hy- percolumn propagates its gradients to the four closest spatial cells in each layer. Locks ensure atomicity of gradient updates, without incurring any performance penalty. This drops training memory for hypercolumns to only 13 MB per image.

We initialize with a version of VGG-16 pretrained on ImageNet, adapting it to grayscale by averaging over color channels in the first layer and rescaling appropriately. Prior to training for colorization, we further fine-tune the network for one epoch on the ImageNet classification task with grayscale input. As the original VGG-16 was trained without batch normalization, scale of responses in internal layers can vary dramatically, presenting a problem for learning atop their hypercolumn concatenation. We use the alternative of balancing hypercolumns so that each layer has roughly unit second moment (E[X2] ≈ 1).

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200521165057.png)

### Colorful Image Colorization




### Reference

[^1]: Larsson, G., Maire, M., Shakhnarovich, G.: Learning representations for automatic colorization. European Conference on Computer Vision (2016)

[^2]: Zhang, R., Isola, P., Efros, A.A.: Colorful image colorization. In: ECCV (2016)
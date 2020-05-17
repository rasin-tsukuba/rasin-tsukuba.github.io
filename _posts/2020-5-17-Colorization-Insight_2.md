---
layout:     post
title:      Colorization Insight 2
subtitle:   Related Works and Techniques
date:       2020-05-17
author:     Rasin
header-img: img/post-colorization-insight-2.jpg
catalog: true
tags:
    - Computer Vision
    - Colorization
    - Deep Learning
---

> Header Image: Reddit: [Snake Charmer _ Morocco_ 1950](https://www.reddit.com/r/Colorization/comments/gkjhsx/snake_charmer_morocco_1950) 

# Evolution

## Pure Hand Colorization

Hand colorization is laborious and time consuming. The first film colorization methods were hand done by individuals. Now, some artists are still using mannual colorization method to colorize old pictures. In Reddit, there are still some popular colorization forums, such as [Colorization](https://www.reddit.com/r/Colorization/) and [ColorizedHistory](https://www.reddit.com/r/ColorizedHistory/).

For example, in order to colorize a still image an artist typically begins by segmenting the image into regions, and then proceeds to assign a color to each region. Thus, the artist is often left with the task of manually delineating complicated boundaries between regions. Colorization of movies requires, in addition, tracking regions across the frames of a shot. Existing tracking algorithms typically fail to robustly track non-rigid regions, again requiring massive user intervention in the process. [^1]

It is worth mentioning that even though this is a labor intensive job and very expensive in making videos, the hand colorization can give the best colorization effect than any other (semi-)automatic techniques. 

## Semi-automatic Colorization

Unfortunately, automatic segmentation algorithms often fail to correctly identify fuzzy or complex region boundaries,such as the boundary between a subject’s hair and her face. 

### Colorization using Optimization

In 2004, Levin[^1] has proposed a new interactive colorization technique that requires neither precise manual segmentation, nor accurate tracking. The user indicates how each region should be colored by **scribbling the desired color** in the interior of the region, instead of tracing out its precise boundary.

![Levin example](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200517092539.png)

This algorithm is working in YUV color space, where \\(Y\\) is the monochromatic luminance channel, which we will refere to simply as intensity, while \\(U\\) and \\(V\\) are the chrominance channels, encoding the color. The algorithm is given as input an intensity volum \\(Y(x,y,t)\\) and outputs two color volumes \\(U(x,y,t)\\) and \\(V(x,y,t)\\).

Levin wished to impose the constraint that two neighboring pixels, **r, s** should have similar colors if their intensities are similar. Thus, to minimize the difference between the color U(**r**) at pixel **r** and the weighted average of the colors at neighboring pixels:

##
J(U)=\sum _{\boldsymbol{r}}\left( U(\boldsymbol{r})-\sum _{s\in N(\boldsymbol{r})} w_{\boldsymbol{rs}} U(\boldsymbol{s})\right)^2
##

where \\(w_{\boldsymbol{rs}}\\) is a weighting function that sums to one, large when $Y(\boldsymbol{r})$ is similar to \\(Y(\boldsymbol{s})\\), and small when the two intensiies are different. Similar weighting functions are used extensively in image segmentation algorithms, where they are usually referred to as affinity functions.

He has experimented with two weighting functions. The simplest one is commonly used by image segmentation algorithms and is bsed on the squared difference between the two intensities:

$$
w_{\boldsymbol{rs}} \propto e^{-(Y(\boldsymbol{r}) - Y(\boldsymbol{s}))^2/2\sigma_{\boldsymbol{r}^2}}
$$

A second weighting function is based on the normalized correlation between the two intensities:

$$
w_{\boldsymbol{rs}} \propto 1 + \frac{1}{\sigma_{\boldsymbol{r}}^2}(Y(\boldsymbol{r})-\mu_{\boldsymbol{r}})(Y(\boldsymbol{s}) - \mu_{\boldsymbol{r}})
$$

where \\(\mu_{\boldsymbol{r}}\\) and 


### Reference

- [Wikipedia-Film-colorization](https://en.wikipedia.org/wiki/Film_colorization)

[^1]: LEVIN, A., LISCHINSKI, D., AND WEISS, Y. 2004. Colorization using optimization. ACM Transactions on Graphics 23, 689– 694.
 


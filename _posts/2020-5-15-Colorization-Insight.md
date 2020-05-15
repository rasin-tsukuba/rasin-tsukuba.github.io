---
layout:     post
title:      Colorization Insight
subtitle:   Some thoughts and ideas
date:       2018-06-05
author:     BY
header-img: img/post-colorization-insight.jpg
catalog: true
tags:
    - Computer Vision
    - Colorization
    - Deep Learning
---

> Header Image: Reddit: [Jerry Stiller with his wife, Anne Meara, 1960s](https://www.reddit.com/r/Colorization/comments/gi4aky/jerry_stiller_with_his_wife_anne_meara_1960s/) 

# Introduction

## What is Colorization

Colorization is a term introduced by Wilson Markle in 1970 to describe computer-assisted process he invented for adding color to black and white movies or TV programs.

The term is now used generically to describe any technique for turning grayscale(mostly) images or videos to colorized version, without losing its original content.

## Why Colorization

Most images and videos in old age is in grayscale or monocolor. The aim of colorization is to get vivid and realistic visual effect, bring people back to their old age, and arise their memory. In video, such as movies and documentaries, colorized version can improve the watching experience. We take some small experiences and find out that human eyes are more appeal to colorized version of images and videos. Thus, colorized version is always better than grayscale version in visual experience. Moreover, colorized version contains more information.

![An colorized example](https://cdn.fstoppers.com/styles/large-16-9/s3/wp-content/uploads/2012/09/Feature-Image-Size.jpg)


In Biology and Medical area, colorization can aid professional pathologists judge and classify medical images. Colored medical image can seperate the difference between tissues and organs. 

![T2 MR Brain Images. From: Colorization and Automated Segmentation of Human T2 MR Brain Images for Characterization of Soft Tissues](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/post-colorization-insight-example2.png)

# Evolution

## Pure Hand Colorization

Hand colorization is laborious and time consuming. The first film colorization methods were hand done by individuals. Now, some artists are still using mannual colorization method to colorize old pictures. In Reddit, there are still some popular colorization forums, such as [Colorization](https://www.reddit.com/r/Colorization/) and [ColorizedHistory](https://www.reddit.com/r/ColorizedHistory/).

For example, in order to colorize a still image an artist typically begins by segmenting the image into regions, and then proceeds to assign a color to each region. Thus, the artist is often left with the task of manually delineating complicated boundaries between regions. Colorization of movies requires, in addition, tracking regions across the frames of a shot. Existing tracking algorithms typically fail to robustly track non-rigid regions, again requiring massive user intervention in the process. [^1]

It is worth mentioning that even though this is a labor intensive job and very expensive in making videos, the hand colorization can give the best colorization effect than any other (semi-)automatic techniques. 

## Semi-automatic Colorization

Unfortunately, automatic segmentation algorithms often fail to correctly identify fuzzy or complex region boundaries,such as the boundary between a subject’s hair and her face. 


### Reference

- [Colorized Lincoln Picture](https://fstoppers.com/video/how-amazing-colorization-black-and-white-photos-are-done-5384)
- [T2 MR Brain Images.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0033616)
- [Wikipedia-Film-colorization](https://en.wikipedia.org/wiki/Film_colorization)
- [^1]: Levin, Colorization using Optimization
 


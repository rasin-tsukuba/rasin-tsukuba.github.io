---
layout: post
title: GAMES101-Overview of Computer Graphics
subtitle: Notes and Ideas
date: 2020-06-14
author: Rasin
header-img: img/GAMES-101-1.jpg
catalog: true
tags:
  - Computer Graphics
  - Notes
---
# Overview of Computer Graphics

## Why Study

Fundamental Intellectual Challenges

* Creates and interacts with realistic virtual world
* Requires understanding of all aspects of physical world
* New computing methods, displays, technologies

Technical Challenges

* Math of projections, curves, surfaces
* Physics of lighting and shading
* Representing / operating shapes in 3D
* Animation / Simulation
* 3D graphics software programming and hardware

## Topics

### Rasterization 光栅化

* Project **geometry primitives** (3D triangles / polygons) onto the screen
* Break Projected primitives into fragments (pixels)
* Gold standard in Video Games (Real-time Applications)
  * Real-time: 30 fps and above

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200614183911.png)

### Curves and Meshes

* How to represent geometry in Computer Graphics

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200614184136.png)

### Ray Tracing

* Shoot rays from the camera though each pixel
  * Calculate **intersection** and **shading**
  * **Continue to bounce** the rays still they hit light sources
* Gold standard in Animations / Movies (Offline Applications)

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200614184310.png)

### Animation / Simulation

* Key frame Animation
* Mass-spring System

## Difference Between Computer Vision

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200614184914.png)

* No clear boundaries





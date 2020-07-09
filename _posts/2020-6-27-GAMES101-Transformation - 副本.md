---
layout: post
title: GAMES101-Transformation
subtitle: Notes and Ideas
date: 2020-06-27
author: Rasin
header-img: img/GAMES-101-3.jpg
catalog: true
tags:
  - Computer Graphics
  - Mathematics
  - Tutorials
---
# Transformation

## 2D transformations

- Representing transformations using matrices
- Rotation, scale, shear

### Scale Transform

\\(x' = sx, y' = sy\\)


$$
\begin{bmatrix}
x'\\
y'
\end{bmatrix} =\begin{bmatrix}
s_x & 0\\
0 & s_y
\end{bmatrix}\begin{bmatrix}
x\\
y
\end{bmatrix}
$$

### Reflection Matrix

$$
\begin{bmatrix}
x'\\
y'
\end{bmatrix} =\begin{bmatrix}
-1 & 0\\
0 & 1
\end{bmatrix}\begin{bmatrix}
x\\
y
\end{bmatrix}
$$

### Shear Matrix

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200627153913.png)

$$
\begin{bmatrix}
x'\\
y'
\end{bmatrix} =\begin{bmatrix}
1 & a\\
0 & 1
\end{bmatrix}\begin{bmatrix}
x\\
y
\end{bmatrix}
$$

### Rotate

Rotation is about the origin \\(0, 0\\), counter-clockwise by default.

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200627154635.png)

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200627162752.png)

$$
R_{\theta } =\begin{bmatrix}
\cos \theta  & -\sin \theta \\
\sin \theta  & \cos \theta 
\end{bmatrix}
$$

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200628144147.png)

## Homogeneous Coordinates

### Translation

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200627163151.png)

$$
x' = x + t_x,\ y' = y+t_y
$$

Translation cannot be represented in matrix form.

$$
\begin{bmatrix}
x'\\
y'
\end{bmatrix} =\begin{bmatrix}
a & b\\
c & d
\end{bmatrix}\begin{bmatrix}
x\\
y
\end{bmatrix} +\begin{bmatrix}
t_{x}\\
t_{y}
\end{bmatrix}
$$

Translation is **not** linear transform.

Add a third coordinate in Homogeneous Coordinates:
- 2D point \\(=(x,y,1)^\top\\)
- 2D vector \\(=(x,y,0)^\top\\)

Matrix representation of translations:

$$
\begin{pmatrix}
x'\\
y'\\
w'
\end{pmatrix} =\begin{pmatrix}
1 & 0 & t_{x}\\
0 & 1 & t_{y}\\
0 & 0 & 1
\end{pmatrix} \cdot \begin{pmatrix}
x\\
y\\
1
\end{pmatrix} =\begin{pmatrix}
x+t_{x}\\
y+t_{y}\\
1
\end{pmatrix}
$$

## Affine Transformations

Affine map = linear map + translation

$$
\begin{bmatrix}
x'\\
y'
\end{bmatrix} =\begin{bmatrix}
a & b\\
c & d
\end{bmatrix}\begin{bmatrix}
x\\
y
\end{bmatrix} +\begin{bmatrix}
t_{x}\\
t_{y}
\end{bmatrix}
$$

Using homogeneous coordinates:
$$
\begin{pmatrix}
x'\\
y'\\
1
\end{pmatrix} =\begin{pmatrix}
a & b & t_{x}\\
c & d & t_{y}\\
0 & 0 & 1
\end{pmatrix} \cdot \begin{pmatrix}
x\\
y\\
1
\end{pmatrix} 
$$

### 2D transformations

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200627165001.png)

### Inverse Transform

\\(M^{-1}\\) is the inverse of transform \\(M\\) in both a matrix and geometric sense

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200627165227.png)

### Composing Transform

Matrix multiplication is **not** commutative.

Matrices are applied right to left:

$$
T_{( 1,0)} \cdot R_{45} \cdot \begin{bmatrix}
x\\
y\\
1
\end{bmatrix} =\begin{bmatrix}
1 & 0 & 1\\
0 & 1 & 0\\
0 & 0 & 1
\end{bmatrix}\begin{bmatrix}
\cos 45\degree  & -\sin 45\degree  & 0\\
\sin 45 & \cos 45\degree  & 0\\
0 & 0 & 1
\end{bmatrix}\begin{bmatrix}
x\\
y\\
1
\end{bmatrix}
$$

Sequence of affine transforms \\(A_1, A_2, A_3, \dots\\), compose by matrix multiplication.

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200627191315.png)

## 3D Transforms

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200627191857.png)

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200627191941.png)

First affine transform, then translation.
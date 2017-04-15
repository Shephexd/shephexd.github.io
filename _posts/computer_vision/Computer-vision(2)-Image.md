---
title: Computer vision(2)-Imaging
layout: post
categories: 
- Computer vision
tags:
- Algorithm
- Linear algebra
- Computer vision
- Robotics
---





The `computer vision` is one of the difficult and important part on machine learning. So, how can we make the computer *see* and *recognize* the image? It is advanced question about image processing. 

Imagine that the computer see and recognize the image like human being. It is related to the robot vision for the robotics.

This post will be introduced about what the computer vision is and the application using `computer vision`.



<!--more-->

## Images in computer

In computers, images can be represented as an array of numbers

*Intensity images = photograph-like images showing light intensities.*
*range images = encoding distance, acquired with specialized sensors such as laser scanners*



We are living in 3D coordinate system, but when we take a picture its coordinate is 2D. So Camera convert 3D(real world) to 2D(picture). Then camera lens work as optical system having optical point.



To get sharp images, all rays coming from a single point $P$, must converge to same point on image plane to point $p$.

- Reducing aperture to a pinhole
  - Problem with amount of light
- Having an optical system of lenses and apertures
  - Depth of field limited




- focal length(zooming) $\uparrow$  $\Rightarrow$ amount of light $\downarrow$  , as the imaged area shrinks.
- aperture size $\uparrow$ $\Rightarrow$ amount of light $\uparrow$
- Exposure time $\uparrow$ $\Rightarrow$ amount of light $\uparrow$



## Perspective camera



### Fundamental projection equations

- $O$ - center of focus of projection
- $F$ - Focal length
- $Z$ - optical axis along $Z$
- $p$ - Image of $P$

Non-linear transform



### Weak-perspective camera

$$
x=f\frac{X}{Z}=X\frac{f}{Z}\\
y=f\frac{Y}{Z}=Y\frac{f}{Z}\\
\text{If average the depth of scene is much larger than the relative of scene points}
$$





### Digital images

- Monochromatic(gray-level) images have values from 0(black) to 255(white)
- Color images have three components(three monochromatic images) representing intensities of Red, Green and Blue
- Spectral images with several components representing intensities of various like RGB



##  Camera parameters

The point $p$ on images can match a point $P$ on 3-D world.



### Extrinsic parameters

Link camera frame with world frame, describe the pose(location and orientation) of camera



- Camera pose in world coordinate frame
  - Translation and rotation separately, or together as a homogeneous transform
  - coordinate transformation $C_P =\ ^{C}R_W(\ ^WP - {}{}{}{}{}{}^WP_{CORG}) =\ ^CT_W\ ^WP$
- Camera 



### Intrinsic parameters

Link pixel coordinate to camera coordinates, describe optics and CCD array



- Perspective projection (focal length $f$)
- Camera to pixel coordinate transformation
- Distortion by imperfect optics(not real pinhole)



### Camera to pixel transformation

Let $n_x$ and $n_y$ be pixel coordinate and $x(,y)$ be camera frame coordinates of the same pixel
$$
x=-(n_x-o_x)s_x\\
y=-(n_y-o_y)s_y\\
(o_x,o_y) =\text{the coordinates of images center}\\
(s_x,s_y) =\text{the pixel size in millimeters in horizontal and vertical directions}
$$
Intrinsic parameters are $f_x, o_x, o_y, s_x, s_y$



## Lens distortion

Lens distortion can be modeled with two parameters $k_1$ and $k_2$
$$
x=x_d(1+k_1r^2+k_2r^4)\\
y=y_d(1+k_1r^2+k_2r^4)\\
\text{where } (x_d,y_d) \text{ is the distorted point, }r^2=x^2_d + y^2_d
$$


## Perspective camera model


$$
\begin{pmatrix}
u\\
v\\
w
\end{pmatrix}
=
K
\begin{pmatrix}
^CR_W\\
^CP_W
\end{pmatrix}
\begin{pmatrix}
^WX\\
^WY\\
^WZ\\
1
\end{pmatrix}
=
K
\begin{pmatrix}
^CR_W\\
^CP_W
\end{pmatrix}
\ ^WP
\\
K = 
\begin{pmatrix}
-f/s_x & 0 &o_x\\
0 & -f/s_y & o_y\\
0 & 0 & 1\\
\end{pmatrix}
$$



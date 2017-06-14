---
title: Computer vision(3)- Image processing
layout: post
categories: 
- Computer vision
tags:
- Algorithm
- Linear algebra
- Computer vision
- Robotics
---

The reason why we analyze images is to get the information and features.

So, We should know how to get the information from images. Many algorithms exist for filtering, edges and corner. It is used for many way to extract and exaggerate features.

Image processing also important to remove noise in images and transform images before extracting features.

In this post, i will introduce some algorithm for image processing.

<!--more-->



## noise in images

Two images are never exactly the same



### Models for image noise

Additive random noise
$$
\hat{I}(i,j)=I(i,j)+n(i,j)\\
\text{random signal: }n(i,j)
$$


#### Gaussian noise

Known as *white noise*

$n(i,j)$ is a zero-mean Gaussian random process



#### Impulsive noise

- Known as *spot, peak or salt and pepper noise*
- Caused by transmission errors and faulty elements(pixels) of CCD array



### Removing noise

- Transform Color images and color 



#### Linear filtering

Let $I$ be $N\times M$ image and $A$ the $m \times m$ convolution kernel. Then filtered version $I_A$ of I is computed by the below.
$$
I_A(i,j) = I *A = \sum_{h=-m/2}^{m/2}\sum_{k=-m/2}^{m/2}A(h,k)I(i-h,j-k)
$$


#### Mean filtering

Replace a pixel by the mean of its neighborhood(size)
$$
A_{avg} = \frac{1}{9}
\begin{pmatrix}
1 & 1 & 1\\
1 & 1 & 1\\
1 & 1 & 1
\end{pmatrix}
$$

- Sharp signal variations are lost, image becomes blurred.
- Impulsive noise is not completely removed.



#### Gaussian smoothing

$$
G(h,k) = exp\left (-\frac{h^2+k^2}{2\sigma^2} \right )
$$

Better than mean filter, no frequency domain sid lobes.

But similar problems as in mean filter: blurring, impulsive noise not handled well



#### Median filtering

This filtering is one of non linear filtering methods.

Replace each pixel by the median of its neighborhood.

Usually slower than linear filtering.



## Color

Energe distibution over different wavelengths of visible light, dominant wavelength.



### Color models

color image composed of 3 gray-level images, each representing one color components.



#### RGB

- Red
- Greedn
- Blue



#### HSI

- Hue - 'Color'
- Saturation - 'Purity'
- Intensitiy - 'amount'



#### RGB to HSI

$$
\theta = \cos^{-1}\left( \frac{\frac{1}{2}((R-G)+(R-B))}{((R-G)^2+(R-B)(G-B))^{1/2}}\right)\\
H = \begin{cases}
\theta &\text{ if } B \le G\\ 
360 - \theta & \text{ if } B \gt G
\end{cases}\\
S = I - \frac{3}{R+G+B}\min(R,G,B)\\
I = \frac{1}{3}(R+G+B)
$$


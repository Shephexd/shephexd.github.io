---
title: Computer vision(5) - image features
layout: post
categories: 
- Computer vision
tags:
- Algorithm
- Linear algebra
- Computer vision
---





To recognize image on the computer, the feature is one of the important part for it. And there are some algorithms to detect features like edge and corner.





And depending on feature type, we can consider which algorithm is useful for our image.



<!--more-->



## What is image features?

Image features have two meaning, global and local. Normally, I will introduce global and local, but the main topic is about the local feature detection.



- Global

  Recognition properties are important.

  - average grey level
  - area in pixels



- Local

  Part of an image with some special properties. In the local features, the location is important
  - lines
  - corners
  - circles
  - textured regions



The good features should be **meaningful** and **detectable**.



Actually, it is almost impossible to get perfect features from the real world.



## Edge detection

Edge points are pixels where image values undergo a sharp variation.

It means that we can find the edge in case we know which border has a sharp variation.



### basic steps for edge detection

1. smooth out noise
2. enhance edges using a filter which responds to edges
3. localize edges



We need to find an optimal linear edge enhancement filter for removing noise.



Good detection

- Minimize probability of false positive and false negatives(low error)
- maximizing signal to noise ratio



Good localization

- edges must be as close as possible to true edges



Single response

- A detector must return only one point corresponding to each true edge point, minimizing the number of false edges due to noise.



Solution to the optimization problem is difficult to express in closed form but `first derivation of Gaussian` is close to the optimal operator.


$$
G(x) = e^{-\frac{x^2}{2\sigma^2}}\\
G'(x) = -\frac{x}{\sigma^2}e^{-\frac{x^2}{2\sigma^2}}\\
$$


### 2D case for edge detection



#### Generalization

- Calculate directional derivatives $I_x$ and $I_y$ (gradient)

- Calculate magnitude M and normal n

  ​

$$
G(x,y) = e^{\frac{x^2+y^2}{2\sigma^2}}\\
M(x,y) = \sqrt{I^2_x(x,y)+I^2_y(x,y)}\\
n(x,y) = tan^{-1}(\frac{I_y(x,y)}{I_x(x,y)})
$$



#### Edge localization

1. non-maximum suppression

   Thinning wide edges resulting from the convolution

2. Thresholding

   - defining the threshold such that the local maxima is defined as an edge
   - Finding the threshold is difficult
   - Low threshold causes false contours
   - High threshold fragments true edges



## Corner detection



### Algoritm

1. Compute image gradient over entire image

2. For each point $p$

   1. Form atrix C using neighborhood of $p$
   2. Compute $\lambda_2$, the smaller eigenvalue of $C$
   3. If $\lambda_2 \gt t$, save coordinates of p into list $L$

3. Sort $L$ in decreasing order of $\lambda_2$

4. Go through $L$ and for each point $p$, delete points further in $L$ which belong to neighborhood of $p$

   ​

## Segmentation by thresholding



## Region features



## Connected component analysis



```sudo
L(x,y)=0 for all x,y,c = 1

For y = 1 to height
	For x = 1 to width
		If I(x,y) = FOREGROUND
			IF L(x,y-1)=0 and L(x-1,y)<>0
				L(x,y) = L(x-1,y)
			ELSE IF L(x-1,y)<>0 and L(x-1,y)=0
				L(x,y) = L(x,y-1)
			ELSE IF L(x,y-1)=0 and L(x-1,y)=0
				L(x,y) = c
				c=c+1
			ELSE
				L(x,y) = L(x,y-1)
```


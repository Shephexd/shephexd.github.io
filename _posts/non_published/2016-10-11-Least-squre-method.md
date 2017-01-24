---
layout: post
title: Least square method
categories:
- Machine learning
tags:
- Regression
- Machine learning
- Linear algebra
- Matlab
---



"Least squares" means that the overall solution minimizes the sum of the squares of the errors made in the results of every single equation. - [Wikipedia](https://en.wikipedia.org/wiki/Least_squares)

**LSM use "Least squares" to make estimated model.**
Defintely, It is one of the methods to get a parameter. To minimize between sum of data and the power of residual


Using this method, we can get regression model like below.

$$
y = ax_1 + ax_1 + ax_2 + ax_3 + ... + ax_n + b
$$

<!--more-->

### What is residual?
Residual is how far the data from the estimated model.


## LSM calculation
There are two kind of methods to get LSM, Algebraic and Analytical.


### Algerbraic method

$$
y = X\beta + \epsilon
$$

$$y$$ is an n-by-1 vector of responses.
$$\beta$$ is coefficient of model.
$$X$$ is the n-by-m design matrix for the model.
$$\epsilon$$ is an n-by-1 vector of errors.

$$
y_1 = ax_1+b\\
y_2 = ax_2+b\\
.\\
.\\
.\\
y_n = ax_n+b
$$

$$
    \begin{bmatrix}
    y_1 \\
    y_2 \\
    y_3 \\
    . \\
    . \\
    . \\
    y_n \\
    \end{bmatrix}
    =
    \begin{bmatrix}
    x_1 & 1\\
    x_2 & 1\\
    x_3 & 1\\
    . & .\\
    . & .\\
    . & .\\
    x_n & 1\
    \end{bmatrix}
    *
    \begin{bmatrix}
    a \\
    b \\
    \end{bmatrix}
$$

$$
AX=B
$$

The least-squares solution to the problem is a vector b, which estimates the unknown vector of coefficients β. The normal equations are given by

$$
(X^TX)b=X^Ty
$$

**It is estimation of inverse matrix for A by using pseudo inverse.**

pseudo inverse
: A has no inverse matrix, This method can estimate the inverse matirx of A.

$$
X = pinv(A)B\\
=
(X^TX)b=X^Ty
$$



### Analytical method

## LSM Sample

## Restirction of LSM
It is not working well when data set have a outlier or more. Because it based on the distance between estimated model and data set. Thus, outlier can be a problem for the model.

If your data set has a outlier or more, you'd better change the method to estimate model like more robust model like RANSAC, M-estimator.



## Notes

It is based on the blog[^dark_blog], my class and the official site of matlab[^matlab].


[^dark_blog]: [dark_blog](http://darkpgmr.tistory.com/56), Korean blog about mathmatics and machine vision.
[^matlab]: [Matlab officail introduction](https://se.mathworks.com/help/curvefit/least-squares-fitting.html) about LSM

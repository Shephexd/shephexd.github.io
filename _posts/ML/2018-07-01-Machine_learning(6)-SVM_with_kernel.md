---
title: Machine learning(5) - SVM
layout: post
categories:
- Machine learning
tags:
- Machine learning
- Tensorflow
typora-root-url: /Users/shephexd/Documents/github/pages/
---

This post is related to the previous post, `SVM`. This post is about the advanced way to solve `non-linear problem` and choose proper `hyper paramters` for `SVM`.



If your data set is `linear` , it is enough to use `linear classifcation` like (`SVM`, `Logistic regression`)



Unless, you can't get the fine model for your dataset.

Here is a solution, the `kernel`. In this post, i will describe how `kernel` can solve  `non-linear` problems.



<!--more-->



Non-linear decision boundary can't be solved by `linear model`.

The equation of linear model is drawn like below


$$
\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots +\theta_4 x_1^2 +\theta_4 x_1^2 + \theta_5 x_2^2 \ge 0 \\
h_\theta(x) = \cases{1 & if  $\theta^Tx \ge 0$\\ 0 & otherwise}
$$




## What is the kernel?

`Kernel` is a `function` to map features into `non-linear dimension` .





One of the example for `kernel function` is `Gaussian kernel` to get similarity between samples and selected points.


$$
f_n = similarity(x_, \ell^{(n)}) =\exp \left(\frac{\lVert x - \ell^{(n)} \rVert^2}{\lVert 2 \sigma^2 \rVert} \right)
$$


The linear model we think is like the below equation.


$$
\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 \ge 0
$$


By mapping the sample through `Gaussian kernel` , the sample can be useful features for `non-linear model`.




$$
f_1 = similarity(x_, \ell^{(1)}) =\exp \left(\frac{\lVert x - \ell^{(1)} \rVert^2}{\lVert 2 \sigma^2 \rVert} \right) \\
f_2 = similarity(x_, \ell^{(2)}) =\exp \left(\frac{\lVert x - \ell^{(2)} \rVert^2}{\lVert 2 \sigma^2 \rVert} \right) \\
\vdots \\
f_n = similarity(x_, \ell^{(n)}) =\exp \left(\frac{\lVert x - \ell^{(n)} \rVert^2}{\lVert 2 \sigma^2 \rVert} \right)
$$


Then we can write `the model with Gaussian kernel`.
$$
\theta_0 + \theta_1 f_1 + \theta_2 f_2 + \theta_3 f_3 \ge 0
$$



## SVM with kernels

As i said the kernel function is to map feature onto other space($R^{(n+1)}$), $x_i \to f_i$



Given $(x^{(1)}, y^{(1)}), (x^{(1)}, y^{(1)}) \dots (x^{(m)}, y^{(m)})$,  

Choose $x^{(1)} \rightarrow f^{(1)},  x^{(2)} \rightarrow f^{(2)} \dots  x^{(m)} \rightarrow f^{(m)} $



$$
f_1 = similarity(x, f^{(1)}) \\
f_2 = similarity(x, f^{(2)}) \\
\\
f = 
\begin{bmatrix}
f_0 \\ f_1 \\ \vdots \\ f_m
\end{bmatrix}
, f_0 = 1
$$



### Hypothesis

Given $x$, compute features $ f \in R^{n+1}$  
Predict $y=1$ if $\theta^T f \ge 0 $ $(\theta^T f =\theta_0 + \theta_1 f_1 + \dots + \theta_m f_m  )$



We can train `SVM` with kernel function as we did without kernel.


$$
\text{minimize } J(\theta)  = C \sum_{i=1}^m y^{(i)} cost_1(\theta^T f^{(i)}) + (1 - y^{(i)})cost_0(\theta^T f^{(i)}) + \frac{1}{2} \sum_{j=1}^m \theta_j^2
$$


## SVM parameters



$C$ is the parameter for regularization. This value works like $\lambda$ we used for `Linear regression` and `Logistic regression`.



You can think $C$ value is similar to $\frac{1}{\lambda}$.

-   Large $C$: lower `bias`, higher `variance`
-   Small $C$: higher `bias`, lower `variance`



If you want to apply `Gaussian kernel` for your `SVM`, Then other parameter is $\sigma^2$. This parameter have an effect on how the feature is smooth.



-   Large $\sigma^2$ : Feature $f_i$ vary `more smoothly`.  
    Higher `bias` , lower `variance`
-   Small $\sigma^2$ : Feature $f_i$ vary `less smoothly`.  
    Lower `bias`, Higher `variance`






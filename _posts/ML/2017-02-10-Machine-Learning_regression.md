---
layout: post
title: Machine learning(1) - Linear regression
published: True
categories: 
- Machine learning
Tags:
- Machine learning
- Regression
- Data mining
---

[TOC]



How can the computer think by self? Normally, they can't think like human, they will use their way to learn and think. Machine learning 



This post is based on the Stanford video lecture by Andrew Ng.

<!--more-->



## What is the machine learning?

There are two definition about the machine learning.

1. The field of study that gives computer the ability to learn without being explicitly programmed
2. A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P. If its performance at tasks n T as measured by P, improves with experience E.



### Supervised learning

Computer can learn from  **a given data** set that we already **know** our outputs are. There is a `relation(model)` between the input and the output.

- Regression

  To predict result within a `continuous` output.

- Classification

  To predict result in a `discrete` output.



### Unsupervised learning

Computer can solve the problem with **little or no idea** what our results are.

- Clustering

  Make clusters having similar characters.

  ​

### Reinforcement learning

Computer can learn from the experience. From the reward, computer can recognize and decide the best action to get the best reward in current environment.



## Linear regression

In statistics, linear regression is n approach for modeling the relationship between a scalar dependent variable y and one or more explanatory variables(or independent variables) denoted $X$. - [Wikipedia][1]



Simply, linear regression predicts a real-valued output based on an input value.

trivial model

$$
l_0 = y + \beta(noise) \\
(US)(V^Tl) = y+\beta \\
T_\alpha l_\alpha ≈ y \\
$$





### Linear regression with one variable

A very simple regression form.
$$
y = x_0+ax_1
$$



### Cost function

The `cost function` is a function to get the value that the estimated value is how far from the real value.

To minimize the cost function we should choose proper $\theta_0,\theta_1$ so that $h_0(x)$ is close to y for our training. 



$hypothesis: h_0(x) = \theta_0 + \theta_1x$
$m=training\ set\ size$



#### Minimize the Cost function:

$$
\min_{\theta_0,\theta_1}\ J(\theta
_0,\theta
_1) = \frac{1}{2m}\sum_{i=1}^m(h_0(x^{(i)} - y^{(i)})^2
$$



## Parameter learning



### Gradient descent

It is an algorithm to find the minimized value of cost function.



Update the parameters until convergence,

First Initialize parameters $\theta_0,\theta_1$.


$$
\theta
_0 := \theta
_0 - \alpha\frac{1}{m}\sum_{i=1}^mh_0(x^{(i)} - y^{(i)})\\
\theta_1 := \theta_1 - \alpha\frac{1}{m}\sum_{i=1}^mh_0(x^{(i)} - y^{(i)})x^{(i)}\\
(\alpha = learning\ rate)
$$



- If $\alpha$ is too small slow convergence
- If $\alpha$ is too large $J(\theta)$ may not decrease and converge

*Choose sufficiently small $\alpha$*



In the case the function is not `convex function` that don't have local optimize,The result might be local optimized value.



#### Simple example using numpy

```python
import numpy as np

X = np.array([[1,2,4,6,7,8,11],[1,1,1,1,1,1,1]])
Y = np.array([2,4,7,9,11,13,15])

# y = Wx <- y = a_1x + a_2

a = 0.1; #learning rate
W = np.random.rand(2)
print(W.size)
LEN = W.size
for i in range(1,100):
    grad = (sum(W.dot(X) - Y))/LEN
    print(sum(W.dot(X) - Y))
    print(W)
    W = W - a*grad
```





### feature scaling

Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly **without normalization**.

Another reason why feature scaling is applied is that `gradient descent` converges **much faster** with feature scaling than without it.



`min-max normalization`: $x'=\frac{x-\min(x)}{\max(x)-\min(x)}$







### Normal equation

$$
\theta \lt R^{n+1}\\
J(\theta_0,\theta_1,...,\theta\_m) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)} - y^{(i)})^2\\
J(\theta) = a\theta^2+b\theta+c\\
\frac{d}{d\theta}J(\theta)= ... 0 (\text{for every } j)
$$

$$
\theta = (X^TX)^{-1}X^Ty\\
(X^TX)^{-1} \text{ is inverse of matrix $X^TX$.}
$$

what if $X^TX$ is non-invertible?(singular/degenerate)



$M$ training examples, $n$ features Gradient descent



### Difference between Gradient descent and Normal equation



#### Gradient descent

- Need to choose $\alpha$.
- Needs many iterations.
- Work well even when n is large.



#### Normal equation

- No need to choose $\alpha$.
- Don't need to iterate.
- Slow if $n$ is very large.



> $O(n^3)$ to calculate $X_n^{-1}$



## Consideration for Linear regression

### Model of complexity

- Dimension
- Number of $X$ variables

### Cross validation

### Log transformation

### overfitting



### Linear regression with multiple variables

$$
hypothesis: h_0(x) = \theta_0 + \theta_1x + \theta_2x + \theta_3x + ... + \theta_n x
\\
parameters: \theta
\\
Cost function: J(\theta_0,\theta_1,\theta_3,...,\theta_n)
$$



## Polynomial regression



## Tensorflow code for linear regression



<script src="https://gist.github.com/Shephexd/0d00f3b4d9aff9f0a9d1b425b3a3c100.js?file=basic_tf_LR.py"></script>

[1]:	https://en.wikipedia.org/wiki/Linear_regression
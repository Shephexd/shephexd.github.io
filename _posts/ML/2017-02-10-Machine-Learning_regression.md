---
layout: post
title: Machine learning(1) - Linear regression
published: True
categories:
- Machine learning
tags:
- Machine learning
- Tensorflow
- Python
---



How can the computer think by self? Normally, they can't think like human, they will use their way to learn and think.

This post is based on the Stanford video lecture,"Machine learning", by Andrew Ng.

<!--more-->



## What is the machine learning?

There are two definition about the machine learning.

1. The field of study that gives computer the ability to learn without being explicitly programmed
2. A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P. If its performance at tasks n T as measured by P, improves with experience E.



### Supervised learning

Computer can learn from  **a given data** set that we already **know** our outputs are. There is a `relation(model)` between the input and the output.

- Regression

  : To predict result within a `continuous` output.

- Classification

  : To predict result in a `discrete` output.



### Unsupervised learning

Computer can solve the problem with **little or no idea** what our results are.

- Clustering

  : Make clusters having similar characters.



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

To minimize the cost function we should choose proper $\theta_0,\theta_1$ so that $h_\theta(x)$ is close to y for our training.



$hypothesis: h_\theta(x) = \theta_0 + \theta_1x$
$m=training\ set\ size$



#### Minimize the Cost function:

$$
\min_{\theta_0,\theta_1}\ J(\theta
_0,\theta
_1) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x_{i}) - y_{i})^2
$$



## Parameter learning



### Gradient descent

It is an algorithm to find the minimized value of cost function.



Update the parameters until convergence,

First Initialize parameters $\theta_0,\theta_1$.


$$
\theta
_0 := \theta
_0 - \alpha\frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})\\
\theta_1 := \theta_1 - \alpha\frac{1}{m}\sum_{i=1}^m ((h_\theta(x^{(i)}) - y^{(i)})x^{(i)})\\
(\alpha = learning\ rate)
$$



- If $\alpha$ is too small slow convergence
- If $\alpha$ is too large $J(\theta)$ may not decrease and converge

*Choose sufficiently small $\alpha$*



In the case the function is not `convex function` that don't have local optimize,The result might be local optimized value.




$$
\frac{\partial}{\partial \theta_j}J(\theta) = \frac{\partial}{\partial \theta_j} \frac{1}{2} (h_\theta(x) - y)^2\\
=  2 \cdot \frac{1}{2} (h_\theta(x) - y) \cdot \frac{\partial}{\partial \theta_j} (h_\theta(x) - y)\\
=   (h_\theta(x) - y) \cdot \frac{\partial}{\partial \theta_j} \left( \sum_{i=0}^n \theta_ix_i - y \right) \\
= (h_\theta(x) - y)x_j
$$


#### LR example using numpy

<script src="https://gist.github.com/Shephexd/e4d259394d557968a5bc8b6d8684f9b1.js?file=LR.py"></script>

<script src="https://gist.github.com/Shephexd/e4d259394d557968a5bc8b6d8684f9b1.js?file=LR.png"></script>

### feature scaling

Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly **without normalization**.

Another reason why feature scaling is applied is that `gradient descent` converges **much faster** with feature scaling than without it.



`min-max normalization`: $x'=\frac{x-\min(x)}{\max(x)-\min(x)}$





### Normal equation




$$
\theta = (X^TX)^{-1}X^Ty\\
(X^TX)^{-1} \text{ is inverse of matrix $X^TX$.}
$$

what if $X^TX$ is non-invertible?(singular/degenerate)



$M$ training examples, $n$ features Gradient descent



#### NE example using numpy

<script src="https://gist.github.com/Shephexd/e4d259394d557968a5bc8b6d8684f9b1.js?file=NE.py"></script>



### Difference between Gradient descent and Normal equation



#### Gradient descent

- Need to choose $\alpha$.
- Needs many iterations.
- Work well even when n is large.
- $O(kn^2)$




#### Normal equation

- No need to choose $\alpha$.
- Don't need to iterate.
- Slow if $n$ is very large.
- $O(n^3)$ to calculate $(X^TX)^{-1}$




## Linear regression with multiple variables

$$
hypothesis: h_0(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_3 + ... + \theta_n x_n\\
x = 
\begin{bmatrix}
x_0\\
x_1\\
x_2\\
\vdots\\
x_n
\end{bmatrix}
\in
R^{n+1}
\ \ \
\theta = 
\begin{bmatrix}
\theta_0\\
\theta_1\\
\theta_2\\
\vdots\\
\theta_n
\end{bmatrix}
\in
R^{n+1}

\\
h_\theta(x) = \theta^Tx
$$



### Gradient descent with multiple features


$$
Cost function: J(\theta_0,\theta_1,\theta_3,...,\theta_n)\\
J(\theta)= \frac{1}{2m}\sum_{i=1}^m(h_\theta x^{(i)} -y^{(i)})^2
$$


Repeat{

$\theta_j:=\theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta) $

}



$$\frac{\partial}{\partial \theta_j} J(\theta) =\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)} $$



### Feature scaling

Feature scaling can make the gradient descent speed  up by reducing the iteration. When the range of dataset is too big, the time for learning will be slow because of the step and oscillation. So feature scaling make the range $-1 \le x \le 1$ or $-0.5 \le x \le -0.5$ to make the range small and It will be helpful for speed up. 



#### Mean normalization

$x1 = \frac{x1-\mu}{\max-\min}$





### MLR example using numpy

<script src="https://gist.github.com/Shephexd/e4d259394d557968a5bc8b6d8684f9b1.js?file=MLR.py"></script>



### Learning rate

In the gradient descent algorithm, The learing rate $\alpha$ is important to decrease the cost function.

How can we select the best $\alpha$?



$J(\theta)$ should decrease after every iteration.

The convergence of graident descent is different depending on applications.



- The cost function increases after each iteration  
  Use smaller $\alpha$
- The cost function decreases slightly after each iteration  
  Use bigger $\alpha$



For sufficiently small $\alpha$, $J(\theta)$ should decrease on every iteration. 

But if $\alpha$ is too small, gradient descent can be slow to converge.



*Choose proper $\alpha$ it depends on application.*



#### Features and polynomial regression

Our hypothesis function can be non-linear.



$$h_\theta(x) = \theta_0 + \theta_1x +  \theta_2x^2 + \theta_3x^3$$

  



### Normal equation

Method to solve for $\theta$ analytically.


$$
\theta \in R^{n+1}\\
J(\theta_0,\theta_1,\dots,\theta_m) = \frac{1}{2m}\sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})^2\\
\frac{\partial}{\partial \theta_j}J(\theta) = 0 (\text{for every }j)
$$

$$
\theta = (X^TX)^{-1}X^Ty\\
$$




$y = X$ 

$\theta = (X^TX)^{-1}X^Ty$



*What if $X^TX$ is non-invertible? (singular\degenerate)*

- Redundant features (linearly dependent)
- Too many features( $ m \le n $)  
  Delete some features, or use regularization.



### Difference between gradient descent and Normal equation



#### Gradient Descent

##### advantages

- more faster thna normal equation - $O(kn^2)$
- works well even when $n$ is large.

##### disadvantages

- Need to choose $\alpha$
- Needs many iterations



   

#### Normal equation

#####  advantages

- No need to choose $\alpha$
- Don't need to iterate



##### disadvantages

- Need to compute $(X^TX)^{-1}$  : $O(n^3)$
- Slow if $n$ is ver large.



## Tensorflow code for linear regression



<script src="https://gist.github.com/Shephexd/0d00f3b4d9aff9f0a9d1b425b3a3c100.js?file=basic_tf_LR.py"></script>

[1]:	https://en.wikipedia.org/wiki/Linear_regression

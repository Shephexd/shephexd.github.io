---
layout: post
title: Machine learning(3) - Regularization
published: True
categories: 
- Machine learning
tags:
- Machine learning
- Python
- Tensorflow
- Matlab
---

What is over fitting and under fitting? How can we avoid this problem?

Are you sure your model is fine for the other cases that is not shown in your training set?

If your model is trained to predcit only your train data, the model is overfitted.



<!--more-->



There are three cases, when your model is validated.

- `under fitting` has **high bias**.
- `over fitting` has **High variance**.
- `Just right`



## Overfitting

If we have too many features, the learned hypothesis may fit the training set very well, but fail to **generalize** to new examples.



### Solution

1. Reduce number of features
   - Maunally select which features to keep
   - Model selection algorithm
2. Regularization
   - Keep all the features but reduce magnitude/ values of parameters $\theta_j$
   - Works well when we have a lot of features, each of which contributes a bit to predicting $y$



## Regularization

The idea of regularization is to avoid the parameter should have similar value. It means that the most parameter will be effective, not biased.

So, when you train your model, you can use this idea by wrting some equation in your cost function to make your parameter not biased.



Small values for parameters $\theta_0,\theta_1, \dots, \theta_n$  

- Simpler hypothesis
- Less prone to overfitting



### Intuition

- Features: $x_1,x_2, \dots, x_{100}$
- Parameters: $\theta_0,\theta_1,\theta_2,\dots,\theta_{100}$


$$
J(\theta) = \frac{1}{2m}\\
\sum^m_{i=1}(h_\theta(x^{(i)}-y^{(i)})^2 ) + \lambda \sum_{i=1}^n \theta^2_j
$$

$\lambda$ is a regularization parameter.



> Q: What if $\lambda$ is set to an extremely large value?
>
> A: The model results in underfitting.





## Regularized linear regression

In the linear regression, the line will be drawn by training. What if there are some outliers?

Your model can't avoid overfitting.

Look at the below equation. you can see there is one more expression on your gradient descent process. 

$\frac{\lambda}{m} \theta_j $ means you don't want to make any value be extremly large than other.



### with gradient descent


$$
\begin{align}
\text{Repeat }\{\\
& \theta_0 := \theta_0 - \alpha \frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_0^{(i)}\\
&\theta_j := \theta_j - \alpha \left[ \frac{1}{m}\sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}+\frac{\lambda}{m}\theta_j \right] & &j \in \{1,2,\dots,n \}
\\
\}
\end{align}
$$

$$
\theta_j := \theta_j(1-\alpha \frac{\lambda}{m}) - \alpha  \frac{1}{m}\sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
$$




### with Normal equation


$$
\begin{align}
&
X = 
\begin{bmatrix}
(x^{(1)})^T\\
\vdots \\
(x^{(m)})^T\\
\end{bmatrix}
&&
y = 
\begin{bmatrix}
(y^{(1)})\\
\vdots \\
(y^{(m)})\\
\end{bmatrix}
\\
\\
&
\min_\theta J(\theta)
\\
& \theta = \left( X^TX + \lambda \cdot L

\right) ^{-1}X^Yy  \\
&\text{where }L= \begin{bmatrix}
0 \\
& 1 \\
& & 1 \\
&&&\ddots \\
&&&& 1 \\
\end{bmatrix}
\end{align}
$$


#### non-invertibility

If $m \lt n$, $X$ will be non-invertible.   
if $m = n$, $X$ may be non-invertible.

if $m \le n$, $X^TX$ is non-invertible. but, when we add the term $\lambda \cdot  L $, then $X^TX + \lambda \cdot L$ becomes invertible.



## Regularized logistic regression





### Cost function

$$
J(\theta) = - \left[ 
\frac{1}{m} \sum_{i=1}^m 
y^{(i)}log(h_\theta(x^{(i)})) +
(1-y^{(i)})log(1-h_\theta(x^{(i)}))
\right]
+ \frac{\lambda}{2m}\sum^n_{j=1}\theta^2_j
$$



### Gradient descent


$$
\begin{align}
\text{Repeat }\{\\
& \theta_0 := \theta_0 - \alpha \frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_0^{(i)}\\
&\theta_j := \theta_j - \alpha \left[ \frac{1}{m}\sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}+\frac{\lambda}{m}\theta_j \right] & &j \in \{1,2,\dots,n \}
\\
\}
\end{align}
$$

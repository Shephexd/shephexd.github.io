---
layout: post
title: Regression
published: True
Categories:
- Machine learing
Tags:
- Machine learning
- Regression
- Data mining
---
Linear regression predicts a real-valued output
based on an input value.

trivial **Model**
$$
l_0 = y + ß(noise)\\
(US)(V^Tl) = y+ß\\
T_∂l_∂ ≈ y
$$

Very simple regression form.
$$
y = x_0+ax_1
$$


<!--more-->

## Linear regression

$$
hypothesis: h_0(x) = \theta_0 + \theta_1x
$$

### Cost function

Choose ∂_0,∂_1 so that h_0(x) is close to y for our traing examples(x,y)

$$
minimize\ J(\theta
_0,\theta
_1) = \frac{1}{2m}∑_{i=1}^m(h_0(x^{(i)} - y^{(i)})^2
$$

Squared error function

### Gradient descent algorithm
It is a algorithm in find the minimized value of cost function.

$$
while(convergnece)\{\\
\theta
_j := \theta
_j - \alpha\frac{\theta
}{∂\theta
_j}J(\theta_0,\theta_1)\
(for\ j = 0\ and\ j = 1)
\\
\}
$$
$$
\theta
_0 := \theta
_0 - \alpha\frac{1}{m}∑_{i=1}^m(h_0(x^{(i)} - y^{(i)})\\
\theta_1 := \theta_1 - \alpha\frac{1}{m}∑_{i=1}^m(h_0(x^{(i)} - y^{(i)})x^{(i)}

\alpha = learning\ rate
$$

If $$\alpha$$ is too small slow convergence
If $$\alpha$$ is too large $$J(\theta)$$ may not decrease and converge

Choose sufficiently small $$\alpha$$

### Model of complexity
- Dimension
- Number of X varables

#### Cross validation

#### Log transformation

#### overfitting


## Linear regression with multiple variables

$$
hypothesis: h_0(x) = \theta_0 + \theta_1x + \theta_2x + \theta_3x + ... + \theta_nx
\\
parameters: \theta
\\
Cost function: J(\theta_0,\theta_1,\theta_3,...,\theta_n)
$$


## Logistic regression


## Polynomial regression

## PCR(Principal component regression)

## PLS(Partial least squares regression)

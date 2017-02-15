---
layout: post
title: Linear Regression
published: False
Categories:
- Machine learning
Tags:
- Machine learning
- Regression
- Data mining
---

In statistics, linear regression is **an approach for modeling the relationship between a scalar dependent variable y and one or more explanatory variables** (or independent variables) denoted X. - [Wikipedia][1]



Simply, **Linear regression predicts a real-valued output based on an input value.**

trivial **Model**

$$
l\_0 = y + \beta(noise) \\
(US)(V^Tl) = y+\beta \\
T\_\alpha l\_\alpha ≈ y \\
$$

$$
hypothesis: h\_0(x) = \theta\_0 + \theta\_1x
$$

<!--more-->



## Linear regression with one variable

Very simple regression form.
$$
y = x_0+ax_1
$$


## feature scaling

Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization.

Another reason why feature scaling is applied is **that gradient descent converges much faster with feature scaling than without it**.

$$
x'={\frac  {x-{\text{min}}(x)}{{\text{max}}(x)-{\text{min}}(x)}}
$$

## Cost function

Choose ∂\_0,∂\_1 so that h\_0(x) is close to y for our traing examples(x,y)

$$
minimize\ J(\theta
\_0,\theta
_1) = \frac{1}{2m}∑_{i=1}^m(h\_0(x^{(i)} - y^{(i)})^2
$$

Squared error function

### Gradient descent algorithm
It is a algorithm in find the minimized value of cost function.

$$
while(convergence)\{\\
\theta
\_j := \theta
\_j - \alpha\frac{\theta
}{∂\theta
\_j}J(\theta\_0,\theta\_1)\\
(for\ j = 0\ and\ j = 1)
\\
\}
$$
$$
\theta
\_0 := \theta
_0 - \alpha\frac{1}{m}∑_{i=1}^m(h\_0(x^{(i)} - y^{(i)})\\
\theta\_1 := \theta\_1 - \alpha\frac{1}{m}∑\_{i=1}^m(h\_0(x^{(i)} - y^{(i)})x^{(i)}
\alpha = learning\ rate
$$

If $$\alpha$$ is too small slow convergence
If $$\alpha$$ is too large $$J(\theta)$$ may not decrease and converge

Choose sufficiently small $$\alpha$$

### Normal equation


$$
\theta = (X^TX)^{-1}X^Ty\\
(X^TX)^{-1}\ is\ inverse\ of\ matrix\ X^TX.
$$

what if $$X^TX$$ is non-invertible?(singular/degenerate)



M training examples, n features
Gradient descent
- Need to choose $$\alpha$$.
- Needs many iterations.
- works well even when n is large.

Normal equation
- No need to choose $$\alpha$$.
- Don't need to iterate.
- slow if n is very large.

> > $$O(n^3)$$ to calculate $$inv(X\_n)$$


## Consideration for Linear regression

### Model of complexity
- Dimension
- Number of X variables

### Cross validation

### Log transformation

### overfitting



## Linear regression with multiple variables

$$
hypothesis: h\_0(x) = \theta\_0 + \theta\_1x + \theta\_2x + \theta\_3x + ... + \theta\_n x
\\
parameters: \theta
\\
Cost function: J(\theta\_0,\theta\_1,\theta\_3,...,\theta\_n)
$$

## Polynomial regression

[1]:	https://en.wikipedia.org/wiki/Linear_regression

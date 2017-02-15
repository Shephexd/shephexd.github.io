---
layout: post
title: Machine learning(2) - Logistic regression
published: True
categories: 
- Machine learning
Tags:
- Machine learning
- Classification
- Data mining
---



In the previous post, we can predict the value using simple linear regression model. Let's think about different situations.

- SPAM(Y/N)
- Fraudulent(Y/N)
- Tumor(Y/N)

In the case the output is discrete like example, Linear regression cannot predict correct answer. Then how can we deal with the discrete case? 



<!--more-->



## What is Logistic regression?

In statistics, logistic regression(logit regression, or logit model) is a regression model where the dependent variable (DV) is categorical. - [Wikipedia][1]



It is used to classify categorial variables, *Pass/Fail, Win/Lose and Alive/Dead*.


$$
y\in {0,1} \begin{cases}
0,  & \text{"Negative class"} \\
1, & \text{"Positive class"}
\end{cases}
$$

$$
\text{Want } 0≤h_{\theta}(x) ≤ 1\\
h_\theta(x) = g(\theta^Tx)\\
g(z) = \frac{1}{1+e^{-z}}
$$


$g(z)= \frac{1}{1+e^{-z}}(\text{ Sigmoid function})$
$h_\theta(x) = \text{estimated probability that y=1 on input }x$
$= P(y=1|x;\theta)$



## Decision boundary

`Decision boundary` is a settle point to separate the region where the hypothesis the hypothesis predicts that y is equal to one and zero.


$$
\begin{cases}
\text{if }h_\theta(x) \ge 0.5 &   y=1\\
\text{if }h_\theta(x) \lt 0.5 & y=0\\
\end{cases}
$$
$g(z)\ge 0.5$ when, $z\ge0$.

$h_\theta(x) = g(\theta^T x) \ge 0.5$ whenever $\theta^Tx \ge 0$



### Linear boundary

$$
h\theta(x) = g(\theta0 + \theta1x1 +\theta2x2)\\
g(z) = \frac{1}{1+e^{-z}}
$$




$$
if\ h_\theta(x) ≥ 0.5,\ y=1 \\otherwise,\ y=0
$$


### Non-linear boundary

$$
h\theta(x) = g(\theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_1^2 + \theta1x2^2)\\
g(z) = \frac{1}{1+e^{-z}}
$$


$$
h_\theta(x) = g(\theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_1^2 + \theta_1x_2^2)
$$


## Cost function

The cost is the value when the answers are wrong.


$$
J(\theta) = cost(h_\theta(x^{(i)},y) = \frac{1}{m}\sum_{i=1}^{m}Cost(H_\theta(x^{(i)},y^{(i)}))
$$

$$
Cost(h_\theta(x),y) = \begin{cases}
-log(h_\theta (x)) &  if\ y=1\\
-log(1-h_\theta(x)) & if\ y=0\\
\end{cases}
$$

>  y = $0$ or $1$ always





### Gradient Descent Algorithm for Logistic


$$
J(\theta) = -\frac{1}{m}[\sum_{i=1}^m y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta (x^{(i)}))]
$$
To fit parameters $\theta$:


$$
h_\theta(x) = \frac{1}{1+e^{-\theta^TX}}\\
$$


$\frac{dJ(\theta)}{d\theta} = (h_\theta(x^i)-y^i)x_j^i$

To make a prediction given new $x$:
output $h_\theta(x)$

$$
\min_\theta J(\theta):\\
Repeat\ \{
\\
\theta_j := \theta_j - \alpha\sum_{i=1}^m(h_\theta(x^i)-y^i)x_j^i\\
\}
$$

## Advanced optimization

Cost function$J(\theta)$. Want $\min_\theta J(\theta)$.

Given $\theta$, we have code that can compute.

- $J(\theta)$
- $\frac{d}{d\theta_j}J(\theta)$



There are some algorithms for optimization.

- Gradient descent
- Conjugate gradient
- BFGS
- L-BFGS



##### Advantages

- No need to manually pick $\alpha$
- Often faster than gradient descent



##### Disadvantage

- More complex



## Multiclass classification

In the case the problem is not binary. How can we deal with logistic regression?

For example

- Email (Work/Friends/Family)
- Weather (Sunny/Cloudy/Rain/Snow)



Just think like there are three kinds of classification like

- Email( Work or not/Friend or not /Family or not)



## Regularization



### Underfitting

`Underfitting` or `High bias`

It means that this algorithm doesn't fit the data well.



### Overfitting

`Overfitting` or `High variance`

It means that this algorithm too fit the data to use other data set.





## Python code



### python code with numpy

<script src="https://gist.github.com/Shephexd/e4d259394d557968a5bc8b6d8684f9b1.js?file=logistic.py"></script>

 

### python code with tensorflow

 <script src="https://gist.github.com/Shephexd/0d00f3b4d9aff9f0a9d1b425b3a3c100.js?file=logistic.py"></script>
---
layout: post
title: Machine learning(2) - Logistic regression
published: True
categories: 
- Machine learning
tags:
- Machine learning
- Python
- Tensorflow
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
$= P(y=1\vert x;\theta)$

$P(y=0\vert x; \theta) = 1 - P(y=1\vert x;\theta)$



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
- $\frac{dJ(\theta)}{d\theta_j}$



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




### Softmax

In the two class case, we use the below equation to predict $\hat{y}$.


$$
H_L(x)=Wx\\
z=H_L(x),g(z)\\
g(z)=\frac{1}{1+e^{-2}}\\
H_R(x)=g(H_L(x))
$$
We can denote this equation to matrix expression.
$$
\begin{bmatrix}
w_1&w_2&w_3
\end{bmatrix}
\begin{bmatrix}
x_1\\
x_2\\
x_3
\end{bmatrix}
=
\begin{bmatrix}
w_1x_1+w_2x_2+w_3x_3
\end{bmatrix}
$$




Then make many $H_R(x)$ to predict multinomial values.


$$
\begin{bmatrix}
w_{A1}&w_{A2}&w_{A3}\\
w_{B1}&w_{B2}&w_{B3}\\
w_{C1}&w_{C2}&w_{C3}
\end{bmatrix}
\begin{bmatrix}
x_1\\
x_2\\
x_3
\end{bmatrix}
=
\begin{bmatrix}
w_{A1}x_1+w_{A2}x_2+w_{A3}x_3\\
w_{B1}x_1+w_{B2}x_2+w_{B3}x_3\\
w_{C1}x_1+w_{C2}x_2+w_{C3}x_3
\end{bmatrix}
\\
=
\begin{bmatrix}
\hat{Y_A}\\
\hat{Y_B}\\
\hat{Y_C}
\end{bmatrix}
$$


The prediction value $\hat{Y}$ can be changed by softmax function into the range 0~1.


$$
S(y_i) = \frac{e^{y_i}}{\sum_je^{y_j}}
$$


### Cost function - Cross entropy



When the prediction is right, there is small or no penalty. In other hand, there is big penalty by the cost function.


$$
Y=L(label)\\
D(S,L)=-\sum_iL_i log(S_i)
$$
It is same to the equation using in the logistic regression.
$$
D(S,L)=-\sum_iL_i log(S_i) \\
= ylog(H(x))-(1-y)log(1-H(x))
$$



### Gradient descent






## Python code



### python code with numpy

<script src="https://gist.github.com/Shephexd/e4d259394d557968a5bc8b6d8684f9b1.js?file=logistic.py"></script>

 

### python code with tensorflow

 <script src="https://gist.github.com/Shephexd/0d00f3b4d9aff9f0a9d1b425b3a3c100.js?file=logistic.py"></script>
---
title: Machine learning(10) - Anomaly detection
layout: post
categories:
- Machine learning
tags:
- Machine learning
- Tensorflow
- Anomaly detection
typora-root-url: /Users/shephexd/Documents/github/pages/
---



How do we detect whether value is anomalous? Like alerting system based on log data, the system must know the some input is different from normal cases.



-   Fraud detection
-   Manufacturing
-   Monitoring log



In this post, I will introduce how to make an Anomaly detection algorithm using `Gaussian`. 



<!--more-->



## Anomaly detection

Normally, the number of anomaly values are much smaller than normal ones.



### Gaussian Distribution

Assuming that our data $x_i$ satisfy  `Gaussian distribution`.



$$
\begin{align}
&x \in R, \\
&\text{if $x$ is a distributed Gaussian with mean $\mu$, variance $\sigma^2$}\\
& x \sim N(\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma}}\exp(-\frac{(x-\mu)^2}{2\sigma^2})
\end{align}
$$



### Anomaly detection algorithm

1.  Choose features $x_i$ that you think might me indicative of anomaly examples.

2.  Fit parameters $\mu_1, \dots, \mu_n,\sigma^2_1,\dots,\sigma^2_n$  
    
    $$
    \begin{align}
    & \mu_j = \frac{1}{m}\sum^m_{j=1}x_j^{(i)} \\
    & \sigma^2_j = \frac{1}{m}\sum^{m}_{j=1} (x_j^{(i)} - \mu_j)^2
    \end{align}
    $$

3.  Given new example $x_k$, compute $p(x)$  
    
    $$
    \begin{align}
    &p(x) = \prod p(x_j;\mu_j,\sigma_j^2) = \prod^n_{j=1}\frac{1}{\sqrt{2n}\sigma_j} \exp(-\frac{(x_j-\mu_j)^2}{2\sigma^2})\\
    &\text{Anomaly if }p(x) \lt \epsilon
    \end{align}
    $$




## Evaluate anomaly detection algorithm

When developing learning algorithm(like `choosing features` etc), making decision is much easier if we have a way of evaluating our learning algorithm.



### Evaluation



-   Training set:  $x^{(i)}, x^{(2)}, \ldots, x^{(m)}$
-   Cross validation set:  $$(x^{(1)}_{cv}, y^{(1)}_{cv}), \dots, (x^{(m_{cv})}_{cv}, y^{(m_{cv})}_{cv})$$
-   Test set:  $$(x^{(1)}_{test}, y^{(1)}_{test}), \dots, (x^{(m_{test})}_{test}, y^{(m_{test})}_{test})$$





Fit model $p(x)$ on training set ${x^{(i)}, \ldots, x^{(m)}}$

on a cross validation/test example predict


$$
\hat y = 
\begin{cases} 
1 & \text{if } p(x) \lt \epsilon \text{ (anomaly)} \\
0 & \text{if } p(x) \ge \epsilon \text{ (normal)} \\
\end{cases}
$$


Possible evaluation metrics

-   `True positive`, `False positive`, `False negative`, `True negative`
-   `Precision` and `Recall`
-   `F1 score`

And $\epsilon$ can be determined by cross validation set





In Anomaly detection,  number of positive(anomaly) examples are very small and can be many different **types**. But,  Number of negative(normal) examples are large.



## Features for anomaly detection

What if our cases cannot be covered with `Gaussian`?



### Non-Gaussian features

Like below transformations make values look like `Gaussian`.



-   $x \to log(x)$
-   $x \to x^{\frac{1}{2}}$
-   $x \to x^{\frac{1}{n}}$



Also, you can create other features if you need

Assuming that you want to detect anomalies on your streaming log data.

You can use the features you defined like below.


$$
x_5 = \frac{\text{CPU load}}{\text{netework traffic}}, x_6 = \frac{(\text{CPU load})^2}{\text{netework traffic}}
$$


### Multivariate Gaussian distribution

The multivariate Gaussian distribution is different from normal Gaussian distribution.


$$
p(x;\mu, \Sigma) = 
\frac{1}{(2\pi)^{\frac{n}{2}}\vert\Sigma\vert^{\frac{1}{2}} }
\exp(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x - \mu)) = p(x) \\
\mu = \frac{1}{m} \sum_{i=1}^m x^{(i)}, \Sigma = \frac{1}{m} \sum_{i=1}^m (x^{(i)} - \mu)(x^{(i)} - \mu)^T
$$

$$
\hat y = 
\begin{cases} 
1 & \text{if } p(x) \lt \epsilon \text{ (anomaly)} \\
0 & \text{if } p(x) \ge \epsilon \text{ (normal)}
\end{cases}
$$


### Differences with original Gaussian model



Let’s see the differences between original one and multivariate Gaussian model



| `Original`                                                   | `Multivariate Gaussian`                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| $p(x) =\\ p(x;\mu_1, \sigma_1^2) \times \dots \times p(x;\mu_n, \sigma_n^2)$ | $p(x;\mu, \Sigma) = \\ \frac{1}{(2\pi)^{\frac{n}{2}}\vert\Sigma\vert^{\frac{1}{2}} } \exp(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x - \mu)) $ |
| Manually create features to capture anomaly where $x_1, \dots , x_n$ | Automatically captures, **Correlations** between features    |
| Computationally cheaper                                      | Computationally expensive                                    |
| OK even if $m$ is small                                      | Must have $ m \gt n $ or else $\Sigma$ is **non-invertible** |



>   If there is no correlations between features, Original and Multivariate Gaussian are same model.
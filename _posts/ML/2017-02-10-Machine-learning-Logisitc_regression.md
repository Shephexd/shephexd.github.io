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
- Farudulent(Y/N)
- Tumor(Y/N)

In the case the output is discrete like example, Linear regression cannot predict correct answer. Then how can we deal with the discrete case? 



<!--more-->



## Logistic regression




$$
\text{want }0\le h_\theta(x)\le1\\
h_\theta(x) = g(\theta^T x)\\
g(z)= \frac{1}{1+e^{-z}}(\text{ Sigmoid function})\\
h_\theta(x) = \text{estimated probability that y=1 on input }x\\
= P(y=1|x;\theta)
$$



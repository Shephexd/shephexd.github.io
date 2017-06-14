---
title: Machine learning(4) - parameter fitting
layout: post
categories: 
- Machine learning
tags:
- Machine learning
---


$$
\frac{(a(1+r)^n-1)}{r}
$$




Evaluating a learning algorithm

<!--more-->

## Debugging a learning algorithm

- Get more traing examples
- Try smaller sets of features
- Try getting additional features
- Try adding polynomial features
- Try decreasing $\lambda$
- Try increasing $\lambda$

$x_1,x_2, x_3, \dots, x_{100} $



## Machine learning diagnostic



## Evaluating a hypothesis



To find the model that can minimize the trainning error and this model should generalize to new examples not in training set.



Divide our data set into 70% of training set and 30% of test set.

$m_{test}$ =. nu



Our purose is finding parameter that can make the training error $J(\theta)$ to be **low** and the test error $J_{test}(\theta)$ to be **high**. 



1. Learn parameter $\theta$ from training data (minimizing traing error $J(\theta)$) 

2. Compute test set error 
   $$
   J_{test}(\theta)=\frac{1}{2m_{test}}\sum_{i=1}^{m_{test}}y_{test}^{(i)}logh_\theta(x^{(i)}_{test})
   $$
   ​
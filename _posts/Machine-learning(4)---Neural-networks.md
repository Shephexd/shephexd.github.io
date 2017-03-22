---
layout: post
title: Machine learning(4) - Neural networks
published: True
categories: 
- Machine learning
tags:
- Machine learning
- Python
- Tensorflow
- Matlab
---



We can solve many problem by using linear and logistic regression. In this post, I will introduce about Neural network to solve non-linear problem.

So why do we need to use neural network even though the linear and logtistic regression is quite good algorithm?



<!--more-->



## Non-linear Hypothesis



## Neural network

Algorithms that try to mimic the brain.



Input layer

Hidden layer

Output layer



$a_i^{(j)} = $ "activation" of unit $i$ in layer $j$.

$\Theta^{(j)}$ = matrix of weights controlling function mapping from layer $j$ to layer $j + 1$



$h_\theta(x) = $  



## Intuitions



### AND Problem



#### AND logic

| $x_1$ | $x_2$ | $y$  |
| :---: | :---: | :--: |
|   0   |   0   |  0   |
|   0   |   1   |  0   |
|   1   |   0   |  0   |
|   1   |   1   |  1   |



### OR Problem



#### OR logic

| $x_1$ | $x_2$ | $Y$  |
| :---: | :---: | :--: |
|   0   |   0   |  0   |
|   0   |   1   |  1   |
|   1   |   0   |  1   |
|   1   |   1   |  1   |



### NOT Problem



#### NOT logic

| x1   | Y    |
| ---- | ---- |
| 0    | 1    |
| 1    | 1    |



### XOR Problem



#### XOR logic

| $x_1$ | $x_2$ | $Y$  |
| :---: | :---: | :--: |
|   0   |   0   |  0   |
|   0   |   1   |  1   |
|   1   |   0   |  1   |
|   1   |   1   |  0   |

 

### XNOR Problem


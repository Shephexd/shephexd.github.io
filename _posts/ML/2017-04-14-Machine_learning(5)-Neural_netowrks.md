---
title: Machine learning(5) - Neural network
layout: post
categories:
- Machine learning
tags:
- Machine learning
---




<!--more-->

## Gradient checker



To make sure your gradient descent is right, you can use approximation for derviation to compare gradient descent.



Two side approximation for derivation
$$
\frac{J(\theta+\epsilon) - J(\theta-\epsilon)}{2\epsilon}
$$





## Initial value of $\Theta$



### Zero initalization

$\Theta_{ij}^{(l)}=0$ For all $,i,j,l$

$\Theta_{01}^{(1)} = \Theta^{(1)}_{02}$



All of your your headed unit have same values if you use zero initialization for neural n



### Random initialization: Symmetry breaking

Initialize each $\Theta^{(l)}_{ij}$ to a random value in [$-\epsilon$, $\epsilon$]

$ -\epsilon \le \Theta_{ij}^{(l)} \le \epsilon$



## Putting it together



- Number of input units: Dimension of features $x^{(i)}$
- Number of output units: Number of classes
- Number of hidden units: 1 hidden layer or more than 1.



## Training a neural network

1. Randomly initalize weights
2. Implements forward propagation to get $h_\Theta(x^{(i)})$ for any $x^{(i)}$
3. Implement code to compute cost function $J(\Theta)$
4. Implement backprop to compute partial derivatives $\frac{\partial}{\partial \Theta_{jk}^{(l)}}J(\Theta)$
5. Use gradient checking to compare $\frac{\partial}{\partial \Theta_{jk}^{(l)}}J(\Theta)$ computed using backpropagation vs using numerical estimate of gradient of $J(\Theta)$
6. Use gradient descent or advanced optimization method with backpropagation to try to minimize $J(\Theta)$ as. function of parameters $\Theta$

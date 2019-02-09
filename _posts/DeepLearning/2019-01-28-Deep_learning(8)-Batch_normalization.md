---
layout: post
title: Deep learning(8) - Batch Normalization
published: True
categories:
- Deep learning
tags:
- Machine learning
- Neural network
- Deep learning
- Python
- Tensorflow
typora-root-url: /Users/shephexd/Dropbox/github/pages/
---

In this post, I will introduce the way to speed up training for Neural network with `batch normalization`.

Normalization is helpful to be converged with gradient descent by reducing the oscillation error.

So, how we can it add into Neural network layer called `Batch Normalization Layer`



<!--more-->

> This post is based on the video lecture[^1] 



## Normalization

Normalizing inputs can speed up learning.



![normalize_input](/assets/post_images/DeepLearning/normalize_input.jpeg)



Let's see the above image. With the left figure, an unnormalized cost function, Gradient descent will be slower because of the oscillation.





### Input normalization

$$
\mu = \frac{1}{m} \sum_{i} x^{(i)} \\
x = x - \mu \\
\sigma^2 \frac{1}{m} \sum_{i}(x^{(i)})^2 \\
x = x / \sigma^2
$$





## Batch noramilziation in NN

In the neural network, the output layer of previous layer is input of next layer.

So, we can normalize the output layer of preivous layer before forwarding to next layer as input.



*Inutition: normalize the output of preivous layer before forwarding to make training fatser* 





### Output of NN normalization


$$
\begin{align}
&\text{ Given some intermediate value in NN: } z^{(1)}, \dots, z^{(n)}\\
&\mu = \frac{1}{m} \sum_i z^{(i)} \\
&\sigma^2 = \frac{1}{m} \sum_i ( z- \mu)^2 \\
&z^{(i)}_{norm} = \frac{z^{(i)} - \mu}{\sqrt{\sigma^2 + \epsilon}} \\
&\tilde{z}^{(i)} = \gamma \cdot z_{norm}^{(i)} + \beta\\
\\
&\text{use $\tilde{z}$ instead of $z^{[\ell](i)}$ } \\
\\
&\gamma, \beta: \text{leranable parameter}
\end{align}
$$



$$
\text{if }\gamma=\sqrt{\sigma^2 + \epsilon}, \beta = \mu\\
\text{then } \tilde{z}^{[\ell](i)} = z^{(i)}
$$





### Adding batch Norm to a network



![2_neural_network](/assets/post_images/DeepLearning/2_neural_network.PNG)

$$
\begin{align}
X \overset{x^{[\ell]}, b^{[\ell]}}{\rightarrow} Z^{[\ell]} \overset{\text{Batch norm}}{\rightarrow} \tilde{Z}^{[\ell]}
\end{align}
$$





### Why does batch Norm work?



*Batch Normalization is helpful to solve covariate shift by reducing the amount that the distribution of hidden unit values shift to speed up.*



### Batch Norm at Test process


$$
\mu = \frac{1}{m}\sum_{i}Z^{(i)} \\
\sigma^2 = \frac{1}{m}\sum_{i}(Z^{(i)} - \mu)^2 \\
Z_{norm}^{(i)} = \frac{Z^{(i)}-\mu}{\sqrt{\sigma^2 + \epsilon}} \\
\tilde{Z}^{(i)} = \gamma Z_{norm}^{(i)} + \beta
$$



$\mu, \sigma^2$: *estimate using exponential weighted average(across minibatch)*







[^1]: https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning	"Deep learning specialization"
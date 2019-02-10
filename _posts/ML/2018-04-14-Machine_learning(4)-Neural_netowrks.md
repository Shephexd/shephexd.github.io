---
title: Machine learning(4) - Neural network
layout: post
categories:
- Machine learning
tags:
- Machine learning
- Tensorflow
- Neural network
typora-root-url: /Users/shephexd/Documents/github/pages/
---



The origins of Neural network is a try to mimic the human's brain.

In this post, i will introduce how Neural network works and how to implement.




<!--more-->



## Neural Model

Neural is a component of our brain. Before introducing Neural network, i need to introduce how our brain works.



the brain is primarily composed of neurons connected to each other. Each neuron send the signal through synapse when the input(stimulus) is activated.



Researchers researching neural network want to solve the complex problem by adapting this neural system.

![neural-network](/assets/post_images/DeepLearning/neural-network.png)



Like the above image, Input can be passed through four nodes and can be transfered to next five nodes. Each node works like a neuron, and each edge works like synapse. And the output layer determine the decision.





## The structure of Neural network



### A neural unit

So, look at the most simple neural model. As i said before, each node works like a neuron. Change the text to equation. 

$$
x = \begin{bmatrix}
x_0\\
x_1\\
x_2\\
x_3
\end{bmatrix}
, 
\theta = \begin{bmatrix}
y_0\\
y_1\\
y_2\\
y_3
\end{bmatrix}
\\
y = h_\theta(\theta^Tx)
$$



When the $h_\theta$ is a `sigmoid function`, Then y will be `0`  or `1`.  If the y is `0`, then we think the neuron is activated, otherwise not activated.



### Forward propagation

The neural network means that the connection of neural units. The output of one neural unit will be a input of next neuron. Also, some neurons output can be inputs of a neuron. this process is called `forward propagation`.



-   $a_i^{(j)}$ = activation of unit i in layer j
-   $\theta^{(j)}$ = matrix of weights controlling function mapping from layer $j$ to layer $j+1$
-   $a_1^{(2)} = g(\theta_{10}^{(1)}x_0 + \theta_{11}^{(1)}x_1 + \theta_{12}^{(1)}x_2 + \theta_{13}^{(1)}x_3 )$
-   $a_2^{(2)} = g(\theta_{20}^{(1)}x_0 + \theta_{21}^{(1)}x_1 + \theta_{22}^{(1)}x_2 + \theta_{23}^{(1)}x_3 )$
-   $a_3^{(2)} = g(\theta_{30}^{(1)}x_0 + \theta_{31}^{(1)}x_1 + \theta_{32}^{(1)}x_2 + \theta_{33}^{(1)}x_3 )$
-   $h_\theta(x) = a_1^{(3)} = g(\theta_{10}^{(2)}x_0 + \theta_{11}^{(2)}x_1 + \theta_{12}^{(2)}x_2 + \theta_{33}^{(2)}x_3 )$





>   If neural network has $s_j$ units in layer $j$, $s_j+1$ units in layer $j+1$, then $\theta^{(j)}$ will be of dimension $s_{j+1} \times (s_j + 1)$



### Neural units

-   Number of input units: Dimension of features $x^{(i)}$
-   Number of output units: Number of classes
-   Number of hidden units: 1 hidden layer or more than 1.



## Training a neural network

1.  Randomly initalize weights
2.  Implements forward propagation to get $h_\Theta(x^{(i)})$ for any $x^{(i)}$
3.  Implement code to compute cost function $J(\Theta)$
4.  Implement back propagation to compute partial derivatives $\frac{\partial}{\partial \Theta_{jk}^{(l)}}J(\Theta)$
5.  Use gradient checking to compare $\frac{\partial}{\partial \Theta_{jk}^{(l)}}J(\Theta)$ computed using back propagation vs using numerical estimate of gradient of $J(\Theta)$
6.  Use gradient descent or advanced optimization method with back propagation to try to minimize $J(\Theta)$ as. function of parameters $\Theta$



Like the other machine learning algorithms, training Neural network is same. Define a cost function and minimize it.

The difference from other algorithms is Neural network can have multiple layers. Training process must go back from cost function to input features through hidden layers.



I wrote down about the detail how neural network can be trained on my [post](http;//http://shephexd.github.io/deep%20learning/2017/06/10/Deep_learning-Learning.html) for deep learning.



## Implementation with Tensorflow


### Perceptron

 <script src="https://gist.github.com/Shephexd/0d00f3b4d9aff9f0a9d1b425b3a3c100.js?file=perception.py"></script>


### Neural Network


 <script src="https://gist.github.com/Shephexd/0d00f3b4d9aff9f0a9d1b425b3a3c100.js?file=NN.py"></script>


### Neural Network for MNIST

 <script src="https://gist.github.com/Shephexd/0d00f3b4d9aff9f0a9d1b425b3a3c100.js?file=DNN_MNIST.py"></script>


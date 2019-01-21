---
layout: post
title: Deep learning(1) - Introduction to Deep learning
published: True
categories:
- Deep learning
tags:
- Machine learning
- Neural network
- Deep learning
- Python
- Tensorflow
typora-root-url: /Users/shephexd/Documents/github/pages/
---



Deep learning start from the idea to make thinking machine.

Then how to make the thinking machine?

People use the brain to think something. If computer can imitate the human's brain, Can computer think like human?

<!--more-->

This post is based on the video lecture[^1] and wildml[^2]







`Andrew Ng` said deep learning is electricity. It means Deep learning will be used widely and the most important component for our life.



Then What `Deep learning` is different from `Machine learning`?

  

## Deep learning

The main component of deep learning is Neural network that is one of the ML model.

The Neural network's idea is to mimic the human's brain using neurons and synapses. 



Imagine you build a construction with the Lego bricks. The construction is depending on how you stack each brick.



-   Neuron: Lego brick
-   Neurons: Stacked Lego bricks



Like the Lego bricks, The neural networks can be build by stacking to each neuron.

Let's see how we create a neuron and stack it.





## Perceptron

This algorithm is designed by `Frank Rsenblatt` in 1957. 



In the `Perceptron`, The output$(1,0)$ is one signal after calculating with weights and inputs. 

When the input signal is enough big, then the output signal will be 1, otherwise 0.


$$
y = 
\begin{cases}
0 & (w_1x_1+w_2x_2 \le \theta)
\\
1& (w_1x_1+w_2x_2 \gt \theta)
\end{cases}
$$
$x$ is input signal and y is output signal. If the out value that calculated by the weighs and input signal is bigger than $\theta$, it will be 1, otherwise 0. 



### Application for perceptron



#### AND logic

| $x_1$ | $x_2$ | $y$  |
| :---: | :---: | :--: |
|   0   |   0   |  0   |
|   0   |   1   |  0   |
|   1   |   0   |  0   |
|   1   |   1   |  1   |



#### OR logic

| $x_1$ | $x_2$ | $Y$  |
| :---: | :---: | :--: |
|   0   |   0   |  0   |
|   0   |   1   |  1   |
|   1   |   0   |  1   |
|   1   |   1   |  1   |



#### NAND logic

| $x_1$ | $x_2$ | $Y$  |
| :---: | :---: | :--: |
|   0   |   0   |  1   |
|   0   |   1   |  1   |
|   1   |   0   |  1   |
|   1   |   1   |  0   |





### Limitation of Perceptron

`Perceptron` is the way to solve linear problem. It means that it can't solve non-linear problem correctly. Then, how can we solve non linear problem correctly by using some ways?

The solution is using two combined linear systems.



For example,
$$
s1=NAND(x_1,x_2)\\
s2 = OR(x_1,x_2)\\
y=AND(s_1,s_2)
$$


#### XOR logic

| $x_1$ | $x_2$ | $s_1$ | $s_2$ | $Y$  |
| :---: | :---: | :---: | :---: | :--: |
|   0   |   0   |   1   |   0   |  0   |
|   0   |   1   |   1   |   1   |  1   |
|   1   |   0   |   1   |   1   |  1   |
|   1   |   1   |   0   |   1   |  0   |



It is called as "Multi Layer Perceptron(MLP)". It is a basic concept of neural networks. But the problem is the weights because we set the parameters manually not automatically.





## Neural network

As i said before, the MLP can solve complex problem like non-linear problem. however, we don't know the way to set the parameters(weights) automatically not manually. So, there is neural network to solve and make it set the parameters automatically.





### Concepts

Neural have many `synapses`. Synapses can be activated by the `signal` being enough to activate the synapse.

for thinking computer a computer has the `equation` if the input value is enough big to activate the `activation function`



![neural-network](/assets/post_images/DeepLearning/neural-network.png)



> synapse = activation function  
>
> signals = input values







[^1]: https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning	"Deep learning[Coursera]"
[^2]: http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/	"wild ml blog"


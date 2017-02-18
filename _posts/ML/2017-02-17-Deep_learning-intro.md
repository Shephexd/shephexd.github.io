---
layout: post
title: Deep learning(1) - Basic Neural network concepts
published: True
categories:
- Machine learning
tags:
- Machine learning
- Neural network
- Deep learning
- Python
- Tensorflow
---



Deep learning start from the idea to make thinking machine.

Then how to make the thinking machine?

People use the brain to think something. If computer can imitate the human's brain, Can computer think like human?



This post is based on the [video lecture](https://www.inflearn.com/course/기본적인-머신러닝-딥러닝-강좌/) and [wildml post](http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/)



<!--more-->

## Neural network

Neural have many `synapses`. Synapses can be activated by the `signal` being enough to activate the synapse.

for thinking computer a computer has the `equation` if the input value is enough big to activate the `activate function`



synapse = activate function

signals = input values



Using this `Neural network`, Computer can solve the AND/OR problems!



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





### The first problem for NN

But the first problem is occurred by XOR problems.

`One logistic regression` unit cannot separate XOR!





#### XOR

| $x_1$ | $x_2$ | $Y$  |
| :---: | :---: | :--: |
|   0   |   0   |  0   |
|   0   |   1   |  1   |
|   1   |   0   |  1   |
|   1   |   1   |  0   |

 

How to computer can solve the `non-linear problem` by Neural network?



Use the `multi layers`!

#### Solution

If we use the MLP, we can solve the XOR problem. but, no one can calculate the each `bias` and `weight`



> No one on earth had found a viable way to train - Marvin Minsky



#### Logic for solution

| $x_1$ | $x_2$ | $y_1$ | $y_2$ | $Y$  |
| :---: | :---: | :---: | :---: | :--: |
|   0   |   0   |   0   |   1   |  0   |
|   0   |   1   |   0   |   0   |  1   |
|   1   |   0   |   0   |   0   |  1   |
|   1   |   1   |   1   |   0   |  0   |







## Back propagation

Using the multi layer perception, we can calculate the hypothesis for the XOR problem.

But how can the machine learn the weight and bias from the multilayer?

Here is a solution called `back propagation`.





The input data affect the result by passing the weights. 

Then, using chain rule, the error change the weights by passing from the last layer to first layer.



### Partial derivative

`Chain rule` is $\frac{df(g(x))}{dx}=\frac{df}{dx} = \frac{df}{dg}\cdot\frac{dg}{dx}$



It is basic equation for the partial derivative.

The chain rule is an essential idea for back-propagation.



### Back propagation process



1. forward calculation

   ​
   $$
   x
   \overset{\mathtt{W_1x+b_1}}{\longrightarrow}
   z_1
   \overset{\mathtt{tanh(z_1)}}{\longrightarrow}
   a_1
   \overset{\mathtt{W_2a_1+b_2}}{\longrightarrow} 
   z_2
   \overset{\mathtt{softmax(z_2)}}{\longrightarrow}

   a_2=\hat{y}
   $$
   ​

2. backward calculation

   ​
   $$
   x
   \overset{\mathtt{ \frac{dz_1}{dW_1} + \frac{dz_1}{db_1} }}{\longleftarrow}
   z_1
   \overset{\mathtt{ \frac{da_1}{dz_1} }}{\longleftarrow}
   a_1
   \overset{\mathtt{  \frac{dz_2}{dW_2} + \frac{dz_2}{db_2} }}{\longleftarrow} 
   z_2
   \overset{\mathtt{ \frac{d\hat{y}}{dz_2} }}{\longleftarrow}

   a_2=\hat{y}
   $$




#### cost function(loss function)

$$
L(y,\hat{y}) = -\frac{1}{N}\sum_{n \in N}\sum_{i \in N} y_{n,i}log\hat{y}_{n,i}
$$



To update our `weights` on the networks, we need to calculate $\frac{dL}{dw}$ how much the value affect the function. Using chain rule of partial derivative, We can derivative $L(y,\hat{y})$ by our weights $W$.


$$
\sigma_3 = y - \hat{y}\\
\sigma_2 = (1 - tanh^2z_1) \cdot \sigma_3W^T_2\\
\frac{dL}{dW_2}=a_1^T\sigma_3\\
\frac{dL}{db_2}=\sigma_3\\
\frac{dL}{dW_1}=x_1^T\sigma_2\\
\frac{dL}{db_1}=\sigma_2\\

$$



$$
x\overset{\mathtt{W_1x_1+b_1}}{\longrightarrow} \fbox{Input} 
\overset{\mathtt{tanh}}{\longrightarrow} \fbox{Hidden 1} 
\overset{\mathtt{W_2a_2+b_2}}{\longrightarrow} \fbox{Hidden 2} 
\overset{\mathtt{softmax}}{\longrightarrow} \fbox{Output} 
$$




Let me show the expression using matrix for back propagation.

In the single case we can get the value easily.



$$
y = Wx \\
y=
\begin{bmatrix}
w_{1} \cdots w_{n}
\end{bmatrix}
\begin{bmatrix}
x_{1} \\
\vdots \\
x_{n}\\
\end{bmatrix}\\
z = \frac{1}{1+e^{-y}}\\
$$

$$
\frac{df}{dw_n}=\frac{dy}{dw_n} \cdot \frac{dz}{dy}\\
\frac{dz}{dy} = \frac{e^x}{(e^x+1)^2}
$$



### The problem in the back propagation

In the case the neural network is deep, the `back propagation` can't affect the first weights.



#### Solution for this problem



>  Neural networks with many layers really could be trained well, If the weights are initialized in a clever way.



It make rebrand the name to `Deep learning`



- Our labeled datasets were thousands of times too small.
- Our computers were millions of times too slow.
- We initialized the weights in a stupid way.
- We used the wrong type of non-linearity.





## Convolutional Neural networks

A researcher research the cat's brain to analysis the  brain's reaction when the cat watch the image.

The result is that the partition of neural is activated not total neural to recognize the image.



## Examples



### Image recognition

The image net is a competition for the computer vision to recognize image's categories.



### Image caption

Computer can explain the image with categories like human being.



###  Chat bot

Computer can communicate with human by understanding sentences and 




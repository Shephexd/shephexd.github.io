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



This post is based on the [video lecture](https://www.inflearn.com/course/기본적인-머신러닝-딥러닝-강좌/)



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







## Backpropagation

It is the solution to solve the XOR problem.



The input data affect the result by passing the weights. And the error change the weights by passing from the last to beginning.



### The problem in the backpropagation

In the case the neural network is deep, the `backpropagation` can't affect the first weights.



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
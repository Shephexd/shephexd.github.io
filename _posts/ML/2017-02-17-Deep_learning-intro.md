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

for thinking computer a computer has the `equation` if the input value is enough big to activate the `activation function`



synapse = activation function

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



Here is an example about a neural network.


$$
x\overset{\mathtt{}}{ \longrightarrow } 
\overset{\mathtt{W_1x_1+b_1}}{ \fbox{Input} } 
\overset{\mathtt{tanh}}{\longrightarrow}
\overset{\mathtt{W_2a_2+b_2}}{\fbox{Hidden 1} } 
\overset{\mathtt{softmax}}{\longrightarrow} 
{\fbox{Output}}
$$


#### forward calculation

$$
x
\overset{\mathtt{W1x+b1}}{\longrightarrow}
z_1
\overset{\mathtt{tanh(z_1)}}{\longrightarrow}
a_1
\overset{\mathtt{W2a1+b_2}}{\longrightarrow} 
z_2
\overset{\mathtt{softmax(z_2)}}{\longrightarrow}
a_2=\hat{y}
$$



#### backward calculation

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



### The problem in the back propagation

In the case the neural network is deep, the `back propagation` can't affect the first weights.



#### Solution for this problem

>  Neural networks with many layers really could be trained well, If the weights are initialized in a clever way.



It make rebrand the name to `Deep learning`



- Our labeled datasets were thousands of times too small.
- Our computers were millions of times too slow.
- We initialized the weights in a stupid way.
- We used the wrong type of non-linearity.




## Setting for the Neural network



### Activation function

In the neural networks, the output from previous nodes is used as input for current node. And the output values from previous nodes can be activated by `activation function`.

The `activation functions` have some different equation.



#### Sigmoid

`Sigmoid` is not working well for deep learning. Because it is `non-linearity` function it occur  `vanishing gradient`. It means the effect from last nodes to first nodes is too small.

The `Sigmoid` is used only for the output value for last nodes on neural network.


$$
f(x) = \frac{1}{(1+e^{-x})}
$$

#### ReLU(Rectified Linear Unit)

For solving `vanishing gradient`, ReLU is used for deep learning.


$$
f(x) = \begin{cases}
x & \text{if }x \gt 0 \\
0 & \text{if } x \le 0
\end{cases}
$$


```python
def relu(x):
    return x*(x>0)

def deriv_relu(x):
    return 1*x(>0)
```



#### Leaky ReLU


$$
f(x) = \begin{cases}
x & \text{if }x \gt 0 \\
0.1x & \text{if } x \le 0
\end{cases}
$$



#### Max out

$$
\max(w_1^Tx + b1, w_2^Tx + b_2)
$$



#### ELU

$$
f(x) = \begin{cases}
x & \text{if }x \gt 0 \\
\alpha (exp(x) - 1) & \text{if } x \le 0
\end{cases}
$$



#### tanh

$$
tanh(x)
$$



### Initialize weight values wisely



#### Restricted Boltmann machine(RBM)

It is not used anymore. But the concept is interesting.

For initializing the weight values, Find the minimum weights between input weights and updated weights after forwarding and backwarding.

It is also called as `Encoder` and `Decoder`.



##### Fine tuning

The RBM machine can pre-train the neural networks each nodes iteratively. First and second nodes, Second and Third nodes, and $\cdots$.



#### Other ways

But there are more easy way for weight initialization. Simple methods are OK.

- Xavier initialization

  `np.random.randn(fan_in, fan_out)/np.sqrt(fan_in)`

- He's initialization 

  `np.random.randn(fan_in,fan_out)/sp.sqrt(fan_in/2)`



### Overfitting

Overfitting can be checked by test data.

- **Very high accuracy** on the training dataset
- **Poor accuracy** on the test data set



####  Solution for overfitting

- More training data
- Reduce the number of features(*Not for deep learning*)
- Regularization



#### Regularization

Let's not have too big numbers in the weight
$$
cost = cost + \lambda \sum w^2
$$


#### Drop out

"*Randomly set some neurons to zero in the forward pass*"

`Drop out` is a way to avoid overfitting in neural network. It is simple idea that just kill some nodes randomly.



For training, we can use the dropout, but not for the evaluation.

```python
dropout_rate = tf.placeholder("float")
_L1 = tf.nn.relu(tf.add(tf.matmul(X,X1), B1))
L1 = tf.nn.dropout(_L1, dropout_rate)

# Train
sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys, dropout_rate: 0.7})

# Evaluation
print ("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels, dropout_rate: 1}))
```





#### Model ensemble







## Neural networks

### CNN

A researcher research the cat's brain to analysis the  brain's reaction when the cat watch the image.

The result is that the partition of neural is activated not total neural to recognize the image.



#### RNN(LSTM)

This network is useful to learn the language model having sequences data. It is widely used in the language, chat bot and machine translation.





### Applications



#### Image recognition

The image net is a competition for the computer vision to recognize image's categories.



#### Image caption

Computer can explain the image with categories like human being.



####  Chat bot

Computer can communicate with human by understanding sentences and 

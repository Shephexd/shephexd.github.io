---
layout: post
title: Deep learning(1) - Basic Neural network concepts
published: True
categories:
- Deep learning
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

## Perceptron

This algorithm is designed by Frank Rsenblatt in 1957. 



In the perceptron, The output$(1,0)$ is one signal after calculating with weights and inputs. 

When the input signal is enough big, then the output signal will be 1, otherwise 0.


$$
y = 
\begin{cases}
0 & (w_1x_1+w_2x_2 \le \theta)
\\
1& (w_1x_1+w_2x_2 \gt \theta)
\end{cases}
$$
$x$ is input signal and y is output signal. If the out value that calculated by the weighs and input singal is bigger than $\theta$, it will be 1, otherwise 0. 



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





### Limitation of perceptron

Perceptron is the way to solve linear problem. It means that it can't solve non-linear problem correctly. Then, how can we solve non linear problem correctly by using some ways?

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



It is called as "Multi Layer Perceptron(MLP)". It is a basic copcept of neural networks. But the problem is the weights because we set the parameters manually not automatically.





## Neural network

As i said before, the MLP can solve coplex problem like non-linear problem. however, we don't know the way to set the parameters(weights) automatically not manually. So, there is neural network to solve and make it set the parameters automactially.





### Concepts

Neural have many `synapses`. Synapses can be activated by the `signal` being enough to activate the synapse.

for thinking computer a computer has the `equation` if the input value is enough big to activate the `activation function`



![neural-network](/assets/post_images/DeepLearning/neural-network.png)



> synapse = activation function  
>
> signals = input values



### Activation function

Activation function is activated by the big enough input. When the proper input comes into activation function, activation function returns output depending on its function.



There are three kind of widely used function for activation.



#### Step function

$$
y = \begin{cases}
1 & \text{if }x\gt0\\
0 & \text{if }x\le1
\end{cases}
$$



![step_function](/assets/post_images/DeepLearning/step_function.png)

```python
def step_function(x):
    return np.array(x > 0, dtype=np.int)

# Step function
x = np.arange(-5.0, 5.0, 0.1)
y = step_function2(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()
```





#### Sigmoid

$$
h(x) = \frac{1}{1+e^{-x}}
$$



![sigmoid](/assets/post_images/DeepLearning/sigmoid.png)

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid function
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
```



#### ReLU(Rectified Linear Unit)

$$
y = \begin{cases}
x & \text{if }x\gt0\\
0 & \text{if }x\le1
\end{cases}
$$



![relu](/assets/post_images/DeepLearning/relu.png)



```python
def relu(x):
    return np.maximum(0, x)

# Relu function
x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-0.1, 5.1)
plt.show()
```





This kind of activation functions are called "`Non-linear function`". Because it can be expressed by using one line. Then, Why do we need to use non-linear function instead of linear function like $f(x)=ax+b$? 

Because of the depth of networks. If we use the linear function as an activation function in the networks, The networks can't be learned enough.

Assuming that we use linear function $h(x) = cx $ as an activation function. 


$$
h(x) = cx\\
y(x) = h(h(h(x)))\\
y(x) = c*c*c*x = c^3x
$$
In this case, We will loss the benefit to use multi-layer networks.



### Output layer



#### Softmax function

$$
y_k = \frac{\exp({a_k})}{\sum^n_{i=1}\exp({a_i})}
$$





The solution by machine learning has learning and inference. In learning process, Model learn the data, In inference process, model infer the feasible result based on learning data.



## Learning Neural Network



### 



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

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

In the classification problem, to solve the problem we need to set up proper the number of output neurons. If we want to classify digit number of 0 to 9, we need to set up 10 of neurons as the output layer.  



The solution by machine learning has learning and inference. In learning process, Model learn the data, In inference process, model infer the feasible result based on learning data.  



If you already know the logistic function, it is perfectly same with using `softmax function`.



#### Softmax function

Each node of output layer will have the value between 0 to 1. And the sum of all node is 1. It can be considered as the probability of likelihood for prediction.




$$
y_k = \frac{\exp({a_k})}{\sum^n_{i=1}\exp({a_i})}\\
\text{But, we have to change for Implementing code to avoid overflow problem in softmax function.}
$$



$$
\begin{align}
y_k &= \frac{\exp(a_k)}{\sum_{i=1}^n\exp(a_i)}\\
&= \frac{C\exp(a_k)}{C\sum_{i=1}^n\exp(a_i)}\\
&= \frac{\exp(a_k+logC)}{\sum_{i=1}^n\exp(a_i+logC)}\\
&= \frac{\exp(a_k+C')}{\sum_{i=1}^n\exp(a_i+C')}
&\end{align} \\
\text{normally, $C$ is selected for maximum number in the array.}
$$




### Inference on neural network

$$
\begin{align}
&X & W1 && W2& &W3 & & \rightarrow &&Y\\
&N\times M & M \times 50 && 50 \times 100& & 100 \times 10 & & &&N \times 10
\end{align}
$$


$$
\begin{align}
& a1 = X \times W1\\
& z1 = activation(a1) \\
& a2 = a2 \times W2\\
& z2 = activation(a2)\\
& a3 = z2 \times W3\\
& Y = softmax(a3)
\end{align}
$$


### Batch process

When we learn the model for prediction, we are used to the whole data set, $N \times M$. 





What if data size is too big to learn one time? We have to divide our data set with proper size. The solution is using batch process.


$$
\text{I will select $100$ as the batch size for our model.}\\
\begin{align}
&X & W1 && W2& &W3 & & \rightarrow &&Y\\
&100\times M & M \times 50 && 50 \times 100& & 100 \times 10 & & &&100 \times 10
\end{align}
$$


```python
batch_size = 100
accuracy_cnt = 0

for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
    print("Accuracy:", str(float(accuracy_cnt) / len(x)))
```



It make the learning process faster to decrease processing time on the numerical computing library.

It is also helpful to prevent overflow and memory exhaustion by reducing burden on the `BUS` on `CPU` or `GPU`.



## Sample code



```python
import pickle
import numpy as np

def get_data():
    (x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=True, one_hot_label=False)

    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
        
        return network
    

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = sigmoid(a3)
    
    return y


x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
    print("Accuracy:", str(float(accuracy_cnt) / len(x)))
```


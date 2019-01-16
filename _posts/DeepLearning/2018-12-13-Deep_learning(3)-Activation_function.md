---
layout: post
title: Deep learning(3) - Activation function
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



In the previous post, I introduced the small neural unit, Logistic regression being considered neural network without hidden layers.



The main difference of neural network from logistic regression is stacking layers called `hidden layers`.

When stacking each hidden layer, we also stack activation function. 

This post is about the activation function and why we need it.



<!--more-->





## Activation function


Activation function is activated by the big enough input. When the proper input comes into activation function, activation function returns output depending on its function.



### Kind of Activation functions

There are some activation functions to pass the input to next layer. Also, the result of activation function of last layer can be used for output.



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



### Why non-linear activation function is needed?

You can make deep neural network by just stacking the hidden layers. Is that true?

Think about two questions below.



1.  Why activation function is needed?
2.  why activation function is `non-linear`?



The output of deep neural network Stacked hidden layers without activation function is too much bigger through the each layer. Using activation function, 





This kind of activation functions are called "`Non-linear function`". Because it can be expressed by using one line. Then, Why do we need to use non-linear function instead of linear function like $f(x)=ax+b$? 

Because of the depth of networks. If we use the linear function as an activation function in the networks, The networks can't be learned enough.

Assuming that we use linear function $h(x) = cx $ as an activation function. 



$$
h(x) = cx\\
y(x) = h(h(h(x)))\\
y(x) = c*c*c*x = c^3x
$$


In this case, We will loss the benefit to use multi-layer networks.

There are other activation functions we can use as an option.





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





## Output layer

In the classification problem, to solve the problem we need to set up proper the number of output neurons. If we want to classify digit number of 0 to 9, we need to set up 10 of neurons as the output layer.  



The solution by machine learning has learning and inference. In learning process, Model learn the data, In inference process, model infer the feasible result based on learning data.  



If you already know the logistic function, it is perfectly same with using `softmax function`.



### Softmax function

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



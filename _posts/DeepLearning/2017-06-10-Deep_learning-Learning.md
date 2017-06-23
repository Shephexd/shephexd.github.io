---
layout: post
title: Deep learning(2) - Learning and Testing in Nueural networks
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



In the previous post, the topic is about the how neural networks work. In this post,  i will introduce the way to learn the neural network.

In learning process, we will compute the `cost function`.

<!--more-->





## Learning Neural Network

There are three ways to develop the algorithms.

1. Algorithm by human
2. Extracting feature by human's algorithm, use Machine Learning alogorithm.
3. Use Deep Learning





### Loss function

`Loss function` is one of index to show the performance of model. The reason why we use loss function instead of using accuracy, `loss function` is used for fitting the parameter in `Back propagation process`.

`Loss function` is useful to get the partial derivative of weight variables.



#### Mean squared error(MSE)


$$
E=\frac{1}{2}\sum_k(y_k - t_k)^2
$$

$$
\begin{align}
&\text{If data size is N,}\\  
&E = -\frac{1}{2N}\sum_n \sum_k(y_k - t_k)^2
\end{align}
$$





#### Cross entropy error(CEE)

$$
E = -\sum_k t_k\log y_k
$$


$$
\begin{align}
&\text{If data size is N,}\\  
&E = -\frac{1}{N}\sum_n\sum_k t_k\log y_k 
\end{align}
$$



$$
\begin{bmatrix}
1 \\
0 \\
\vdots \\
0
\end{bmatrix}^T
\times
\ \begin{bmatrix}
0.6 \\
0.01 \\
\vdots \\
0.4
\end{bmatrix} = -1 \times \log 0.6 = 0.51
$$




### Mini-batch learning

The data size is too big, we can use batch processing. Also, In learning process, we can use batch for learing called `mini-batch`. To reduce the learning time, we can select random sample from traing data set. Then the seleted random data set will be used for training samples.



## Optimization


$$
f(x_0,x_1) = x_0^2+x_1^2
$$
![x^2+y^2](/assets/post_images/DeepLearning/x^2+y^2.png)



The blue point is small point that has smallest value. We called it as `Saddle point`



How can we find this point automatically?



Using derivative, you can find the value to be getting close to `saddle point`



![gradient_sample](/assets/post_images/DeepLearning/gradient_sample.png)



#### Derivative


$$
f'(x)=\lim_{h\rightarrow 0} \frac{f(x+h)-f(x-h)}{2h}
$$


```python
def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) /(2*h)
        x[idx] = tmp_val
        
    return grad
```



#### Gradient descent

$$
x_0 = x_0 - \alpha \frac{\partial f}{\partial x_0}\\
x_1 = x_1 - \alpha \frac{\partial f}{\partial x_1}\\
$$



```python
def gradient_descent(f,init_x, lr=0.1, step_num=100):
    x = init_x
    acc_x = list()
    for i in range(step_num):
        acc_x.append(x.copy())
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x,np.array(acc_x)
```



#### Example

```python
def f(x):
    return x[0]**2 + x[1]**2


init_x = np.array([-3.0, 4.0])
x, history_x = gradient_descent(f,init_x)

plt.plot( [-5, 5], [0,0], '-b')
plt.plot( [0,0], [-5, 5], '-b')
plt.plot(history_x[:,0],history_x[:,1], 'o')
np.arange()
plt.show()
```



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



### Simple example

$$
\begin{align}
& z= x\times y\\
& \frac{\partial z}{\partial x} = y\\
& \frac{\partial z}{\partial y} = x\\
\end{align}
$$

$$
\begin{align}
z= x + y\\
\frac{\partial z}{\partial x} = 1\\
\frac{\partial z}{\partial y} = 1\\
\end{align}
$$



```python
import numpy as np


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
        
    
    def forward(self, x, y):
        self.x = x
        self.y = y
    
        return x*y
    
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        
        return dx, dy
    
class AddLayer:
    def __init__(self):
        pass
    
    def forward(self, x, y):
        return x+y
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
    

if __name__=='__main__':
    apple = 100
    apple_num = 2
    orange = 150
    oragne_num = 3
    tax = 1.1

    #layers
    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer = MulLayer()

    #forwad propagation
    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, oragne_num)
    all_price = add_apple_orange_layer.forward(apple_price,orange_price)
    price = mul_tax_layer.forward(all_price, tax)

    print("Forward| Price:",price)

    #Back propagation
    dprice = 1
    dall_price, dtax = mul_tax_layer.backward(dprice)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price)

    print("Backward| apple price:",dapple," apple num:", dapple_num)
    print("Backward| orange price:",dorange, " orange num:",dorange_num)
    print("Backward| tax:", dtax, " all price:",dall_price)
```



#### output

```
Forward| Price: 715.0000000000001
Backward| apple price: 2.2  apple num: 110.00000000000001
Backward| orange price: 3.3000000000000003  orange num: 165.0
Backward| tax: 650  all price: 1.1
```







### Gradient check

Derivative can be used for checking error of backpropagation. Derivative is easy to make but it is slow comparing to back propagation.

It is obvious that the difference between derivative and backpropagation. If not, it might be error in our codes for back propagation or derviative.

This function is called as `Gradient check`



### The problem in the back propagation

In the case the neural network is deep, the `back propagation` can't affect the first weights.



#### Solution for this problem

> Neural networks with many layers really could be trained well, If the weights are initialized in a clever way.



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

#### Solution for overfitting

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







## Neural networks

### CNN

A researcher research the cat's brain to analysis the  brain's reaction when the cat watch the image.

The result is that the partition of neural is activated not total neural to recognize the image.



### RNN(LSTM)

This network is useful to learn the language model having sequences data. It is widely used in the language, chat bot and machine translation.





### Applications



#### Image recognition

The image net is a competition for the computer vision to recognize image's categories.



#### Image caption

Computer can explain the image with categories like human being.



#### Chat bot

Computer can communicate with human by understanding sentences and 




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



![gradient_sample](/assets/post_images/DeepLearning/gradient_sample.png)


---
layout: post
title: Logistic Regression
published: False
Categories:
- Machine learning
Tags:
- Machine learning
- Regression
- Data mining
- Classification
---

In statistics, logistic regression(logit regression, or logit model) is a regression model where the dependent variable (DV) is categorical. - [Wikipedia][1]

> > Linear regression is good for predicting to other values. But, when we use the linear as a classifier, It will have some problems. Also, the result is not between 0 and 1.

<!--more-->


## What is logistic regression?

The Logistic regression is a regression for **classification**.
It is used to classify categorial variables, *Pass/Fail, Win/Lose and Alive/Dead*.

$$y\in {0,1}
\begin{cases}
0,  & \text{"Negative class"} \\
1, & \text{"Positive class"}
\end{cases}
$$

$$
\text{Want } 0≤h_{\theta}(x) ≤ 1\\
h_\theta(x) = g(\theta^Tx)\\
g(z) = \frac{1}{1+e^{-z}}
$$

$$
h_\theta(x) = \text{estimated probability that y=1 on input x}\\
h_\theta(x) = P(y=1|x;\theta)
$$


## Decision boundary

Suppose predict 
$$if\ h_\theta(x) ≥ 0.5,\ y=1 \\otherwise,\ y=0$$

$$
h_\theta(x) = g(\theta_0 + \theta_1x_1 +\theta_2x_2)
\\g(z) = \frac{1}{1+e^{-z}}
$$

$$
\theta = \begin{bmatrix}
4\\ 1\\ -5 
\end{bmatrix}
$$

$$
\text{Predict y=1 if } 4 + x_1 + -5x_2 > 0\\
Then,x_1 + -5x_2 > -4\ is\ decision\ boundary\ as\ a\ settle\ point $$

### Non-linear decision boundaries

$$h_\theta(x) = g(\theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_1^2 + \theta_1x_2^2)
\\g(z) = \frac{1}{1+e^{-z}}
$$

$$
\theta = \begin{bmatrix}
-1\\0\\0\\1\\1
\end{bmatrix}
$$

$$
Predict\ y=1\ if\ -1 + x_1^2 + x_2^2 ≥ 0
\\x_1^2+x_2^2 = 1
$$

**In much higer polynoimals, The decision boundaries may have strange shape not like a circle and elipse.**

$$h_\theta(x) = g(\theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_1^2 + \theta_1x_2^2
$$

## Cost function

Linear regression

$$
J(\theta) = cost(h_\theta(x^{(i)},y) = \frac{1}{m}\sum_{i=1}^{m}Cost(H_\theta(x^{(i)},y^{(i)})
$$

$$
Cost(h_\theta(x),y) = \begin{cases}
-log(h_\theta (x)) &  if\ y=1\\
-log(1-h_\theta(x)) & if\ y=0\\
\end{cases}
$$

**y = 0 or 1 always**

### Gradient Descent Algorithm for Logistic

$$
J(\theta) = -\frac{1}{m}[\sum_{i=1}^m y^{(i)}logh_\theta(x^{(i)}+(1-y^{(i)})log(1-h_\theta (x^{(i)}))]
$$

To fit parameters $$\theta$$:

$$ min_\theta J(\theta) $$

$$
h_\theta(x) = \frac{1}{1+exp^{-\theta^TX}}
$$

To make a prediction given new x:
output $$h_\theta(x)$$
while {  
$$
\theta_j := \theta_j - \alpha\sum_{i=1}^m(h_\theta(x^i)-y^i)x_j^i
$$  
}

## Python code for tensorflow
```python
import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack = True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))


h = tf.matmul(W,X)
hypothesis = tf.div(1., 1.+tf.exp(-h))

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

#Minimize
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print (step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))

print ('---------------------------------')
print (sess.run(hypothesis, feed_dict={X:[[1], [2], [2]]}) > 0.5)
print (sess.run(hypothesis, feed_dict={X:[[1], [5], [5]]}) > 0.5)
print (sess.run(hypothesis, feed_dict={X:[[1, 1], [2 ,4], [3, 2]]}) > 0.5)
```
```
#train.txt
#x0 x1 x2 y
1   2   1   0
1   3   2   0
1   3   4   0
1   5   5   1
1   7   5   1
1   2   5   1
```
[1]:	https://en.wikipedia.org/wiki/Logistic_regression
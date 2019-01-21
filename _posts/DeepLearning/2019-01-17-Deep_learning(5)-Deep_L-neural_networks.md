---
layout: post
title: Deep learning(5) - Deep L-neural network
published: True
categories:
- Deep learning
tags:
- Machine learning
- Neural network
- Deep learning
- Python
- Tensorflow
typora-root-url: /Users/shephexd/Dropbox/github/pages/
---

from the previous post, we can study how to train neural network from loss function with back propagation.



In this post, we will study about the different types of neural network with layer sizes.

The simple meaning of deep neural network is the neural network with many hidden layers.

> Actually, Deep learning doesn't mean just deeper neural networks.



<!--more-->



## L-neural network



Let's see the kinds of neural networks from the simplest one to deeper. 



### Logistic regression



![logistic_regression](/assets/post_images/DeepLearning/logistic_regression.png)




$$
\sigma \left(
\begin{bmatrix}
x_1 &
x_2 &
x_3 
\end{bmatrix}
\cdot
\begin{bmatrix}
w_1 \\
w_2 \\
w_3 
\end{bmatrix}
+ b 
\right)
= a = \hat{y}
$$




### 1-hidden layer network



![neural-network](/assets/post_images/DeepLearning/neural-network.png)


$$
\sigma\left(
\begin{bmatrix}
x_1 &
x_2 &
x_3 &
x_4
\end{bmatrix}
\cdot
\begin{bmatrix}
w_{11}^{[1]} & w_{21}^{[1]} & w_{31}^{[1]} \\
w_{21}^{[1]} & w_{22}^{[1]} & w_{23}^{[1]} \\
w_{31}^{[1]} & w_{32}^{[1]} & w_{33}^{[1]} \\
w_{31}^{[1]} & w_{32}^{[1]} & w_{33}^{[1]}
\end{bmatrix}
+ b^{[1]}
\right)
= 
\begin{bmatrix}
a_1^{[1]} &
a_2^{[1]} &
a_3^{[1]} 
\end{bmatrix}
$$


$$
\sigma\left(
\begin{bmatrix}
a_1 &
a_2 &
a_3 
\end{bmatrix}
\cdot
\begin{bmatrix}
w_{11} & w_{21} & w_{31} \\
w_{21} & w_{22} & w_{23} \\
w_{31} & w_{32} & w_{33}
\end{bmatrix}
+ b
\right)
= 
\begin{bmatrix}
a_1 &
a_2 &
a_3 
\end{bmatrix}
$$




### 2-hidden layer network



![2_neural_network](/assets/post_images/DeepLearning/2_neural_network.PNG)





###  5-hidden layer network



![5_neural_network](/assets/post_images/DeepLearning/5_neural_network.png)






$$
z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]} \\
a^{[l]} = g^{[l]}(z^{[l]})
$$





> (Informally) There are functions you can compute with a **small** L-layer deep neural network that shallower networks require exponentially more hidden units to compute.





## Getting matrix dimensions right




$$
\begin{align}
& z^{[l]}= W^{[l]} \cdot X + b^{[l]}\\
& (3, 1) = (3, 2) \times (2, 1) + (3, 1)
\end{align}
$$





### Builing blocks of deep neural network



$$
\begin{align}
& Layer : W^{[l]}, b^{[l]} \\
& Forward : a^{[l-1]} (Input),\ a^{[l]}(Output) \\
& Z^{[l]} = W^{[l]} \cdot a^{[l-1]} + b^{[l]} \\
& cache: Z^{[l]}\\
\\
& Backward : da^{[l]} (Input),\ da^{[l-1]}(Output) \\
& cache: Z^{[l]}, dw^{[l]}, db^{[l]}
\end{align}
$$








## Forward and Backward propagation





### Forward propagation



$$
\begin{align}
& Input: a^{[l-1]} \\
& Ouput: a^{[l]}, cache(z^{[l]}) \\
& Z^{[l]} = W^{[l]} \cdot a^{[l-1]} + b^{[l]} \\
& a^{[l]} = g^{[l]}(z^{[l]})
\end{align}
$$



### Backward propagation



$$
\begin{align}
&Input: da^{[l]} \\
&Ouput: da^{[l-1]}, dW^{[l]}, db^{[l]} \\
& dZ^{[l]} = da^{[l]} * g'^{[l](z^{[l]})}\\
& dW^{[l]} = dZ^{[l]} \cdot a^{[l-1]} \\
& db^{[l]} = dZ^{[l]} \\
& da^{[l-1]} = W^{[l]^{T}} \cdot dZ^{[l]}
\end{align}
$$


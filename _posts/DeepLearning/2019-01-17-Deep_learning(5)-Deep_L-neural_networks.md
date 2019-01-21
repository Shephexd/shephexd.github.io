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



<!--more-->



> Actually, Deep learning doesn't mean just deeper neural networks.





> This post is based on the video lecture[^1]



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
z^{[\ell]} = W^{[\ell]} a^{[\ell-1]} + b^{[\ell]} \\
a^{[\ell]} = g^{[\ell]}(z^{[\ell]})
$$





> (Informally) There are functions you can compute with a **small** L-layer deep neural network that shallower networks require exponentially more hidden units to compute.





## Getting matrix dimensions right




$$
\begin{align}
& z^{[\ell]}= W^{[\ell]} \cdot X + b^{[\ell]}\\
& (3, 1) = (3, 2) \times (2, 1) + (3, 1)
\end{align}
$$


$$
\begin{align}
& W^{[1]}:& (n^{[1]}, n^{[2]}) \\
& W^{[2]}:& (n^{[2]}, n^{[1]}) \\
& \vdots \\
& W^{[\ell]}:& (n^{[\ell]}, n^{[\ell-1]}) \\
\\
& X:& (n^{[\ell - 1]}, 1) \\
& dW^{[\ell]}: & (n^{[\ell]}, n^{[\ell - 1]}) \\
& db: & (n^{[\ell]}, 1) \\
& \hat{y}:& (n^{[l]}, 1)
\end{align}
$$




### Building blocks of deep neural network



$$
\begin{align}
& Layer\ \ell : W^{[\ell]}, b^{[\ell]} \\
& Forward : a^{[\ell-1]} (Input),\ a^{[\ell]}(Output) \\
& Z^{[\ell]} = W^{[\ell]} \cdot a^{[\ell-1]} + b^{[\ell]} \\
& cache: Z^{[\ell]}\\
\\
& Backward : da^{[\ell]} (Input),\ da^{[\ell-1]}(Output) \\
& cache: Z^{[\ell]}, dw^{[\ell]}, db^{[\ell]}
\end{align}
$$



$$
\begin{align}
&X & W1 && W2& &W3 & & \rightarrow &&\hat{y}\\
&(N, M) & (M, 50) && (50, 100) & & (100, 1) & & && (N, 1)
\end{align}
$$







## Forward and Backward propagation





### Forward propagation



$$
\begin{align}
& Input: a^{[\ell-1]} \\
& Ouput: a^{[\ell]}, cache(z^{[\ell]}) \\
& Z^{[\ell]} = W^{[\ell]} \cdot a^{[\ell-1]} + b^{[\ell]} \\
& a^{[\ell]} = g^{[\ell]}(z^{[\ell]})
\end{align}
$$



### Backward propagation



$$
\begin{align}
&Input: da^{[\ell]} \\
&Ouput: da^{[\ell-1]}, dW^{[\ell]}, db^{[\ell]} \\
& dZ^{[\ell]} = da^{[\ell]} * g'^{[\ell](z^{[\ell]})}\\
& dW^{[\ell]} = dZ^{[\ell]} \cdot a^{[\ell-1]} \\
& db^{[\ell]} = dZ^{[\ell]} \\
& da^{[\ell-1]} = W^{[\ell]^{T}} \cdot dZ^{[\ell]}
\end{align}
$$





[^1]: https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning	"Deep learning[Coursera]"****
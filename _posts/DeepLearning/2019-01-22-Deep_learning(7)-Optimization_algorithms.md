---
layout: post
title: Deep learning(7) - Optimization Algorithms
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



One of the most important part of deep learning is to find the proper weights of model.

In this post we will discuss the ways of optimization for training process.

There are many ways to train with gradient descent.

<!--more-->

> This post is based on the video lecture[^1] and the lecture note[^2]





## Optimization Algorithms



When we don’t know the way to go descent, how can we find the path?

To find the parameters that minimize the cost of model is similar problem. We will follow the way that the cost is minimized.





### Mini-batch algorithm



Vectorization allows you to efficiently compute on $m$ examples.



**But**, with big data, computing huge matrix is hard to be dealt.


$$
n_x: n_0 \text{ features} \\
m: n_0 \text{ samples} \\
X = [x^{(1)}, x^{(2)}, \dots, x^{(m)}] \\
Y = [y^{(1)}, y^{(2)}, \dots, y^{(m)}] \\
$$


What if $m=5,000,000​$,

$5,000​$ Mini-batches of $1,000​$ epoch

Mini_batch $t$: $x^{\{t\}}$, $y^{\{t\}}$
$$
with\ 5,000\ batch\ size\\
X = [x^{(1)}, x^{(2)}, \dots, x^{(m)}] \rightarrow [x^{\{1\}}, x^{\{2\}}, \dots, x^{\{5000\}}] \times m/5000 \\
Y = [y^{(1)}, y^{(2)}, \dots, y^{(m)}] \rightarrow [y^{\{1\}}, y^{\{2\}}, \dots, y^{\{5000\}}] \times m/5000 \\
$$



### Mini-batch gradient descent



*The idea is dividing data into mini sample data sets and iterating like batch gradient algorithms*





![batch_vs_mini_batch_gradient](/assets/post_images/DeepLearning/batch_vs_mini_batch_gradient.png)



#### sudo code


$$
\begin{align}
repeat\{\\
&\text{for }t=1, ..., 5000\{ \\
&    \text{Forwoard Propagation on } x^{\{t\}} \\
& Z^{[1]} = w^{[1]} \times x^{\{t\}} + b^{[1]} \\
& A^{[1]} = g^{[1]}(z^{[1]}) \\
& \vdots \\
& A^{[L]} = g^{[L]}(z^{[L]}) \\
& \text{compute cost }J=\frac{1}{1000}\sum_{i=1}^\ell L(\hat y^{(i)}, \hat y^{(i)})
+ \frac{\lambda}{2 \times 1000} \sum_{\ell} \Vert w^{[\ell]} \Vert ^2_F
\\
& \text{Back Propagation to compute gradient }J^{\{t\}}\text{ by }(x^{\{t\}}, y^{\{t\}})
\\
& W^{[\ell]} = W^{[\ell]} - \alpha dw^{[\ell]} \\
& b^{[\ell]} = b^{[\ell]} - \alpha db^{[\ell]}
\\
&\}
\\\}
\end{align}
$$




### Stochastic gradient descent

`Stochastic gradient descent` (often shortened to *SGD*), also known as **incremental** gradient descent, is an iterative method for optimizing a differentiable objective function, a stochastic approximation of gradient descent optimization. [^3]



Intuition: *Sampling randomly to compute gradient descent instead of using all samples*.



Normally, SGD will be diverged but need to check whether the diverged point may be not proper because of the biased samples.



If minibatch = 1: Stochastic gradient descent

- Fast but too noisy. 
- Loose speed up from vectorization.



minibatch = m: Batch gradient Descent

- Fast but too noisy. 
- Loose speed up from vectorization.



![sgd_vs_mini_batch](/assets/post_images/DeepLearning/sgd_vs_mini_batch.png)





## Optimization methods



### Exponentially weighted average



Let’s see the examples of the temperature of London.



![temperature_in_london](/assets/post_images/DeepLearning/temperature_in_london.PNG)



To find the trend of temperatures from the noise data, we will use `exponentially weighted average`.




$$
\begin{align}
&v_0 = 0 \\
&v_1 = 0.9v_0 + 0.1\theta_1 \\
&v_2 = 0.9v_1 + 0.1\theta_2 \\
&v_3 = 0.9v_2 + 0.1\theta_3 \\
&\ \ \ \ \ \ \ \ \ \ \vdots \\
&v_t = 0.9 v_{t-1} + 0.1\theta_t
\\
\\
& v_t = \beta v_{t-1} + (1 - \beta) \cdot \theta_t \\
& \beta = 0.9 \approx \text{10 days temperature} \\
& \beta = 0.98 \approx \text{50 days temperature} \\
& v_t \text{ approximately averaging over days } \approx \frac{1}{1-\beta} \text{ days temperature}
\end{align}
$$


#### Understanding exponentially weighted averages

The meaning of below equation is value is getting smaller through the time.


$$
\begin{align}
& v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot \theta_t \\
& v_{100} = 0.1 \cdot \theta_{100} + 0.9 \cdot (0.1\cdot \theta_{99} + 0.9\cdot v_{98}) \\
& = 0.1 \cdot \theta_{100} + 0.1 \cdot 0.99 \cdot \theta_{99} + 0.1 \cdot (0.9)^2 \cdot \theta_{98} + 0.1 \cdot (0.9)^3 \cdot \theta_{97} + \dots

\end{align}
$$



$$
(1- \epsilon)^{\frac{1}{\epsilon}} = \frac{1}{e} \approx 0.35 \\
\epsilon = 1 - \beta \\
$$


With `Exponentially weighted averages`, we can get the denoised sample from the data. 

With proper $\beta$, we can get proper trend of our samples.



![exponentially_weighted_average](/assets/post_images/DeepLearning/exponentially_weighted_average.PNG)





### Bias correction

When adapting `Exponentially weigthed average` with $v_0 = 0$, the bias for initial phase may not be correct. Because of the initial phase value can start with small value.

To make the bias correct for initial phase estimation.


$$
v_t = \beta \cdot v_{t-1} + (1 - \beta )\theta_t \\
v_0 = 0 \\
v_1 = 0.9 \cdot v_0 + (1 - \beta) \cdot v_1 = (1 - \beta)
$$


To prevent these bias for initial estimation, we will use the `Bias correction`.


$$
v_t := \frac{v_t}{1 - \beta^t}
$$


When $t$ is larger value, the $\beta^t$ is closer to 0. It means the effect of `bias correction`  will disappear. But with small $t$, the `bias correction` will make the value get better estimation for initial learning.



> Normally, people don’t care about the `bias correction`. Because it is only for the initial estimation. When you need to get better initial estimation, you should use `bias correction`.





### Gradient Descent with momentum

*The basic idea is to compute the exponentially weighted averages of gradient to  use it as a gradient.*

Gradient descent with momentum will work faster than normal gradient descent.



![gradient_example](/assets/post_images/DeepLearning/gradient_example.PNG)



Like the above image, The gradient descent will be diverged into red dot with vertical oscillation. If our learning rate $\alpha$ is large, the gradient descent will do overshooting.

Then different viewpoint is vertical oscillation cause the problems, overshooting, slow learning.



With exponentially weighted averages, we can make it straight forward by decreasing the vertical movement.



The idea is to get the trend of gradient descent and make it path forward.



on iteration $t$, compute $dw, db$ on the current mini-batch
$$
v_{dw} = \beta \cdot v_{dw} + (1- \beta) \cdot dw \\
v_{db} = \beta \cdot v_{db} + (1- \beta)\cdot db\\
W := W - \alpha \cdot v_{dw}\\ 
b := b - \alpha \cdot v_{db}
$$



> Normally, you don’t need to use bias correction for gradient with momentum because it will be warmed up after some iterations.





In the above equation, hyperparameters are $\alpha, \beta$.



> The most common value for $\beta$ is 0.9 averaging last 10 samples.





### RMS(root mean square) prop

Assuming the $b$ is vertical axis, and $w​$ is horizontal axis.

As we do gradient descent with momentum, decreasing oscillation is helpful to speed up.




$$
S_{dw} = \beta\cdot S_{dw} + (1-\beta) \cdot dW^2 \\
S_{db} = \beta \cdot S_{db} + (1 -  \beta)\cdot db^2 \\
w := w - \alpha \frac{dw}{\sqrt{S_{dw}}} \\
b := b - \alpha \frac{db}{\sqrt{S_{db}}}
$$


Also, it works well in the higher dimension not only $2D​$.



Make it sure the values $S_{dw}, S_{db}$ is not closer to zero. If not, the values $w, $b will be exploded.

We can edit the equation For numerical stability.


$$
S_{dw} = \beta\cdot S_{dw} + (1-\beta) \cdot dW^2 \\
S_{db} = \beta \cdot S_{db} + (1 -  \beta)\cdot db^2 \\
w := w - \alpha \frac{dw}{\sqrt{S_{dw} + \epsilon}} \\
b := b - \alpha \frac{db}{\sqrt{S_{db} + \epsilon}} \\
ex) \epsilon= 10^{-8}
$$


With ADAM, you can use large learning rate $\alpha$ than before. It make your algorithm path forward.



### ADAM(adaptive moment estimation)

  
$$
\begin{align}
&V_{dw} = 0, S_{dw}=0, V_{db} = 0, S_{db} = 0 \\
\\
&\text{on iteration: } t\\
& V_{dw} = \beta_1 \cdot V_{dw} + (1- \beta) dw , \ V_{db}= \beta_1 \cdot V_{db} + (1-\beta^t) \\
& S_{dw} = \beta_2 \cdot S_{dw} + (1- \beta_2) dw^2 , \ S_{db} = \beta_2 \cdot S_{db} + (1- \beta_2) 
\\
\\
& V^{correct}_{dw} = V_{dw}/(1-\beta^t_1), \ 
V_{db}^{correct} = V_{db} / (1-\beta^t_1) \\
& S^{correct}_{dw} = S_{dw}/(1 - \beta_2^2), \
 S_{db}^{correct} = S_{db} / (1 - \beta_2^T) \\

& W := W - \alpha \cdot \frac{V_{dw}^{correct}}{\sqrt{S_{dw}^{correct} + \epsilon}} \\
& b := b - \alpha \cdot \frac{V_{db}^{correct}}{\sqrt{S_{db}^{correct} +\epsilon}}
\\
\\
& \alpha: 0.1 \text{learning rate(need to be tune)}\\
& \beta_1: 0.9 \text{(weighted average for $dw$)}\\
& \beta_2: 0.999 \text{(weighted average for $dw^2$)}\\
& \epsilon: 10^{-8}
\end{align}
$$


## Learning rate decay

For `gradient descent`, we use the $\alpha$ as learning rate. After many iterations, we don’t want to learn too fast.



1 epoch: 1 pass through data



### Decreasing learning rate



#### Ex1

$$
\alpha := \frac{1}{1 + \text{decay rate} \times \text{epoch num}} \cdot \alpha
$$





#### Ex2

$$
\alpha = 0.95 \\
\alpha := 0.95^{\text{t}} \cdot \alpha_0 \\
t: \text{epoch number}
$$



#### Ex3


$$
\alpha = 0.95 \\
\alpha : = \frac{k}{\sqrt{t}} \cdot \alpha \\
t: \text{epoch number}
$$


#### Ex4

$$
\alpha =0.95 \\
\frac{k}{\sqrt{t}} \cdot \alpha \\
t: \text{epoch number}
$$





## The problem of local optima



Normally, It will be converged at local optima. Because deep neural network model is not convex function.



![local_optima](/assets/post_images/DeepLearning/local_optima.PNG)



When we think about the local optima, it looks like the left figure that looks like there are many local optima.

But, the most point of 0 gradient is saddle point like the right figure.



In the higher dimensions like $20,000D​$ , there are many concave points and convex points. It will be crossed at the saddle point If there are not concave points or convex points. 



So, what I want to say is the learning algorithm in the lower dimension may not work in the higher dimension because of the saddle point.



- Unlikely to get stuck in a bad local optima
- Plateaus can make learning slow



## Training model process

You will wonder how You can train your models and build up.

Depending on your computing resource, You can select one.



1. Baby sitting one
2. Training many models in parallel





[^1]: https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning	"Deep learning specialization"
[^2]: http://cs231n.stanford.edu/	"Stanford CS231n"

[^3]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent
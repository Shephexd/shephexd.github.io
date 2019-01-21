---
layout: post
title: Deep learning(6) - Practical aspect of Deep learning
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



We've studied about the process of traning models. How to train and test our model with data?

How do we evaluate our model after training?

In this post, let's see the way of training and testing to improve your model performance as you expect.





<!--more-->

> This post is based on the video lecture[^1] and the lecture note[^2]



## Practical aspect of Deep learning



The learning process of deep learning is iterating below process.


$$
learn \rightarrow code \rightarrow experiment \rightarrow learn \rightarrow \dots \rightarrow experiment
$$




### parameters

In each cycle, you will update the parameters to make your model better.



- Layers
- Hidden units
- Learning rate
- Activation function





### Train /Dev/ Test sets



In normal machine learning system, Divde data into train, dev and test sets.

`Training set` is for fitting model. 

`dev set` is for fitting hyper parameters. 

`Test set` is for validation trained model.



- Training Set: 70%
- Dev Set: 30%

or 

- Training Set: 60%
- Dev Set: 20%
- Test Set: 20%



**But**, big data for Deep learning

- Training Set: 98%
- Dev Set: 1%
- Test Set: 1%



#### Dealing with mismatched train / test distirbution

Assuming that you want to train the cat pictures from web for your app users.

If your dev and test set is from users using your app, the distribution is mismathced.



*"Make sure dev and test set com from same distribution"*



## Bias and Variance



**High bias(Under fitting)** means the prediction power is lower.

**High variance(Overfitting)** means the prediction power is high only for train set.



![bias_variance](/assets/post_images/DeepLearning/bias_variance.png)



We want to train the model to use generally.



*"Bias variance trade off"*



High bias $\rightarrow​$ Bigger network, Train Longer, more parameters

High variance $\rightarrow$ More Data, Regularization





## Regulariazation

The goal of regularization is to prevent overfitting by penalizing the biased weights.

If one of the weights is too big, it will affect the result of model too much.

Let's see the regularization techinque called weight decay or $L_n $ Regularization to smooth the weights.



### Logistic regerssion


$$
\begin{align}
&\min J(w, b) = \frac{1}{m} \sum_{i=1}^m L(\hat y^{(i)}, y^{(i)}) + \frac{\lambda}{2m} \Vert w \Vert_2^2 \\
& L_2 \text{ Regularization}: \Vert w \Vert_2^2 = w^Tw \\
& L_1 \text{ Regularization}: \frac{\lambda}{2m} \sum_{2m}^m \vert w_j \vert = \frac{\lambda}{2m} \Vert w \Vert_1 \\
& (w \text{ will be sparse})
\end{align}
$$




### Neural networks



#### Forbenius norm


$$
\Vert X \Vert _F^2 = \sqrt{
    \sum_{i=1}^m\sum_{j=1}^n \vert a_{ij} \vert ^2 
}
$$


#### Weight decay


$$
\begin{align}
W^{[l]} = & W^{[l]} - \alpha[{(dw)}^{[l+1]} + \frac{\lambda}{m}w^{[l]}] \\
=& W^{[l]} - \frac{\alpha \lambda}{m}w^{[l]} - {(dw)}^{[l+1]}
\end{align}
$$




#### Dropout

Intution: *Can't rely on any one feature, so have to spread out weights*



![dropout](/assets/post_images/DeepLearning/dropout.png)





**In the training process**, random neurons on each layer are dropped out.

But, you should not use dropout **on test and validation process**.



> Cost function is noet well defined when you use Dropout.



## Tips for training



### Early stopping

In normal train process, we define the $N$ iterations for epochs.

Too many iterations may cause `overfitting`. To avoid it, Early stopping is useful.

The meaning is stopping training when our conditions are fullfilled.



- When your cost function is not updated after epoch comparing to previous epoch.
- When the cost function is enough low.



### Normalization

Normalization is one of preprocessing technique to make the data in similar range.



![prepro1](/assets/post_images/DeepLearning/prepro1.jpeg)





1. Subtract mean, $x := x - \mu​$
2. normalize variance, $\sigma^2 = var(x) = E((X - \mu)^2), x \leftarrow (x - \mu) / \sigma$





## Debugging

When you want to verify the calculation is right, debugging is necessary.



### Numerical Approximation of gradients



After you implement back propagation, sometimes you cannot sure the calculated gradient is correct.

If the graident is not correct, the model is getting worse after training.




$$
\frac{
    f(\theta + \varepsilon) - f(\theta - \varepsilon)
}{
    2 \varepsilon
}
\approx g(\theta) \\

f'(\theta) = 
\lim_{\varepsilon}
\frac{
    f(\theta + \varepsilon) - f(\theta - \varepsilon)
}{
    2 \varepsilon
}
$$


### Gradient check


$$
d \theta_{\varepsilon}[i] = \frac{
    J(\theta_1, \theta_2, \dots, \theta_i+ \epsilon, \dots) -
    J(\theta_1, \theta_2, \dots, \theta_i- \epsilon, \dots)
}
{
    2\epsilon
} \\
\approx d\theta[i] = \frac{dJ}{d\theta_i}
$$




Check the below equation with proper $N$
$$
\frac{
    \Vert d\theta_{epsilon} - d\theta \Vert_2
}
{
    \Vert d\theta_{epsilon} \Vert + \Vert d\theta \Vert_2
}
\approx 10^{-N}
$$


#### Need to know

- Don't use on training process (only for debugging)
- If algorithm fails gradient check, loock at components to try to identify.

- Remeber regulariation
- Doesn't work with dropout





[^1]:	https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning	"Deep learning specialization"

[^2]: http://cs231n.stanford.edu/	"Stanford CS231n"


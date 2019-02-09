---
layout: post
title: Deep learning(9) - Introduction to ML strategy
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

Before more advanced deep learning algorithm, Let’s look at the way to debugging your NN model.

There are something keeping in your mind to train the model.

Obviously, It is depending on your goal you want to achieve. 

Let’s see how you can diagnose your model with some debugging ideas.



<!--more-->

>This post is based on the video lecture[^1] 



## Introduction to ML strategy

Analyzing your ML problem to save your time.

There are some options you can select to improve your model.



- Collect more data
- Collect more diverse training set
- Train algorithm longer with `gradient descent`
- Try `adam` instead of `gradient descent`
- Try bigger or smaller network
- Try `dropout`
- Add $L_2$ regularization
- Change Network architecture



If you have much time and computing power, you can do it all.

But, knowing what idea should be useful for your problem saves your time.



### Orthogonalization

Make the tunings do only on thing it make easy to tune parameters.



Assuming that there are three controllable parameters,$p_1, p_2, p_3$, for tuning model.

*Independent parameters can be tuned easily.*





### Chain of Assumption in Machine Learning

Before adapting debugging ideas, You should think about the latent problems your model might have.

Step by step is helpful to solve one problem by tuning one parameter.



$$
\begin{CD}
        \text{Fit training set well on cost function} @>{\text{Underfitted?}}>> 
        \text{Higher network, Use Adam }\\
        @VVV @. \\
        \text{Fit dev set well on cost function} @>{\text{Overfitted?}}>> 
        \text{Regularization}\\
        @VVV @. \\
        \text{Fit test set well on cost function} @>{\text{generalized?}}>> 
        \text{Bigger dev set}\\
        @VVV @. \\
        \text{Perform well in real world} @>{\text{Is mis-matched data set?}}>> 
        \text{Change dev set or cost function}\\
    \end{CD}
$$






## Evaluation

How can we evaluate our model performance?



### Single number evaluation metric



![evaluation_metric](/assets/post_images/DeepLearning/evaluation_metric.png)



#### Precision



examples recognized what $\%$ actually are right.


$$
Precision = \frac{tp}{tp + fp}\\
$$




#### Recall

What % of actual cats are correctly recongize.


$$
Precision = \frac{tp}{tp + fn}\\
$$



#### F1-Score

Average of $Prcision$ and $Recall$.


$$
F_1 = 2 \cdot \frac{precision \times recall}{precision + recall}
$$



#### True negative rate


$$
True\ negative\ rate(specificity) = \frac{tn}{tp + fp}
$$



#### Accuracy


$$
Accuracy = \frac{tp + tn}{tp + tn + fp + fn}
$$





## Train/Dev/Test distribution

Recommend dev and test set come from **same distribution**.  
$\rightarrow$ Randomly Shuffle into dev/test



*If you use the dev set from web and use the test set from books, It will be hard to improve and generalize the model.*



> Choose a dev set and test set to reflect data you expect to get in the future and consider important to do well on.



[^1]: https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning	"Deep learning specialization"
---
title: Machine learning(7) - Debugging algorithm
layout: post
categories:
- Machine learning
tags:
- Machine learning
- Tensorflow
typora-root-url: /Users/shephexd/Documents/github/pages/
---

If your learning algorithm have some problems, How can we detect and solve it?



Here is the solution. The one of the most important parts to train and build machine learning is debugging your algorithm. Debugging can give you the insight for your next step. Without debugging, you don't sure your algorithm is trained well without `overfitting` or `underfitting`. 



<!--more-->



## Debugging algorithm



Do you think it is enough if you have low cost for train ?


$$
J(\theta) = \frac{1}{2m} \left[ \sum_{i=1}^m (h_\theta(x^{(i)}) - y ^{(i)})^2 - \lambda \sum_{j=1}^m \theta_j^2  \right]
$$




Suppose you have a trained classifier to predict house price.

How to decrease your error in model?



### Some examples

- Get more training examples
- Try getting additional features
- Try adding polynomial features
- Try decreasing $\lambda$
- Try increasing $\lambda$



## Evaluate hypothesis

To make check whether your model is `generalized`, try evaluate!



### Training and Test set

Divide your dataset into `training set`(70%) and `test set`(30%).

1.  Learn parameters($\theta$) from `training data`

2.  Compute test set error

    
    $$
    J_{Test} = \frac{1}{m_{test}} \sum_{i=1}^{m_{test}} y_{test}^{(i)} \log h_\theta (x^{(i)}_{test}) + (1 - y^{(i)}_{test}) \log h_\theta(x_{test}^{(i)})
    $$

3.  


$$
\begin{array}{r|rr}
&\text{Actual class}\\
               & 1 & 0 &    \\ \hline
\text{predicted class }           1 &   \text{True positive } & \text{False negative} \\
           0 &   \text{False negative} &   \text{True negative} \\
\end{array}
$$

$$
Precision = \frac{\text{True positive}}{\text{True positive + False positive}} \\
Recall = \frac{\text{True positive}}{\text{True positive + False negative}} \\
F_1\ Score = 2 \frac{Precision \times Recall}{Precision + Recall}
$$




### Training, validation and test set

Once parameters are fail to fit generalized values for your model.

-   `Training set` (60%)

    -   To fit trained parameters by `y values`(labels)
    -   $J_{train}(\theta) = \frac{1}{2 m_{train}} \sum_{i=1}^{m_{train}} (h_\theta(x^{(i)}-y^{(i)}))^2$

-   `Cross validation set`(20%)

    -   To fit parameters and select best model
    -   $J_{cv}(\theta) = \frac{1}{2 m_{cv}} \sum_{i=1}^{m_{cv}} (h_\theta(x^{(i)}-y^{(i)}))^2$

-   `Test set` (20%)

    -   To verify your selected model is generalized or not

    -   $J_{test}(\theta) = \frac{1}{2 m_{test}} \sum_{i=1}^{m_{test}} (h_\theta(x^{(i)}-y^{(i)}))^2$

        



### What happens if your model is not generalized 



Assuming your model is trained, the accuracy is not good even for `train data`. Then your model is `underfitted`(`High bias`)

Or the accuracy is good only for `train data`. Then your model is `overfitted`(`High variance`).

Or your model.



-   High bias
    -   $J_{train}(\theta)$ will be high
    -   $J_{cv}(\theta) \approx J_{train}(\theta)$
-   High variance
    -   $J_{train}(\theta)$ will be low
    -   $J_{cv} \gt \gt J_{train}$



The way to diagnosis your model is drawing `learning curve`.



`learning curve` make you know whether increasing number of training examples are helpful for your model.



### What you should check

For `overfitted` model, you should make your model much simpler.

For `underfitted` model, you should make your model well trained.



-   Get more training examples $\Rightarrow$ fixes `high bias`
-   Try smaller sets of features  $\Rightarrow$ fixes `high variance`
-   Try getting additional features  $\Rightarrow$ fixes `high bias`
-   Try adding polynomial features $\Rightarrow$ fixes `high bias`
-   Try decreasing $\lambda$ $\Rightarrow$ fixes `high bias`
-   Try increasing $\lambda$ $\Rightarrow$ fixes `high variance`






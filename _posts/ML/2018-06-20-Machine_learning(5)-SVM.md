---
title: Machine learning(5) - SVM
layout: post
categories:
- Machine learning
tags:
- Machine learning
- Tensorflow
typora-root-url: /Users/shephexd/Documents/github/pages/

---



In this post, I will introduce about the SVM(Support vector machine)

SVM is one of the powerful algorithms for classification.



I will compare with the logistic regression the most common algorithm for classification to make you know what the differences are.



<!--more-->

SVM also called as `Large margin classification`.  

It will be helpful to know the why the name is `Support vector`. The meaning of support vector makes different from other linear classification algorithms.





## What is different from Logistic regression?

First I need to show the differences between Logistic regression and SVM.



### logistic regression

The goal of Logistic regression is to classify $x$.


$$
h_\theta(x) = \frac{1}{1+ e^{(-\theta^T x)}}
$$

$$
\begin{align}
& \text{if }y=1, \text{we want }h_\theta(x) \approx 1 & \text{it means } \theta^Tx  \gt \gt 0 \\
& \text{if }y=0, \text{we want }h_\theta(x) \approx 0 & \text{it means } \theta^Tx  \lt \lt 0 \\
\end{align}
$$



#### Cost function

Cost is defined differently by $y$ label.


$$
\begin{align}
&Cost_0 = 1-\log (h_\theta(x)) &  \text{ if } y=0\\
&Cost_1 = \log (h_\theta(x))  &\text{ if } y=1 \\
\end{align}
$$

It can be written as an equation like below.


$$
\begin{align}
Cost(\hat y, y) &=  -(y Cost_1 +(1-y)Cost_0 ) & \\
& =   -(y  \log h_\theta(x) + (1-y) \log {(1- h_\theta(x))}) &
\end{align}
$$


### SVM



There are two `cost` for SVM.


$$
\begin{align}
&Cost_0 = -\theta^T x + b&  \text{ if } y=0\\
&Cost_1 = \theta^T x - b  &\text{ if } y=1 \\
\end{align}
$$

We want to make the margin larger as much as we can.


$$
\begin{align}
& \text{if }y=1, \text{we want }h_\theta(x) \approx 1 & \theta^Tx  & \ge 1 \\
& \text{if }y=0, \text{we want }h_\theta(x) \approx 0 & \theta^Tx  & \le -1 \\
\end{align}
$$



### Cost function



The below equation is the cost function of `SVM`.


$$
C \sum_{i=1}^m \left[ y^{(i)} cost_1(\theta^Tx^{(i)}) + (1-y^{(i)}) cost_0(\theta^T x^{(i)}) \right] +\frac{1}{2}\sum_{j=1}^n \theta_j^2
$$


-   $C$ is a `hyperparameter` for support vectors  
    It works like a regularization parameter. 

    

    >   If $C$ is too large, the margin error will decrease but it might be `overfitted`.
    >
    >   If $C$ is too small, the margin error will increase but it can be `generalized`.

    



### Cost (logistic vs SVM)

![logit_vs_cost0](/assets/post_images/ML/logit_vs_svm_0.png)

![logit_vs_cost1](/assets/post_images/ML/logit_vs_svm_1.png)



>   The blue one is form `SVM` . the other one is `logistic`.

Two above plot shows the different definition of the cost function for logistic and `SVM`. There is no cost function when $ \lvert x \rvert \le 1$. It means that `SVM` doesn't care for the samples between the two margins.





## Large margin

The meaning of support vector on the vector space can be expressed as the mathematical behind large margin classification.

 

### Inner product

To remind yourself, there is an introduction for inner product of vectors.

Assuming that there are two vectors $u, v$.


$$
u = \begin{bmatrix} u_1 \\ u_2\end{bmatrix}, v = \begin{bmatrix} v_1 \\ v_2\end{bmatrix}
$$

$$
\begin{align}
\left\lVert u \right\rVert &= \text{length of vector } u, \sqrt{u_1^2 + u_2^2} \\
P &= \text{length of project of $v$ onto $u$}\\
u^Tv &= P \cdot \left\lVert u \right\rVert = u_1v_1 + u_2v_2 \\
& = \cos\theta \cdot \left\lVert u \right\rVert \cdot \left\lVert v \right\rVert
\end{align}
$$



### Decision boundary



The cost function of `SVM` is the below equation. We want to minimize the cost function as much as we can.



$$
\min C \sum_{i=1}^m \left[ y^{(i)} cost_1(\theta^Tx^{(i)}) + (1-y^{(i)}) cost_0(\theta^T x^{(i)}) \right] +\frac{1}{2}\sum_{j=1}^n \theta_j^2
$$



There are two parts of cost function.


$$
\begin{align}
&\text{we want to miminize } \sum_{i=1}^m \left[ y^{(i)} cost_1(\theta^Tx^{(i)}) + (1-y^{(i)}) cost_0(\theta^T x^{(i)}) \right] \\
&\text{we want to miminize } \frac{1}{2}\sum_{j=1}^n \theta_j^2
\end{align}
$$



The first equation is about the classifier work well to classify the samples.

Then how about the second equation? Let's talk about the detail of the equation that makes two large margins.




$$
\begin{align} 
\frac{1}{2}\sum_{j=1}^n \theta_j^2 &= \frac{1}{2}(\theta_1^2 + \theta_2^2)\\
& =\frac{1}{2} \left(\sqrt{\theta_1^2 + \theta_2^2}\right)^2 \\
& = \frac{1}{2}  \left\lVert u \right\rVert ^2 \\
\\

s.t.\  & \theta^Tx^{(i)} \ge 1 & \text{ if } y^{(i)}=1 \\
\ &\theta^Tx^{(i)}\le -1 & \text{if }y^{(i)}=0
\end{align}
$$




$$
\begin{align}
& \theta = \begin{bmatrix} \theta_1 \\ \theta_2\end{bmatrix}
 x = \begin{bmatrix} x_1 \\ x_2\end{bmatrix} \\

\theta^Tx &= \theta_1x_1 + \theta_2 x_2 \\
&= P^{(i)} \cdot  \left\lVert \theta \right\rVert \\
&= \cos{\theta} \cdot  \left\lVert x \right\rVert \cdot \left\lVert \theta \right\rVert \\
&\text{ where $P^{(i)}$ is the projection of $x^{(i)}$ onto the vector $\theta$}
\end{align}
$$



In the cost function of `SVM`, we want to minimize $ \left\lVert \theta \right\rVert$, to satisfy the constraints $\left(\theta^T x^{(i)} \ge 1 ,  \theta^T x^{(i)} \le -1 \right)$ $P^{(i)}$ must be large. $P^{(i)}$ is the projection of $x^{(i)}$ onto the vector $\theta$. 





![SVM_hyperplanes](/assets/post_images/ML/svm_max_sep_hyperplane_with_margin.png)



Two drawn hyperplanes by the vector $w$ with constraints are the decision boundaries of `SVM`. Other linear classification algorithms just care about whether the classifiers work well or not, But `SVM` care about the classifier can classify the samples($x^{(i)}$) with **large margin** by drawing the hyperplane as called `Support Vector`.


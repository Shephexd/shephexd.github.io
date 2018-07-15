---
title: Machine learning(9) - PCA(Principal Component Analysis)
layout: post
categories:
- Machine learning
tags:
- Machine learning
- Tensorflow
- Dimensional reduction
typora-root-url: /Users/shephexd/Documents/github/pages/
---



When your data samples are in high-dimensional feature space, you should consider the `curse of dimensionality`[^1] decreasing accuracy of your model.



High dimensional is hard to understand and not always helpful for your data.

In this post, I will introduce about the way to reduce to map high-dimensional features into low-dimensional features by `PCA`.



<!--more-->

## Dimensionality reduction



At first, why do we need to reduce dimensionality?

There are two common purposes.

1.  Data compression  
    $$
    \begin{align}
    &\text{Reduce data from } 2D \rightarrow 1D&\\
    &x^{1} \in R^2 \rightarrow z^{1} \in R^1\\
    &x^{2} \in R^2 \rightarrow z^{2} \in R^1\\
    &\vdots&\\
    &x^{m}\in R^2 \rightarrow z^{m} \in R^1
    \end{align}
    $$

2.  Data visualization  
    Visualizing the data set with many feature is hard. So, most of plots only can represent on $1D$ to $3D$ cases.

​    





## PCA(Principal Component Analysis) 

`PCA` is one of the way to make data lower dimensional case.



### Intuition

Reduce dimension ,$2D \rightarrow 1D$: Find a direction onto which to project the data so as to minimize the projection error.

 Then reduce dimension, $n$-dimension to $k$-dimension: Find $k$ vectors $u^{(1)}, u^{(2)}, \dots, u^{(k)}$ onto which to project the data, so as to minimize the projection error.



Doesn't it sounds like the `Linear regression`?

Let's see the difference between  `PCA` and `Linear regression`

In `PCA`, the goal is to minimize the `projection error`.





### PCA process



#### 1. Data processing



Training set: $x^{(i)}, x^{(2)}, \dots, x^{(m)}$  
Preprocessing(`feature scaling` like `mean normalization`)  
$\mu_j = \frac{1}{m}\sum^m_{i=1}x_j^{(i)} \\ \text{Replace $x_j^{(i)}$with $x_j-\mu_j$}$   
If different features on different scales(ex $x_1$ is size of hours, $x_2$ number of bedrooms), feature scaling is to have comparable range of values  
$$
x_j^{(i)} \leftarrow \frac{x_j^{i}-\mu_j}{s_j}
$$



#### 2. Compute covariance matrix


$$
\Sigma = \frac{1}{m}\sum_{n=1}^n(x^{(i)})(x^{(i)})^T
$$


#### 3. Compute eigenvectors of covariance matrix 

using Singular vector decomposition, we can get $k$ eigenvectors.



```matlab
[U, S, V] = svd(sigma);
Ureduce = U(:, 1:k);
z = Ureduce.T * x;
```



The process of `SVD` is like below.



##### SVD decomposition

$A$ is a matrix, and it can be


$$
A = USV^T
$$


where $U, V$ are the rotation(or `unitary`) matrices and $S$ is a diagonal matrix with singular values $\sigma_1,\dots, \sigma_n$ on the diagonal.

if A is a $m \times n$ matrix, then $U$ is a $m \times m$ square matrix, $V$ is a $n \times n$ matrix and $S$ is  diagonal matrix of size $m \times n$, where the number of diagonal elements is the smaller one of $m$ and $n$.



#### 4. Choosing $k$ 

$k$ is the number of principal components. It means we can project $n$ features into $k$ features using `PCA`



##### Projection error


$$
\frac{1}{m}\sum_{i=1}^m \lVert x^{(i)} - x^{(i)}_{aprrox} \rVert^2
$$


##### Total variation in the data


$$
\frac{1}{m}sum_{i=1}^m \lVert x^{(i)} \rVert^2
$$


Typically, choose $k$ to be smallest value so that, 


$$
\begin{align}
&\frac{\text{Projection error}}{\text{Total variation in the data}}=\\
&\frac{\frac{1}{m}\sum_{i=1}^m \lVert x^{(i)} - x^{(i)}_{aprrox} \rVert^2}{\frac{1}{m}sum_{i=1}^m \lVert x^{(i)} \rVert^2} \le P
,ex)\ P=0.01, 0.05, 0.1
\end{align}
$$


$(1-P) \times 100\%$ of variance is retained





## Advice of applying PCA

Using `PCA`, we can project $n$-feature space into smaller feature space. Then, What we can do with `PCA`?



### applying PCA



1.  `Supervised learning` speed up by decreasing feature size.
2.  Compression
    -   Reduce memory/disk needed to store data
    -   Speed up learning algorithm
3.  Data visualization



Bad use of `PCA` is to prevent `overfitting`.

Use $z^{(i)}$ instead of $x^{(i)}$ to reduce the number of features to $k \lt n$.  
Thus, fewer features, less likely to `overfit`.



It might work OK, But It isn't good way to address `overfitting`.  
If you use `PCA` and get better performance, the same result can be possible by using regularization.  
`PCA` is projection, It means your raw features have some meaning. What about projected features?

Use regularization instead.


$$
\min_\theta \frac{1}{2m}\sum^m_{i=1}(h_\theta(x^{(i)}) - y^{(i)}) + \frac{\lambda}{2m} \sum^n_{j=1} \theta_j^2
$$


First RUN without `PCA`, just with your raw data.  
Only if that doesn't do what you want, then implement `PCA` and consider using $z^{(i)}$.

It is the way to keep the meaning of the raw features.



>    Keep your mind the projected features $z_i$ doesn't have same meaning with $x_i$.
>
>   The best way is to use raw features to keep the features meaning if you can get what you want.





[^1]: https://en.wikipedia.org/wiki/Curse_of_dimensionality


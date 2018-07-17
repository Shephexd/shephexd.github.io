---
title: Machine learning(10) - Recommendation system
layout: post
categories:
- Machine learning
tags:
- Machine learning
- Tensorflow
- Anomaly detection
typora-root-url: /Users/shephexd/Documents/github/pages/
---



One of the most interesting part in Machine learning is knowing what you want based on the your logs.

Like `netflix`, User rates movies using zero to five stars, Then the system will recommend movies you may like by analyzing your personal taste.



<!--more-->



## Content-based recommendation system

Let's see the below table for user's rating for the movies they watched. 

What if $User_0$ watch the movie, `notebook`? Can you guess the rating?

Maybe you can guess based on other people's rating and $User_0$'s taste.

 
$$
\begin{array}{r|rr} \text{Movies}

               & User_0 & User_1 & User_2 &  User_3 &  User_4 \\ \hline
\text{the greatest show man}      &  5 & 4 & 4 & 2 &1 \\
\text{once upon a time}		    &  5 & 4 & ? & 2 &4 \\
\text{notebook}			         &  ? & 4 & 2 & 2 &2 \\
\text{taken}				&  4 & 3 & 2 & 3 &5 \\
\end{array}
$$


For each user $j$, learn a parameter $\theta^{(j)} \in R^3$. Predict user $j$ as rating move $i$ with $(\theta^{(j)})^T x^{(i)}$ stars


$$
\begin{align}
& r(i, j) = 1 \ \text{if user $j$ as rated movie $i$($0$ otherwise)}\\
& y^{(i, j)} = \text{rating by user $j$ on movie $i$(if defined)} \\
& \theta^{(j)} = \text{parameter vector for user $j$}\\
& x^{(i)} = \text{feature vector for movie $i$}\\
& (\theta^{(j)})^T(x^{(i)}) =\text{For user $j$, movie $i$, predicted rating}\\
& m^{(j)} = \text{number of movies rated by user $j$}
\end{align}
$$



#### Optimization for Objective function



$$
\begin{align}
& \text{To learn $\theta^{(j)}$ parameters for user $j$}: \\
& \min_{\theta(j)} \frac{1}{2} \sum_{i:r(i,j)=1} \left( (\theta^{(j)})^T x^{(i)} - y^{(i, j)} \right)^2 + \frac{\lambda}{2} \sum_{k=1}^n \left( \theta^{(j)}_k \right)^2 \\
\\
& \text{To learn $\theta^{(1)}, \theta^{(2)} \dots, \theta^{(n_u)}$}: \\
& \min_{\theta^{(1)}, \theta^{(2)} \dots \theta^{(n_u)}} 
\frac{1}{2} \sum_{j=1}^{n_u} \sum_{i:r(i,j)=1} \left( (\theta^{(j)})^T x^{(i)} - y^{(i, j)} \right)^2 
+ \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n \left( \theta^{(j)}_k \right)^2 \\
\end{align}
$$



#### Optimization algorithm



$$
\begin{align}
& \text{Optimization algorithm}: \\
& J({\theta^{(1)}, \theta^{(2)} \dots , \theta^{(n_u)}}) = 
\frac{1}{2} \sum_{j=1}^{n_u} \sum_{i:r(i,j)=1} \left( (\theta^{(j)})^T x^{(i)} - y^{(i, j)} \right)^2 
+ \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n \left( \theta^{(j)}_k \right)^2 \\

& \text{Gradient descent update:}\\
& \theta_k^{(j)} :=
\theta_k^{(j)} 
- \alpha \sum_{i:r(i,j) = 1} \left( (\theta^{(j)})^Tx^{(i)} - y^{(i,j)}  \right)x_k^{(i)} (\text{for } k =  0) \\
& \theta_k^{(j)} :=
\theta_k^{(j)}- \alpha \sum_{i:r(i,j) = 1} \left( ((\theta^{(j)})^Tx^{(i)} - y^{(i,j)})x_k^{(i)} +\lambda x_k^{(j)}  \right) (\text{for } k \ne 0) \\

\end{align}
$$



## Collaborative filtering

The collaborative filtering means the weight can be estimated by each other.

We will build a model to categorize content for each content and the weight of categorized contents for each user.



There are two hypothesis.

1.  Each user like the movies depending on the movie categories.
2.  Each user have different weights for the movie categories.



The main differences from `content based recommendation` are the weight for categories is added and two weights are updated simultaneously.



$x^{(1)}, \dots, x^{(n_m)}$ are movie categories for each movie.

$\theta^{(1)}, \dots, \theta^{(n_u)}$ are the weights of movie categories for each user.


$$
\begin{align}
&\text{Estimate the weights for the taste of users}\\
&\text{Given } x^{(1)}, \dots, x^{(n_m)} \text{ and  movie ratings}\\
&\text{Can estimate } \theta^{(1)}, \dots, \theta^{(n_u)} \\
\\
&\text{Estimate the weights of contents}\\
&\text{Given } \theta^{(1)}, \dots, \theta^{(n_u)}\\
&\text{Can estimate } x^{(1)}, \dots, x^{(n_m)} \\
\end{align}
$$


### Optimization function


$$
\begin{align}
&\text{for a user:}\\
&\text{Given } \theta^{(1)}, \dots, \theta^{(n_u)} \text{ to learn } x^{(i)} \\
& \min \frac{1}{2} \sum_{j:r(i,j)=1} 
\left( ( \theta^{(j)} )^T  x^{(i)} - y^{(i, j)} \right) + \frac{\lambda}{2} \sum_{k=1}^n  \left( x_k^{(i)}\right)^2 \\
\\
&\text{for users:}\\
&\text{Given } \theta^{(1)}, \dots, \theta^{(n_u)} \text{ to learn } x^{(1)}, \dots, x^{(n_m)} \\
& \min_{x^{(i)}, \dots, x^{(n_m)}} \frac{1}{2} \sum_{i=1}^{n_m} \sum_{j:r(i,j)=1} 
\left( ( \theta^{(j)} )^T  x^{(i)} - y^{(i, j)} \right) 
+ \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^n  \left( x_k^{(i)}\right)^2 \\

\end{align}
$$




Minimize $x^{(1)}, \dots, x^{(n_m)}$ and $\theta^{(1)}, \dots, \theta^{(n_m)}$ simultaneously
$$
J(x^{(1)}, \dots, x^{(n_m)}, \theta^{(1)}, \dots, \theta^{(n_u)}) = \\
\frac{1}{2} \sum_{(i,j):r(i,j)=1} \left( (\theta^{(j)})^Tx^{(i)} - y^{(i, j)}\right)^2 
+ \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^n  \left( x_k^{(i)}\right)^2
+ \frac{\lambda}{2} \sum_{i=1}^{n_u} \sum_{k=1}^n  \left( \theta_k^{(j)}\right)^2 \\
$$


### Optimization algorithm

Iterate below process until converge

1.  Initialize $x^{(1)}, \dots, x^{(n_m)}, \theta^{(1)}, \dots, \theta^{(n_u)}$ to small random values.
2.  Minimize $J(x^{(1)}, \dots, x^{(n_m)}, \theta^{(1)}, \dots, \theta^{(n_u)}) $ using gradient descent.


$$
\begin{align}
& x_k^{(i)} := x_k^{(i)} - \alpha \left( 
\sum_{j:r(i, j)=1} 
\left(
(\theta^{(j)})^T x^{(i)} - y^{(i, j)}
\right) \theta_k^{(j)}
+ \lambda x_k^{i}
\right)
\\
& \theta_k^{(j)} := \theta_k^{(j)} - \alpha \left( 
\sum_{i:r(i, j)=1} 
\left(
(\theta^{(j)})^T x^{(i)} - y^{(i, j)}
\right) x_k^{(i)}
+ \lambda \theta_k^{j}
\right)

\end{align}
$$

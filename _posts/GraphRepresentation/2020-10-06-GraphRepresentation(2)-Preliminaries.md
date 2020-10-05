---
layout: post
published: True
title: Probabilistic Graphical Model(2) - Preliminaries
categories:
- Mathematics
tags:
- Mathematics
- Statistics
- MachineLearning
typora-root-url: /Users/shephexd/Dropbox/github/pages/

---

This post is series of the instruction notes for the Coursera lecture[^1]

In this post, I will introduce two preliminaries before studying PGM.

1. Joint Distribution
2. Factor



<!--more-->



## Preliminaries: Joint Distribution

Assuming that there are three variables that determine admission for University.



- Intelligence(I)
  - $i^0(low), i^1(high)$
- Difficulty(D)
  - $d^0(easy), d^1(hard)$
- Grade(G)
  - $g^1(A), g^2(B), g^3(C)$



parameters: $2 \times 2 \times 3 = 12$ 

independent parameters: $11$



### Conditioning

$$
P(\beta\ \vert\ \alpha) = \frac{P(\alpha\ \cap \ \beta)}{P(\alpha)}
$$



To calculate the probability when grade is C, $g^1$

condition on $g^1$: $P(I,D \vert g^1)$



#### Conditioning: Reduction

By summing one variables we can get reduced probability.
$$
\sum_I P(I, D) = P(D)
$$


#### Conditioning: Renormalization


$$
\begin{align}
P(I, D, g^1) =& \sum_{j,k} P(i^j, d^k, g^1)\\
P(i^0, d^0 \vert g^1) =& \frac{P(i^0, d^0, g^1)}{P(I, D, g^1)} \\
\\
&\sum_{j,k} P(i^j, d^k \vert g^1) = 1
\end{align}
$$


#### Margialization

$$
P(D \vert g^1) = \sum_k P(i^k, D \vert g^1)
$$



------

## Preliminaries: Factors

Factor is the node in the PGM. It has values and can connect to other factors. Factor can explain the conditional probability distribution.



- A factor $\phi(X_1, \dots, X_k)$
  - $\phi: Val(X_1, \dots, X_k) \rightarrow R$
- Scope = $\{X_1, \dots, X_k\}$



### Conditional probability Distribution(CPD)


$$
P(G \vert I, D)
$$


In intuitive terms, this means that the value of B is not dependent on the value of A. We can derive this from $P(A,B)=P(A) \times P(B)$ as follows:
$$
\begin{align}
P(A, B) =& P(A) \times P(B) & \text{(by definition of independence)}\\
=&P(B\vert A) \times P(A) & \text{(by chain rule of probabilities)}\\
\text{therefore } P(B \vert A) =& P(B)
\end{align}
$$



### Factor product

$$
\phi(A, B) \times \phi(B, C) = \phi(A, B, C) \\
\phi(A, B, C) \times \phi(C, D) = \phi(A, B, C, D) \\
$$



### Factor Margilnalization

Let $X, Z$ be binary variables, and let $Y$ be a variable

if $\phi(X, Y, Z)$ is the factor, $\varphi(Y,Z) = \sum_X \phi(X, Y, Z)$ 



### Factor Reduction

Let $X, Z$ be binary variables, and let $Y$ be a variable

Assuming that we observe $Y=k$.

 If $\phi(X, Y, Z)$ is the factor, 
we can compute the missing entries of the reduced factor $\varphi(X, Z)$ given that $Y=k$



### Why factors?

- Fundamental building block for defining distributions in high-dimensional spaces
- Set of basic pertaions for manipulating these probability distributions



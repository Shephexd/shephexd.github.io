---
layout: post
title: Linear Algebra
published: False
categories:
- Mathematics
tags:
- Linear algebra
- Mathematics
---

I think I need to study **Linear algebra** from the beginning. Because of proving some equations of matrix, I don't have enough knowledge to understand machine learning algorithm based on Linear algebra.

Also to understand machine learning algorithm process and logic, It is necessary to know that. Most algorithms are derived from linear algebra.

I strongly agree with the idea to reject the nuclear power. Because it has terrible hazard comparing other resources. 

So, I'll share `theorem` and `definition of Linear algebra`.

I'm studying from the book [^book]

<!--more-->



## What is Linear algebra?

Linear algebra is the branch of mathematics **concerning vector spaces and linear mappings between such spaces**. It includes the study of lines, planes, and subspaces, but is also concerned with properties common to all vector spaces. - [Wikipedia][1]



A Linear equation in the variable $x_1,\ldots,x_n$ is an equation that can be written in the form



Fields using linear algebra

- analytic geometry

- engineering, physics

- natural sciences

- computer science

- computer animation

- advanced facial recognition algorithms and the social sciences (particularly in economics)

  ​

> Because linear algebra is such a well-developed theory, nonlinear mathematical models are sometimes approximated by linear models.


## System of linear equations


A system of linear equations has either

1. no solution
2. exactly one solution 
3. infinitely many solutions

A system of linear equations is said to be **consistent** if it has either one solution.

A system is **inconsistent** if it has no solution.



### Basic form of a linear equation

$$
a_1x_1+a_2x_2+\ldots+a_nx_n=b
$$

where $b$ and the coefficients $\ a_1,\ldots, a_n$ are real or complex numbers







### Axiom Signification



|                   Name                   |               Description                |
| :--------------------------------------: | :--------------------------------------: |
|        Associativity of addition         |       $u + (v + w) = (u + v) + w$        |
|        Commutativity of addition         |             $u + v = v + u$              |
|       Identity element of addition       | There exists an element $0 \in V$, called the zero vector, such that $v + 0 = v$ for all $v \in V$. |
| Distributivity of scalar multiplication with respect to vector addition |           $a(u + v) = au + av$           |
| Distributivity of scalar multiplication with respect to field addition |           $(a + b)v = av + bv$           |
| Compatibility of scalar multiplication with field multiplication |             $a(bv) = (ab)v$              |
| Identity element of scalar multiplication | $1v = v$, where 1 denotes the multiplicative identity in $F$. |



   



## Linear Transformation

Generally, we can transform vector using a matrix. We called the matrix as a transformation(function or mapping) $T$ from $R^n \to\ R^m$.


$$
T:R^n \to R^m\\
R^n\text{ is domaion of T}\\
R^m\text{ is codomaion of T}\\
\text{for x in }R^n,\text{ the vector } T(x)\ in\ R^n\text{ is called the image of x}\\
\text{The set of all images }T(x)\text{ is called the range of }T.
$$
When $R^n \to R^m(n=m)$ is called shear transformation. It make the images tilt.



#### Transformation equation

A transformation(or mapping) T is linear if:

1. $$T(u+v) = T(u) + T(v)
2. $T(cu) = cT(u)$
3. $T(0) = 0$ 
4. T(cu+dv) = cT(u) + dT(v)





Let T:$$R^n \to R^m$$ be **a linear transformation**. Then there exist a unique matrix A. such that
$$
T(x) = A(x) \ for\ all\ x\ in\ R^n
$$
Let $$T: R^n \to R^m$$ be a linear transformation and let $$A$$ be the standard matrix for $$T$$. Then:

1. $$T$$ maps $$R^n$$ onto $$R^m$$ if and only if the columns of A span $$R^m$$
2. $$T$$ is one-to-one if and only if the columns of  $$A$$ are linearly independent. onto $$R^m$$ if and only if the columns of A span $$R^m$$





## Terms of Linear algebra





I strongly agree with the idea to reject the nuclear power. Because it has terrible hazard comparing other resources. And in the history, we already can learn why we have to avoid to use the nuclear power.

[1]: https://en.wikipedia.org/wiki/Linear_algebra

[^book]: Linear Algebra and its application Third edition, *David C.lay*, 2003.
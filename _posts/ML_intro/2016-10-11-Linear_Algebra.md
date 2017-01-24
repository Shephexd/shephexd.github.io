---
layout: post
title: Linear Algebra
published: true
categories:
- Mathmatics
tags:
- Linear algebra
- Mathmatics
---

I think I need to study **Linear algebra** from the beginning. Because of proving some equations of matrix, I don't have enough knowledge to understand machine learning algorithm based on Linear algebra.

Also to understand machine learning algorithm process and logic, It is necessary to know that. Most algorithms are derived from linear algebra.



So, I'll share **theorem** and **definition of Linear algebra**.

I'm studying from the book [^book]

<!--more-->



## What is Linear algebra?

Linear algebra is the branch of mathematics **concerning vector spaces and linear mappings between such spaces**. It includes the study of lines, planes, and subspaces, but is also concerned with properties common to all vector spaces. - [Wikipeida][1]



A Linear equation in the variable $$x_1,\ldots,x_n$$ is an equation that can be written in the form



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

where $$b$$ and the coefficients $$\ a_1,\ldots, a_n$$ are real or complex numbers





### Axiom Signification



|                   Name                   |               Description                |
| :--------------------------------------: | :--------------------------------------: |
|        Associativity of addition         |        u + (v + w) = (u + v) + w         |
|        Commutativity of addition         |              u + v = v + u               |
|       Identity element of addition       | There exists an element 0 ∈ V, called the zero vector, such that v + 0 = v for all v ∈ V. |
| Distributivity of scalar multiplication with respect to vector addition |            a(u + v) = au + av            |
| Distributivity of scalar multiplication with respect to field addition |            (a + b)v = av + bv            |
| Compatibility of scalar multiplication with field multiplication |              a(bv) = (ab)v               |
| Identity element of scalar multiplication | 1v = v, where 1 denotes the multiplicative identity in F. |



   



### Linear independence

The Linear independence is important in Linear algebra. Because it means that each vector has no relation each others. 



#### Linearly independent

$$
\text{if}\ x_1v_1+x_2v_2+\ldots+x_pv_p = 0 \\
\text{has only the trivial solution}
$$

It can show using matrix. matrix A are linearly independent and only if the equation $$Ax=0$$ has only trival solution.

> The trivial solution means $$x_1,x_2,\ldots, x_p are all zero$$





Then How about dependent? It means each vector has relation among them.



#### Linearly dependent

$$
\text{if there exist weights }c_1,c_2,\ldots,c_p \text{ not all zero, such that}\\
\ c_1v_1+c_2v_2+\ldots+c_pv_p = 0
$$





### Linear Transformation

Generally, we can transform vector using a matrix. We called the matrix as a transformation(function or mapping) $$T$$ from $$R^n \to\ R^m$$.


$$
T:R^n \to R^m\\
R^n\text{ is domaion of T}\\
R^m\text{ is codomaion of T}\\
\text{for x in }R^n,\text{ the vector } T(x)\ in\ R^n\text{ is called the image of x}\\
\text{The set of all images }T(x)\text{ is called the range of }T.
$$
When $$R^n \to R^m(n=m)$$ is called shear transformation. It make the images tilt.



#### Transformation equation

A transformation(or mapping) T is linear if:

1. $$T(u+v) = T(u) + T(v)$$ **
2. $$T(cu) = cT(u)$$ **
3. $$T(0) = 0$$ **
4. $$T(cu+dv) = cT(u) + dT(v)$$ **





Let T:$$R^n \to R^m$$ be **a linear transformation**. Then there exist a unique matrix A. such that
$$
T(x) = A(x) \ for\ all\ x\ in\ R^n
$$
Let $$T: R^n \to R^m$$ be a linear transformation and let $$A$$ be the standard matrix for $$T$$. Then:

1. $$T$$ maps $$R^n$$ onto $$R^m$$ if and only if the columns of A span $$R^m$$
2. $$T$$ is one-to-one if and only if the columns of  $$A$$ are linearly independent. onto $$R^m$$ if and only if the columns of A span $$R^m$$





## Terms of Linear algebra

### Matrix and vector

$$
A = 
\begin{bmatrix}
a_{11} & \cdots & a_{1n}\\
\vdots & \ddots & \vdots\\
a_{m1} & \cdots & a_{mn}\\
\end{bmatrix}
$$

$$
X = \begin{bmatrix}
x\_1\\
\vdots\\
x\_n
\end{bmatrix}
$$

### square matrix

$$
A = 
\begin{bmatrix}
a_{11} & \cdots & a_{1n}\\
\vdots & \ddots & \vdots\\
a_{n1} & \cdots & a_{nn}\\
\end{bmatrix}
$$

### identity matrix

$$
A = 
\begin{bmatrix}
1 & 0 & \cdots & 0\\
0 & 1 & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & 1\\
\end{bmatrix}
$$

### transpose of a matrix or a vector

1. $$(A^T)^T = A $$ 
2. $$(A+B)^T = A^T + B^T$$
3. $$\text{For any scalar r, } (ra)^T = ra^T$$
4. $$(AB)^T = B^TA^T$$


$$
A = 
\begin{bmatrix}
a_{11} & \cdots & a_{1n}\\
\vdots & \ddots & \vdots\\
a_{m1} & \cdots & a_{mn}\\
\end{bmatrix}
$$


$$
A^T = 
\begin{bmatrix}
a_{11} & \cdots & a_{m1}\\
\vdots & \ddots & \vdots\\
a_{1n} & \cdots & a_{nm}\\
\end{bmatrix}
$$

$$
X = \begin{bmatrix}
x\_1\\
\vdots\\
x\_n
\end{bmatrix}
$$


$$
X^T = \begin{bmatrix}
x\_1 \cdots 
x\_n
\end{bmatrix}
$$


### determinant

### trace

### diagonal matrix

$$
A = \\
diag(a\_1,a\_2,\cdots,a\_n) = \\
\begin{bmatrix}
a\_1 & 0 & \cdots & 0\\
0 & a\_2 & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & a\_n\\
\end{bmatrix}
$$


### eigenvalue, eigenvector

### eigen decomposition

### characteristic equation

### Cayley-Hamilton theorem

### matrices with specific conditon

#### Orthgonal matrix
$$
AA^T=A^TA=E
$$

#### Symmetrix matrix
$$
A^T=A
$$

#### Unitray matrix

#### Hermitian matrix

### SVD(Singular value decomposition)

$$
A = U\Sigma V^T\\
A^TA = V(\Sigma^T\Sigma)V^T\\
AA^T = U(\Sigma \Sigma^T)U^T
$$

$$
\Sigma=\\
\begin{bmatrix}
\sigma & 0 & \cdots & 0\\
0 & \sigma & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & \sigma\\
0 & 0 & \cdots & 0\\
\end{bmatrix}\\
\\
or\\
\begin{bmatrix}
\sigma & 0 & \cdots & 0&0\\
0 & \sigma & \cdots & 0&0\\
\vdots & \vdots & \ddots & \vdots&0\\
0 & 0 & \cdots & \sigma & 0\\
\end{bmatrix}
$$







[1]: https://en.wikipedia.org/wiki/Linear_algebra

[^book]: Linear Algebra and its application Third edition, *David C.lay*, 2003.
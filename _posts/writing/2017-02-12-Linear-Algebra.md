---
layout: post
title: Linear Algebra
published: True
categories:
- Mathematics
tags:
- Linear algebra
- Mathematics
---



I think I need to study **Linear algebra** from the beginning. Because of proving some equations of matrix, I don't have enough knowledge to understand machine learning algorithm based on Linear algebra.

Also to understand machine learning algorithm process and logic, It is necessary to know that. Most algorithms are derived from linear algebra.



So, I'll share `theorem` and `definition of Linear algebra`.

I'm studying from the book [^Book]

<!--more-->



## What is Linear algebra?

Linear algebra is the branch of mathematics *concerning **vector** spaces and **linear mappings** between such spaces*. It includes the study of lines, planes, and subspaces, but is also concerned with properties common to all vector spaces. - [Wikipedia][1]



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

A system of linear equations has either properties below.

1. no solution
2. exactly one solution 
3. infinitely many solutions



- A system of linear equations is said to be `consistent` if it has either one solution.
- A system is `inconsistent` if it has no solution.



### The basic form of a linear equation

$$
a_1x_1+a_2x_2+\ldots+a_nx_n=b
$$

where $b$ and the coefficients $\ a_1,\ldots, a_n$ are real or complex numbers



#### Matrix and vector

$$
A = 
\begin{bmatrix}
a_{1} & \cdots & a_{n}\\
\end{bmatrix}
\ \
X = \begin{bmatrix}
x_1\\
\vdots\\
x_n
\end{bmatrix}
\\ a_n\text{ is column vectors}
$$



The linear equation can be denoted by the product of $A$ and $X$, $AX=b$



### Homogeneous equation

A system of linear equation is said to be `homogeneous` if it can be written in the form $AX=0$, where $A$ is a $m \times n$ matrix and 0 is the vector in $R^m$.

Such a system $AX=0$ always has at least one solution $X=0$(zero vector in $R^n$). This zero solution is called *the trivial solution*



> The homogeneous equation $AX =0$ has a nontrivial solution if and only if the equation has at least one free variable 



### Non-homogeneous

`Non-homogeneous` equation is denoted by $AX=b$. If it is consistent for some given b, assume that the solution for this equation is $p$. Then the solution set of $AX=b$ is the set of all vectors of the form $w=p+v_n$ ($v_n$ is any solution of the homogeneous equation $AX=0$)





## Linear independence

The `Linear independence` is important in Linear algebra. Because it means that each vector has no relation each others. 



### Linearly independent

$$
\text{if}\ x_1v_1+x_2v_2+\ldots+x_pv_p = 0 \\
\text{has only the trivial solution}
$$

It can show using matrix. The matrix $A$ is `linearly independent` and only if the equation $$Ax=0$$ has only a trivial solution.



The `trivial solution` means $x_1,x_2,\ldots, x_p$ are all zero.





### Linearly dependent

$$
\text{if there exist weights }c_1,c_2,\ldots,c_p \text{ not all zero, such that}\\
\ c_1v_1+c_2v_2+\ldots+c_pv_p = 0
$$

It can be expressed by matrix.


$$
\begin{bmatrix}
v_{1} &v_{2} & \cdots & v_{n}\\
\end{bmatrix}

\begin{bmatrix}
c_{1}\\
c_{2}\\
\vdots\\
c_{n}\\
\end{bmatrix}
=
\begin{bmatrix}
0\\
0\\
\vdots\\
0\\
\end{bmatrix}
$$




## Linear Transformation

Generally, we can transform vector using a matrix. We called the matrix as a `transformation`(function or mapping) $T$ from $R^n \to\ R^m$.
$$
T:R^n \to R^m\\
R^n\text{ is domaion of T}\\
R^m\text{ is codomaion of T}\\
\text{for x in }R^n,\text{ the vector } T(x)\ in\ R^n\text{ is called the image of x}\\
\text{The set of all images }T(x)\text{ is called the range of }T.
$$
When $R^n \to R^m(n=m)$ is called shear transformation, It make the images tilt



### Transformation equation

A `transformation(or mapping)` $T$ is linear if:

1. $T(u+v) = T(u) + T(v)$
2. $T(cu) = cT(u)$
3. $T(0) = 0$ 
4. $T(cu+dv) = cT(u) + dT(v)$



Let T:$R^n \to R^m$ be  `linear transformation`. Then there exist a unique matrix A. such that
$$
T(x) = A(x) \ for\ all\ x\ in\ R^n
$$
Let $T: R^n \to R^m$ be a `linear transformation` and let $A$ be the standard matrix for $T$. Then:

1. $T$ maps $R^n$ onto $R^m$ if and only if the columns of A span $R^m$
2. $T$ is one-to-one if and only if the columns of  $A$ are linearly independent. onto $R^m$ if and only if the columns of $A$ span $R^m$



## Invertible

If matrix $A$ has the property like $det(A) \ne 0$, Matrix $A$ is `invertible`. 

`Inverse matrix` is denoted by $A^{-1}$. 

$A^{-1}A=I$ and $AA^{-1}=I$ 



### Properties of Invertible matrix

- $(A^{-1})^{-1} = A$
- $(AB)^{-1}=B^{-1}A^{-1}$
- $(A^T)^{-1} = (A^{-1})^T$



## Subspace

A `subspace` of $R^n$ is any set $H$ in $R^n$ that has three properties.

1. The zero vector is in $H$.
2. For each $u$ and $v$ in $H$, the sum $u+v$ is in $H$.
3. For each $u$ in $H$ and each scalar $c$, the vector $cu$ is in $H$.



The `column space` of a matrix $A$ is the set col $A$ of all linear combination of the columns of A.

The `null space` of a matrix $A$ is the set Null $A$ of all solutions to the homogeneous equation $AX=0$.



A basis for a subspace $H$ of $R^n$ is a linearly independent set in $H$ that spans $H$. The pivot columns of a matrix $A$ form a basis for the column space of $A$.



## Dimension and Rank



## Determinant



## Terms of Matrices

### Matrix and vector

$$
A = 
\begin{bmatrix}
a_{11} & \cdots & a_{1n}\\
\vdots & \ddots & \vdots\\
a_{m1} & \cdots & a_{mn}\\
\end{bmatrix}

X = \begin{bmatrix}
x_1\\
\vdots\\
x_n
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

1. $(A^T)^T = A $
2. $(A+B)^T = A^T + B^T$
3. For any scalar r,  $(ra)^T = ra^T$
4. $(AB)^T = B^TA^T$

$$
A = 
\begin{bmatrix}
a_{11} & \cdots & a_{1n}\\
\vdots & \ddots & \vdots\\
a_{m1} & \cdots & a_{mn}\\
\end{bmatrix}
\ 
A^T = 
\begin{bmatrix}
a_{11} & \cdots & a_{m1}\\
\vdots & \ddots & \vdots\\
a_{1n} & \cdots & a_{nm}\\
\end{bmatrix}
$$

$$
X = \begin{bmatrix}
x_1\\
\vdots\\
x_n
\end{bmatrix}
\ 
X^T = \begin{bmatrix}
x_1 \cdots 
x_n
\end{bmatrix}
$$



### determinant

### trace

### diagonal matrix

$$
A = diag(a_1,a_2,\cdots,a_n) = 
\begin{bmatrix}
a_1 & 0 & \cdots & 0\\
0 & a_2 & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & a_n\\
\end{bmatrix}
$$

### eigenvalue, eigenvector

### eigen decomposition

### characteristic equation

### Cayley-Hamilton theorem

### matrices with specific condition

#### Orthogonal matrix

$$
AA^T=A^TA=E
$$

#### Symmetric matrix

$$
A^T=A
$$

#### Unitary matrix

#### Hermitian matrix

### SVD(Singular value decomposition)

$$
A = U\Sigma V^T\\
A^TA = V(\Sigma^T\Sigma)V^T\\
AA^T = U(\Sigma \Sigma^T)U^T
$$

$$
\Sigma= \begin{bmatrix}
\sigma & 0 & \cdots & 0\\
0 & \sigma & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & \sigma\\
0 & 0 & \cdots & 0\\
\end{bmatrix} \ or \
\begin{bmatrix}
\sigma & 0 & \cdots & 0&0\\
0 & \sigma & \cdots & 0&0\\
\vdots & \vdots & \ddots & \vdots&0\\
0 & 0 & \cdots & \sigma & 0\\
\end{bmatrix}
$$



[1]: https://en.wikipedia.org/wiki/Linear_algebra

[^Book]: Linear Algebra and its application Third edition, *David C.lay*, 2003.
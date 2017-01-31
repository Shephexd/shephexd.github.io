---
layout: post
title: Linear and nonlinear inverse problem
published: True
categories:
- Mathematics
tags:
- Linear algebra
- Mathematics
---



Linear algebra is useful to solve some problems. But, In the real, There are many nonlinear problems. To solve this kind of problem, we can assume a nonlinear problem like linear problem by using some methods.

<!--more-->



## What is Linear algebra?

Linear algebra is the branch of mathematics **concerning vector spaces and linear mappings between such spaces**. It includes the study of lines, planes, and subspaces, but is also concerned with properties common to all vector spaces. - [Wikipedia][1]



A Linear equation in the variable $$x_1,\ldots,x_n$$ is an equation that can be written in the form



Fields using linear algebra

- analytic geometry

- engineering, physics

- natural sciences

- computer science

- computer animation

- advanced facial recognition algorithms and the social sciences (particularly in economics)

  ​

So, I'll share **theorem** and **definition of Linear algebra**.

I'm studying from the book [^book]









## Linear and nonlinear inverse problem

11

### Norm

Norm can be called as a distance like `Manhattan`, `Euclidean`

It can map nonlinear function to linear function.



`Norm` is mapping: $\Vert \cdot \Vert : V \to R^N = [0,\infty)$

1. $\Vert f\Vert=0$ if and only if $f=0$ (zero elements)
2. $\Vert\alpha f\Vert=\Vert\alpha\Vert\ \Vert f \Vert$
3. $\Vert f+g \Vert \le \Vert f \Vert + \Vert g\Vert$ 



#### Different Norms

$$
||V||_2 = \sqrt{v_1^2+v_2^2+v_3^2} : Euclidean\ Norm
$$



### Normed space

If $f,g$ belongs to a normed space V then should lead

- $f+g \in V$
- $\alpha \cdot f \in V, \forall \alpha \in R^n $
- $f+g= g+f, h+(f+g) = (h+f)+g$



Normed space is Vector space $\oplus$ Norm define over it.



#### Finite Dimensional normed spaces

The set ${\phi_1, \phi_2,…,\phi_n} :=S$ is a spanning set for the space V if any element $f \in V $ can be written in form $f=\sum_{i=1}^m\alpha_i\phi_i,\alpha_i \in n $.

$\phi_i$ are called `Spanning functions`



### Inner product

We define the `Inner product`, $<\cdot,\cdot>:V\times V \to R$ a mapping.

`Inner product` has properties.

- $\langle f,g \rangle=\langle g,f \rangle$
- $\langle \alpha f, g \rangle = \alpha \langle f, g\rangle$
- $\langle f+g,h \rangle = \langle f,h \rangle + \langle g, h \rangle$
- $\langle f, f \rangle \gt 0, \langle f, f \rangle = 0 \mathtt{\ if\ and\ only\ if} f=0$



Another useful equation in inner product is $cos\theta = \frac{\langle x, y \rangle}{\Vert x \Vert \Vert y \Vert}$



### Orthogonality



#### Orthogonal projections

`orthogonal projection` of vector $f$ onto the subspace $V$ is $\Vert f - g \Vert$ is **minimized**.


$$
\Vert f - g \Vert^2 = \langle f-g,f - g \rangle \\
= \Vert f \Vert ^2 + \Vert g \Vert ^2 - \langle f,g  \rangle - \langle g,f  \rangle \\
= \Vert f \Vert ^2 + \Vert g \Vert ^2 - 2 \langle f,g  \rangle\\
= \Vert f \Vert^2 + \Vert g \Vert ^2 - 2 \langle f, g \rangle\\
=\langle\sum_{i=1}^nc_i\phi_i, \sum_{j=1}^nc_j\phi_j \rangle - \langle \sum_{i=1}^nc_i\phi_i, g \rangle + \langle g, g \rangle \\
= \sum_{i=1}^n c_i\sum_{j=1}^n \langle \phi_i\phi_j \rangle - 2\sum_{i=1}^n c_i \langle \phi_i,g \rangle + \langle g, g\rangle\\
$$


Assuming the case of $n=3$,


$$
{c_1^2 \langle \phi_1, \phi_1 \rangle} + c_2^2 \langle \phi_2, \phi_2 \rangle + c_3^2 \langle \phi_2, \phi_2 \rangle\\
 + 2c_1c_2 \langle \phi_1, \phi_2 \rangle + 2c_1c_3 \langle \phi_1, \phi_3 \rangle + 2c_2c_3 \langle \phi_2, \phi_3 \rangle \\
- 2c_1 \langle \phi_1, g \rangle - 2c_2 \langle \phi_2, g \rangle - 2c_3 \langle \phi_3, g \rangle +\langle g, g \rangle\\
$$


$S(c_1,c_2,c_3)$ We have to find the values to minimmize $\Vert f - g \Vert$.


$$
\frac{d S}{d c_1} := 2c_1 {\langle \phi_1, \phi_1 \rangle}  + 2c_2 {\langle \phi_1, \phi_2 \rangle} + 2c_3{\langle \phi_1, \phi_3 \rangle} - 2 {\langle \phi_1, g \rangle}\\
= 2\sum_{i=1}^3c_i\langle \phi_1, \phi_i \rangle -2 {\langle \phi_1, g \rangle} = 0\\

\frac{d S}{d c_2} := 2c_1 {\langle \phi_2, \phi_1 \rangle}  + 2c_2 {\langle \phi_2, \phi_2 \rangle} + 2c_3{\langle \phi_2, \phi_3 \rangle} - 2 {\langle \phi_2, g \rangle}\\
= 2\sum_{i=1}^3c_i\langle \phi_2, \phi_i \rangle -2 {\langle \phi_2, g \rangle} = 0\\

\frac{d S}{d c_3} := 2c_1 {\langle \phi_3, \phi_1 \rangle}  + 2c_2 {\langle \phi_3, \phi_2 \rangle} + 2c_3{\langle \phi_3, \phi_3 \rangle} - 2 {\langle \phi_3, g \rangle}\\
= 2\sum_{i=1}^3c_i\langle \phi_3, \phi_i \rangle -2 {\langle \phi_3, g \rangle} = 0\\

\left\{ 
\begin{array}{c}
\sum_{i=1}^3 c_i\langle \phi_1, \phi_i \rangle = {\langle \phi_1, g \rangle} \\
\sum_{i=1}^3 c_i\langle \phi_2, \phi_i \rangle = {\langle \phi_2, g \rangle} \\
\sum_{i=1}^3 c_i\langle \phi_3, \phi_i \rangle = {\langle \phi_3, g \rangle} 
\end{array}
\right.
$$


by expression using matrix,


$$
\begin{bmatrix}
        {\langle \phi_1, \phi_1 \rangle} & {\langle \phi_1, \phi_2 \rangle} & {\langle \phi_1, \phi_3 \rangle} \\
        {\langle \phi_2, \phi_1\rangle} & {\langle \phi_2, \phi_2 \rangle} & {\langle \phi_2, \phi_3 \rangle} \\
        {\langle \phi_3, \phi_1 \rangle} & {\langle \phi_3, \phi_2 \rangle} & {\langle \phi_3, \phi_3 \rangle} \\
\end{bmatrix}
\begin{bmatrix}
        c_1 \\
        c_2 \\
        c_3 \\
\end{bmatrix}
=
\begin{bmatrix}
	{\langle \phi_1, g \rangle} \\
	{\langle \phi_2, g \rangle} \\
	{\langle \phi_3, g \rangle} \\
\end{bmatrix}
$$




`Orthonormalization` :  $V_i = \frac{1}{\Vert\phi_i\Vert}\cdot\phi_i$



#### Orthonormal basis

We say we have `orthogonal basis` for the space $V$ if $V = span\ \{ \phi_1,\dots,\phi_n\} \ where\ \Vert \phi_i \Vert = 1$ and $\Vert \phi_i, \phi_j \Vert = \begin{cases}1  & \text{if $i=j$} \\0 & \text{if $i\neq j$}\end{cases} \forall i,j=1,\dots,n$ 


---
title: Computer vision(8) - Localization
layout: post
categories:
- Computer vision
tags:
- Algorithm
- Linear algebra
- Computer vision
---



3D localization (3D $\rightarrow$ 2D)

<!--more-->


$$
\left[ x_i, y_i \right]^T = \left[ \frac{fX_i}{Z_i}, \frac{fY_i}{Z_i} \right]^T
$$

$$
\begin{pmatrix}
\tilde p_i\\
1
\end{pmatrix} = 
\begin{pmatrix}
\tilde x_i\\
\tilde y_i\\\
1
\end{pmatrix}
=
K^{-1}
\begin{pmatrix}
\bar x_i\\
\bar y_i\\\
1
\end{pmatrix}
$$

$$
x_i = \frac{^CX_i}{^CZ_i} = \frac{r_{11}X_i+r_{11}X_i+r_{11}X_i+t_x}{1}
$$

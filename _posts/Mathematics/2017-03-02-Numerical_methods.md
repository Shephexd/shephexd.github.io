---
layout: post
title: Numerical methods
categories:
- Mathematics
tags:
- Mathematics
- Linear algebra
---



Numerical analysis is the study of algorithms that use numerical approximation for the problems of mathematical analysis.

We can estimate the value with small error by using the numerical algorithm.



<!--more-->

## Preliminaries

`Machine epsilon`  : $\epsilon_{mach} = 2^{-52}$   
`Relative rounding error` $\le\ \frac{1}{2}\epsilon_{mach}$   
Another type of error is `approximation error`  



## Approximation for derivative

Even if we don't know the actual function, we can estimate the derivative value.   



$f'(x) = \lim_{h\rightarrow0}\frac{f(x+h)-f(x)}{h}$  
$f'(x) \simeq \frac{f_1 - f_0}{h}$  where $f_1 = (x+h)$ and $f_0 = f(x)$  

`Two point forward-difference` formula, if $h \gt 0$  
`Two point backward-difference` formula, if $h \lt 0$



### Lagrange interpolating polynomials

To formulate the difference formulas with `error approximation`, start by approximating the function using `Lagrange interpolating polynomials`.

Let $\{ x_k, f(x_k) = P(x_k) \}\ k=0,1,\cdots, m$  
be the set of evaluation points, then  
$P_n(x) = f(x_0)L_{n,0}(x)+ \dots + f(x_m)L_{n,m}(x) = \sum_{k=0}^m f(x_k)L_{n,k}(x)$  
where $\prod_{i=0, i \neq k}^n \frac{x-x_i}{x_k-x_i} \forall k=0, \cdots, m$



### Error terminology

If $x_0, x_1, \dots x_n$ are distinct number in the interval $[a,b]$, and if $f\in \xi(x)$ in $(a,b)$ exists with $f(x) = P_n(x) + R(x)$   
where $R_x = \frac{f^{n+1}(\xi(x))}{n+1}(x-x_0)(x-x_1)\cdots(x-x_n)$   



### Example

Approximate $f′(x_0)$  
$$
\text{Let } x_1 = x_0 + h, h \neq 0\text{ and }x_1 \in [a, b]\\
f(x)= \sum_n^1 f(x_k)L_{n,k}(x) + \frac{(x-x_0)(x-x_1)}{2!} f''(\xi)\\
f'(x) = D_x(f(x))\\
f'(x) = \frac{f(x_0+h) - f(x_0)}{h} - \frac{h}{2}f''(\xi)
$$

where $f ′′(ξ) ≤ M = max \vert f ′′(ν) \vert$, and thus the **truncation** $ν∈[a,b]$  

**error** $E(f ′(x_0)) ≤ \frac{\vert h \vert}{2}M$



For example, the real result is $0.12344493424$ and our approximation value is $0.1234491250$. Then, the error is the difference between real and approximation, $0.12344493424 - 0.1234491250= -4.190760000008509e^{-06}$



### Three point approximation equation


$$
f'(x_0) = \frac{1}{2h}(-3f(x_0) + 4f(x_0+h) - f(x_0 + 2h)) + \frac{h^2}{3}f^{(3)}(\xi_0)\\
f'(x_0) = \frac{1}{2h}( f(x_0 + h) - f(x_0 - h)) + \frac{h^2}{6}f^{(3)}(\xi_1) (\text {centered formual})
$$


- Two-point formulas: $E=O(h)$
- Three-point formulas: $E=O(h^2)$

*For too small $h$ rounding error start to grow.*



### Rounding error example

$f(x_0 \pm h)  = \hat{f}(x_0 \pm h) + e_{\pm h}$  
Now $\vert e_{\pm h}\vert \lt \epsilon$  and $f'(x_0) = \frac{1}{2h}(f(x_0 + h) - f(x_0 -h) - \frac{h^2}{6}f^{(3)}(\xi))$

*Define $h$ where total error(approximation error + rounding error) is at minimum.*


$$
f'(x_0) = \frac{1}{2h}(f(x_0 + h) - f(x_0 -h) - \frac{h^2}{6}f^{(3)}(\xi) )\\
= \frac{1}{2h}(\hat{f}(x_0 + h) - \hat{f}(x_0 -h)) + \frac{1}{2h}(e_h - e_{-h})- \frac{h^2}{6}f^{(3)}(\xi)
$$

$$
E(h) = |f'(x_0) -\frac{1}{2h}(\hat{f}(x_0+h) - \hat{f}(x_0 - h))|\\
= |\frac{1}{2h}(e_h - e_{-h}) -  \frac{h^2}{6}f^{(3)}(\xi) |\\
\le |\frac{e_h}{2h}| + |\frac{e_{-h}}{h^2}|  + |\frac{h^2}{6}f^{(3)}(\xi)| \le \frac{\epsilon}{h}+\frac{h^2}{6}M = E_{upper}(h)
$$

The `total error upper bound` is **minimum** at  

$E'_{upper}(h)  = - \frac{\epsilon}{h^2}+\frac{2h}{6}M=0 \rightarrow h_{opt} = (\frac{3\epsilon}{M})^{1/3}$





### Five point formulas

$$
f'(x_0) = \frac{1}{12h}( -25f(x_0) + 48f(x_0 + h) -36f(x_0 + 2h) + 16f(x_0 + 3h) -3f(x_0 + 4h)) + \frac{h^4}{5}f^{(5)}(\xi_0) \\
f'(x_0) = \frac{1}{12h}(f(x_0-2h) - 8f(x_0 - h) + 8f(x_0 + h) -f(x_0 + 2h) ) + \frac{h^4}{30}f^{(5)}(\xi_1)(\text {centered formual})
$$



### second derivative

$$
f''(x_0) \simeq \frac{1}{h^2}(f(x_0 - h) - 2f(x_0) + f(x_0) + h))
$$



### Partitial derivatives


$$
\frac{\partial u(x,y)}{\partial x} \simeq \frac{u(x+h,y) - u(x,y) }{ h }
$$

$$
\frac{\partial^2 u(x,y)}{\partial x \partial y} \simeq \frac{1}{4h^2}(u(x+h,y+h) - u(x+h,y-h) = u(x-h,y+h) + u(x-h,y-h) )
$$





## Approximation for integral

Numerical evaluation of integrals
$$
J = \int_a^bf(x)dx
$$


### Rectangular Rule


$$
\int_a^bf(x)dx \simeq h \sum_{i=1}^nf^{*}_i\\
\text{where }h=(b-a)/n
$$


### Trapzoidal Rule

$$
\int_a^bf(x)dx \simeq \frac{h}{2}f(a) +  h \sum_{i=1}^{n-1}f(x_i) + \frac{h}{2}f(b)\\
\text{where }h=(b-a)/n
$$



### Simpson's rule

$$
\int_a^bf(x)dx \simeq \frac{h}{3} \left( f(x_0)
+ 2\sum_{i=1}^{n/2-1} f(x_{2i})
+ 4\sum_{i=1}^{n/2} f(x_{2i-1}) + f(x_n)\right)\\
\text{where } h=(b-a)/2m,n=2m
$$



### Error estimation



#### Trapezoidal rule

$$
\epsilon = -\frac{b-a}{12}h^2f''(\xi)
$$



#### Simpson's rule

$$
\epsilon = -\frac{b-a}{180}h^4f^{(4)}(\xi)
$$





### Gaussian quardrature

$$
\int_a^bf(x)dx = \frac{b-a}{2} \int_{-1}^1 f\left( \frac{b+a+(b-a)t}{2} \right)dt \simeq \frac{b-a}{2} \sum_{i=1}^nw_if(\hat{t_i})\\
\hat{t_i} = (b+a +(b-a)t_i)/2\\
x = (b+a +(b-a)t)/2, \ dx=(b-a)/2dt
$$



## Inital value problem



### Euler's Method

$$
y(t_{n+1}) \simeq y_{n+1} = y_n + f(t_n,y_n)h\\
h=step\ size
$$



### Improved Euler's Method

The rate of the change can be approximated as $\frac{1}{2}(f(t_n, y_n) + f(t_{n+1},y_n + f(t_n, y_n)h))$
$$
y(t_{n+1}) = \simeq y_{n+1} = y_n + \frac{h}{2}(f(t_n,y_n)+f(t_{n+1} + y_n + f(t_n,y_n)h)
$$


### Runge-Kutta Method


$$
y_{n+1} = y_n + \frac{h}{6}\left( k_{n,1} + 2k_{n,2} + k_{n,3} + k_{n,4}  \right)\\
f(t_n, \phi(t_n)) = k_{n,1} = f(t_n,y_n)\\
k_{n,2} = f(t_n + \frac{h}{2},y_n +\frac{h}{2}K_{n,1})\\
k_{n,3} = f(t_n + \frac{h}{2},y_n + \frac{h}{2}k_{n,2})\\
$$

### Error analysis



How can we get the solution with reasonable error?
If the error of one step $\ge \epsilon_{tol}$, making the step size smaller will be a way to reduce the error. And repeat until the error is reasonable.



### Stiff problem

`Stiff problems` are problems whose solution contains terms involving widely varing time scales.



### Implicit Euler's Method

$$
y(t_{n+1}) \simeq y_{n+1} = y_n + f(t_{n+1},y_{n+1})h
$$



## boundary value problem

a Simple form of two-point boundary value problem of ODE is

$ y'' = f(t,y,y'), y(a) = y_a, y(b) = y_b$



- Shooting method
- finite difference methods
- finite element methods




### Shooting method

$R(s) = y_s(b) - y_b$ where $y_s(t)$ is the solution of inital value problem

$y'' = f(t,y,y'), y(a) = y_a, y'(a)=s$

The solution of original boundary value problem satisfies R(s) = 0





## linear algebra

`A linear system of n equations` in $n$ unknowns $x_1,\dots, x_n$ is a set of equations of the form  




$$
a_{11}x_1 + \dots + a_{1n}x_n = b_1,\\
a_{21}x_1 + \dots + a_{2n}x_n = b_2\\
\vdots\\
a_{n1}x_1 + \dots + a_{nn}x_n = b_n,
$$





The system can be denoted by the matrix expression as
$$
Ax=b
$$
The solution can be obtained by `inverse matrix`
$$
x = A^{-1}b
$$
In numerical calculations, the inverse matrix not **explicitly** created.
But various algorithms, most of them based on matrix factorization methods can be used:

- Gauss elimination
- LU-decomposition
- cholesky's method
- QR-decomposition
- several iterative methods



### Matrix as a mapping

​       
$$
A = \begin{pmatrix}
a & 0 \\
0 & b\\
\end{pmatrix}
$$
Matrix $A$ can map into the circle, $Ax=y$.  
$Ax = (y_1,y_2) = (ax_1,bx_2)^T$, so the image is the ellipses $\frac{y_1^2}{a^2} + \frac{y_2^2}{b^2} = 1$.







### SVD decomposition

$A$ is a matrix, and it can be
$$
A = USV^T
$$
where $U, V$ are the rotation(or `unitary`) matrixes and $S$ is a diagonal matrix with singular values $\sigma_1,\dots, \sigma_n$ on the diagonal.

if A is a $m \times n$ matrix, then $U$ is a $m \times m$ square matirx, $V$ is a $n \times n$ matrix and $S$ is  diagonal matrix of size $m \times n$, where the number of diagonal elements is the smaller one of $m$ and $n$.



#### Unitray matrix

a complex square matirx $U$ is `unitary` if its conjugate transpose $(U^*)$ is also its inverse



$U$ is a unitray matrix.
$$
U^*U = UU^* = I
$$
$A^*=(\overline{A})^T = \overline{A^T}$



### Conditiong

How much noise in the data y of an equation system $Ax=y$ effect the accuracy of the solution $x$ depends on the amount of noise.



`Well-conditioned`: if **small** changes in the data cause only **small** changes in the solution.

`ill-conditioned`: if **small** changes in the data cause only **large** changes in the solution.



### Condition number



$k(A) = \sigma_1/\sigma_n$

$k(A) = \Vert A \Vert \Vert A^{-1} \Vert  $



1. A linear system of equations $Ax=y$ whose condition number $k(A)$ is **small** is `well-conditioned`.
2. A large condition number indicated `ill-conditioning`.
3. The smallest possible value of $k(A)$ is 1, and it is achieved by rotation matrixes.



$Ax=b$

if $n>m$, the system $Ax=b$ is called *overdetermined*  
and use Least square(LSQ)   
$$
\min_x \Vert Ax-b \vert_2 = \min_x \sum_i((Ax)_i - b_i)^2
$$
if $n < m, the system Ax = b is called *underdetermined*  
and use SVD  to get a solution



### Pseudoinverse

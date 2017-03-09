---
layout: post
title: Algorithm - Discrete optimization
categories: Algorithm
tags:
- Algorithm
- Matlab
- Mathematics
---





Heuristics



<!--more-->

```matlab
intlinprog() %minimize
% -f(x) minimize -> f(x) maximize
```



A graph is a pair $G = (V,E)$ where $V$ is a set of $vertices$ or $nodes$, $E$ is a set of $edges$. 

Vertices $i \in V$ and $j \in V $ are $adjacent$ (or $neighbors$) if $[i,j] \in E$.


$$
a_{ij}=\begin{cases}
1 & & if[i,j] \in E\\
0 & & \text{otherwise}
\end{cases}
$$



## Terminology

Weighted graph
: each $edge\ [i,j]$ has a $weight\ w_{ij} $associated with it.

Comlete graph
: there is an edege between every pair of vertices i.e all vertices are adjacent to each other.

Connected graph
: there is a path between every pair of vertices.

Planar graph
: can be darwn in the plane without crossing edges.

Tree
: a connected graph with no cycles.

Hemiltonian path
: a path that includes every vertex (exactly once)

Hemiltonian cycle
: a cycle that includes every vertex (exactly once)

Eulerian walk / circult
: a walk / closed walk in which each edge appears exactly once.

Graph $G = (V,E)$ has an Eulerian circuit if and only if it is connected and the degree on every node is even.  
Graph $G = (V,E)$ has an Eulerian circuit if and only if it is connected and the degree on every node is even.



## Graph theory

There are many fields using and adopting `graph theory` 

- telecommunications
- data transmission
- transportation and logistics
- neural networks
- traffic



1. Salesman problem
2. ​
3. KNAPSACK problem





- 100 files to be stored in a USB stick. whcih to select, if all don't fit?  
  *$\rightarrow$ values, sizes, USB capacitiy*
- 10 investment possibilities with captal investments $w_i$, expected returns $v_i$



Integer linear programming model(ILP)  
$\min f = \sum^n_{j=1}c_jx_j$  

subject to constrain
$\sum_{i=1}^n w_ix_i \le M $  		$x_i = \begin{cases} 1 & & \text{if item $i$ selected} \\ 0 & & \text{if not} \end{cases}$

1. 2 
2.  3
3. 3
4. 4
5. 4
6. 4
7. 4
8. 4
9. 4
10. 4 
11. ​
12. Minimum weight edge cover
13.   
14. Vertext coloring, chromatic number  
    Color the vertices of a graph with a minimum number of colors such that adjacent vertices have different colors.
15. ​
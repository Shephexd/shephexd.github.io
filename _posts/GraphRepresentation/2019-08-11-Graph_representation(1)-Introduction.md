---
layout: post
published: True
title: Probabilistic Graphical Model(1) - Introduction
categories:
- Mathematics
tags:
- Mathematics
- Statistics
- MachineLearning
typora-root-url: /Users/shephexd/Dropbox/github/pages/
---

This post is series of the instruction notes for the Coursera lecture[^1]

The Proabilistic Graphical Model let us understand the world with relations and factors.

Also, Most of the machine learning model can be represened as an probabilistic graph. 



There are three main subjects for `Probabilistic Graph Models`

- Representation
- Learning
- Reasoning



<!--more-->



### Required

- Basic probability theory
- Some programming
- Some algorithms and data structures



### Recommended

- Machine learning
- Simple optimization
- Matlab



### Other Issues

- Honor code
- Time management(10 - 15 hrs / week)
- Discussion forum & study groups



### What you'll learn

- Fundamental methods
- Real-world applications
- How to use these methods in your work





## Probabilistic Graphical Model



- Predisposing factors
- Symptoms
- test results
- diseases
- treatment outcomes



millions of pixels or thousands of superpixels

each needs to be labeled





### Models

Delclarative representation to represnt world





domain expert elicitation into model

data learning for model







#### Probabilitistc(uncertainty)

- Partial knowledege of state of the world
- Noisy observation
- phenomena not covered by our model
- Inherent stochasticity





### Probability theory

- Declarative representation with clear semantics
- powerful reasoning patterns
- Established learning methods



### Graphical

To represent complex systems based on the graphic theory of computer science



Random variables $X_1, \dots, X_n$

Joint distribution $P(X_1, \dots, X_n)$



for example, in the $N$ binary valued distribution over $2^N$ possible states.





### Graphical models



#### Bayesian networks

- directed graph



#### Markov networks

- Undirected graph
  (Undirecte graph over 4 random variables A, B, C, D)



### Graphcial representaiton

- Intuitive & compact data structure
- Efficient reasoning using general-purpose algorithms
- Sparse parameterization
  - Feasible elicitation
  - Learning from data



#### Many Application

- Medical diagnosis
- Fault diagnosis
- Natural language processing
- Traffic Analysis
- Image segmentation





### Textual information Extraction

![스크린샷 2019-02-01 12.15.35](/assets/post_images/probabilistic_graph_representation/textual_information_extraction.png)



#### Representation

- Directed and undirected
- Temporal and plate model



#### Inference(reasoning)

- Exact and approximate
- Decision making



#### Learning

- parameters and structure
- with and without complete data





## Preliminaries: Joint Distribution

- Intelligence(I)
  - $i^0(low), i^1(high)$
- Difficulty(D)
  - $d^0(easy), d^1(hard)$
- Grade(G)
  - $g^1(A), g^2(B), g^3(C)$





parameters: $2 \times 2 \times 3 = 12$ 

idependent parameters: $11$





### Conditioning

condition on $g^1$



#### Conditioning: Reduction



#### Conditioning: Renormalization

$P(I, D, g^1)$





#### Margialization

$P(I, D)$





------

## Preliminaries: Factors



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





### General factors

![image-20190201133039014](/Users/shephexd/Library/Application%20Support/typora-user-images/image-20190201133039014.png)



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





## Bayesian Networks



### Semantis & Factorization

The student, $P(G, D, I, S, L)$

- **G**rade
- Cour **D**ifficulty
- Student **I**ntelligence
- Student **S**AT
- Reference **L**etter



### Bayesian Network

1. Grade might be depend on `Court difficulty`, and `Student Intelligence`.
2. Intelligence might have an effect on `Grade`, and `Student SAT`
3. Grade might have an effect on the `Reference Letter`



![image-20190208114241727](/Users/shephexd/Library/Application Support/typora-user-images/image-20190208114241727.png)





> The network consist of five CPDs.  
> CPD: Conditional probability Distribution

![image-20190208114609234](/Users/shephexd/Library/Application Support/typora-user-images/image-20190208114609234.png)



- `Diffculty` and `Intelligence` are uncodnitional probability distrbution because they are not depend on other nodes.

- `Grade` , `Letter` and `SAT` are uncodnitional probability distrbution because they are not depend on other nodes.



The above network can be represented by the equation,
$$
P(D, I, G, S, L) = P(D)P(I)P(G \vert D, I)P(S \vert I) P(L \vert G)
$$




A Bayesian Network is:

- A directed acyclic graph(DAG) G whose nodes represent the random variables $X_1, \dots, X_n$.
- For each node $X_i$ a CPD $P(X_i \vert \text{Par}_G(X_i))$
- THe Bayesian Network represents a joint distribution via the chain rule for Bayesian Networks.  
  $P(X_1, \dots, X_n) = \prod_i P(X_i \vert \text{Par}_G(X_i))$


$$
\begin{align}
\sum_{D, I, G, S, L} P(D, I, G, S, L) =& \sum_{D, I, G, S, L} P(D) P(I) P(G \vert I, D) P(S \vert I)P(L \vert G) \\
=& \sum_{D, I, G, S} P(D) P(I) P(G \vert I, D) P(S \vert I)\sum_L P(L \vert G) \\
=& \sum_{D, I, G} P(D) P(I) P(G \vert I, D) \sum_S P(S \vert I) \\
=& \sum_{D, I} P(D) P(I) \sum_G P(G \vert I, D) \\
\sum_X P(X \vert Y) = 1
\end{align}
$$



### Reasoning



#### causal reasoning

causal reasoning form top node to the bottom.


$$
P(l^1) \approx 0.5 \\
P(l^1 \vert i^0) \approx 0.39 \\
P(l^1 \vert i^0, d^0) \approx 0.51
$$


#### Evidential Reasoning

When student get a low rank, we can do reasoning like the course might be hard or the intelligence of student might be not enough for that course.


$$
P(d^1) = 0.4 \\
P(d^1 \vert g^3) \approx 0.63 \\
$$

$$
P(i^1) = 0.3 \\
P(i^1 \vert g^3) \approx 0.08 \\
$$





#### Intercausal Reasoning

When student get a low rank and class is hard,


$$
P(d^1) = 0.4 \\
P(d^1 \vert g^3) \approx 0.63 \\
$$

$$
P(i^1) = 0.3 \\
P(i^1 \vert g^3) \approx 0.08 \\
P(i^1 \vert g^3, d^1) \approx 0.11 \\
$$
![image-20190208185433008](/Users/shephexd/Library/Application Support/typora-user-images/image-20190208185433008.png)



### When can X influence y?

$(X)$ is observed value.

- $X \rightarrow y$
- $X \leftarrow y$
- $X \rightarrow W \rightarrow y$
- $X \leftarrow W \leftarrow y$
- $X \rightarrow W \leftarrow y$





### When can X influence y Given evidence about Z



![image-20190211201148502](/Users/shephexd/Library/Application Support/typora-user-images/image-20190211201148502.png)



|                  $W \notin Z$                  |                     $W \in Z$                     |
| :--------------------------------------------: | :-----------------------------------------------: |
|                       O                        |                         X                         |
|                       O                        |                         X                         |
|                       O                        |                         X                         |
| X: if $W$ and or of its descendants not in $Z$ | O: either $W$ or one of its descendants is in $Z$ |





### Active Trails

- A trail $X_1 - \dots - X_n$ is active given $Z$ if:
  - for any v-structure $X_{i-1} \rightarrow X_i \leftarrow X_{i+1}$ we have that $X_i$ or one of its descendants $\in Z$
  - No other $X_i$ is in $Z$



The node that can be linked via forward direction can be affected if there is no observed node between start and end node.

Even if the node that is not linked via forward direction but backward direction, When the no







In a Bayesian network, the conditional probability distribution associated with a variable is the conditional probability distribution of that variable given its parents. If the probabilities for 2 of E's possible values are known, then the probability of the 3rd is also known because the probabilities of E's possible values must sum to 1.



To calculate the required values, we can apply Bayes' rule. For instance,



$$
\begin{align}
&P(A=1\vert T=1,P=1) P(A=1,T=1,P=1)P(T=1,P=1) 
\\=&P(A=1,T=1,P=1)P(A=0,T=1,P=1)+P(A=1,T=1,P=1)
\end{align}
$$


We can then use the chain rule of Bayesian networks to substitute the correct values in, e.g.,


$$
P(A=1,T=1,P=1)=P(P=1)×P(A=1)×P(T=1\vert P=1,A=1)
$$


This example of inter-causal reasoning meshes well with common sense: if we see a traffic jam, the probability that there was a car accident is relatively high. However, if we also see that the president is visiting town, we can reason that the president's visit is the cause of the traffic jam; the probability that there was a car accident therefore drops correspondingly.



![image-20190215133913091](/Users/shephexd/Library/Application Support/typora-user-images/image-20190215133913091.png)



![image-20190214200901589](/Users/shephexd/Library/Application Support/typora-user-images/image-20190214200901589.png)





## Idependence



- For events $\alpha, \beta, P \vDash \alpha \bot $ if:
  - $P(\alpha, \beta) = P(\alpha) \cdot P(\beta)$
  - $P(\alpha \vert \beta) = P(\alpha)$
  - $P( \beta \vert  \alpha) = P(\beta)$

- For random variables $X, y, P \vDash X \bot y$ if:
  - $P(X, y) = P(X) \cdot P(y)$
  - $P(X \vert y) = P(X)$
  - $P(Y \vert X) = P(y)$





### Conditional Independence

- For (Sets of) random variables $X, Y, Z$
  - $P \vDash (X \bot y \vert Z ) if$
    - $P(X, y \vert Z) \ P(X \vert Z) \cdot P(y \vert Z)$
    - $P(X \vert y, Z) = P(X \vert Z)$
    - $P(y \vert X, Z) = P(y \vert Z)$
    - $P(X, Y, Z) \propto \phi_1(X, Z) \phi_2(y, Z)$



![image-20190215150806665](/Users/shephexd/Library/Application Support/typora-user-images/image-20190215150806665.png)



There are two coins. One is pair coin, other one is biased coin.

The first toss is depend on the coin you pick.

But second toss is not depend on the first toss, It is only depend on the coin you pick.



- $P \cancel{\vDash} X_1 \bot X_2$
- $P \vDash (X_1 \bot X_2 \vert C)$





### Bayesian Network

- $P(X, y) = P(X)P(y)$, If X, Y are idenpendent.

- $P(X, Y, Z) \propto \phi_1(X, Z) \phi_2(y, Z)$, if $(X \bot y \vert Z)$

  



- Factorization of a distribution $P$ implies independencise that hold in $P$
- If $P$ factorizes over $G$, can we read these independencies from the structure of $G$?





### Flow of influence & d-separation



Definition: $X$ and $y$ are $d$-separated in $G$ given $Z$ if there is no active trail in $G$ between X and y given Z.



$D-sep_G(X, y \vert Z)$



Factorization 



Theorem: If $P$ factorizes over $G$, and $d-sep_G(X, y \vert Z)$, then $P$ satisfies ($X \bot y \vert Z$)


$$
\begin{align}
P(D, I, G, S, L) &= P(D)P(I)P(G \vert D, I) P(S \vert I) P(L \vert G) \\
\\
P(D, S) &= \sum_{G, L, I} P(D)P(I)P(G \vert D, I)P(S \vert I)P(L \vert G)\\
&= \sum_{I}^i P(D)P(I)P(S \vert I) \sum_{G}^g \left( \cancel{P(G \vert D, I)} \sum_L \cancel{P(L \vert G)}\right)\\
&=P(D)(\sum_I P(I)P(S \vert I)) \\
&= \phi_1(D) \phi_2(S) \\
\Rightarrow P \vDash D \bot S
\end{align}
$$




*Any node is d-separated from its non-descendants given its parents.*  
$\rightarrow$ If P factorizes over G, then in P, any variable is independent of its non-descendants given its parents.



Simply, when we know the parents of a node, the other node that is non-descendants of its node is independent. 



### I-maps

- d-separation in G $\Rightarrow$ P satisfies corresponding independence statement

$$
I(G) = \{(X \bot y \vert Z) : \text{d-sep}_G(X, y \vert Z)\}
$$

- Definition: If P satisfies $I(G)$, we say that $G$ is an I-map(independency map) of P



Factorization $\Rightarrow$  If P factorizes over G, then G is an I-map for P.



[^1]: https://www.coursera.org/learn/probabilistic-graphical-models


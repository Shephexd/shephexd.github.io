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

To study about Probabilistic Graphical Model, you should know about basic knowledge such as probability, programming, basic algorithms and data structures.  

If you have already studied about machine learning and optimization, it would be better to understand. 

This model is useful to represent real world problem that can't be reprsented because of complexity.



## Probabilistic Graphical Model

The meaning of Probabilistic Graphical Model is the model structured by probabilistic graph. Then we can tractable the model by following the connected nodes. 



### Probabilitistc(uncertainty)

- Partial knowledege of state of the world
- Noisy observation
- phenomena not covered by our model
- Inherent stochasticity



#### Probability theory

- Declarative representation with clear semantics
- powerful reasoning patterns
- Established learning methods



### Graphical

To represent complex systems based on the graphic theory of computer science



Random variables $X_1, \dots, X_n$

Joint distribution $P(X_1, \dots, X_n)$



for example, in the $N$ binary valued distribution over $2^N$ possible states.



#### Graphical models

**Bayesian networks**

- directed graph



**Markov networks**

- Undirected graph
  (Undirecte graph over 4 random variables A, B, C, D)



#### Graphcial representaiton

- Intuitive & compact data structure
- Efficient reasoning using general-purpose algorithms
- Sparse parameterization
  - Feasible elicitation
  - Learning from data



### Models

- Delclarative representation to represnt world
- domain expert elicitation into model
- data learning for model



[^1]: https://www.coursera.org/learn/probabilistic-graphical-models


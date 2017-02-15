---
layout: post
title: Bayesian probability
published: True
categories: 
- Mathematics
tags:
- Mathematics
- Probability
- Bayesian
---



This post is based on blog[^1] and lecture material in LUT[^2]

Bayesian is one of the important method to classify the values using probability. This theory is based on **conditional probability** and **posteriori probability**.

Features are random variables $x ∈ R^D$ , the so called feature space.
Classification in the most probable class $ω_i$ thus minimizes error probability.

<!--more-->

#### The Bayesian classification rule

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
\\ where\ A\ and\ B\ are\ events\ and\ P(B)\ ≠\ 0.
$$



- Conditional Probability $P(A \vert B)$

  : the probability of event A when B is known (to occur)


- A priori probability $P(w_i)$

  : the probability(proportion) of class $w\_i$ without knowing any measurement.


- Class conditional probability $p(x \vert w_i)$

  : the probability of getting measurement value x when the class is $w_i$


- A posteriori probability P($w_i \vert x$)

  : the probability of class $w_i$ when the measured value is x

  ​

> The uppercase P(•) denotes probability, whereas lowercase p(•) denotes probability density.



## Bayesian probability

There are two kind of method to calculate a probability.

$$
P(w_i|A) = \frac{P(A|w_i)P(w_i)}{P(A)}
$$

### ML(Maximum Likelihood)

$$
MAX(\ P(A|w_i) )
$$

### MAP(Maximum A Posteriori)

$P(w_i)$ is fixed constant value, thus you don't need to care it when you select the maximum value using $MAP$ method.

$$
(w_i|A) = {P(A|w_i)P(w_i)}
$$

$$
MAX(\ P(w_i|A)\ )
$$


**MAP is more accurate than ML because it considers prior probability.** When the prior probability is same. then the result will be same.

**If the prior probabilities are same. (MAP=ML)**

### Minimum error classification
In the two classes, the Bayesian classification decide,


$$
\text{if } P(w_1|x) > P(w_2|x), decide\ w_1,
\\ otherwise,\ decide\ w_2
$$


### The bayesian classification

$$
P(w\_i|x) = \frac{p(x|w_i)P(w_i)}{p(x)}
$$


$$
p(x) = \sum_{i=1}^M{p(x|w_i)P(w_i)}
$$

$$
posterior = \frac{likelihood * prior}{evidence}
$$



#### Probability of error

$$
P(error|x) =  \begin{cases}{P(w_1|x)} & \text{ if } decision\ is\ w_2
\\P(w_2|x) & \text{ if } decision\ is\ w_1
\end{cases}
$$

#### Average error

$$
\int_{-\infty}^{\infty}P(error|x)p(x)dx
$$



#### Simplyfying the classification using MAP

**Probability of evidence p(x) is same for both classes.** 
It is noly a scaling factor and does not affect the decision.

$$
If\ p(x|w_1)P(w_1) \> p(x|w_2)P(w_2),\\ decide\ w_1
\\ otherwise,\ decide\ w_2
$$

If the a prior probabilties are **equal**

$$
If\ p(x|w_1) \> p(x|w_2),\ decide\ w_1
\\ otherwise,\ decide\ w_2
\\
$$

## Multiclass Bayesian Classification

$$
Decide\ w_i\ \text{ if }  P(w_i|x) \> P(w_j|x)\ ∀j ≠ i
$$

Choose the class for which the a posteriori probability is the highest.

The classfication rule divides the feature space into decision $$R\_i$$.

### Minimum risk classfication

**Different(incorrect and correct) decision may have different consequences.**

Loss function $l_i$ for decision $i$ describes how much expected loss is associated with that decision.

$$
l\_i = \sum_{k=1}^{M}{\lambda_{ki}p(w_k|x)}
$$

The decision rule
$$
Decide\ w_i\text{ if } \ l_i \lt l_j ∀j≠i
$$

$$
l\_i = \sum_{k=1}^{M}{\lambda_{ki}p(x|w\_k)P(w\_k)}
$$

#### Two-class case
Expected losses
$$
l_1=\lambda_{11}p(x|w_1)P(w\_1)+\lambda_{21}p(x|w_2)P(w_2)
$$
$$l\_2=\lambda_{12}p(x|w\_1)P(w\_1)+\lambda_{22}p(x|w\_2)P(w\_2)$$


$$
\text{Decide } w_i \text{ if } l_1<l_2 \text{ that is if }\\
(\lambda_{21}-\lambda_{22})p(x|w\_2)P(w\_2) \lt (\lambda_{12}-\lambda_{11})p(x|w\_1)P(w\_1)
$$

- Usually both $(\lambda_{21}-\lambda_{22})$ and $(\lambda_{12}-\lambda_{11})$ are positive, so the decision rule becomes.

  ​
$$
Decide\ w_i\ \text{if } \frac{p(x|w_1)}{p(x|w_2)}\>\frac{P(w_2)\lambda_{21}-\lambda{22}}{P(w_1)\lambda_{12}-\lambda{11}}
$$

### Discriminat functions and decision surface

$$
P (ω_i |x) − P (ω_j |x) = 0
$$
$$
Decide\ ω_i\text{ if  }\ g_i(x) \> g_j(x) ∀j ≠ i
$$



## Normal distribution

## Univariate and multivariate normal distribution

## Bayesian classfier



[^1]: [dark's blog][1], 베이지언 확률(Bayesian Probability)
[^2]: lecture material for Pattern recognition class in the Lappeenranta university of technology.


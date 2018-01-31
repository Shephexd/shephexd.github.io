---
layout: post
title: Bayesian probability
published: True
categories: 
- Mathematics
tags:
- Mathematics
- Probability
---

This post is based on blog[^1] and lecture material in LUT[^2]

Bayesian is one of the important method to classify the values using probability. This theory is based on **conditional probability** and **posteriori probability**.


Features are random variables $x ∈ R^D$ , the so called feature space.
Classification in the most probable class $ω_i$ thus minimizes error probability.

<!--more-->

## Introduction to Bayesian probability

Bayes' theorem describes the probability of event, based on prior knowledge of conditions that might be related to the event.

One of the widely used application of Bayes' rule is Bayesian inference, a particular approach to statistical inference.

### Bayesian' theorem
$$ 
\begin{align}
P(A \vert X)& = \frac{P(B \vert A)P(A)}{P(X)}  \\
& \propto P(X \vert A) P(A) \\
\end{align}\\
\text{where $A$ and $X$ are events and $P(X) \neq 0$}
$$

- $P(A \vert X)$ is the priori probability
- $P(X)$ is the posteriori probability


### example for throwing a coin

```python% matplotlib inline
from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('font', family='Malgun Gothic')

import scipy.stats as stats

figsize(11, 9)


dist = stats.beta
n_trials = [0, 1, 2, 3, 4, 5, 8, 15, 50, 500]
data = stats.bernoulli.rvs(0.5, size=n_trials[-1])
x = np.linspace(0, 1, 100)

for k, N in enumerate(n_trials):
    sx = plt.subplot(len(n_trials) / 2, 2, k + 1)
    plt.xlabel('$p$, probability for front', fontsize=13) \
        if k in [0, len(n_trials) - 1] else None
    plt.setp(sx.get_yticklabels(), visible=False)
    heads = data[:N].sum()
    y = dist.pdf(x, 1 + heads, 1 + N - heads)
    plt.plot(x, y, label='%dth trial, \n front: %d' % (N, heads))
    plt.fill_between(x, 0, y, color='#348ABD', alpha=0.4)
    plt.vlines(0.5, 0, 4, color='k', linestyles='--', lw=1)
    #plt.vlines(colors=,data=,hold=,label=,linestyles=,x=,ymax=,ymin=,)
    leg = plt.legend()
    leg.get_frame().set_alpha(0.4)
    plt.autoscale(tight=True)
plt.suptitle('Updated posterior probability', y=1.02, fontsize=14)

plt.tight_layout()
```


## Probability distribution function

Probability distribution function is depend of the type of random variable.

$Z$ is discrete type
: A random sample can be selected from the list of specific values.

$Z$ is exponential type
: A random sample can be selected from the range of values

$Z$ is mixed type
: The random value can be selected from the discrete type and exponential type distribution.


### discrete case
if the random variable $Z$ is discrete, the distribution is called *Probability mass function*.

The most useful probability mass function is *Poisson-distribution*.
$$
P(Z = k) = \frac{\lambda^k e ^{-\lambda}}{k!}
$$

- $\lambda$ is a parameter to decide the shape of distribution

more increase the $\lambda$, probability is assigned to bigger number, 
more decrease the $\lambda$, probability is assigned to smaller number.

$$
Z \sim \text{Poi($\lambda$)}\\
E[Z \vert \lambda] = \lambda
$$

### Continuous case
If the random variable $Z$ with a parameter $\lambda$ follows exponential distribution, $Z$ is *exponential*.






#### The Bayesian classification rule


$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$  
where $A$ and $B$ are events and $P(B) \neq 0$



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


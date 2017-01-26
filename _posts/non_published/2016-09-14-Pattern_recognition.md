---
layout: post
published: False
title: Pattern Recognition
categories:
- Machine learning
tags:
- Data analysis
- Big data
---

Pattern recognition is a basic concept of machine learning. It w



It is for  contents in my course called "Pattern Recognition" in Lappeenranta University technology. It starts from the definition of Pattern Recognition to Making a model. I hope it can be useful for students and engineers.



<!--more-->

## What is Pattern Recognition?

Pattern recognition is a branch of machine learning that focuses on the **recognition of patterns** and **regularities in data**, although it is in some cases considered to be nearly synonymous with machine learning. - [Wikipedia]

>> Related fields Machine learning, Data mining, Application areas

### Pattern recognition system
Generally, The pattern recognition system gets data from a sensor. 
Using this data, the system extract features and classify to decide a decision.

$$
R^M -> R^D, D<<M
$$

### The process of pattern recognition

1. Data collection
    - How much data is needed?
    - What is measured?
2. Feature selection
    - Which features provide good class separability and generalization?
    - Do we need pre-processing of features?
3. Classifier design
    - Which classifier works the best?
4. System evaluation
    - How should the performance be measured?

### Important concepts
- class, model
- a feature, a feature vector
- feature extraction
- training patterns, training samples, training data
- validation patterns, validating samples, validation data
- testing patterns, testing samples, test data
- cost, risk
- classifier
- decision boundary
- generalization vs over-fitting
- supervised vs unsupervised learning
- classification vs regression

## Taxonomy of learning methods
Basically, There are two kinds of methods recognizing patterns. **Supervised learning**,  **Unsupervised learning** and **Reinforcement learning**.

### Supervised learning
Supervised learning needs training sets to make an objective function,a model. Training data consist of  observation values. each training data has results which are correct or not. After training, a model can be made. And it can predict which is correct or not. Using this process, we can get the accuracy of the model. Basically, it is used in Machine learning.

**finding a mapping from future space into space with minimal error.**

- Linear regression
- Handwriting recognition
- Fraud detection
- spam filtering

### Unsupervised learning
Unsupervised learning doesn't need training data. Input data is latent values. Also, no desired output exists. In almost case, unsupervised learning makes groups which have some similar characters.

**cluster patterns into groups with minimal within-group differences and maximal between-group differences.**

**cluster patterns for groups with minimal different features between groups**

- clustering, grouping with customers who have similar characters.

### Reinforcement learning
Reinforcement learning is an area of machine learning inspired by behaviorist psychology, concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward. [Wikipedia](https://en.wikipedia.org/wiki/Reinforcement_learning) 

**classes are not known before training**

In a specific environment,  the agent recognizes current status and decide to make maximum benefit or minimum cost.

It is similar to the game theory or Dynamic planning.


## methodological taxonomy

### Geometric
Feature space is divided into parts where each part represents a class.

### Statistical
Features are random variables with statistical properties.
One of example is the bayesian probability. 

### Neural-networks
It is based on "Black-Box"methods for constructing a transform/mapping from features to classes.
It is similar to our brain work. There are input and output data, however, It is hard to explain why it is.



## The algorithms in Pattern Recognition

Classification
: Classification is used to classify data. There three kinds of color dots. green, blue, red. In this case, the line can classify two kinds of colors. However, two blue dots exists. They are outliers in this graph.

Clustering
: It makes groups have similar characters. In this graph, each dot is pointed in x-axis and y-axis. It is one of unsupervised learning to make groups.

Regression
: It is the linear regression is one of supervised learning. Using this model, We can predict the value if another item is added. The line which has ordinary least square is made.


## feature selection and processing

Because a large number of features requires a lot of training data for the data to be representative, we need to select some important features.

Also, some features are dependent on each other.

### The reason for feature selection
Separability: features very different for objects **in different classes**(inter-class variation)
Generalization: features similar for objects **in the same class** (intra-class variation)

true positive (TP)
: eqv. with hit

true negative (TN)
: eqv. with correct rejection

false positive (FP)
: eqv. with false alarm, Type I error

false negative (FN)
: eqv. with miss, Type II error

#### True positive rate TPR(Sensitivity)
$$
\frac{FP}{TN+FP} = Sensitivity
$$

#### True positive rate FPR(Fall-out)
$$
\frac{FP}{TN+FP} =1 -  Specificity
$$

#### Specificity
$$
Specificity = \frac{TN}{TN+FP}
$$


feature
: a feature is an individual measurable property of a phenomenon being observed.

training data
: the data using to find out predictive relationships and to learn model(objective function).

test data
: the data using to test the learned model from training data to assess the relationship.


decision boundary
: Decision boundary is a saddle point to divide classes When values are on the graph, The classifier will separate one side of the decision boundary as some values in one class and another side as other values in other class.
Sometimes, the values near Decision boundaries are ambiguous to separate to one class.

Separability
: features very different for objects in different classes (cf. inter-class variation).

Generalization
: features similar for objects in the same class (cf. intra-class variation).
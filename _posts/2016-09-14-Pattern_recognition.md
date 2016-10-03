---
layout: post
published: True
title: Pattern Recognition
categories:
- Machine learning
tags:
- Data analysis
- Big data
---

It is for arrangement of contents in my course called "Pattern Recognition" in Lappeenranta University technology. It starts from the definition of Pattern Recognition to Making a model. I hope it can be useful for students and engineers.

## What is Pattern Recognition?
Pattern recognition is a branch of machine learning that focuses on the **recognition of patterns** and **regularities in data**, although it is in some cases considered to be nearly synonymous with machine learning. - [Wikipedia](https://en.wikipedia.org/wiki/Pattern_recognition)


### Supervised learning and Unsupervised learning
Basically, There are two kinds of methods recognizing patterns. **Supervised learning** and **Unsupervised learning**.

#### Supervised learning
Supervised learning needs training sets to make an objective function,a model. Training data consist of  observation values. each training data has results which are correct or not. After training, a model can be made. And it can predict which is correct or not. Using this process, we can get the accuracy of the model. Basically, it is used in Machine learning.

- Linear regression
- Handwriting recognition
- Fraud detection
-  spam filtering

#### Unsupervised learning
Unsupervised learning doesn't need training data. Input data is latent values. Also, no desired output exists. In almost case, unsupervised learning makes groups which have some similar characters.

- clustering, grouping with customers who have similar characters.

<!--more-->

### The algorithms in Pattern Recognition

Classification
: Classification is used to classify data. There three kinds of color dots. green, blue, red. In this case, the line can classify two kinds of colors. However, two blue dots exists. They are outliers in this graph.

Clustering
: It makes groups have similar characters. In this graph, each dot is pointed in x-axis and y-axis. It is one of unsupervised learning to make groups.

Regression
: It is the linear regression is one of supervised learning. Using this model, We can predict the value if another item is added. The line which has ordinary least square is made.

### Pattern Recognition System Design
1. Data collection
2. Feature selection
3. Classifier design
4. System evaluation

## feature selection and feature processing


### Definition
feature
: a feature is an individual measurable property of a phenomenon being observed.

training data
: the data using to find out predictive relationships and to learn model(objective function).

test data
: the data using to test the learned model from traing data to assess the relationship.


decision boundary
: Decision boundary is one of boundary between classes. When values are on the graph, The classfier will seperate one side of the decison boundary as some values in one class and other side as other values in other class.
Sometime, the values near Decision boundaries are ambigious to seperate to one class.

Separability
: features very different for objects in different classes (cf. inter-class variation).

Generalization
: features similar for objects in the same class (cf. intra-class variation).


### Problem 2

The class means for Feature 1 are 3(Class1) and 7(Class2).

Feature 1	| means  | STDs
----------|--------|-----
Class 1 	| 3		  | 2
Class 2   | 7 	  | 1

Feature 2	| means  | STDs
----------|--------|-------
Class 1 	| 5		  | 0.2
Class 2   | 6 	  | 0.2

$$FDR_1  = \frac{(µ_1 - µ_2)^2}{( σ_1^2 + σ_2^2)} $$

#### Fisher Discriminant ratio(FDR)

$$Feature 1: \frac{(3 - 7)^2}{(2^2 + 1^2)} =  \frac{16}{5} = 3.2$$

$$Feature 2 \frac{(5 - 6)^2}{0.2^2 + 0.2^2} = \frac{1}{0.08} = 12.5$$

**Feature2 is more bigger than Feature 1, thus It is more useful in Classification**

### Problem 3


![Feature 1 vs Feature 2](/assets/post_images/Pattern_recognition/feature_graph.png)


### Problem 4

4. Feature normalization

$$
\begin{pmatrix}        0 \\        0 \\\end{pmatrix} ,
\begin{pmatrix}        0 \\        1 \\\end{pmatrix} , \begin{pmatrix}        2 \\        1 \\\end{pmatrix} ,
\begin{pmatrix}        2 \\        0 \\\end{pmatrix} ,
\begin{pmatrix}        1 \\        10 \\\end{pmatrix}
$$

1. Min-max normalization
x 0 - 2  
y 0 - 10
x1 =

$$ = \frac{ x_{ik} - x^{min}_k }{x^{max}_k - x^{min}_k}
$$

#### Calculation  
1. Min-Max normalization

$$
\begin{pmatrix}        0 \\        0 \\\end{pmatrix} ,
\begin{pmatrix}        0 \\        0.1 \\\end{pmatrix} , \begin{pmatrix}        1 \\        0.1 \\\end{pmatrix} ,
\begin{pmatrix}        1 \\        0 \\\end{pmatrix} ,
\begin{pmatrix}        0.5 \\        1 \\\end{pmatrix}
$$

2. Mean-variance normalization(standardization)

```matlab
N = length(x);
o = sqrt(sum(((x - mean(x)).^2)) / (N-1));

res = (x - mean(x)) / o;
```

$$ µ_x = 1, µ_y = 2.4 $$
$$σ_x = 1, σ_y = 4.2778$$
$$xyˆ_{ik} = [-1,    -1,     1,     1 ,    0]$$
$$yˆ_{ik} = [-0.5844   -0.3506   -0.3506   -0.5844    1.7532]$$


$$
\begin{pmatrix}        -1 \\        -0.5844 \\\end{pmatrix} ,
\begin{pmatrix}        -1 \\        -0.3506 \\\end{pmatrix} , \begin{pmatrix}        1 \\        -0.3506 \\\end{pmatrix} ,
\begin{pmatrix}        1 \\        -0.5844 \\\end{pmatrix} ,
\begin{pmatrix}        0 \\        1.7532 \\\end{pmatrix}
$$


3. Softmax-scaling

```matlab

y_ik = (x-mean(x)) / o;
res = (1 + exp(y_ik)).^(-1);

```

$$ t_{ik} = \frac{1}{1+ e^{-s_{ik}}}$$


$$
\begin{pmatrix}        0.7311 \\        0.6367 \\\end{pmatrix} ,
\begin{pmatrix}        0.7311 \\        0.5811 \\\end{pmatrix} , \begin{pmatrix}         0.5811 \\        0.1 \\\end{pmatrix} ,
\begin{pmatrix}        0.2689  \\        0.6367 \\\end{pmatrix} ,
\begin{pmatrix}        0.5000 \\        0.1447 \\\end{pmatrix}
$$

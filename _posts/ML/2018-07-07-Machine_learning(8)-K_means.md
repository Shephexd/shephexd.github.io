---
title: Machine learning(8) - K-means Clustering
layout: post
categories:
- Machine learning
tags:
- Machine learning
- Tensorflow
- Clustering
typora-root-url: /Users/shephexd/Documents/github/pages/
---



Previous post is about the supervised learning trained from labeled data. But many cases, we afford to have labeled data set to train.

The other way to make your model for your data is `Unsupervised learning` trained from unlabeled data.



<!--more-->



## Unsupervised learning

In the supervised learning training set is constructed with labeled data, $y$.

But Unsupervised learning can be trained by only training data without label.



### Examples

Some cases like below It is natural the data is unlabeled.

-   Market segmentation
-   Social network analysis
-   Organize computing clusters
-   Astronomical data analysis



## K-means algorithm

`K-means` clustering is the one of the clustering algorithms. It create some centroid point to make clusters for your unlabeled data.

The parameters $K$ is how many centroids you want. Then your train data will be labeled as one of clusters included into a centroid.



### Learning process

To make $K$ clusters for your data, You should have a `hypothesis` for your data set.

The `hypothesis` is that the data can be generated from $K$  centroids with some noises.



#### Input

-   K (number of clusters)
-   Training set ($\{x^{(1)}, x^{(2)}, \dots, x^{(m)} \}, x^{(i)} \in R^n$})



#### Initialize

-   Should have $K \lt m $
-   Randomly pick $K$ training examples.  
    set $\mu_1, \dots , \mu_K$ equal to these $K$ examples.



#### Learning

1.  Randomly initialize $K$ cluster centroids $\mu_1, \mu_2, \dots , \mu_k (\mu_k \in R^{n})$
2.  Find the proper centroid by updating centroid points


$$
\begin{align}
\text{Repleat} &&\text{Cluster assignment}\\
& \text{for $i=1$ to $m$}\\
& c^{(i)} := \text{index(from $1$ to $K$) of cluster, centroid closet to $x^{(i)}$} \\
\\
\text{Repleat} &&\text{Updating Centroids}\\
& \text{for $i=1$ to $K$}\\
& \mu_K := \text{average(mean) of points assigned to cluster $K$}
\end{align}
$$


#### Optimization

Your goal is selecting the $K$ clusters that make your cost($J(c^{(i)}, \dots, c^{(m)}, \mu_1, \dots, \mu_K)$) minimize.


$$
\begin{align}
&c^{(i)} = \text{index of cluster($1,2,\dots, K$ to which example $x^{(i)}$ is currently assigned )}\\
&\mu_K = \text{cluster cetroid $K$ ($\mu_K \in R^n$)}\\
&\mu_c^{(i)} = \text{cluster centroid of culster to which example $x^{(i)}$ has been assigned}\\
\\
&J(c^{(i)}, \dots, c^{(m)}, \mu_1, \dots, \mu_K) = \frac{1}{m}\sum^{m}_{i=1} \lVert x^{(i)} - \mu_c^{(i)} \rVert^2
\end{align}
$$




## How to evaluate

As much as you increase the number of clusters,$K$, the cost is much smaller.

But it is `overfitting` the cluster will be just like your training data.



One of the way to select proper $K$ is find the `elbow point`.

 As K increases, the amount of cost reduction will decrease. Find $K$ value where the amount of decreasing cost decreases smoothly.



But, Some case the amount of decreasing cost is always similar as $K$ increases.

In this case you should select proper $K$ for your $K$- means algorithm depending on your purpose.

For instance, If your purpose is dividing your customer groups into 3 levels, You will select $K$ as 3.



## Code example

I wrote the implemented codes for `K-means` with `numpy` .



### K-means with numpy

```python
import numpy as np


def get_random_sample(n_samples):
    samples = []
    for i in range(n_samples):
        if np.random.random() > 0.5:
            samples.append([np.random.normal(0.0, 0.6), np.random.normal(0.0, 0.9)])
        else:
            samples.append([np.random.normal(3.0, 0.5), np.random.normal(2.0, 0.8)])
    return np.array(samples)

def get_new_centroid(samples):
    return np.mean(samples)

def index_of_min_centroid(distances):
    return np.argmin(distances)

def distance_between_twopoint(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

def cost_function(samples, c, centroids):
    J = 0
    for idx, sample in enumerate(samples):
        index_of_centroid = int(c[idx])
        J += np.sum((sample - centroids[index_of_centroid])**2)
    return J/len(samples)

X = get_random_sample(2000)
K = 2
c = np.zeros(len(X))

prev_cost = 0


for step in range(250):
    for idx, sample in enumerate(X):
        candi_distance = [distance_between_twopoint(sample, centroid) for centroid in centroids]
        c[idx] = index_of_min_centroid(candi_distance)

    for idx in range(K):
        samples_assigned_into_centroid = X[c==idx]
        centroids[idx] = get_new_centroid(samples_assigned_into_centroid)
    
    cost = cost_function(X, c, centroids)
        
    if step % 50 == 0:
        check_points.append(centroids.copy())
        print(cost)

        if prev_cost == cost:
            break

        prev_cost = cost
```



![k-means result](/assets/images/articles/ML/kmeans_result.png)


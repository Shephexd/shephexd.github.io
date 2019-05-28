---
layout: post
title: Deep learning(14) - Introduction to word embedding
published: True
categories:
- Deep learning
tags:
- Machine learning
- Neural network
- Deep learning
- Python
- Tensorflow
typora-root-url: /Users/shephexd/Documents/github/pages/
---

The one of the difference in NLP(Natural Langugage Processing) comparing to other tasks is the vector representation.



For example, RGB Image can be represented like $28 \times 28 \times 3$.



Then how about the text?

Can you represent the sentences like "I eat an apple", "I ate an apple"?



In this post, i will introduce the way to represent word to embed into vector space.

Then, Sentence can be represented by the combination of words.

<!--more-->



> this post is based on the blog post[^1] and lecture[^2]



## Word representation

To processs sentences on machine, we need to convert sentence to vector.



### one-hot representation

One of wiedly used word representation is `one-hot representation` labelling as a binary vector that is all zero except the index of integer.




$$
Man = \begin{bmatrix}
0 \\ 0 \\0 \\ \vdots \\ 1\\ 0 \\ \vdots  \\0
\end{bmatrix}

Woman = \begin{bmatrix}
0 \\ 0 \\0 \\ \vdots \\0 \\ 1\\ \vdots  \\0
\end{bmatrix}

Apple = \begin{bmatrix}
1 \\ 0 \\0 \\ \vdots \\0 \\ 0\\ \vdots  \\0
\end{bmatrix}


Orange = \begin{bmatrix}
0 \\ 1 \\0 \\ \vdots \\0 \\ 0\\ \vdots  \\0
\end{bmatrix}
$$




Using this representation, you can distingiush the words but can't calculate distance between the words. The word distance between `man` and `women` might be closer than the word distance between `women` and `orange`.



## Word embedding

The idea of `word embedding` is to featurize the word vector by word embedding.

The word can be represended with the featurized factors. One word can be used with some other words having similar meaning.


$$
\begin{array}{r|rr} \text{}
               & man & woman & king & queen & apple & orange \\ \hline
\text{Gender}       &  -1	& 1 	   & -0.96 	& 0.97 &0.00 & 0.01 \\
\text{Royal}	 &  0.01	   & 0.02	 & 0.93	  & 0.95 & 0.00 & 0.01 \\
\text{Age}		&  0.03 &	0.00	& 0.7 & 0.69 & 0.03 &-0.02 \\
\vdots  &\vdots & \vdots &\vdots& \vdots & \vdots \\
\text{food}				&  0.09 & 0.01 & 0.01 & 0.02 & 0.95 &0.96 \\
\vdots  &\vdots & \vdots &\vdots& \vdots & \vdots \\
\end{array}
$$


Normally `T-sne` is one of the effective way to visualize the embedded words, $300D \to 2D$





### Transfer learning and word embeddings

1. Learn word embeddings from large text corpus(or Download pre-trained embedding online)
2. Transfer embedding to new task with smaller training set
3. (Optional) Continue to finetune the word embedding with new data set.



### Analogies

The distance between man and woman is similar to the distance between king and queen.




$$
\begin{align}
e_{man} - e_{woman} =& \begin{bmatrix}
-2 \\ 0 \\ \\ \vdots \\0
\end{bmatrix}
\\
e_{king} - e_{queen} =& \begin{bmatrix}
-2 \\ 0 \\ \\ \vdots \\0
\end{bmatrix}
\end{align}
$$



Lingustic regularities in continuous space word representations


$e_{man} - e_{woman} \approx e_{king} - e_{?}$




$$
sim(u, v) = u^Tv
$$




### Embedding matrix

with 10,000 vocabulary size

Embedding for $j$ th word.



$$
E \cdot O_{j} = 
\begin{bmatrix}
a_{1,1}& \dots & a_{1, 10000}\\
a_{2,1} &\dots  & a_{2, 10000}\\
&\vdots \\
a_{j,1} &\dots  & a_{j, 10000}\\
&\vdots \\
a_{300,1} & \dots  & a_{300, 10000}\\
\end{bmatrix}
\begin{bmatrix}
0 \\ 0 \\ \vdots  \\1 \\  \vdots \\0
\end{bmatrix}
= e_{j}
$$






But, multiplying with one-hot vector is not efficient.

In practice, use specialized function to look up an embedding instead of multiplying with one-hot vector.



### Language model

How can we use `word embedding` for language model?


$$
\begin{align}
&\text{I} 		\to 	O_{4563} \to E \to e_{4563}\\
&\text{want} 	\to 	O_{863} \to E \to e_{863}\\
&\text{a} 		\to 	O_{136} \to E \to e_{136}\\
&\text{glass} 	\to  O_{34} \to E \to e_{34}\\
&\text{of} 		\to 	O_{3584} \to E \to e_{3584}\\
&\text{orange} 	\to 	O_{6164} \to E \to e_{6164}\\
&\text{juice} 	\to 	O_{6257} \to E  \to e_{6257}\\
\\


&e_k = 
\begin{bmatrix}
e_{4563} & 
e_{863} &
e_{136} & 
e_{34} &
e_{3584} &
e_{6164} &
e_{6257} 
\end{bmatrix}
\\
&\text{softmax}(e_k) = \hat{y}

\end{align}
$$

the meaning of $\hat y$ is probabiltiy that if you randomly pick a word nearby "ants", that it is a "car".





## Word2vec

Let's see the more detail for word embedding with `word2vec` model.



There are two ways to train `word2vec`.

- CBOW
- Skip gram



Before explaining word2vec for multiple context words, let's see the detail for one word context.



### One-word context



- $V$: Vocabulary size
- $N$: Hidden layer size
- $h$: hidden layer matrix
- $W$: input to hidden weight matrix
- $v_{w_I}$: row $i$ of $W$, the vector representation of the input word $w_I$
- $v^\prime_{w_j}$: $j$-th column of the matrix $W^\prime$
- $W^\prime $: hidden to output weight matrix
- $E$: loss function
- $EH$: $N$-dim vector, it is the sum of the output vectors of all words in vocabulary, weighted by their prediction error
- $e$: prediction error



####  Forward propagation



$$
\begin{align}
& h = W^T x = W^T_{(k, :)} = v^T_{w_I} \\

& u_j = {v^\prime_{w_j}}^T h \\
& y_j = softmax(u_j)\\
\\
& x_k \xrightarrow[{W_{V \times N}}]{} 
h_i  \xrightarrow[W^{\prime}_{N \times V}]{} 
u_j \xrightarrow[softmax]{} y_j\\

\\
& P(w_J \vert w_I) = y_j = 
\frac{
\exp({v^{\prime}_{w_j}}^Tv_{w_I})
}
{
    \sum_{j^\prime=1}^V \exp\left( {v^\prime_{w_{j^\prime}}}^T v_{w_I} \right)
}
\end{align}
$$



#### Backwoard propagation



##### Define loss function for derivation



$$
\begin{align}
\max p(w_O \vert W_I) = & \max y_{j^*} \\
= & \max \log y_j^* \\
= & u_j^* - \log \sum_{j^\prime = 1}^V \exp(u_{j^\prime}) := -E
\\
& j^* : \text{the index of the actual output word in the output layer}
\end{align}
$$



##### Update $W^\prime$



$$
\begin{align}
& t_j=
	\begin{cases}
	1 & \text{where }j = j^* \\
	0  &\text{otherwise} \\
	\end{cases}\\
\\
&\frac{\part E}{\part u_j} = y_j - t_j := e_j\\
\\
& \frac{\part E}{\part w^\prime_{ij}} = \frac{\part E}{\part u_j} \cdot \frac{\part u_j}{\part w^\prime_{ij}} = e_j \cdot h_i \\
\\
& {w^\prime_{ij}}^{(new)} =  {w^\prime_{ij}}^{(old)} - \eta \cdot e_j \cdot h_{(i,:)} \\
&or \\
& {v^\prime_{w_{j}}}^{(new)} =  {v^\prime_{w_j}}^{(old)} - \eta \cdot e_j \cdot h &\text{for }j=1,2,\dots, V \\

\end{align}
$$



##### Update $W$



$$
\begin{align}
& \frac{\part E}{\part h_i} = \sum_{j=1}^V \frac{\part E}{\part u_j} \cdot \frac{\part u_j}{\part h_i}= \sum_{j=1}^V e_j \cdot w_{ij}^\prime := EH_i \\
& \frac{\part E}{\part w_{ki}} = \frac{\part E}{\part h_i} \cdot \frac{\part h_i}{\part w_{ki}} = EH_i \cdot x_k\\

& \frac{\part E}{\part W} = x \otimes EH = x \cdot EH^T \\
& v^{(new)}_{w_I} = v^{(old)}_{w_I} - \eta EH^T
\end{align}
$$





### CBOW(Continuous Bag-of-Word) Model



`The` `quick` [**brown**] `fox` `jumps` over the lazy dog.

With 2 window size, last four words on left and right are selected for target word.



| Context | Target |
| :-----: | :----: |
|   the   | brown  |
|  quick  | brown  |
|   fox   | brown  |
|  jumps  | brown  |







### Skip gram Model



When the sentence is below and selected context word is `brown`.

`The` `quick` [**brown**] `fox` `jumps` over the lazy dog.

With 2 window size, last four words on left and right are selected for target word.



| Context | Target |
| :-----: | :----: |
|  brown  |  the   |
|  brown  | quick  |
|  brown  |  fox   |
|  brown  | Jumps  |



The goal of `skip-gram` is to embed from selected context word to nearby being two words behind and to words ahead.



#### Model

- Vocabulary size: 10K
- Embedding size: 300




$$
O_c \to E \to e_c \to O \to \hat{y}\\
\theta_t = \text{parameter associate with output }t\\
p(t \vert c) = \frac{e^{\theta_t^Te_c}}{\sum_{j=1}^{10,000} e^{e_j^Te_c}} \\
L(\hat y, y)  = -\sum_{i=1}^{10,000} y_i \log{\hat y_i}
$$




Normally, we can use `softmax` for loss function to tarin emebedding matrix. 


$$
L(\hat y, y)  = -\sum_{i=1}^{10,000} y_i \log{\hat y_i}
$$


But, the problem with `softmax` classification is the value is small



## Optimization

With huge bag of words, the training process is too slow because the pair of words only can be used to train   self.



### Negative sampling

Negative sampling is the idea to train the other words that are not closer.

Sampling negative case is helpful to improve training time by updating the words that are not 



| Context | Word  | is_target |
| :-----: | :---: | :-------: |
| orange  | juice |     1     |
| orange  | king  |     0     |
| orange  |  fox  |     0     |
| orange  | Jumps |     0     |


If we don't apply negative sampling, we only can train the pair, (orange, juice). But with negative sampling we can train other pairs, `[(orange, king),(orange, fox),(orange, jumps)]` with negative label. It increases the distance between `orange` and the words, `king`, `fox`, `jumps`.



### Subsampling

The most frequent words like (`the`, `and`, `or`  … ) may not helpful to train the context of words.

Because these words can be appeared in most of sentences. 


$$
P(w_i) = (\frac{z(w_i)}{\epsilon} + 1) \times \frac{\epsilon}{z(w_i)}
$$


- $z(w_i)$ is the frequency of the word, $w_i$
- $\epsilon$ can be $0.01, 0.001, \dots$



$P(w_i)$ means the probability to sample a word, $w_i$.

In code implementation, We will pick up the word, $w_i$, when the random probability is bigger than $P(w_i)$



## Conclusion

In this post is about the basic idea of word vector representation.

The goal of word vector is the word can be represented as a vector and we expect the similar vectors can have similar meaning.

And embedding word can be a proper solution to avoid curse of dimensions.



[^1]: http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
[^2]: https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning
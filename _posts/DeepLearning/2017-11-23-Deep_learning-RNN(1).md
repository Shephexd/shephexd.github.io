---
layout: post
title: Deep learning RNN(1) - Intro
published: True
categories:
- Deep learning
tags:
- Machine learning
- Neural network
- Deep learning
- Python
- Tensorflow
---



z<!--more-->



## The difference between NN vs RNN



![nn_vs_rnn](/Users/shephexd/Documents/github/pages/assets/post_images/DeepLearning/nn_vs_rnn.png)





$$
P(w_1,\dots, w_m) = \prod^m_{i=1}P(w_i\vert w_1, \dots, w_{i-1})
$$



## Concept of RNN



Recurrent Neural Networks(RNN) are widely used for processing sequential data.
$$
\begin{align}
Y_{(t)} &= \phi(X_{(t)} \cdot W_x + Y_{(t-1)} \cdot W_y + b)\\
& =\phi (  
\begin{bmatrix}
X_{(t)} & Y_{(t-1)}
\end{bmatrix}
\cdot
W + b
 \text{ with } 
\begin{bmatrix}
W_{x} \\ W_{y}
\end{bmatrix}

\end{align}
$$


- $Y_{(t)}$ is an $m \times n_{neurons}$
- $X_{(t)}$ is an $m \times n_{inputs}$
- $W_x$ is an $ n_{inputs}\times n_{neurons}$
- $W_y$ is an $m_{neurons} \times n_{neurons}$
- $b$ size $n_{neurons}$



- $\Theta$ is shared
- Next node has a parameter of previous node





### Memory Cells

the output of a recurrent neuron at time step $t$ is a function of all the inputs from previous time steps, so it has a form of *memory*.



### Input and Output Sequences

RNN can simultaneously take a sequence of inputs and produce a sequence of outputs.



### Example






## Unfolding Computational Graph



### Classic form of a dynamical system




$$
s^{(t)} = f(s^{(t-1)};\theta)
$$



#### Example

$$
\begin{align}
s^{(3)} &= f(s^{(2)};\theta)\\
&=f(f(s^{(t=1)};\theta)
;\theta)\\
\end{align}
$$



### Dynaimical system with external signal $x^{(t)}$


$$
s^{(t)} = f(s^{(t-1)},x^{(t)};\theta)
$$


### RNN

$$
\begin{align}
h^{(t)}& =g^{(t)}(x^{(t)}, x^{(t-1)}, x^{(t-2)}, \dots, x^{(2)}, x^{(1)}) \\\
&=f(h^{(t-1)}, x^{(t)}; \theta) \\
\end{align}
$$





## Limitation of RNN



### Vanishing/exploding gradient problem

When you try to train RNN on long sequences, you suffer 



#### Solution

1. good parameter initialization
2. nonsaturating activation functions(e.g. *ReLU*)
3. Batch Normalization
4. Gradient Clipping
5. Faster optimizer





## LSTM(Long Short-Term Memory)


$$
\begin{align}
i_{(t)} &= \sigma(W_{xi}^T \cdot x_{(t)} + W_{hi}^T \cdot h_{(t-1)} + b_i) \\
f_{(t)} &= \sigma(W_{xf}^T \cdot x_{(t)} + W_{hf}^T \cdot h_{(t-1)} + b_f) \\
o_{(t)} &= \sigma(W_{xo}^T \cdot x_{(t)} + W_{ho}^T \cdot h_{(t-1)} + b_o) \\
g_{(t)}& = \tanh(W_{xg}^T \cdot x_{(t)} + W_{hg}^T \cdot h_{(t-1)} + b_g)\\

c_{(t)} &= f_{(t)} \otimes c_{(t-1)} + i_{(t)} \otimes g_{(t)}\\
y_{(t)} &= h_{(t)} = o_{(t)} \otimes \tanh(c_{(t)})
\end{align}
$$


- $c_{(t)}$ is for long-term memory status.  

  $c_{(t)}$ will drop some memories through the *forget gate*, and get some memories through the *input gate*.

- $h_{(t)}$ is for short-term memory status.



The output of $\tanh$ is between -1 and 1. So, the output value can be remebered for short-time.



The output of $\sigma$ is between 0 and 1. So, the output value can be decreased through the time step.



## GRU(Gated Recurrent Unit)

The GRU cell is a simplified version of the **LSTM cell**, and it seems to perform just as well.



- Both state vectors are merged into a single vector $h_{(t)}$


$$
\begin{align}
z_{(t)} &= \sigma(W_{xz}^T \cdot x_{(t)} + W_{hz}^T \cdot h_{(t-1)} + b_z) \\
r_{(t)} &= \sigma(W_{xr}^T \cdot x_{(t)} + W_{(hr)}^T \cdot h_{(t-1)} + b_r)\\
g_{(t)} &= \tanh(W_{xg}^T \cdot x_{(t)} + W_{hg}^T \cdot (r_{(t)} \otimes h_{(t-1)}) + b_g) \\
h_{(t)} &= (1-z_{(t)}) \otimes h_{(t-1)} + z_{(t)} \otimes g_{(t)}
\end{align}
$$

- Both state vectors are merged into a single vector h_{(t)}.
- A single gate controller controls both the forget gate and the input gate. If the gate controller output is 1, the input gate is one and the forget gate is closed. If it output is 0.
- There is no output gate. the full state vector is output at every time step.





---
layout: post
title: Reinforcement learning(2) - Q-learning(Nondeterministic)
published: True
categories: 
- Reinforcement learning
tags:
- Machine learning
- Tensorflow
- Reinforcement learning
- OpenAI gym
---

Realworld is not deterministic environment. We can't sure our decision has same result in same environment. It is called stochastic environment.



### Non-deterministic


$$
\hat Q(s,a) \leftarrow r + \alpha[\gamma \max_{a'} \hat Q (s',a')]\\
\alpha:\text{learning ratio}\\
\gamma:\text{discount ratio}\\
\hat Q : \text{approximation value for }Q\\
$$

$$
Q(s,a) \leftarrow (1- \alpha)Q(s,a) + \alpha[r+\gamma \max_{a'} \hat Q (s',a')]
$$




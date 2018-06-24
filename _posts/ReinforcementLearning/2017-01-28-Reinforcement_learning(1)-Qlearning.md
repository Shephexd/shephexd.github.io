---
layout: post
title: Reinforcement learning(1) - Q-learning
published: True
categories: 
- Reinforcement learning
tags:
- Machine learning
- Tensorflow
- Reinforcement learning
- OpenAI gym
---





인공 지능에 한 분야로 컴퓨터가 스스로 현재 상태를 인지하고, 선택가능한 행동들 중 보상이 가장 크게 예측되는 행동을 하게 된다.



OpenAI GYM과 Tensorflow환경에서  Q-learning과 Dq learning 등의 알고리즘의 구현을 실습해보려 합니다.



Sung Kim교수님의 [인터넷 강의](https://www.youtube.com/watch?v=dZ4vw6v3LcA&index=1&list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG)를 수업을 참고하여 작성하였습니다.



<!--more-->

## What is the reinforcement learning?



Basic idea: We can learn from **past experiences**.



#### Objects

- Environment
- Actor



#### Basic rule

1. Actor's action can change the environment.
2. After a action, observation(state) is changed.
3. After actions, Actor can get the reward.



## Environment

- Python

- Tensorflow

  `sudo apt-get install python-pip python-dev`

  `pip install tensorflow` or `pip install tensorflow-gpu`

- OpenAI Gym

  `sudo apt install cmake`

  `apt-get install lib g-dev`

  `sudo -H pip install gym`

  `sudo -H pip install gym[atari]`



### OpenAI GYM

A toolkit for developing and comparing reinforcement learning algorithms. It supports teaching agents everything from walking to playing games like Pong or Go. - [OpenAI](https://gym.openai.com)



```python
import gym
env = gym.make("FrozenLake-v0")
observation = env.reset() #environment reset
for _ in range(1000):
    env.render() # Show the environment
    action = env.action_space.sample() # Your agent here
    observation, reward, done, info = env.step(action)
    # observation, reward
    # doen whether the game is over.
    # info additional info
```



```python
import gym
from gym.envs.registration import register
import sys,tty,termios

env = gym
```



## Q-Learning

Agent don't know which action is right and good for self. If agent can ask the action to someone, it will be helpful. Q can answer this question.



### Q function

Also, it can be called "state action value function"

Q can reply the question from the agent, In the present state, do any action. you might get the reward(quality)



#### Input parameter

- state
- action

#### Output parameter

- Quality(reward)



### Policy

$Q$(state, action)



#### Optimal policy with Q


$$
\max Q = \max_{a'}Q(s1,a')\\
\pi^*(s) = \arg\max_{a}Q(s,a)
$$

1. Find MAX rewad by action.
2. Select argument for MAX reward.



### Learning Q

Assum $Q$ in $s'$ exists



- I am in $s$

- when i do action $a$, I'll go to $s'$

- when i do action $a$, I'll get to $r$

- $Q$ in $s'$, $Q(s',a')$ exist

  ​

$$
Q(s,a) \leftarrow r+ \max_{a'} Q(s',a')\\
R^*(t) = r_t + \max R(t+1)\\

R_t = r_t + r_{t+1} +r_{t+2} + \cdots + r_{n}
$$



### Algorithm

For each $s,a$ *initialize* table entry $\hat Q(s,a) \leftarrow 0$

*Observe* current state $s$

*Do forever:*

- *Select* an action $a$ and execute it

- *Receive* immediate reward $r$

- *Observe* the new state $s'$

- *Update* the table entry for $\hat Q(s,a)$ as follows:

  $$\hat Q(s,a) \leftarrow r + \max_{a'}(s',a')$$

- $s \leftarrow s'$






## Example - Frozen lake

Simple game for going from start to goal with avoiding hole.



|  S   |  F   |  F   |  F   |
| :--: | :--: | :--: | :--: |
|  F   |  H   |  F   |  H   |
|  F   |  F   |  F   |  H   |
|  H   |  F   |  F   |  G   |

*S = Start point*      *H = Hole*  

*G = Goal*      *F=Path*  



### First environment

| S(A) |  0   |  0   |  0   |
| :--: | :--: | :--: | :--: |
|  0   |  0   |  0   |  0   |
|  0   |  0   |  0   |  0   |
|  0   |  0   |  0   |  0   |



### Final environment (Example)

| S(A) |  R   |  D   |  L   |
| :--: | :--: | :--: | :--: |
|  D   |  -1  |  D   |  -1  |
|  R   |  D   |  D   |  -1  |
|  -1  |  R   |  R   |  1   |

*U= Up*      *D = Down*  

*L= Left*      *R=Right*  



### Dummy Q-learning (Python)

<script src="https://gist.github.com/Shephexd/9aa0a0c00970f2ff73be060a23bdbdca.js?file=dummy_qlearning.py"></script>



## What is the problem in dummy Q-learning?



### Exploit vs Exploration

Exploit: Visit to somewhere have never been to before

Exploration: Visit to the reasonable way.



How to solve the problem?



### E-greeedy

```python
e = 0.1
if random < e:
    a = random
elif:
    a = argmax(Q(s,a))
```

But after many step, we don't need to exploit many times.



### decaying E-greedy

```python
for i in range(1000):
  e = 0.1/(i+1)
  if random < e:
      a = random
  elif:
```



### Add random noise

```python
a = argmax((Qs,a) + random_values)

for i in range(1000):#decaying
	a = argmax((Qs,a) + random_values/(i+1))
```

Using radom values, the selection will be changed some times.



#### The difference between E-greedy and Add random noise.

E-greedy select the random value in stead of the best way in case the random value is smaller than e value.

Add random noise method has high probability to select second or third best value even if the best one is decreased by noise.



#### Discounted reward

It is similar to the `Depreciation`  in the economy. The best is to get the reward now than future.



### Q-learning equation

$$
\hat Q(s,a) \leftarrow r + \gamma \max_{a'} \hat Q (s',a')\\
\gamma:\text{discount ratio}\\
\hat Q : \text{approximation value for }Q\\
$$

$\hat Q$ converges to $Q$.

- In deterministic words
- In finite states




## Q-learning code (python)



### Q-learning with noisy

<script src="https://gist.github.com/Shephexd/9aa0a0c00970f2ff73be060a23bdbdca.js?file=qlearning_noisy.py"></script>



### Q-learning with e-greedy

<script src="https://gist.github.com/Shephexd/9aa0a0c00970f2ff73be060a23bdbdca.js?file=qlearning_egreedy. py"></script>
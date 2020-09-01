---
layout:     post
title:      David Silver - Reinforcement Learning Note 2
subtitle:   Markov Decision Process
date:       2020-08-30
author:     Rasin
header-img: img/rl-note-1.png
catalog: true
tags:
    - Reinforcement Learning
    - Markov Decision Process
---

> Lecture: [Markov Decision Process](https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf) 

## Learning Goals

- Understand the Agent-Environment interface
- Understand what MDPs (Markov Decision Processes) are and how to interpret transition diagrams
- Understand Value Functions, Action-Value Functions, and Policy Functions
- Understand the Bellman Equations and Bellman Optimality Equations for value functions and action-value functions

## Markov Processes

### Introduction to MDPs

Markov decision processes formally describe an environment for reinforcement learning, where the environment is *fully observable*: the current *state* completely characterises the process.

Almost all RL problems can be formalised as MDPs.

### Markov Property

Definition: A state \\(s_t\\) is Markov if and only if

$$
\mathbb{P}[s_{t+1}|s_t] = \mathbb{P}[s_{t+1} | s_1, \dots, s_t]
$$

The state captures all relevant information from the history. Once the state is known, the history may be thrown away. The state is sufficient statistic of the future.

### State Transition Matrix

For a Markov state \\(s\\) and successor state \\(s'\\), the **state transition probability** is defined by:

$$
\mathcal{P}_{ss'} = \mathbb{P}[s_{t+1} = s' \mid s_t = s]
$$

**State transition matrix** \\(\mathcal{P}\\) defines transition probabilities from all states \\(s\\) to all successor state \\(s'\\).

$$
\mathcal{P} = \begin{bmatrix}
\mathcal{P}_{11} & \cdots  & \mathcal{P}_{1n}\\
\vdots  & \iddots  & \vdots \\
\mathcal{P}_{n1} & \cdots  & \mathcal{P}_{nn}
\end{bmatrix}
$$

where each row of the matrix sums to 1.

### Markov Process

A Markov process is a memoryless random process, i.e. a sequence of random states \\(s_1, s_2, \cdots \\) with the Markov property.

Definition: A Markov Process (or Markov Chain) is a tuple \\(\left< \mathcal{S}, \mathcal{P} \right> \\)

- \\(\mathcal{S}\\) is a finite set of states
- \\(\mathcal{P}\\) is a state transition probability matrix: \\(\mathcal{P}_{ss'} = \mathbb{P}[s_{t+1} = s' \mid s_t=s])

## Markov Reward Processes

### Introduction

A Markov chain with values.

Definition: A Markov Reward Process is a tuple \\(\left< \mathcal{S}, \mathcal{P} \right> \\)

- \\(\mathcal{S}\\) is a finite set of states
- \\(\mathcal{P}\\) is a state transition probability matrix: \\(\mathcal{P}_{ss'} = \mathbb{P}[s_{t+1} = s' \mid s_t=s])
- \\(\mathcal{R}\\) is a reward function, \\(\mathcal{R}_s=\mathbb{E}[r_{t+1} \mid s_t = s] \\)
- \\(\gamma\\) is a discount factor, \\(\gamma \in [0, 1])

### Return

Definition: The return \\(G_t\\) is the total discounted reward from time-step \\(t)

$$
G_t = R_{t+1} + \gamma R_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

- The discount \\(\gamma \in [0, 1]\\) is the present value of future rewards
- The value of receiving reward \\(R\\) after \\(k+1\\) time-steps is \\(\gamma^k R)
- This value immediate reward above delayed reward
  - \\(\gamma\\) close to 0 leads to *myopic* evaluation
  - \\(\gamma\\) close to 1 leads to *far-sighted* evaluation

### Value Function

The value function \\(v(s)\\) gives the long-term value of state \\(s\\).

Definition: The state value function \\(v(s)\\) of an MRP is the expected return starting from state \\(s\\)

$$
v(s) = \mathbb{E}[G_t \mid s_t=s]
$$

### Bellman Equation for MRPs

The value function can be decomposed into two parts:

- immediate reward R_{t+1}
- discounted value of successor state \\(\gamma v(S_{t+1})\\)

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200830161338.png)

The Bellman equation can be expressed concisely using matrices:

$$
v = \mathcal{R} + \gamma \mathcal{P}v
$$

where \\(v\\) is a column vector with one entry per state

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200901101328.png)

The bellman equation is a linear equation. It can be solved directly:

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200901101507.png)

Direct solution only possible for small MRPs. There are many iterative methods for large MRPs:

- Dynamic Programming
- Monte-Carlo Evaluation
- Temporal-Difference Learning

## Markov Decision Process

A Markov decision process (MDP) is a Markov reward process with decisions. It is an environment in which all states are Markov.

Definition: A Markov Decision Process is a tuple \\(\left< \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \mathcal{\gamma} \right> \\)

- \\(\mathcal{S}\\) is a finite set of states
- \\(\mathcal{A}\\) is a finite set of actions
- \\(\mathcal{P}\\) is a state transition probability matrix: \\(\mathcal{P}_{ss'}^a = \mathbb{P}[s_{t+1} = s' \mid s_t=s, a_t=a])
- \\(\mathcal{R}\\) is a reward function, \\(\mathcal{R}_s^a=\mathbb{E}[r_{t+1} \mid s_t = s, a_t = a] \\)
- \\(\gamma\\) is a discount factor, \\(\gamma \in [0, 1])

### Policies

Definition: A policy \\(\pi\\) is a distribution over actions given states:

$$
\pi(a|s) = \mathbb{P}[a_t = a \mid s_t = s]
$$

A policy fully defines the behavior of an agent. MDP policies depend on the current state. Policies are stationary (time-independent).

$$
a_t ~ \pi(\cdot \mid s_t), \forall t>0
$$

Given an MDP \\(\mathcal{M}\\) and a policy \\(\pi\\), the state sequence \\(s_1, s_2, \cdots \\) is a Markov process \\(\left< \mathcal{S}, \mathcal{P}^\pi \right>\\). The state and reward sequence \\(s_1, r_2, s_2, \cdots\\) is a Markov reward process \\(\left< \mathcal{S}, \mathcal{P}^\pi, \mathcal{R}^\pi, \gamma \right>\\), where:

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200901104214.png)

### Value Function

Definition: The *state-value* function \\(v_\pi (s)\\) of an MDP is the expected return starting from state \\(s\\), and then following policy \\(\pi\\):

$$
v_\pi(s) = \mathbb{E}_\pi [G_t \mid s_t = s]
$$

Definition: The *action-value* function \\(q_\pi (s, a)\\) is the expected return starting from state \\(s\\), taking action \\(a\\), and then following policy \\(\pi\\):

$$
q_\pi(s, a) = \mathbb{E}_\pi [G_t \mid s_t = s, a_t = a]
$$

### Bellman Expectation Equation

The *state-value* function can again be decomposed into immediate reward plus discounted value of successor state:

$$
v_\pi (s) = \mathbb{E}_\pi [R_{t+1} + \gamma v_\pi (s_{t+1}) \mid s_t = s]
$$

$$
v_\pi (s) = \sum_{a\in \mathcal{A}} \pi(a \mid s) q_\pi (s, a)
$$

The *action-value* function can similarly be decomposed:

$$
q_\pi (s, a) = \mathbb{E}_\pi [R_{t+1} + \gamma q_\pi(s_{t+1}, a_{t+1} \mid s_t = s, a_t = a]
$$

$$
q_\pi (s, a) = \mathcal{R}_s^a + \gamma \sum_{s'\in\mathcal{S}} \mathcal{P}_{ss'}^a v_\pi (s')
$$

$$
v_\pi (s) = \sum_{a\in \mathcal{A}} \pi(a \mid s) (\mathcal{R}_s^a + \gamma \sum_{s'\in\mathcal{S}} \mathcal{P}_{ss'}^a v_\pi (s'))
$$

$$
q_\pi (s, a) = \mathcal{R}_s^a + \gamma \sum_{s'\in\mathcal{S}} \mathcal{P}_{ss'}^a \sum_{a'\in \mathcal{A}} \pi(a' \mid s') q_\pi (s', a')
$$

Matrix Form:

$$
v_\pi = \mathcal{R}^\pi + \gamma \mathcal{P}^\pi v_\pi
$$

### Optimal Value Function

Definition: The *optimal state-value function* \\(v_* (s)\\) is the maximum value function over all policies:

$$
v_* (s) = \max_\pi v_\pi (s)
$$

The *optimal action value function* \\(q_* (s, a)\\) is th maximum action-value function over all policies:

$$
q_*(s, a) = \max_\pi q_\pi (s, a)
$$

The optimal value function specifies the best possible performance in the MDP. An MDP is solved when we know the optimal value function.

### Optimal Policy

Define a partial ordering over policies:

$$
\pi \geq \pi' if v_\pi(s) \geq v_{\pi'}(s), \forall s
$$

Theorem: For any MDP:

- There exists an optimal policy \\(\pi_*\\) that is better than or equal to all other policies, \\(\pi_* \req \pi, \forall \pi\\)
- All optimal policies achieve the optimal value function, \\(v_{\pi_*}(s) = v_*(s)\\)
- All optimal policies achieve the optimal action-value function, \\(q_\pi_*(s, a) = q_*(s, a)\\)

### Finding an Optimal Policy

An optimal policy can be found by maximising over \\(q_*(s, a)\\):

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200901112713.png)

There is always a deterministic optimal policy for any MDP. If we known \\(q_*(s,a)\\), we immediately have the optimal policy.

### Bellman Optimality Equation for \\(v_*\\)

$$
v_*(s) = \max_a q_*(s, a)
$$

$$
v_\pi (s) = \max_a \mathcal{R}_s^a + \gamma \sum_{s'\in\mathcal{S}} \mathcal{P}_{ss'}^a v_* (s')
$$

### Bellman Optimality Equation for \\(Q_*\\)

$$
q_*(s,a) = \mathcal{R}_s^a + \gamma \sum_{s'\in\mathcal{S}} \mathcal{P}_{ss'}^a v_* (s')
$$

$$
q_\pi (s, a) = \mathcal{R}_s^a + \gamma \sum_{s'\in\mathcal{S}} \mathcal{P}_{ss'}^a \max_{a'} q_* (s', a')
$$

### Sloving the Bellman Optimality Equation

Bellman Optimality Equation is non-linear. There is no closed form solution in general. But there are many iterative solution methods like Value Iteration, Policy Iteration, Q-learning, Sarsa and so on.

## Extensions to MDPs

- Infinite and continuous MDPs
- Partially observable MDPs
- Undiscounted, average reward MDPs

## Summary

- Agent & Environment Interface:
  - At each step \\(t\\) the agent receives a state \\(s_t\\)
  - performs an action \\(a_t\\)
  - and receives a reward \\(r_{t+1}\\)
  - The action is chosen according to a policy function \\(\pi\\)
- The total return \\(G_t\\) is the sum of all rewards starting from time \\(t\\)
  - Future rewards are discounted at a discount rate \\(\gamma^k\\)
- Markov property:
  - The environment's response at time \\(t+1\\) depends only on the *state and action representations* at time \\(t\\)
  - The future is independent of the past given the present.
  - Even if an environment doesn't fully satisfy the Markov property we still treat it as if it is
  - and try to construct the state representation to be approximately Markov.
- Markov Decision Process (MDP):
  - Defined by a state set \\(S\\)
  - action set \\(A\\) 
  - and one-step dynamics \\(p(s',r | s,a)\\). 
  - In practice, we often don't know the full MDP (but we know that it's some MDP).
- The Value Function \\(v(s)\\) estimates how "good" it is for an agent to be in a particular state \\(s\\)
  - More formally, it's the expected return \\(G_t\\) given that the agent is in state \\(s\\)
  - \\(v(s) = E[G_t | S_t = s]\\)
  - Note that the value function is specific to a given policy \\(\pi\\).
- Action Value function: \\(q(s, a)\\) estimates how "good" it is for an agent to be in state \\(s\\) and take action \\(a\\). 
  - Similar to the value function, but also considers the action.
- The Bellman equation expresses the relationship between the value of a state and the values of its successor states
  - It can be expressed using a "backup" diagram.
  - Bellman equations exist for both the *value function* and the *action value function*.
- Value functions define an ordering over policies
  - A policy \\(p_1\\) is better than \\(p_2\\) if \\(v_{p_1}(s) >= v_{p_2}(s)\\) for all states \\(s\\). 
  - For MDPs, there exist one or more optimal policies that are better than or equal to all other policies.
- The optimal state value function \\(v*(s)\\) is the value function for the optimal policy. 
  - Same for \\(q*(s, a)\\). 
  - The Bellman Optimality Equation defines how the optimal value of a state is related to the optimal value of successor states. 
  - It has a "max" instead of an average.
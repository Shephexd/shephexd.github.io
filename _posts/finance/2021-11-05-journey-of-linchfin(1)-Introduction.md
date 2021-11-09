---
layout: post
title: Journey of LinchFin(1) - Introduction
published: True
categories:
- Finance
tags:
- MachineLearning
- Finance

---



`LinchFin` is a library for asset investors. In this article, I want to introduce about the concept and flow of `LinchFin`.

<!--more-->



## Introduction

How many skills and knowledge do you have to be a successful investors? Financial market looks lie a creature threatening us. Do you have any weapon for the battle? You will notice decreasing your asset values if we are not prepared to protect these threat. Preparing your weapons and gather your crew to be more smart investor who notice risks and opportunities in advance.
The goal of project `LinchFin (LF)` is to give insight and information for investor who consider investment for multiple assets. To achieve this goal, We will serve some functions from simple to sophisticated ones. Like lego blocks, combining and linking simple ones can be worthy.
We will serve core tools to build your own portfolio managing pipeline yourself. I hope you can be familiar being investing with these tools and data analytics. After that share your strategies that make you get rewards and influence on our platform.
This project aim the financial platform for lead investors who want to manage and share their strategies and followers who want to study and follow up their strategies.



## Functions

`LinchFin` give helpful functions to improve your investment like below.



### Analytics

1. Asset selection
   1. Asset Screening
   2. Asset clustering
   3. Asset Network
   4. Universe management
2. Factor Analysis
   1. Macro factors
   2. Technical factors
3. TimeSeries Modeling
   1. Dynamic rebalancing policy
4. Portfolio optimization
   1. Mean Variance optimization (MVO)
   2. Risk Parity
   3. Hierarchical risk parity(HRP)



### Simulation & Evaluation

1. Risk Analysis
   1. Internal risk
   2. External risk
2. Trading strategies Evaluation
3. Backtest Simulation
   1. Static simulation
      1. stationary
      2. accrual
   2. Dynamic simulation
      1. Dynamic path finding with rebalancing



## Data Driven Development

Based on the Data Driven Development, I designed structure of `LinchFin`. 



### Entities

Each entity has their identity and attribute.

> *Many objects are not fundamentally defined by their attributes, but rather by a thread of continuity and identity. - Domain-Driven Design, Eric Evan -*



#### Asset

Investable item (stock/bond/ETF)



#### AssetClass

class of asset



#### Portfolio

Asset with weights, The sum of weights must be 1.



### ValueTypes

ValuesTypes is object having no identity.


> *Many objects have no conceptual identity. These objects describe characteristics of a thing.  - Domain-Driven Design, Eric Evan -*



#### Feature

Numerical or Categorical value



#### Metric

number value( $0 \le x \ \le 1$)



#### TimeSeries(TimeSeriesRow)

Time series data (prices, index)



### Aggregates

The set of entities



#### AssetUniverse

The set of Asset



#### Cluster

Grouped assets



### AssetNetwork

The graph represeiontation of assets



## Conclusion

Before describing detail, I want to give a brief concept of `LinchFin`.
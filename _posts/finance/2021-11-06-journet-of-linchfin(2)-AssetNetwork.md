---
layout: post
title: Journey of LinchFin(2) - AssetNetwork
published: True
categories:
- Finance
tags:
- MachineLearning
- Finance
- Python
- Neo4j
---

What is the best way to represent the relations of assets?
I suggest a way to represent the asset's relation called, asset network, based on the graph theory.


<!--more-->



## What is AssetNetwork
 
I suggest a way to represent the asset's relation based on the Probabilistic Graphical Models, called `AssetNetwork`. 

Assuming that each asset has asset class and sub class to classify its category. `Asset A` can be belonging to asset `class C`. 

We can call that The asset `class C` has the `asset A`. 

Based on this idea, we can generate the network with having relations.



## Why AssetNetwork

Based on the graph theory, the graph has nodes and edges. To represent asset network, I suggest a simple example.


### Node

Node is asset or its classes.

- Asset
- AssetClass
- AssetSubClass



### Edge

Edge is implicit or explicit relation between assets and asset class. The relation can be created or deleted conditionally.

- has (explicit)
- is close to (implicit)



![asset-network](/assets/images/articles/LinchFin/asset-network.png)


## Conclusion

Using Asset network, we can query to find the specific asset on the condition like replaceable or compatible.  
Also, the asset node can be clustered based on the implicit or explicit edge conditions. 
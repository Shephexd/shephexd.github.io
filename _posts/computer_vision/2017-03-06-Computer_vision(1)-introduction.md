---
title: Computer vision(1)-introduction
layout: post
categories: 
- Computer vision
tags:
- Algorithm
- Linear algebra
- Computer vision
- Robotics
---



The `computer vision` is one of the difficult and important part on machine learning. So, how can we make the computer *see* and *recognize* the image? It is advanced question about image processing. 

Imagine that the computer see and recognize the image like human being. It is related to the robot vision for the robotics.

This post will be introduced about what the computer vision is and the application using `computer vision`.



<!--more-->



## Introduction

There are related fields with computer vision.

- Image processing
- Pattern recognition
- photogrammetry




###  Subjects

- camera models, calibration
- Image processing
- Motion and tracking
- stereopsis




### prerequisite for robotics

- Mechanical engineering
- Mathematics
- Control theory
- electronic engineering
- Computer science




## Computer vision



Point $P$ in frame $W$ is *world(global) frame*.
$$
^WP = \begin{pmatrix} P_x\\ P_y\\ P_z\end{pmatrix}
$$
Rotation matrix $R$, $W \to B$
$$
^WR_B = \begin{pmatrix} ^WX_B\\ ^WY_B\\ ^WZ_B\end{pmatrix}
$$

### Relative pose

Position and orientation is different depending on the view. For example a point P can be described by coordinate vectors relative to either select ${A}$ or ${B}$.


$$
^AP = {^A\xi_B} \cdot {^BP}
$$


You can think ${^A\xi_B}$ as a motion it move $A$ to $B$.

  

  

![](/assets/post_images/cv/muti_coordinate.jpg)

- *world frame,* ${O}$
- *robot frame,* $R$
- *camera frame*, $C$
- *object*, $B$


### Frame-to-Frame mapping

![](/assets/post_images/cv/coordinate_mapping.jpg)

$^AP=^AR_{B}{^BP} + ^AP_{BORG}$





### Homogeneous transform

$P$ is *homogeneous* position vector
$T$ is $4 \times 4$ matrix called *homogeneous transform*.


$$
\begin{align}
&P=
\begin{pmatrix}
x,y,z,1
\end{pmatrix}^T\\

&\begin{pmatrix}
^AP\\
1
\end{pmatrix}
=
\begin{pmatrix}
^AR_B & ^AP_{BORG}\\
0\ 0\ 0 & 1\\
\end{pmatrix}
\begin{pmatrix}
^BP\\
1\\
\end{pmatrix}\\
&^AP=^AT_B{^BP}

\end{align}
$$



$$
^AT_C=^AT_B{^BT_C}\\
^AT_B^{-1}=^BT_A=\begin{pmatrix}
^AR_B^T & -^AR^T_B{^AP}_{BORG}\\
0\ 0\ 0 & 1\\
\end{pmatrix}
$$




The $3 \times 3$  *transformation matrix* can take care of the **translation** and **rotation**, But we may also need **scaling**.



The $3 \times 3$  *transformation matrix* matrix can all these (**translation**, **rotation** and **scaling**).

The transform matrix is obtained by matrix multiplication of the **part transforms**  
$translation * rotation * scaling $



### Coordination transformation

![](/assets/post_images/cv/transform_ex.jpg)




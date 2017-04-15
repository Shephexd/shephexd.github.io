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




### Basic of robotics



#### Pose

Basic of robotics is pose (description of position and orientation).  
Coordinate system(frame) attatched to eacth object.  
Transforms between coordinate systems necessary.  



#### kinematics

It is about science of motion without regard to causing forces, Position, velocity, accerlation



#### Links and joint types



### Velocities and singularities

#### Trajectory

#### Control





## Coordinate System



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




### Rotation matrix

Rotation matrix $^VR_B$ describes how point are transformed from frame $\{B\}$ to Fram $\{V\}$ when the frame is rotated.
$$
\begin{pmatrix} \frac{V_y} {V_y}\end{pmatrix} = \begin{pmatrix} cos \theta & -sin \theta\\ -sin \theta & cos \theta\end{pmatrix} \begin{pmatrix} B_x\\ B_y\end{pmatrix}
$$


Rotation matrix has some features.

1. Orthogonal
2. Has orthogonal columns
3. Columns are unit vectors
4. Inverse = transpose


$$
\begin{pmatrix} A_x\\ A_y\end{pmatrix} =
  \begin{pmatrix} V_x\\ V_y\end{pmatrix} +  \begin{pmatrix} x\\ y\end{pmatrix} =
\begin{pmatrix} cos \theta & -sin \theta\\ -sin \theta & cos \theta\end{pmatrix}
\begin{pmatrix} B_x\\ B_y\end{pmatrix} +  \begin{pmatrix} x\\ y\end{pmatrix} =
\begin{pmatrix} cos \theta & -sin \theta & x\\ -sin \theta & cos \theta & y\end{pmatrix}
\begin{pmatrix} B_x\\ B_y \\1\end{pmatrix}
$$

$$
\begin{pmatrix} A_x\\ A_y \\1\end{pmatrix} 
=\begin{pmatrix} ^AR_B & t \\ 0_{1\times2} & 1 \end{pmatrix}
\begin{pmatrix} B_x\\ B_y \\1\end{pmatrix}\\
{t=\text{transformation and} ^AR_B =\text{orientation} }
$$


### Frame-to-Frame mapping

pose = position + orientation

$\{B\}=\{^AR_B,\ ^AP_{BORG}\}$

![](/assets/post_images/cv/coordinate_mapping.jpg)

$^AP=\ ^AR_{B}{^BP} +\ ^AP_{BORG}$





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


$$
\begin{pmatrix} ^AV\\ 0\end{pmatrix} 
=\begin{pmatrix} ^AR_B &\ ^AP_{BORG} \\ 0\ 0\ 0 & 1 \end{pmatrix}
\begin{pmatrix} ^BV\\ 0\end{pmatrix}\\
\Leftarrow\Rightarrow 
^AV=\ ^A
\Leftarrow\Rightarrow
$$




### Forward Kinematics

The kinematic chains is formed by a sequence of homogeneous transfomr, each describing one link.
$$
^0T_N =\ ^0T_1\ ^1T_2 \cdots \ ^{N-1}T_{N}
$$


### Coordination transformation

![](/assets/post_images/cv/transform_ex.jpg)




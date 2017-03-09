---
layout: post
title: tensorflow GPU install
categories: Development
tags:
- Development
- Tensorflow
- CUDA
---



Ubuntu 16.04 LTS version Tensorflow install.



<!--more-->



## Version info

- Ubuntu 16.04 LTS
- CUDA 8.0
- CUDNN 5.1
- Tensorflow 1.0 (recent version)



To send a file from local to remote server, `scp` is useful command.

`scp -P PORT_NUMBER FILE_NAME USER@REMOTE_IP:DIR`



## CUDA Download

[CUDA download Link](https://developer.nvidia.com/cuda-downloads)



```bash
wget -c https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo dpkig -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo apt update
sudo apt install cuda
```



### Environment setting and test

```bash
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
cuda-install-samples-8.0.sh ~
cd ~/NVIDIA_CUDA-8.0_Samples/5_Simulations/nbody
make
./nbody
```



## CUDNN Download

[CUDNN nvida download](https://developer.nvidia.com/cudnn)

To download cuDNN, you need to register your account.  
And after that, you can download the compressed file and uncompress and move to cuda directory.



```bash
tar -xvf cudnn-8.0-linux-x64-v5.1.tar

cd ~/cuda/lib64
sudo mv * /usr/local/cuda/lib64/
cd ~/cuda/include
sudo mv * /usr/local/cuda/include/
```



### Install library

`sudo apt-get install libcupti-dev`



## Anaconda setting

```bash
conda create -n tf
source activate tf
pip install tensorflow-gpu
```



## Tensorflow test



```python
import tensorflow as tf
# Creates a graph.
with tf.device('/gpu:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print sess.run(c)
```


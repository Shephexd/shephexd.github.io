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
- CUDNN 6.0
- Tensorflow 1.4 (recent version)



To send a file from local to remote server, `scp` is useful command.

`scp -P PORT_NUMBER FILE_NAME USER@REMOTE_IP:DIR`



## Install process





1. NVIDIA driver install
2. CUDA install
3. Environment setting
4. CUDnn install
5. NVIDIA docker install
6. Anaconda install
7. Tensorflow Install





## NVIDIA driver install

```bash
sudo apt-get purge nvidia-*
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-375
sudo reboot
```





## CUDA Download

[CUDA download Link](https://developer.nvidia.com/cuda-downloads)



```bash
wget -c https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo apt update
sudo apt install cuda
```



### Environment setting and test

```bash
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

cd $LD_LIBRARY_PATH
cd ../bin/
bash cuda-install-samples-8.0.sh ~
cd ~/NVIDIA_CUDA-8.0_Samples/5_Simulations/nbody

make
./nbody
```



## CUDNN Download

[CUDNN nvidia download](https://developer.nvidia.com/cudnn)

To download cuDNN, you need to register your account.  
And after that, you can download the compressed file and uncompress and move to cuda directory.



```bash
tar -xvf cudnn-8.0-linux-x64-v6.0.tar

cd ~/cuda/lib64
sudo mv * /usr/local/cuda/lib64/
cd ~/cuda/include
sudo mv * /usr/local/cuda/include/
```



### Install library

`sudo apt-get install libcupti-dev`





## NVIDIA Docker Install

```bash
curl -L https://nvidia.github.io/nvidia-docker/gpgkey | \
sudo apt-key add -
sudo tee /etc/apt/sources.list.d/nvidia-docker.list <<< \
"deb https://nvidia.github.io/libnvidia-container/ubuntu16.04/amd64 /
deb https://nvidia.github.io/nvidia-container-runtime/ubuntu16.04/amd64 /
deb https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64 /"
sudo apt-get update
```



```Bash
sudo apt-get install nvidia-docker2
sudo pkill -SIGHUP dockerd
```



## Anaconda

### Install

```bash
wget -c https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
bash Anaconda3-5.0.1-Linux-x86_64.sh
```



### Anaconda setting

```bash
conda create -n tf python=3.5
source activate tf
pip install tensorflow-gpu
```



## Tensorflow test

```python
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
```





```python
import tensorflow as tf
# Creates a graph.
with tf.device('/gpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print sess.run(c)
```


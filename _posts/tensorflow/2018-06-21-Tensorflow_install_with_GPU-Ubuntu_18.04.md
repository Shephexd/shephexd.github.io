---
layout: post
title: tensorflow GPU install - Ubuntu 18.04
categories: Tensorflow
tags:
- Development
- Tensorflow
- CUDA
---





서버 업그레이드 후, 개발 환경 설정을 위하여 텐서플로우를 설치하려고 봤더니, 이전과는 버젼이 많이 달라져서 새로 개발 환경 설정 방법에 대해 정리해보고자 한다.



<!--more-->

우분투는 18.04부터 Unity 환경에서 Gnome환경으로 돌아왔다. 개인적으로는 Gnome을 선호하고, Unity의 잦은 버그로 겪었던지라 이번 업데이트 버젼을 설치하였다.

 

## Version info

- Ubuntu 18.04 LTS
- CUDA 9.0
- CUDNN 7.0
- Tensorflow 1.8 (recent version)
- python 3.6



To send a file from local to remote server, `scp` is useful command.

`scp -P PORT_NUMBER FILE_NAME USER@REMOTE_IP:DIR`



## Install process





1. NVIDIA driver install
2. CUDA install
3. Environment setting
4. CUDnn install
5. Anaconda install
6. Tensorflow Install





## NVIDIA driver install

```bash
sudo ubuntu-drivers list
sudo apt-get remove nvidia-*
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-390
sudo reboot
```



드라이버가 정상적으로 설치되어 있는지는 해당 명령어로 확인 가능하다

```bash
nvidia-smi
```



정상적으로 설치된 경우, 그래픽 카드의 정보가 나타난다.



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

CuDNN을 설치하기 위해서는 Nvidia developer에 회원가입 및 동의 후 설치 가능하다.

`CUDA 7.0`을 위한 `CuDNN 9.0 리눅스`을 설치하면 된다.



[cuDNN v7.0.5 Library for Linux](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.5/prod/9.0_20171129/cudnn-9.0-linux-x64-v7)



```bash
tar -zxvf cudnn-9.0-linux-x64-v7.tgz
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64
sudo cp cuda/include/cudnn.h /usr/local/cuda-9.0/include
```



### Install library

`sudo apt-get install libcupti-dev`



## Anaconda

### Install

```bash
wget -chttps://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
bash Anaconda3-5.0.1-Linux-x86_64.sh
```



### Anaconda setting

```bash
conda create -n tf python=3.6
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


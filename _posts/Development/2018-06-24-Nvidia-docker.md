---
layout: post
published: True
title: Nvidia docker를 이용한 Tensorflow GPU환경 구축
categories: Server
tags:
- Linux
- Environment
- Tensorflow
- Nvidia-docker
- Docker

typora-root-url: /Users/shephexd/Documents/github/pages/
---



Docker 이미지를 이용하여 Tensorflow 컨테이너를 쉽게 구성할 수 있다. Docker에서는 GPU 그래픽카드를 기본 설정으로는 지원하지 않기 때문에, Nvidia Docker를 설치하여야 한다.



Nvidia-docker github[^nvidia-docker] 의 설치가이드를 참조하였다.



현재 설치 가이드의 환경은 다음과 같다.

- Ubuntu 18.04
- Geforce GTX 1070
- Docker 18.03



<!--more-->



## Nvidia Docker

Nvidia 도커의 시스템 구조는 아래의 이미지에 설명되어 있다.

![nvidia-docker](/assets/post_images/docker/nvidia-docker.png)



Nvidia-docker는 도커 엔진 위에서 작동하는 컨테이너들을 OS에 설치되어 있는 그래픽 카드 드라이버를 연결시켜 주는 역활이다. 즉, 컨테이너는 가상화 되어있지만, GPU 자원은 컨테이너가 공유하는 구조로 되어 있다.



## Prerequisite

현재 컴퓨터에 설치되어 있는 Nvidia 그래픽 드라이버가 설치되어 있어야한다.



해당 명령어로 설치된 그래픽 카드 드라이버가 설치되어 있는지 확인 할 수 있다.

```bash
nvidia-smi
```



Docker에서 그래픽 자원에 접근가능한지 확인해보자.

```bash
docker run --rm nvidia/cuda nvidia-smi
```

> 도커의 기본 설정에서는 그래픽 카드 자원의 공유가 되어 있지않아 에러가 발생한다.





## Install Nvidia Docker



1. Add & Update repository

   ```bash
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \\n  sudo apt-key add -\ndistribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \\n  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update
   ```

   



2. Install nvidia-docker

   ```bash
   sudo apt-get install -y nvidia-docker2
   ```

   

3. Process reboot

   ```bash
   sudo pkill -SIGHUP dockerd
   ```

   

4. Check

   ```bash
   docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
   ```

   

## Use prepared tensorflow-gpu image



한글 자연어 처리 패키지와 tensorflow-gpu를 이용하여 도커 컨테이너를 동작시킬 수 있다.'

- Mecab
- Konlpy
- Tensorflow-GPU



### 실행 방법

```bash
mkdir notebooks
docker run --runtime=nvidia -p 5000:8888 -v $PWD/notebooks:/data/notebooks shephexd/mecab_parser
```

> -p: 로컬 호스트의 5000포트와 8888(컨테이너의 주피터 노트) 포트를 매핑시켜 준다.
>
> -v: 현재 Directory의 notebooks와 도커 컨테이너의 /data/notebooks를 매핑 시켜준다.
>
> (컨테이너와 로컬호스트 간의 공유 폴더 개념)



http://127.0.0.1:5000 으로 동작중인 Docker container의 Jupyter-notebook에 접근가능하며, 해당 노트북에서 작업한 파일은 현재 경로의 notebooks 폴더에 저장된다. 



### Dockerfile

해당 도커 이미지의 설정은 다음과 같다.

```dockerfile
FROM tensorflow/tensorflow:latest-gpu-py3
MAINTAINER shephexd, shephexd <shephexd@google.com>

RUN apt-get update
RUN apt-get install wget openjdk-8-jdk cmake automake git vim language-pack-ko -y

RUN locale-gen ko_KR.UTF-8
ENV LANG ko_KR.UTF-8
ENV LANGUAGE ko_KR.UTF-8
ENV LC_ALL ko_KR.UTF-8

WORKDIR /home
RUN wget -c https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
RUN wget -c https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.0.1-20150920.tar.gz
RUN tar -xvf mecab-0.996-ko-0.9.2.tar.gz;tar -xvf mecab-ko-dic-2.0.1-20150920.tar.gz

WORKDIR /home/mecab-0.996-ko-0.9.2
RUN ./configure;make;make check;make install
RUN ldconfig
WORKDIR /home/mecab-ko-dic-2.0.1-20150920
RUN ./autogen.sh;./configure;make;make install

WORKDIR /home
RUN git clone https://bitbucket.org/eunjeon/mecab-python-0.996.git

WORKDIR mecab-python-0.996
RUN python3 setup.py build;python3 setup.py install

RUN pip install Jpype1 konlpy requests seaborn flask

WORKDIR /home
RUN rm mecab-0.996-ko-0.9.2.tar.gz mecab-ko-dic-2.0.1-20150920.tar.gz
RUN git clone https://github.com/Shephexd/mecab_flask
RUN echo "exec jupyter-notebook --allow-root &" >> run.sh
RUN echo "exec python /home/mecab_flask/mecab_parser.py" >> run.sh
RUN chmod +x /home/run.sh

RUN mkdir -p /data/notebooks
VOLUME ["/data/notebooks"]

WORKDIR /data
CMD ["/bin/bash","/home/run.sh"]
```



[^nvidia-docker]: https://github.com/NVIDIA/nvidia-docker


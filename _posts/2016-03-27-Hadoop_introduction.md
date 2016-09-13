---
layout: post
published: True
title: Hadoop Introduction - install
categories: 
- server
tags:
- Development
- big data
---

## What is Hadoop?
하둡은 **하둡은 대용량 데이터를 분산 처리할 수 있는 자바 기반의 오픈소스 프레임워크** 의미한다.

#### Function
하둡은 분산 파일 시스템은 HDFS(Hadoop Distributed File System)에 데이터를 저장하고, 분산 처리 시스템인 맵리듀스를 이용하여 데이터를 처리.

#### History
하둡은 구글이 논문으로 발표한 GFS(Google File System)와 맵리듀스(Map Reduce)를 2005년에 더글 커딩이 구현한 결과물. 처음에는 오픈소스 검색 엔진인 너치(Nutch)에 적용하기 위해 시작하다가 독립적인 프로젝트로 만들어진 후, 2008년에 아파치 최상위 프로젝트로 승격.

#### Hadoop Ecosystem
하둡이 비즈니스에 효율적으로 적용될 수 있도록 다양한 서브 프로젝트가 구성되었고, 이러한 서브 프로젝트가 상용화되면서 하둡 에코시스템(Hadoop  Ecosystem)이 구성

### Hadoop guide

#### 실행 모드 결정(Execute mode setting)
1. 독립 실행(Standalone) 모드
  - Basic Execute Mode. 하둡 환경설정 파일에 아무런 설정을 하지 않고 실행하면 로컬 장비에서만 실행되기 때문에 로컬(local) 모드라고도 불림. 하둡에서 제공하는 데몬을 구동하지 않기 때문에 분산 환경을 고려한 테스트는 불가능. 단순히 맵리듀스 프로그램을 개발하고, 해당 맵리듀스를 디버깅하는 용도로만 적합한 모드.

2. 가상 분산(Pseudo-distributed) 모드
  - 하나의 장비에 모든 하둡 환경설정을 하고, 하둡 서비스도 이 장비에서만 제공하는 방식을 말함. HDFS와 맵리듀스와 관련된 데몬을 하나의 장비에서만 실행하기 때문에 입문자들이 테스트 환경을 구성할 때 사용.

3. 완전 분산(Fully Distributed) 모드
  - 여러 대의 장비에 하둡이 설치된 경우, 하둡으로 라이브 서비스를 하게 될 경우 이와 같은 방식으로 구성.

<!--more-->

### Hadaoop2
- YARN
- dd

### Hadoop 2.6 install guide
1. Protocol Buffer install
- 프로토콜 버퍼는 구글에서 공개한 직렬화 라이브러리.
- 바이너리 데이터를 통해 이기종간 통신하기 위해 구글에서 오픈소스로 공개.

Download
`wget "http://protocolbuf.googlecode.comd/files/protocolbuf-2.5.0.tar.gz"`

2. Hadoop2 Download
`wget "http://mirror.apache-kr.org/hadoop/common/hadoop-2.6.0/hadoop-2.6.0.tar.gz"`


environment:ubuntu 14.04

```sh
sudo apt-get update #update repository
sudo apt-get install default-jdk #haddop based on java, to install hadoop we need jdk(Java Development Kit)

java -version # check the java version
sudo apt-get install ssh #hadoop use ssh to communicate with other node.
sudo apt-get install rsync #rsync is remote synchronize daemon
ssh-keygen -t dsa -P ' ' -f ~/.ssh/id_dsa
cat ~/.ssh/id_dsa.pub >> ~/.ssh/authorized_keys
wget -c http://mirror.olnevhost.net/pub/apache/hadoop/common/hadoop-2.6.0/hadoop-2.6.0.tar.gz
sudo tar -zxvf hadoop-2.6.0.tar.gz
sudo mv hadoop-2.6.0 /usr/local/hadoop
update-alternatives --config java
```

`sudo gedit ~/.bashrc`

```bash
          #Hadoop Variables
          export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64
          export HADOOP_HOME=/usr/local/hadoop
          export PATH=$PATH:$HADOOP_HOME/bin
          export PATH=$PATH:$HADOOP_HOME/sbin
          export HADOOP_MAPRED_HOME=$HADOOP_HOME
          export HADOOP_COMMON_HOME=$HADOOP_HOME
          export HADOOP_HDFS_HOME=$HADOOP_HOME
          export YARN_HOME=$HADOOP_HOME
          export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
          export HADOOP_OPTS="-Djava.library.path=$HADOOP_HOME/lib"
```

```sh
source ~/.bashrc
cd /usr/local/hadoop/etc/hadoop
sudo gedit hadoop-env.sh
```

```bash
          #The java implementation to use.
          export JAVA_HOME="/usr/lib/jvm/java-7-openjdk-amd64"
```
	sudo gedit core-site.xml
```xml
          <configuration>
                  <property>
                      <name>fs.defaultFS</name>
                      <value>hdfs://localhost:9000</value>
                  </property>
          </configuration>
```

`sudo gedit yarn-site.xml`

```xml
	
	sudo cp mapred-site.xml.template mapred-site.xml
	sudo gedit mapred-site.xml

```

```xml
          <configuration>
                  <property>
                      <name>mapreduce.framework.name</name>
                      <value>yarn</value>
                  </property>
          </configuration>
```

`sudo gedit hdfs-site.xml`

```xml
          <configuration>
                  <property>
                      <name>dfs.replication</name>
                      <value>1</value>
                  </property>
                  <property>
                      <name>dfs.namenode.name.dir</name>
                      <value>file:/usr/local/hadoop/hadoop_data/hdfs/namenode</value>
                  </property>
                  <property>
                      <name>dfs.datanode.data.dir</name>
                      <value>file:/usr/local/hadoop/hadoop_store/hdfs/datanode</value>
                  </property>
          </configuration>
```

```bash
cd
mkdir -p /usr/local/hadoop/hadoop_data/hdfs/namenode
mkdir -p /usr/local/hadoop/hadoop_data/hdfs/datanode
sudo chown parallels:parallels -R /usr/local/hadoop


hdfs namenode -format
start-all.sh
jps
```

```sh
http://localhost:8088/
http://localhost:50070/
http://localhost:50090/
http://localhost:50075/

#this install guide from below url.
http://chaalpritam.blogspot.in/2015/01/hadoop-260-single-node-cluster-setup-on.html
```


#### Error
>WARN util.NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable

	/usr/local/hadoop/lib/native$ sudo mv * ../../lib/
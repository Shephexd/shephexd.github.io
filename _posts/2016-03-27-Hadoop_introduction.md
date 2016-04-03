---
layout: post
published: True
title: Hadoop Introduction - install
excerpt: Introduction about Hadoop 2.6 and how to install.
categories: Programming, big data
tags:
- Development
- big data
---

###What is Big data?  
빅 데이터의 3대 요소(3V)
1. 크기 (Volume)
  - 기존 파일 시스템에 저장하기 어렵고, 데이터 분석을 위해 사용하던 기존의 방법으로는 소화하기 어려울 정도로 데이터의 양이 증가.
  - 이를 해결하기 위해 확장 가능한 방식으로 데이터를 저장하고 분석하는 분산 컴퓨팅 기법을 적용.
  - 현재 분산 컴퓨팅 솔루션은 Google GFS, Apache Hadoop, 대용량 병렬 처리 데이터 베이스는 EMC GreenPlum, HP Vertica, IBM Netezza, TeraData Kickfre등이 존재.
2. 속도 (Velocity)
  - 실시간 처리와 장기적인 접근으로 분류가 가능.
  - 실시간 처리는 매우 빠른 속도로 다양하게 발생하는 데이터들, 사용자 검색 및 이용 데이터, 로그 데이터 등.
  - 장기적으로 수집된 대량의 데이터를 다양한 분석 기법 등을 이용하여 분석하고 학습 후 활용하는데도 사용. 데이터 마이닝이나 기계 학습, 자연어 처리, 패턴 인식등에 활용.
3. 다양성 (Variety)
  - 다양한 종류의 데이터들이 빅데이터들을 구성. 데이터 정형화의 종류에 따라 정형(Structured), 반정형(semi-structured), 비정형(unstructured)로 분류.
  - 정형 데이터는 일정한 형식을 갖추고 저장되는 정형화된 데이터를 의미.
  - 반정형 데이터는 고정된 필드나 형태로 저장되진 않지만, XML,HTML같은 메타데이터나 스키마가 포함된 데이터.
  - 비정형 데이터는 고정된 필드나 스키마가 포함되지 않은 데이터. 유투븨으 동영상, SNS 사진이나 오디오 등 다양한 종류의 비정형 데이터가 존재.
일반적으로 위의 3대 요소 중 두 가지 이상만 충족된다면 빅데이터로 분류,

###What is Hadoop?
####Definition
하둡은 **하둡은 대용량 데이터를 분산 처리할 수 있는 자바 기반의 오픈소스 프레임워크** 의미한다.

####Function
하둡은 분산 파일 시스템은 HDFS(Hadoop Distributed File System)에 데이터를 저장하고, 분산 처리 시스템인 맵리듀스를 이용하여 데이터를 처리.

####History
하둡은 구글이 논문으로 발표한 GFS(Google File System)와 맵리듀스(Map Reduce)를 2005년에 더글 커딩이 구현한 결과물. 처음에는 오픈소스 검색 엔진인 너치(Nutch)에 적용하기 위해 시작하다가 독립적인 프로젝트로 만들어진 후, 2008년에 아파치 최상위 프로젝트로 승격.

####Hadoop Ecosystem
하둡이 비즈니스에 효율적으로 적용될 수 있도록 다양한 서브 프로젝트가 구성되었고, 이러한 서브 프로젝트가 상용화되면서 하둡 에코시스템(Hadoop  Ecosystem)이 구성

###Hadoop guide
####실행 모드 결정(Execute mode setting)
1. 독립 실행(Standalone) 모드
  - Basic Execute Mode. 하둡 환경설정 파일에 아무런 설정을 하지 않고 실행하면 로컬 장비에서만 실행되기 때문에 로컬(local) 모드라고도 불림. 하둡에서 제공하는 데몬을 구동하지 않기 때문에 분산 환경을 고려한 테스트는 불가능. 단순히 맵리듀스 프로그램을 개발하고, 해당 맵리듀스를 디버깅하는 용도로만 적합한 모드.

2. 가상 분산(Pseudo-distributed) 모드
  - 하나의 장비에 모든 하둡 환경설정을 하고, 하둡 서비스도 이 장비에서만 제공하는 방식을 말함. HDFS와 맵리듀스와 관련된 데몬을 하나의 장비에서만 실행하기 때문에 입문자들이 테스트 환경을 구성할 때 사용.

3. 완전 분산(Fully Distributed) 모드
  - 여러 대의 장비에 하둡이 설치된 경우, 하둡으로 라이브 서비스를 하게 될 경우 이와 같은 방식으로 구성.

###Hadaoop2
- YARN
-

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

sudo gedit ~/.bashrc
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

sudo gedit yarn-site.xml
```xml
          <configuration>
                  <property>
                      <name>yarn.nodemanager.aux-services</name>
                      <value>mapreduce_shuffle</value>
                  </property>
                  <property>
                      <name>yarn.nodemanager.aux-services.mapreduce.shuffle.class</name>
                      <value> org.apache.hadoop.mapred.ShuffleHandler</value>
                  </property>
          </configuration>
```
sudo cp mapred-site.xml.template mapred-site.xml
sudo gedit mapred-site.xml
```xml
          <configuration>
                  <property>
                      <name>mapreduce.framework.name</name>
                      <value>yarn</value>
                  </property>
          </configuration>
```
sudo gedit hdfs-site.xml
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
sudo chown jace:jace -R /usr/local/hadoop
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

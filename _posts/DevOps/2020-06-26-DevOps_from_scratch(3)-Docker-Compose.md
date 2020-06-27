---
layout: post
title: 밑바닥부터 시작하는 DevOps 생존기(3) - docker-compose
published: True
categories:
- Server
tags:
- Development
- Docker
- docker-compose
- DevOps
- CI/CD

---

이전 장에서 Docker를 이용하여 소스코드의 가상환경을 설정하고, 해당 간단한 웹 서비스를 실습하였습니다. 여러 컨테이너 간의 통신을 관리하거나 실행시 파라미터가 많거나 가변적인 경우에는 도커 명령어만으로 관리하기가 쉽지 않습니다. 주로 단일 호스트에서 컨테이너 오케스트레이션으로 사용하기 좋은 `docker-compose`에 대해 소개하려고 합니다.



<!--more-->



## docker-compose란?

`docker-compose` 는 여러 도커 컨테이너들을 정의하고 실행시키는 도커 오케스트레이션 툴입니다.



일반적으로 도커 컨테이너에는 하나의 프로세스가 동작합니다. 각 프로세스를 도커 컨테이너로 구동 시킨 시스템에서 통신을 하기 위해서는 각 컨테이너 간에 통신이 필요합니다. 



> *Compose* is a tool for defining and running multi-container *Docker* applications



## Docker-compose를 이용한 서비스 배포



### 배포 방법

1. `Dockerfile` 에 동작하는 서비스에 대한 환경을 정의합니다.
2. `docker-compose.yaml` 파일에 Dockerfile 빌드에 필요한 파라미터나 서비스에 필요한 다른 컨테이너 이미지를 정의합니다.
3. `docker-compose build` 로 도커 이미지를 생성합니다.
4. `docker-compose up` 명령어로 서비스에 정의한 어플리케이션들을 실행시킵니다.



### 장점

1. 실행이나 빌드시 필요한 파라미터를 `docker-compose.yaml` 에 정의하여 관리할 수 있습니다.
2. 여러 컨테이너 간 통신이 필요하거나 공유 디렉토리가 필요한 서비스를 정의할때 유용합니다.
3. Scale up을 지원합니다.



### 단점

1. 단일 호스트가 아닌 경우에는 각 서버 별로 배포를 진행해야 합니다.



## docker-compose를 이용한 빌드 

이전 장에서 도커로 작성한 flask app을 수정하여, 처음 request를 `redis`에 캐싱하는 flask 서비스를 `docker-compose`를 이용하여 작성하려고 합니다.



이 서비스가 동작하기 위해서는 두가지 컨테이너가 필요합니다.

- echo-server: 실습에서 사용할 Flask web application, 이미지는 로컬에서 빌드하여 사용
- redis: 캐싱에 사용할 인메모리 DB, 도커 허브 이미지 사용



### docker-compose.yaml

```yaml
version: '3'

services:
  echo-server:
    build: .
    image: echo-server:latest
    environment:
      - TZ=Asia/Seoul
    ports:
      - "5001:5000"
  redis:
    image: redis:latest
    expose:
      - 6379
```



1. 호스트의 5001번 포트의 요청을 echo-server의 5000번 포트로 포워딩합니다
2. redis 컨테이너와 echo-server 간 통신을 위해서 redis 컨테이너의 6379 포트를 expose합니다. (호스트에서는 접근 불가)



>  외부에 노출시킬 컨테이너는 port로 정의하고, 컨테이너 간 통신만 사용할 경우에는 expose를 사용합니다.



### config.json

```json
{
  "ENV": "DEVELOPMENT",
  "MESSAGE": "DevOps",
  "REDIS": {
    "host": "redis",
    "port": "6379"
  }
}
```



echo-server에서 redis 컨테이너를 접근할 때는 호스트의 IP 대신 서비스에 정의된 이름(ex: `redis`)으로 통신이 가능합니다. 



### 이미지 빌드

```bash
docker-compose build
```



### 이미지 pull

```bash
docker-compose pull
```



### 서비스 실행

```bash
docker-compose up
```

```bash
docker-compose up -d  #daemon 모드로 실행
```



### 로그 확인

```
docker-compose logs
```



전체 실습 코드는 github [^1]에 올려두었습니다.



## Conclusion

단일 호스트 환경에서 여러 컨테이너 간 서비스를 정의할 때 유용한 `docker-compose`에 대해 소개하였습니다. 실제 CI/CD를 구축할 때 `docker-compose`를 이용하면 빌드 및 실행에 필요한 명령어를 간소화할 수 있습니다.

다음에는 CI/CD에 유용한 `Jenkins`를 이용하여 배포 workflow를 작성하는 방법에 대하여 소개하려고 합니다.





[^1]: https://github.com/Shephexd/echo-server
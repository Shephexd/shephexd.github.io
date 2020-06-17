---
layout: post
title: 밑바닥부터 시작하는 DevOps 생존기(2) - Docker
published: True
categories:
- Server
tags:
- Development
- DevOps
- CI/CD

---

로컬에서 문제없이 동작하던 코드라고 하더라도, 설치된 라이브러리나 OS 버전에 따라 해당 소스코드가 동작하지 않을 수도 있습니다.

이번 편에서는 로컬과 서버의 환경이 상이한 경우에도 쉽게 배포가 가능한 Docker 도입기에 대해 소개하려고 합니다.

<!--more-->



Docker는 `Container Virtualization`를 이용하여 소스코드를 격리된 공간에서 동작하도록 합니다. 

소스코드를 동작하는 가상환경에 격리하고, 이미지화하여 서비스 서버에서 동작시킵니다.



> Docker 설치는 아래의 블로그 링크 혹은 공식 문서 참고 바랍니다.
>
> https://shephexd.github.io/server/2018/06/23/How_to_use_docker.html
>
> https://docs.docker.com/get-docker/



## Docker를 이용한 서비스 배포

도커를 이용한 배포는 간단한  `flask` 실습 코드와 함께 설명하도록 하겠습니다.

아래의 소스코드를 Docker를 이용하여 서비스 서버에 배포하려고 합니다.



### 배포 방법

1. `Dockerfile`을 작성한다.
2. 서버에서 Docker image를 빌드한다.
3. 해당 도커 이미지 내에서 소스코드를 실행시킨다.



### 장점

1. 로컬 환경과 서비스 환경이 달라도 도커 가상환경으로 배포가 가능하다.
2. 이미지 태그만 잘 관리하면 서비스 롤백에 용이하다.



### 문제점

1. 로컬에서 빌드한 시점과 서버에서 빌드한 시점의 차이가 큰 경우, 설치되는 패키지가 상이할 수 있다.
2. 동작 스크립트가 복잡한 경우, 별도의 관리가 필요하다.
3. 매번 각각의 서버에서 빌드 과정을 거쳐야 한다.



## Docker를 이용한 서비스 배포 실습



### 예제 코드

`requirements.txt`

```
click==7.1.2
Flask==1.1.2
itsdangerous==1.1.0
Jinja2==2.11.2
MarkupSafe==1.1.1
Werkzeug==1.0.1
```



`app.py`

```python
from flask import Flask
from settings import configs


app = Flask(__name__)


@app.route('/')
def hello_world():
    return f'Hello World! {configs["MESSAGE"]}'


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
```



`settings/__init__.py`

```python
import os
import json

APP_HOME = os.getenv("APP_HOME", os.path.dirname(__file__))
print(APP_HOME)
configs = json.load(open(os.path.join(APP_HOME, 'config.json')))
```



`settings/config.json`

```json
{
  "ENV": "DEVELOPMENT",
  "MESSAGE": "DevOps"
}
```



python을 사용하여 `flask` web server를 8000번 포트에서 동작시킵니다.

```bash
python app.py
```



### Dockerfile 작성

Dockerfile에는 소스코드를 실행시키기 위한 환경에 필요한 패키지를 설치하고 소스코드를 실행시키기 위한 명령어가 정의되어 있습니다.



```dockerfile
FROM shephexd/python:3.6

WORKDIR /webapp/server/
ADD . /webapp/server/
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```



1. `FROM shephexd/python:3.6`: Base Image를 가져옵니다.
2. `WORKDIR /webapp/server/`: 도커 환경 내의 작업 디렉토리를 /webapp/server/ 로 지정합니다.
3. `ADD . /webapp/server/`: 현재 디렉토리의 파일들을 /webapp/server/ 에 추가합니다.
4. `RUN pip install -r requirements.txt`: requirements.txt에 정의된 패키지를 설치합니다.
5. `CMD ["python", "app.py"]`: python으로 app.py를 실행시킵니다.



### Docker image build

소스코드와 같은 경로에 Dockerfile을 두고, 아래의 도커 이미지 빌드 명령어를 실행합니다.

해당 명령어는 현재 경로에 있는 Dockerfile을 참조하여 이미지를 빌드하고, 이미지 태그를 `simple_flask:0.1`로 지정합니다.



```bash
docker build . -t simple_flask:0.1
```



도커 이미지 목록을 확인하시면 `simple_flask:0.1` 이미지를 확인할 수 있습니다.

```
docker images
```



### Docker 이미지 실행

```bash
docker run -d --rm -p 5001:5000 simple_flask:0.1
```



- `-d`: docker가 데몬 모드로 동작합니다.
- `--rm`: 도커 컨테이너가 정지되면 해당 컨테이너를 삭제합니다.
- `-p`: 호스트의 port를 도커 컨테이너로 포트 포워딩시킵니다. (host: 5001 -> container: 5000)



### Docker 동작 확인

```bash
docker ps  #현재 동작 중인 컨테이너 확인
```



```bash
curl -XGET http://localhost:5001
```



## Conclusion

도커로 소스코드의 동작 환경을 이미지화하는 Dockerfile을 작성하고, 해당 이미지를 동작시키는 방법에 대해 소개하였습니다. 

다음 장에서는 여러 서비스가 결합되거나 실행 파라미터가 많을 때 용이한 `docker-compose` 를 이용한 배포를 소개하도록 하겠습니다.


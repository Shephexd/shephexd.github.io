---
layout: post
published: True
title: Python coding
categories: 
- Programming
tags: 
- python
---

We will stduy about something important for our.

- dd
- ddf
- ff

## THINK LIKE PYTHON
**generally It is important to work like python with python.**  
There are specific methods and statements in python. Python programmers emphasize simple and legibility using that. This pattern will be helpful for your work.

### Check your python version
**파이썬은 버전에 따라 차이점이 존재하기 떄문에 현재 사용중인 파이썬의 버전을 확인하는 것이 중요하다.**

파이썬에서 버전확인을 위해 사용하는 방법은 두 가지가 있다.

<!--more-->
1. **터미널**
- 터미널에서 python명령어를 입력하면 파이썬 콘솔이 실행된다. 이때 --version 플래그를 사용하면 버전 확인이 가능하다.

```bash
python --version
```
**`python 2.7.8` 혹은 `python 3.4.3` 와 같은 결과가 나온다.**

**2. sys 모듈 사용**
- 파이썬의 내장된 sys 모듈 안의 값을 조사하여 런타임에 사용 중인 파이썬의 버전 확인이 가능하다.

```python
import sys

print(sys.version_info)
print(sys.version)
```

**>>> sys.version_info(major=3, minor=5, micro=1, releaselevel='final', serial=0)**
**'3.5.1 |Anaconda 4.0.0 (x86_64)| (default, Dec  7 2015, 11:24:55) \n[GCC 4.2.1 (Apple Inc. build 5577)]'**


> 현재는 파이썬 2와 파이썬 3를 혼용해서 사용하고 있지만, 파이썬 2의 개발은 버그 수정, 보안 강화, 파이썬 2를 3로 포팅하는 기능 이외에는 중지된 상태이다. **만약 새 파이썬 프로젝트를 시작한다면 파이썬 3를 사용하는 것을 추천한다.**

### Follow PEP 8 guide
PEP[^PEP] #8 

### Differences Byte, str, unicode

### Use heler method instead of complicated expression

### How to slice sequence

### Don't use start, end, stride in one slice

### Use list comprehension instead of map and filter

### Don't use list comprehension more than two

### Use generation with large comprehension

### Use enumerate instead of range

### Use zip to process iterator in parallel

### Don't use else after repetitive statement like for and while

### Use benefits of each block in try/except/else/finally

## FUNCTION

[^PEP]: Python Enhancement Proposal. It is style guide about how to construct python code. 
---
layout: post
published: True
excerpt: Basic concept about Regular Expression using python re module.
title: 정규표현식(Regular Expression) 정리
categories: Development,python
tags: Development
---

#### Regular Expression Abstarct
1. '.':문자
  - 임의의 한 문자를 의미합니다.  
    ex) a.c -> abc,aec,avb,afc ...
2. '\*':0회 이상
  - 바로 앞의 문자가 없거나 하나 이상인 경우.  
    ex) s*e -> se,see,ssefe ...
3. '\+':1회 이상
  - 바로 앞의 문자가 하나 이상인 경우  
    ex) s+e -> sse,ssee ...
4. '\?':0 또는 1회
  - 바로 앞의 문자가 없거나 하나 뿐인 경우.  
    ex) th?e -> e, the 두 가지의 경우 매칭
5. '\^':시작
  - 문자열이나 행의 처음을 의미.
6. '\$':끝
  - 문자열의 행의 끝을 의미.
7. '()':하위식
  - 여러 식을 하나로 묶을 수 있음.
8. '\\n':일치하는 n번째 패턴
  - 일치하는 패턴들 중 n번째를 선택.

#### re method
compile()
match()
search()
split()
replace()
test()
exec()
findall()


####example re

```python3
import re
m = re.search("(?<=abc)def","abcdef")
m.group(0)
```

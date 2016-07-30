---
layout: post
published: True
excerpt: Basic concept about Regular Expression using python re module.
title: 정규표현식(Regular Expression)
categories: 
- Development
tags:
- Development
- python
---

# Regular Expression Abstarct
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

# re method
compile()
match()
search()
split()
replace()
test()
exec()
findall()

| Method | parameters |   Description   | return Value |
|:-------|:----|:-----|:----|
|  compile  |(pattern,[flags])|pattern을 컴파일 |re object|
|  match  |(pattern, string,[flags])|string의 앞부터 pattern이 존재하는지 검사 | Match Object instance|
|  search  |(pattern, string,[flags])|string의 전체에 pattern이 존재하는지 검사 | Match Object instance |
|  split  |(pattern, string,[maxplit=0])|pattern을 구분자로 string을 분리 | List |
|  findall  |(pattern, string,[flags])|string에서 pattern을 만족하는 문자열을 찾음 | List |
|  finditer  |(pattern, string,[flags])|string에서 pattern을 만족하는 문자열을 찾음 | Iterator |
|  sub  |(pattern, repl, string,[count=0])|string에서 pattern과 일치하는 부분을 repl로 교체 | str |
|  subn  |(pattern, repl, string,[count=0])|string에서 pattern과 일치하는 부분을 repl로 교체 | Tuple  (결과문자열, 매칭횟수)|
|  escape  |(string)|영문자 숫자가 아닌 문자들을 백슬래쉬 처리해서 리턴.| str |


# example re

```
#include <stdio.h>

void main(void){
  printf("hello world");
}
```

```python
import re
m = re.search("(?<=abc)def","abcdef")
.group(0)


```

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

## Regular Expresssion
**특정한 규칙을 가진 문자열의 집합을 표현하는 데 사용하는 형식 언어이다.** - [Wikipedia](https://en.wikipedia.org/wiki/Regular_expression)

### Basic concept
**주로 패턴(pattern)으로 부르는 정규 표현식은 특정 목적을 위해 필요한 문자열 집합을 지정하기 위해 쓰이는 식이다.**

Boolean "OR"
: 수직선(`|`)은 여러 항목 중 선택을 하기 위해 구분한다. 
이를테면 `gray|grey`는 "gray" 또는 "grey"와 일치한다.

Grouping
: 괄호(`()`)를 사용하면 연산자의 범위와 우선권을 정의할 수 있다. 
이를테면 gray|grey와 gr(a|e)y는 "gray"나 "grey" 집합을 둘 다 기술하는 동일 패턴이다.

Quantification

| Method |   Description   |
|:-------|:----|:-----|:----|
|?|	물음표는 0번 또는 1차례까지의 발생을 의미한다. 이를테면 colou?r는 "color"와 "colour"를 둘 다 일치시킨다.|
|*	|별표는 0번 이상의 발생을 의미한다. 이를테면 ab*c는 "ac", "abc", "abbc", "abbbc" 등을 일치시킨다.|
|+|	덧셈 기호는 1번 이상의 발생을 의미한다. 이를테면 ab+c는 "abc", "abbc", "abbbc" 등을 일치시키지만 "ac"는 일치시키지 않는다.|
|{n}	|정확히 n 번만큼 일치시킨다.|
|{min,}	|"min"번 이상만큼 일치시킨다.|
|{min,max}|	적어도 "min"번만큼 일치시키지만 "max"번을 초과하여 일치시키지는 않는다.|

## Syntax

### POSIC Basic

### POSIC Extend

### String class

'.'
: 임의의 한 문자를 의미  
`ex) a.c -> abc,aec,avb,afc ...`

'\*': 0회 이상
: 바로 앞의 문자가 없거나 하나 이상인 경우.  
`ex) s*e -> se,see,ssefe ...`

'\+': 1회 이상
: 바로 앞의 문자가 하나 이상인 경우  
`ex) s+e -> sse,ssee ...`

'\?': 0 또는 1회
: 바로 앞의 문자가 없거나 하나 뿐인 경우.  
`ex) th?e -> e, the 두 가지의 경우 매칭`

'\^': 시작
: 문자열이나 행의 처음을 의미.

'\$': 끝
: 문자열의 행의 끝을 의미.

'()': 하위식
: 여러 식을 하나로 묶을 수 있음.

'\\n':일치하는 n번째 패턴
: 일치하는 패턴들 중 n번째를 선택.

## re method
compile()
match()
search()
split()
replace()
test()
exec()
findall()

| Method  |   Description   | return Value |
|:-------|:-----|:----|
|  `compile(pattern,[flags])`|pattern을 컴파일 |re object|
|  `match(pattern, string,[flags])`|string의 앞부터 pattern이 존재하는지 검사 | Match Object instance|
|  `search(pattern, string,[flags])` |string의 전체에 pattern이 존재하는지 검사 | Match Object instance |
|  `split(pattern, string,[maxplit=0])` |pattern을 구분자로 string을 분리 | List |
|  `findall(pattern, string,[flags])` |string에서 pattern을 만족하는 문자열을 찾음 | List |
|  `finditer(pattern, string,[flags])` |string에서 pattern을 만족하는 문자열을 찾음 | Iterator |
|  `sub(pattern, repl, string,[count=0])` |string에서 pattern과 일치하는 부분을 repl로 교체 | str |
|  `subn(pattern, repl, string,[count=0])` |string에서 pattern과 일치하는 부분을 repl로 교체 | Tuple  (결과문자열, 매칭횟수)|
|  `escape(string)` |영문자 숫자가 아닌 문자들을 백슬래쉬 처리해서 리턴.| str |


### example re

``` python
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

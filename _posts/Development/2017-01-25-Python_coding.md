---
layout: post
published: True
title: Python coding
categories:
- Development
tags:
- Python
---

파이썬을 주언어로 즐겨 사용해왔지만 내장함수나 파이썬만의 표현 방식에 대해서는 잘 알지 못했었다.  
특히 종종 다른 사람의 코드를 리뷰하던 중에 이해가 안되는 문법이나 표현을 접하는 경우가 있어 이번 기회에 새로 학습하기로 했다.  
**"Think like python"**[^book] 에서는 파이썬만의 확장된 기능과 문법 등을 통해 파이썬을 좀더 파이썬답게 쓰는 방법에 대해서 설명하였다.



<!--more-->



## THINK LIKE PYTHON

generally It is important to work with **python like python**.  
There are specific methods and statements in python. Python programmers emphasize simple and legibility using that. This pattern will be helpful for your work and projects.


### Check your python version

1. Terminal

```bash
python --version
```
`python 2.7.8` or `python 3.4.3`

2. Using sys module

<script src="https://gist.github.com/Shephexd/397b032ef16f41fc9f736a0c7a95d617.js?file=01_version_check.py"></script>



### Follow PEP 8 guide

PEP[^PEP] is guide for the developer to keep in mind to improve productivity and cooperation with others.



1. Whitespace

   - Using space instead of tab.
   - Use 4 space for Indent having meaningful syntax.
   - The maximum length for one sentence is 79.
   - If the expression is too long, use the indent (4 space).
   - In file, separate function and class using two blank lines.
   - In class, method has one blank line for separation.
   - List index, call fund, allocating keyword don't have any space.
   - Allocating variable have space before and after it. `a = 1`

2. naming

   - **Functions, variables, attributes** follow `lowercase_underscore`
   - **Protected instance attributes** follow `_leading_underscore`
   - **Private instance attributes** follow `__double_leading_underscore`
   - **Classes and exceptions** follow `CapitalizedWord`
   - **Module level constants** follow `ALL_CAPS`
   - **In class and instance method,** first parameter(for instance itself) denote `self`
   - **In class method,** first parameter(for class itself) denote `cls`

3. Expression and sentences use

   - **Use inline positive** `if a is not b` instead of negative of positive expression `if not a is b`

   - Consider blank value as a `False`

   - Consider non-blank value as a `True`

     ​

### Differences Byte, str, unicode

<script src="https://gist.github.com/Shephexd/397b032ef16f41fc9f736a0c7a95d617.js?file=03_string_encoding.py"></script>



### Use helper method instead of complicated expression

<script src="https://gist.github.com/Shephexd/397b032ef16f41fc9f736a0c7a95d617.js?file=04_make_helper.py"></script>


### How to slice sequence

<script src="https://gist.github.com/Shephexd/397b032ef16f41fc9f736a0c7a95d617.js?file=05_slice_sequence.py"></script>


### Don't use start, end, stride in one slice

<script src="https://gist.github.com/Shephexd/397b032ef16f41fc9f736a0c7a95d617.js?file=06_slice_caution.py"></script>


### Use list comprehension instead of map and filter

**List comprehension** is more effective way comparing filter and map.

<script src="https://gist.github.com/Shephexd/397b032ef16f41fc9f736a0c7a95d617.js?file=07_list_comprehension.py"></script>



### Don't use list comprehension more than two

avoid the case of using more than two comprehension in one list. Consider to make helper or loop.

<script src="https://gist.github.com/Shephexd/397b032ef16f41fc9f736a0c7a95d617.js?file=08_list_comprehension_caution.py"></script>


### Use generator with large comprehension

When you save the huge data into the list, you'd better use generator for keeping your memory.

<script src="https://gist.github.com/Shephexd/397b032ef16f41fc9f736a0c7a95d617.js?file=09_generator.py"></script>



Keep your mind about generator can print out only **one time** using next()



### Use enumerate instead of range

<script src="https://gist.github.com/Shephexd/397b032ef16f41fc9f736a0c7a95d617.js?file=10_enumerate.py"></script>



### Use zip to process iterator in parallel

<script src="https://gist.github.com/Shephexd/397b032ef16f41fc9f736a0c7a95d617.js?file=11_zip_for_parallel.py"></script>


There are **two problem** when you use `zip`

1. In python2, zip is not generator. So you should use `izip` in  `itertools` .
2. When the length in two list is different, zip doesn't work well. you can use `zip_longest` in `itertools`(`zip_longest` in python2)

Check the length of the lists before using `zip`



### Don't use else after repetitive statement like for and while

When will the `else`  work in the python loop?

<script src="https://gist.github.com/Shephexd/397b032ef16f41fc9f736a0c7a95d617.js?file=12_loop_caution.py"></script>

This is easy to understand. Never use `else` in the loop.



### Use benefits of each block in try/except/else/finally

<script src="https://gist.github.com/Shephexd/397b032ef16f41fc9f736a0c7a95d617.js?file=13_try_except_else.py"></script>



## FUNCTION

Function is first basic tool for python developers. To improve readability and make more easily understandable code, it is useful for reuse and refactoring.



### Raise exception instead of returning None

<script src="https://gist.github.com/Shephexd/397b032ef16f41fc9f736a0c7a95d617.js?file=14_raise_exception.py"></script>



### How to work close in the variable scope

In the python, function is first class.

<script src="https://gist.github.com/Shephexd/397b032ef16f41fc9f736a0c7a95d617.js?file=15_variable_scope.py"></script>

### Use generator instead of returning list

A generator can save your memory comparing a list when the size is too big.

<script src="https://gist.github.com/Shephexd/397b032ef16f41fc9f736a0c7a95d617.js?file=16_return_generator.py"></script>



### Defensive iteration

When you use the iterative loop for a iterator, It might be a problem.

```python

```

<script src="https://gist.github.com/Shephexd/397b032ef16f41fc9f736a0c7a95d617.js?file=17_defensive_iteration.py
"></script>


### To make clear, use variable position argument

<script src="https://gist.github.com/Shephexd/397b032ef16f41fc9f736a0c7a95d617.js?file=18_variable_position_arguemnt.py"></script>



### Offer optional action by using keyword argument

<script src="https://gist.github.com/Shephexd/397b032ef16f41fc9f736a0c7a95d617.js?file=19_keyword_paramters.py"></script>


### Use None and docstring for dynamic basic argument

<script src="https://gist.github.com/Shephexd/397b032ef16f41fc9f736a0c7a95d617.js?file=20_dynamic_default_param.py"></script>


### Emphasize the clearness of argument for keyword.

<script src="https://gist.github.com/Shephexd/397b032ef16f41fc9f736a0c7a95d617.js?file=21_param_for_keywords.py"></script>

### Use helper class

<script src="https://gist.github.com/Shephexd/397b032ef16f41fc9f736a0c7a95d617.js?file=21_param_for_keywords.py"></script>

### Use function instead of class for argument
<script src="https://gist.github.com/Shephexd/397b032ef16f41fc9f736a0c7a95d617.js?file=21_param_for_keywords.py"></script>


[^PEP]: Python Enhancement Proposal. It is style guide about how to construct python code.
[^book]: "Effective python 59: specific ways to write better python" in English, 파이썬 코딩의 기술

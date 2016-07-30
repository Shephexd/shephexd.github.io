---
layout: post
published: False
excerpt: Basic concept about Regular Expression using python re module.
title: System Software
categories: 
- Development
tags: 
- Development
---

[TOC]

#What is System programming?
Computer consists of H/W and S/W.

How to match S/W to H/W or H/W to SW.

```
Coding:Editor -> Source Program:compiler -> Object Program:Linker -> Load:Module -> Memory
```

## Introduction

1. CPU
  - 산술논리연산장치(ALU: Arithmetic Logic Unit)
    : 컴퓨터 명령어 내에 있는 연산자들에 대한 연산과 논리 동작을 처리
  - 제어장치(CU: Control Unit)
    : 제어 신호 발생
2. 기억장치(Memory)
  - RAM(Random Access Memory)
    : 읽기,쓰기 가능, 휘발성
  - ROM(Read Only Memory)
    : 읽기만 가능, 비휘발성
3. 레지스터(Register)
  - 범용 레지스터
  - 특수 목적 레지스터
    1. PC(Program Counter)
      : 다음에 수행할 명령어의 메모리 주소를 가리킴
    2. IR(Instruction Register)
      : 현재 수행중인 명령어를 보관
    3. PSW(Program Status Word)
      : 현재 프로 그램의 상태를 보관, Carry, Overflow, Zero 등의 상태와 interrupt 발생유무를 표시

### 어셈블러(Assembler)
- 어셈블러
  : 어셈블리어를 기계어로 자동적으로 변환해주는 프로그램
- 어셈블리어
  : 기계어와 1:1로 매칭되는 기호(mnemonic or symbol)
- 로더
  : 프로그램들을 기억 장치에 놓고 수행할 수 있또록 준비하는 프로그램

### 로더 기능(Loader function)
1. 할당(Allocation)
: 프로그램을 위한 기억 장소 **할당(allocation)**
2. 연결(Linking)
: 목적 프로그램 간의 기호적 **호출 해결(linking)**
3. 조정(relocation)
: 주소 상수같이 주소에 종속되는 부분을 할당도니 기억 장소에 일치하도록 **조정(relocation)**
4. 적재(Loading)
: 실제적으로 기계어 명령들과 자료를 기억 장치에 **적재(Loading)**

### 매크로 프로세서(Macro Processor)
- macro definition: 특정 구간을 M으로 정의
- macro call: M을 호출
- macro expansion: 매크로 확장

### 컴파일러(Compiler)
- 고급 언어로 작성된 프로그램을 받아 목적 프로그램으로 만드는 프로그램

### 인터프리터(Interpreter)
- 원시 프로그램을 마치 기계어 프로그램인 것처럼 수행하는 프로그램. 기계어 프로그램을 만들지 않고 원시프로그램들이 지시하는 내용을 인터프리터가 직접 대신 실행

**Basic 4 step**
1. statement counter가 가리키는 문장을 source program으로부터 가져온다.
2. statement counter를 증가
3. statement를 분석하여 어떤 동작을 어느 operator에 행해야 할지를 결정
4. 해당 subroutine을 call함으로써 문장을 실행

고급언어 -> 어셈블리어 변환 예제  

```A = B+C+D;```

**위의 코드를 어셈블러로 변환**
``` nasm
LOAD R1,B // B에 있는 내용을 R1에 로드: 581200
ADD R1,C  // R1과 C를 더한 값을 R1에 저장: 5A1300
ADD R1,D  // R1과 D를 더한 값을 R1에 저장: 5A1400
STORE R1,A  // R1의 값을 A에 저장: 501100
```

P.S.
수행 시간(Execution time)
: 사용자의 프로그램을 수행하는데 걸린 시간
컴파일 시간(Compile time)
: 사용자의 원시 프로그램을 번역하는데 걸린 시간
적재 기간(Load time)
: 목적 프로그램을 적재하고 수행하기 위해 준비하는 시간

## Machine Code, Assebly Language

### Computer H/W structure
- 오늘날 대부분의 컴퓨터 구조는 폰노이만이 구축한 stored program computer의 개념에 기초


> MAR(Memory Address Register): 사용한 메모리 address를 보관하며 MAR이 가리키는 번지의 내용을 R/W 가능
> MBR(Memory Buffer Register): memory에 R/W할 data를 보관

|	Read process	|	Write Process	|
|:------------:	|:------------:	|
|	MAR <- addr	|	MAR<-Addr		|
|	Read signal	|	MBR < Data	|
|	MAR <- addr	|	Write signal	|

**명령어 수행단계(instruction cycle)**

1. Fetch cycle
	- indirect cycle
2. execute cycle
	- interrupt cycle : interrupt가  fetch단계에서 걸려도 interrupt cycle 실행


### IBM 360/370 H/W Structure 
1. Memory
	- **Memory basic size - byte**
	- byte - 8bits
	- halfword - 2bytes(16bits)
	- fullword - 4bytes(32bits)
	- double word - 8bytes(64bits)

2. Register
- 32bits 범용 레지스터(general register) 16개
- 64bits 부동 소수점 레지스터(flatting point register) - 4개
- 64bits PSW(Program status word) - 1개

> 유효주소(EA: Effective Address) = Offset(Displacement) + data of Base register + data of index register

**..**

Disp = Symbol Value - base reg data

- ST
- LT
- BT(Base Table)

- RR Format
- RX Format

## Macro and Macro Processor
### Macro 정의

- 형식 매개변수(formal parameter): &p0,&p1,
- 레이블 매개변수(label parameter): &p0
- 매크로 프로세스의 출력은 확장된 원시 프로그램

```
MACRO
ADD3		#Macro definition
A	2,DATA1
A	2,DATA2
A	3,DATA3
MEND
```

# 걍해라


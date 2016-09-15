---
layout: post
published: True
title: Big Data Introduction
categories:
- Big data
tags:
- Big data
---

## What is Big data?  
Big data is a term for data sets that are **so large or complex that traditional data processing applications are inadequate to deal with them**. - [wikipedia](https://en.wikipedia.org/wiki/Big_data)

### 빅 데이터의 3대 요소(3V) - Volume, Velocity, Variety

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

<!--more-->

---
layout: post
title: Storage
published: True
Categories:
- Infra
Tags:
- Infra
- Storages

---

인프라에서 저장소의 역할을 담당하는 스토리지의 개념에 대해서 정리하였습니다. 인프라 관련 도서[^2]를 참고하여 작성하였습니다.



<!--more-->



## 스토리지란?

스토리지는 디지털 데이터를 유지하는데 사용되는 컴퓨터 구성 요소와 기록 매체로 구성되어 있습니다. 일반적으로 전원이 꺼지면 데이터가 손실되는 휘발성 기술을 `메모리`라고 하고, 비휘살성 기술을 `스토리지`라고 부릅니다.

> wikipedia[^1] 참조



## 스토리지의 종류

### DAS

DAS(Directed Attached Storage)로 서버에 직접 연결한 스토리지를 의미합니다. 가장 전통적인 방식으로 서버와 스토리지가 직접 연결되어 있어 별도의 네트워크 연결이 필요없습니다.



### NAS

NAS(Network Attached Storage)는 서버와 스토리지가 각각 이더넷 케이블로 연결되어 통신합니다. 



### SAN

SAN(Storage Area Network)는 SAN 스위치를 통해 서버와 스토리지를 연결하여 성능이나 확장성을 보장하는 연결 방식입니다.



## 스토리지 성능 확장

스토리지의 가용성 및 성능 개선을 위한 방법을 소개해드리고자 합니다. 확보



- 미러링: 같은 데이터를 복제하여 저장
- 스트라이핑: 데이터를 여러 개의 디스크에 분산하여 저장



[^1]: https://en.wikipedia.org/wiki/Computer_data_storage
[^2 ]: 정송화, 김엉선, 전성민, 2018, 개발자도 궁금한 IT 인프라, http://www.kyobobook.co.kr/product/detailViewKor.laf?mallGb=KOR&ejkGb=KOR&barcode=9791188621224
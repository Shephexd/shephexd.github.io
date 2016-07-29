---
layout: post
published: True
title: ubuntu 서버 구축
excerpt: Introduction about basic method to install APM(Apache,Php,Mysql)
categories: Programming
tags:
- server
- ubuntu
- Development
---

##우분투 설치
우분투 [공식사이트]에서 iso이미지 설치가 가능하다.

설치방법

	설치방법
	1. vmware를 통한 가상 구축
	2. *실제 하드디스크에 설치
	3. *듀얼부팅

##APM 설치
우분투에서 가장 서버구축을 하기 위해, 먼저 apm(apache,php,mysql)을 설치합니다.

	sudo apt-get install phpmyadmin mysql-client mysql-server php5-common apache2 php5-mysql

1. 서버는 apache로 선택
2. mysql관리자 비밀번호 설정 후,
3. 아파치에서 한글 설정.
	/etc/apache2/apach2.conf에서 한글 설정을 위하여
	*"AddDefaultCharset utf-8"*
	을 추가하여줍니다. (나중에 수정할때 편하게 주석까지 같이 추가)

##서버 설정

###~로 사용자별 홈페이지에 접속하게 만들기
>리눅스에 각각의 계정에 맞게 다른 홈페이지에 접속하기 위해서 필요한 설정

예를 들면 sample.co.kr/~creator , sample.co.kr/~admin

이렇게 두 곳의 홈페이지를 구별 가능하게 만들기 위한 설정입니다.

각 사용자 계정의 홈 디렉터리에 있는 html 코드 별로 수행되는겁니다.

	cd /etc/apache2/mods-enabled
	ln -s /etc/apache2/mods-available/userdir.load .
	ln -s /etc/apache2/mods-available/userdir.conf .
	vi /etc/apache2/mods-enabled/userdir.conf

에서 두번째줄에 public_html이 있는지 확인합니다.(기본값이에요)


아파치서비스를 재시작하고..

	/etc/init.d/apache2 restart
	cd ~
	mkdir public_html
	cd public_html
	cp /var/www/html/index.html .
디렉토리 생성할 때는 각 사용자 계정 별로 만드는 것을 추천
***생성권한 문제로 인해..물론 바꿀수있지만 번거로움***

	ifconfig | grep "inet addr

본인의 아이피 주소를 확인하고,

	http://192.168.23.254/~creator
	http://127.0.0.1/~creator	#loopback주소

접속하면 apache 초기화면에 접속된다

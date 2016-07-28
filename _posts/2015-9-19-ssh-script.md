---
layout: post
published: True
title: ssh 자동화 스크립트 제작
excerpt: Make ssh auto connection script using python.
categories: Programming
tags:
- hack
- python
- ssh
- development
- ip scanning
---

## 소스코드

#### 여러 ip목록에서 id,pw값을 기준으로   접속을 시도하는 스크립트이다.

```ruby
# This is highlighted code
def foo
  puts 'foo'
end
```

```py
# Here is some in python
import time

def foo():
  print 'foo'
```

```python
"""hydra URL Configuration
The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.9/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""

from django.conf.urls import url
from django.contrib import admin

urlpatterns = [
    url(r'^admin/', admin.site.urls),
]

import requests
from bs4 import BeautifulSoup


def gaon_top_rank(year, week) :
    """
    # 샘플 리스트
    y_li = range(2010, 2016+1)       # y_li[5] = 2015
    w_li = range(1,52+1)             # w_li[45] = 46
    year = str(y_li[5])
    week = str(w_li[43])
    """
    print(year,"년도", week, "주차 가온차트 목록 데이터 수집중...")
    year = str(year)
    week = str(week)
    url = "http://www.gaonchart.co.kr/main/section/chart/online.gaon?nationGbn=T&serviceGbn=ALL&targetTime=%s&hitYear=%s&termGbn=week"  %(week, year)
    res = requests.get(url)
    #print (res.text)

    soup = BeautifulSoup(res.text, 'html.parser')
    subject = list(soup.find_all("td", class_="subject"))
    production = list(soup.find_all("td", class_="production"))
    ranking = list(soup.find_all("td", class_="ranking"))
    param_album = list(soup.find_all("div", class_="chart_play"))

    #print(param_album[0])
    #idx = 0

    top_rank = []
    for idx in list(range(100)) :
        ## 가온차트 정보 수집
        soup0 = BeautifulSoup(str(subject[idx]), 'html.parser')
        title = soup0.p["title"]    # 음원제목

        song_info = soup0.find("p", class_="singer")["title"]
        artist = list(song_info.split(" | "))[0]    # 아티스트명
        album = list(song_info.split(" | "))[1]     # 앨범명

```
##소스코드 설명
####id와 pw값에 대하여 ip_list파일에 대하여 ssh접속시도를 한 후, 결과를 result_ssh에 대한 결과를 저장한다.

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

##소스코드
#### 여러 ip목록에서 id,pw값을 기준으로   접속을 시도하는 스크립트이다.

{% highlight c %}
#include <stdio.h>

printf("%d",111);

{% enhighlight%}

```ruby
# This is highlighted code
def foo
  puts 'foo'
end
```
```python
# Here is some in python
def foo():
  print 'foo'
```

~~~python

import pxssh
import time
import pexpect

data = open('ip_list','r')
result_ssh=open('result_ssh','w')

count=0

id = raw_input()
pw = raw_input()
for target_ip in data:

     ip = target_ip[:-1]

    result_ssh.write(ip)
    print "start ssh to "+ip

    try:
        s = pxssh.pxssh()
        s.login(ip,id, pw,port=22)	# run a command
        s.prompt()			# match the prompt
        print(s.before)		# print everything before the prompt.
        s.sendline('ls -l')
        s.prompt()
        print(s.before)
        s.logout()
        result_ssh.write('\nlogin success\n')

    except pxssh.ExceptionPxssh as e:
        print("pxssh failed on login.")
        result_ssh.writelines('\nlogin fail\n')

    except pexpect.EOF as e:
        print "EOF"
        result_ssh.writelines('\nEOF error\n')

    result_ssh.writelines('=============\n\n')

count+=1
time.sleep(30)
~~~

##소스코드 설명
####id와 pw값에 대하여 ip_list파일에 대하여 ssh접속시도를 한 후, 결과를 result_ssh에 대한 결과를 저장한다.

---
layout: post
title: Debugging C++ with Visual Studio Code
published: True
categories:
- Development
tags:
- Development
- C++
typora-root-url: /Users/shephexd/Dropbox/github/pages/
---



OS X에서 Visual Studio Code를 이용하여 C++코드를 작성하고, 디버깅 하는 방법을 정리해보고자 한다.

[공식 사이트](https://code.visualstudio.com/?wt.mc_id=DX_841432) 에서 다운로드 후 설치 가능하다.



> VisualStudioForMac을 설치해봤지만...현재 Mac Version은 C++ 디버깅을 지원하지 않는다고 한다.



<!--more-->



## 환경 설정

Visual Studio Code에서 디버깅을 하기 위해서는 아래의 3가지가 필요하다.



### 실행 환경

- Mac OS Mojave
- LLDB



1. 확장 도구 설치
2. Build Task 설정
3. Debugging 설정



### 확장 도구 설치

VS extensions에서 C++ 확장자 파일을 작성시, Recommendation으로 나타나는 목록이다.



![c++확장프로그램](/assets/post_images/Development/vs_code_cpp_extensions.png)



맨위의 `C/C++` 이 디버깅을 위한 **필수설치** 확장 도구이다.

`Code > Preference > Extensions` 에서 검색 후 설치 가능하다.



### Build Task 설정

위의 확장 도구 설치 후, 해당 C++ 코드에 대한 Build Task를 설정해야 한다.



`Terminal > Configure Tasks` 에서 작성한 `hello_world.cpp` 파일에 대한 build task를 설정해보자.



#### tasks.json

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "build hello world",
            "type": "shell",
            "command": "g++",
            "args": [
                "-g",
                "hello_world.cpp"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "problemMatcher": [
                "$gcc"
            ]
        }
    ]
}
```



위의 설정을 적용 후, `Terminal > Run Build Task` 선택 후, `build hello world` 를 실행시켜보자.

현재 디렉토리에 `a.out`이라는 실행 파일이 생성된다.



### 디버깅 설정

디버깅(F5)를 실행하면 아래의 기본 설정이 나타난다.

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "(lldb) Launch",
            "type": "cppdbg",
            "request": "launch",
            "program": "enter program name, for example ${workspaceFolder}/a.out",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": true,
            "MIMode": "lldb"
        }
    ]
}
```



기본 설정으로 실행해보니 아래의 에러 메세지가 나타났다.

*launch: program 'enter program name, for example /Users/shephexd/Dropbox/github/Algorithms/a.out' does not exist*



`launch.json` 의 `program` 값이 커맨드 상에서 그대로 실행되는 것으로 보인다.

`"program": "enter program name, for example ${workspaceFolder}/a.out"`



#### launch.json

기본 설정에서 빌드되는 실행 파일 경로만 수정하였다.

```json
{
    "version": "0.2.0",
    "configurations": [

        {
            "name": "(lldb) Launch Hello world",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/a.out",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": true,
            "MIMode": "lldb"
        }
    ]
}
```



"현재 디렉토리에 빌드된 `a.out` 파일을 디버깅한다" 설정이다.



## Debugging C++ with Visual Studio Code

위의 설정을 이용하여, 아래의 간단한 프로그램을 디버깅하려고 한다.

```c++
#include <iostream>

using namespace std;

int main(){
    int a = int();
    cout << "hello world!" << endl;
    cout << a << endl;
    a++;
}
```



`a` 값이 증가하기 바로 전줄에 BreakPoint를 걸어두고, 빌드 후 해당 코드를 디버깅해보자.



![디버깅 화면](/assets/post_images/Development/vs_code_cpp_debugging.png)



## 한계점

OSX에서 LLDB로 디버깅시 아래와 같은 현상이 발생한다.

- 디버깅을 실행시 터미널 창이 자동으로 생성된다. 디버깅 도중 터미널 창을 닫을시, 디버깅이 종료된다.
- 디버깅이 종료되더라도 해당 터미널은 자동으로 닫히지 않는다.



`GDB` 사용 혹은 다른 운영체제에서의 디버깅은 [공식 홈페이지](https://code.visualstudio.com/docs/languages/cpp)를 참고바란다.



## 결론

`C++` 로 작성된 오픈소스를 분석하기 위해 많이 쓴다고하는 `Visual Studio Code`를 이용한 디버깅 환경을 구성해보았다.

다양한 운영체제에서 지원하고 가볍고 UI도 깔끔한 편이라 종종 사용할 수 있을 것 같다.



이 후, `automake` 등 빌드툴로 구성된 프로젝트의 디버깅 환경 구성을 정리해보고자 한다.
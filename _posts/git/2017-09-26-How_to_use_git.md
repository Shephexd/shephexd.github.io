---
layout: post
published: True
title: How to use git
categories:
- Git
tags:
- Git
- Development
---

개발자의 필수도구이자 협업도구로 널리 사용하고 있는 git에 대하여 소개하고, 효과적인 사용방법과 흐름에 대하여 설명하고자 한다.



<!--more-->



## Git이란?

Git는 분산 처리 버젼관리 시스템으로 개발자가 소스코드를 관리하고 협업할 수 있는 도구이다. Git은 기존의 SVN시스템과 다르게 각각의 버젼마다 시간순으로 Snapshot을 가지고 있고, 원격 저장소와 로컬 저장소에 작업 환경이 구축되기 때문에, 서버의 상태와 상관없이 빠르게 로컬에서 코드 관리가 가능하다.



![](/assets/images/articles/git/gitflow.png)





Git에서는 파일을 두가지의 경우(Tracked, Untracked)로 구분한다.

`.gitignored` 파일에 등록된 패턴과 일치하는 파일의 경우에는 Untracked의 파일로 구분하여, File의 변화를 추적하지 않는다.



Tracked 파일은 3가지의 상태로 구분된다.

1. Commited - 로컬에 저장된 상태
2. Modified - 수정된 이후, commit되지 않은 상태
3. Staged - 커밋할 준비가 되어있는 상태



## Git의 주요 용어

Repository

:git의 히스토리, 태그, 소스 등이 저장되는 저장소를 의미하며, 저장소의 위치에 따라 로컬 저장소(Local repository)와 원격 저장소(Remote repository: Github, bitbucket, Gitlab 등)로 구분됨

Working Tree

: 저장소를 어느 한 시점을 바라보는 작업자의 현재 시점

Staging rea

: 저장소에 커밋을 하기전에 커밋을 준비하는 단계

Commit

: 현재 변경된 작업 상태를 점검을 마치며 확정하고 저장소에 저장하는 단계

Head

: 현재 작업중인 Branch를 의미

Branch

: 분기점을 의미하며, 현재 상태를 복사한 분리된 작업환경으로 주로 새로운 이슈나 기능 추가 부분에 대해서 분리하여 작업할 때 사용됨

Merge

: Merge는 현재 작업하던 내용을 다른 Branch와 합치는 것을 의미

Master Branch

: Git을 시작할 때 가장 먼저 생성되는 Branch로 프롤젝트의 다른 Branch들이 최종적으로 Merge되는 Main Branch를 의미



## Git 초기 설정

시스템에서 Git을 사용하기 전에, 사용자의 기본 정보를 등록할 수 있다.



### 설정파일

1. `/etc/gitconfig`: 시스템의 모든 사용자와 모든 저장소에 적용되는 설정으로, `git config --system` 옵션으로 파일을 읽고 쓸 수 있다. 참조의 우선순위가 가장 낮다.
2. `~/.gitconfig`: 특정 사용자의 설정으로, `git config --global` 옵션으로 파일을 읽고 쓸 수 있다.
3. `.git/config`: 특정 디렉토리의 git에 있는 설정파일로, 해당 저장소에만 적용된다. 참조의 우선순위가 가장 높다.



### 설정 명령어

`--system` 옵션을 이용하면 사용자의 모든 시스템에서 사용가능한 gitconfig 파일(`/etc/gitconfig`)에 저장된다.

```bash
git config --system user.name "shephexd"
git config --system user.email "shephexd@gmail.com"
```



`--global ` 옵션을 이용하면 특정 사용자가 사용가능한 gitconfig 파일(`~/.gitconfig`)에 저장된다.

```bash
git config --global user.name "shephexd"
git config --global user.email "shephexd@gmail.com"
```



`--global ` 옵션을 이용하면 특정 사용자가 사용가능한 gitconfig 파일(`.git/config`)에 저장된다.

```bash
git config --global user.name "shephexd"
git config --global user.email "shephexd@gmail.com"
```



사용자 정보를 설정한 후, git에서 사용할 기본 텍스트 편집기를 고를 수 있다. 기본적으로 vi나 vim으로 설정되어 있다.

```bash
git config --global core.editor vim #(vi or emacs are available)
```



Merge 충돌을 해결하기 위해 사용하는 Diff도구 역시 설정가능하다.

```bash
git config --global merge.tool vimdiff
```



### 설정확인하기

```bash
git config --list
```





## Git 시작하기

`git init` 명령어를 통해 현재 디렉토리를 git저장소로 만들 수 있다. 



```bash
git init
git add README.md
git commit -m "initialize project"
```





## Git commit Message rule



1. 제목과 본문을 빈 행으로 분리한다.
2. 제목 행을 `50자로 제한`한다.
3. 제목 행 첫 글자는 `대문자`로 쓴다.
4. 제목 행 끝에 `마침표`를 넣지 않는다.
5. 제목 행에 `명령문`을 사용한다.
6. 본문을 72자 단위로 개행한다.
7. `어떻게` 보다는 `무엇`과 `왜`를 설명한다.





## Git Flow

성공적인 Git branching 모델로 Vincent driessen이 제안한 Gitflow를 소개한다.



![](/assets/images/articles/git/gitflow_with_text.png)



Master

: 프로젝트의 전체 흐름, Tag를 이용한 버전 관리







[Successful-gitflow]: http://nvie.com/posts/a-successful-git-branching-model/


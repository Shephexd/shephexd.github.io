---
layout: post
published: True
title: Stochastic Natural Language Processing
categories:
- Machine learning
tags:
- Data mining
- NLP
- Linear Algebra
---



자연어 처리에서 사용되는 다양한 모델들 중 확률 기반하여 토픽이나 키워드 추출에 사용되는 여러 모델들이 있다.

자연어 처리에 기본적으로 사용되는 확률 기반 모델들(TF-IDF, BM25, N-gram, LDA, LSA, pLSA)를 소개한다.



<!--more-->



## TF-IDF

TF는 Term Frequency를, DF는 Document Frequency를 의미한다. 여기서 IDF는 Inverse Document Frequency로 DF에 역가중치를 준 값을 의미한다. TF-IDF에 주요한 가정은 전체 문서에서의 발생 빈도는 적지만, 단어의 발생빈도는 높은 값을 중요한 키워드라고 본다. 즉, 특정 문서들에서 빈번하게 발생한 키워드를 찾아내는 원리이다.



### TF(Term frequency)

1. Boolean frequency
   $$
   \cases{
   \text{TF}(t,d) = 1 & \text{ if $t$가 문서내에 한번 이상 등장} \\
   \text{TF}(t,d) = 0 & \text{ 그렇지않으면 0} \\
   }
   $$
   ​

2. Log scale frequency
   $$
   \text{TF}(t,d ) = 0.5 + \frac{0.5 \times f(t,d)}{max\{ f(w,d): w \in d\}}
   $$
   ​

3. augmented frequency(증가 빈도)

$$
\text{TF}(t,d ) = 0.5 + \frac{0.5 \times f(t, d)}{max\{f(w,d)  w \in d\}}\\
$$



### IDF(Inverse Document Frequency)

$$
\text{IDF}(t,D) = log \frac{\vert D \vert}{1+\vert \{ d \in D : w \in d\}\vert}
$$

- $\vert D \vert$: 문서 집합 D의 크기 혹은 전체 문서의 수

- $\vert \{ d \in D : t \in d\} \vert$: 단어 t가 포함된 문서의 수.(즉 $tf(t,d) \neq 0$), 단어가 전체 말뭉치 안에 존재하지 않을 경우, 이는 분모가 0이듸므로, 보통 이를 막기 위해 1을 더함

  ​

### TF-IDF equation

$$
\text{TFIDF}(t, d, D) = tf(t,d) \times idf(t,D)
$$



## BM 25

BM25는 검색엔진인 Elastic Search에서 기본으로 사용하는 랭킹함수 해당 키워드와 문서의 관계도를 기반으로 순위 매칭을 통해 가장 유사한 문서를 찾아내기 위한 알고리즘이다. 기본적으로는 $TF-IDF$ 의 아이디어와 유사하지만, optimization parameters와 문서의 길이를 고려한다는 점에서 차이가 있다.



### Equation

$$
score(D, Q) = \sum^n_{i=1}IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1+1)}{f(q_i, D)+k_1 \cdot (1-b +b \cdot \frac{\vert D\vert}{avgdl})}
$$



위의 식에서 TF부분과 IDF부분을 분리해서 보는게 이해하기 쉽다.

#### TF

$$
\frac{f(q_i, D) \cdot (k_1+1)}{f(q_i, D)+k_1 \cdot (1-b +b \cdot \frac{\vert D\vert}{avgdl})}
$$

- $f$: 문서에 매칭된 키워드 수
- $k_1$: tf 가중치 값
- $b$ : field(문서)에 대한 가중치
- $\vert D \vert$: 문서 집합 D의 크기 혹은 전체 문서의 수
- Avgdl: 평균 문서의 길이



#### IDF

$$
\sum^n_{i=1}IDF(q_i)\\
IDF = \frac{log(1+(docCount - docFreq + 0.5))}{docFreq + 0.5}
$$

- docCount: 전체 문서의 개수
- docFreq: 문서에 키워드가 나타난 빈도수




### Advanced algorithm

#### BM11

(for $b=1$)
$$
score(D, Q) = \sum^n_{i=1}IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1+1)}{f(q_i, D)+k_1 \cdot (1-b +b \cdot \frac{\vert D\vert}{avgdl})}\\
= \sum^n_{i=1}IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1+1)}{f(q_i, D)+k_1 \cdot \frac{\vert D\vert}{avgdl}}
$$


#### BM15

(for $b=0$)
$$
{score(D, Q) = \sum^n_{i=1}IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1+1)}{f(q_i, D)+k_1 \cdot (1-b +b \cdot \frac{\vert D\vert}{avgdl})}\\
= \sum^n_{i=1}IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1+1)}{f(q_i, D)+k_1}
}
$$


#### BM25F

문서의 제목, 본문, 부록 등의 값에 다른 가중치를 주고, 길이를 정규화(Normalization) 후 계산





#### BM25+

$$
score(D, Q) = \sum^n_{i=1}IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1+1)}{f(q_i, D)+k_1 \cdot (1-b +b \cdot \frac{\vert D\vert}{avgdl})}
$$



## N-gram

언어학과 확률분야에서 사용되며 n-gram 연속적인 단어의 N개의 집합을 하나의 키워드 묶음으로 처리하는 방법이다. N의 개수에 따라 unigram, bigram, trigram 등으로 불리고, 보통 단위를 문자나 형태소로 묶어서 처리한다. 



| Field                                    | Unit                                     | Sample sequence         | 1-gram sequence                          | 2-gram sequence                          | 3-gram sequence                          |
| ---------------------------------------- | ---------------------------------------- | ----------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| Vernacular name                          |                                          |                         | unigram                                  | bigram                                   | trigram                                  |
| Order of resulting [Markov model](https://en.wikipedia.org/wiki/Markov_model) |                                          |                         | 0                                        | 1                                        | 2                                        |
| [Protein sequencing](https://en.wikipedia.org/wiki/Protein_sequencing) | [amino acid](https://en.wikipedia.org/wiki/Amino_acid) | … Cys-Gly-Leu-Ser-Trp … | …, Cys, Gly, Leu, Ser, Trp, …            | …, Cys-Gly, Gly-Leu, Leu-Ser, Ser-Trp, … | …, Cys-Gly-Leu, Gly-Leu-Ser, Leu-Ser-Trp, … |
| [DNA sequencing](https://en.wikipedia.org/wiki/DNA_sequencing) | [base pair](https://en.wikipedia.org/wiki/Base_pair) | …AGCTTCGA…              | …, A, G, C, T, T, C, G, A, …             | …, AG, GC, CT, TT, TC, CG, GA, …         | …, AGC, GCT, CTT, TTC, TCG, CGA, …       |
| [Computational linguistics](https://en.wikipedia.org/wiki/Computational_linguistics) | [character](https://en.wikipedia.org/wiki/Character_(computing)) | …to_be_or_not_to_be…    | …, t, o, _, b, e, _, o, r, _, n, o, t, _, t, o, _, b, e, … | …, to, o_, _b, be, e_, _o, or, r_, _n, no, ot, t_, _t, to, o_, _b, be, … | …, to_, o_b, _be, be_, e_o, _or, or_, r_n, _no, not, ot_, t_t, _to, to_, o_b, _be, … |
| [Computational linguistics](https://en.wikipedia.org/wiki/Computational_linguistics) | [word](https://en.wikipedia.org/wiki/Word) | … to be or not to be …  | …, to, be, or, not, to, be, …            | …, to be, be or, or not, not to, to be, … | …, to be or, be or not, or not to, not to be, … |



## Latent Semantic Analysis(LSA)

잠재 의미 분석은 자연어 분야에서 문서 집합과 키워드 간의 관계를 분석하는 분포적인 의미를 문서와 키워드 간의 관계 

특히 분포적인 의미에서 문서와 키워드의 관계 집합을 생성함으로써 문서와 키워드에 포함된 의미를 분석한다.

 때때로 Latent Semantic Indexing(LSI)라고도 불린다.



### Occurence matrix

LSA에서 term-document matrix는 키워드의 문서에서의 동시 발생을 표현한다. 이는 키워드에 일치하는 열과 행에 일치하는 문서의 희소행렬이다. 전형적인 예시는 가중치가 부여된 원소의 행렬  TF-IDF 행렬이 있다.



### Ranking lowering

Ranking 감소로 PCA에서 사용되는 것과 같이 SVD를 이용하여 차원 축소를 통하여, 희소행렬의 의미를 저차원의 밀집 행렬로 함축하여 표현하게 된다.



#### Example

${(car), (truck), (flower)} --> {(1.3452 * car + 0.2828 * truck), (flower)}$



#### Equation

$$
X= U\Sigma V^T\\
XX^T = (U\Sigma V^T)(U\Sigma V^T)^T = (U\Sigma V^T)(V{^T}^T\Sigma^T U^T) \\
= U\Sigma (V^TV)\Sigma^T U^T = U(\Sigma \Sigma^T) V^T
= U\Sigma^2 V^T\\

X^TX = (U\Sigma V^T)^T(U\Sigma V^T) = (V{^T}^T\Sigma^T U^T)(U\Sigma V^T) \\
= U\Sigma (V^TV)\Sigma^T U^T = U(\Sigma \Sigma^T) V^T
= U\Sigma^2 V^T
$$


$$
{\begin{matrix}&X&&&U&&\Sigma &&V^{T}\\&({\textbf {d}}_{j})&&&&&&&({\hat {\textbf {d}}}_{j})\\&\downarrow &&&&&&&\downarrow \\({\textbf {t}}_{i}^{T})\rightarrow &{\begin{bmatrix}x_{1,1}&\dots &x_{1,n}\\\\\vdots &\ddots &\vdots \\\\x_{m,1}&\dots &x_{m,n}\\\end{bmatrix}}&=&({\hat {\textbf {t}}}_{i}^{T})\rightarrow &{\begin{bmatrix}{\begin{bmatrix}\,\\\,\\{\textbf {u}}_{1}\\\,\\\,\end{bmatrix}}\dots {\begin{bmatrix}\,\\\,\\{\textbf {u}}_{l}\\\,\\\,\end{bmatrix}}\end{bmatrix}}&\cdot &{\begin{bmatrix}\sigma _{1}&\dots &0\\\vdots &\ddots &\vdots \\0&\dots &\sigma _{l}\\\end{bmatrix}}&\cdot &{\begin{bmatrix}{\begin{bmatrix}&&{\textbf {v}}_{1}&&\end{bmatrix}}\\\vdots \\{\begin{bmatrix}&&{\textbf {v}}_{l}&&\end{bmatrix}}\end{bmatrix}}\end{matrix}}
$$

$$
X_{k}=U_{k}\Sigma _{k}V_{k}^{T}
$$




## Probability Latent Semantic Analysis(pLSA)

다음과 같은 생성 과정을 가정한다.

1. 문서의 주제 분포 $d$로부터 주제 $z$를 선택한다.
2. 주제 $z$가 주어졌을 때 $w_n$은 $p(w_n|z)$로부터 선택한다.

위 과정으로부터 문서와 단어가 나올 결합 확률을 구하면 다음과 같다.
$$
p(d|w_n) = p(d) \sum_np(w_n|z)p(z|d)
$$


### Limitation

pLSI모형은 $p(z|d)$를 통해 하나의 문서가 여러개의 주제를 가질 수 있도록 허용한다. 그러나 $d$는 트레이닝 셋에 속한 문서들에 대한 인덱스이기 때문에 $p(z|d)$는 트레이닝 셋에 포함된 문서에 대해서만 정의된다. 따라서 pLSI는 새로운 문서에 대한 확률을 계산할 수 없으며, 잘 정의된 생성 모형은 아니다.



### Similar model

#### 유니그램 혼합 모형

1. $z$로부터 문서의 주제를 선택한다.
2. 문서의 주제가 주어졌을 때 $w_n$은 $p(w_n|z)$로 부터 선택한다.

위 과정으로부터 문서의 확률을 구하면 다음과 같다.
$$
p(w)=\sum_zp(z)\prod_{n=1}^Np(w_n|z)
$$
각각의 문서가 정확히 하나의 주제를 가진다고 가정한다. 따라서 문서의 갯수가 증가할 때 전집을 효율적으로 모형화하기 어렵다.



## LDA(Latent Dirichlet allocation)

잠재 디리클레 할당, 주어진 문서에 대하여 각 문서에 어떤 주제들이 존재하는지에 대한 확률모형을 생성 후, 추정된 분포를 기반으로 



### Intro

1. LDA는 이산 자료들에 대한 확률적 생성 모형이다. 문자 기반의 자료들에 대해 쓰일 수 있으며 사진 등의 다른 이산 자료들에 대해서도 쓰일 수 있다.[[1\]](https://ko.wikipedia.org/wiki/%EC%9E%A0%EC%9E%AC_%EB%94%94%EB%A6%AC%ED%81%B4%EB%A0%88_%ED%95%A0%EB%8B%B9#cite_note-lda_original-1) 기존의 [정보 검색](https://ko.wikipedia.org/wiki/%EC%A0%95%EB%B3%B4_%EA%B2%80%EC%83%89)분야에서 LDA와 유사한 시도들은 계속 이루어져 왔다.
2. 단어의 교환성(exchangeability),'단어 주머니(bag of words)'. 교환성은 단어들의 순서는 상관하지 않고 오로지 단어들의 유무만이 중요하다는 가정이다. 
   예를 들어, 'Apple is red'와 'Red is apple' 간에 차이가 없다고 생각하는 것이다. 이 가정을 기반으로 단어와 문서들의 교환성을 포함하는 혼합 모형을 제시한 것이 바로 LDA이다.
   하지만 LDA의 교환성의 가정을 확장 시킬 수 도 있다. 단순히 단어 하나를 단위로 생각하는 것이 아니라 특정 단어들의 묶음을 한 단위로 생각하는 방식(n-gram)도 가능하다.
3. LDA 에서 각각의 문서는 여러개의 주제를 가지고 있다. 확률 잠재 의미 분석 (Probabilistic latent semantic analysis, pLSA) 와 비슷하지만 주제들이 디리클레 분포를 따른다고 가정한다.





### Model

 $M$개의 문서아 주어져 있고, 모든 문서는 각각 $k$개의 주제 중 하나에 속할 때,

- 단어는 이산 자료의 기본 단위로 단어집(vocabulary)의 인덱스로 나타낼 수 있다. 단어집의 크기를 $V$ 라 하면, 각각의 단어는 인덱스 $v \in \{1,…,V\}$ 로 대응된다. 단어 벡터 $w$는 $V$-벡터로 표기하며 $w^v=1,w^u=0,u \neq v$ 를 만족한다.
- 문서는 $N$개의 단어의 연속으로 나타내며,  $\mathbf  {w} = (w_1, w_2, \dots, w_N)$ 으로 표기
- 전집은  M개의 문서의 집합으로 나타내며, D = {$\mathbf  {w}_1, \mathbf  {w}_2, \dots, \mathbf  {w}_M$} 으로 표기



1. $ N \sim Poisson(\xi)$
2. $\theta \sim Dir(\alpha)$
3. 문서 내의 단어 $w_n \in \mathbf  {w}$ 에 대해서
   1. $z_n \sim $ Multinomial($\theta$)
   2. $z_n$이 주어졌을 때, $w_n$는 $p(w_n \vert z_n, \beta)$



- ![\alpha ](https://wikimedia.org/api/rest_v1/media/math/render/svg/b79333175c8b3f0840bfb4ec41b8072c83ea88d3)는 $k$ 차원 [디리클레 분포](https://ko.wikipedia.org/wiki/%EB%94%94%EB%A6%AC%ED%81%B4%EB%A0%88_%EB%B6%84%ED%8F%AC)의 매개변수이다.
- $\theta$는 $k$ 차원 벡터이며, $\theta_i$는 문서가 $i$번째 주제에 속할 확률 분포를 나타낸다.
- $z_n$는 $k$ 차원 벡터이며, $z_n%i$는 단어 $w_n$이 $i$번째 주제에 속할 확률 분포를 나타낸다.
- $\beta$는 $k×V$ 크기의 [행렬](https://ko.wikipedia.org/wiki/%ED%96%89%EB%A0%AC) 매개변수로, $\beta_{ij}$는 $i$번째 주제가 단어집의 $j$번째 단어를 생성할 확률을 나타낸다.



여기에서 $w_n$는 실제 문서를 통해 주어지며, 다른 변수는 관측할 수 없는 [잠재 변수](https://ko.wikipedia.org/w/index.php?title=%EC%9E%A0%EC%9E%AC_%EB%B3%80%EC%88%98&action=edit&redlink=1)이다.

이 모형은 다음과 같이 해석될 수 있다. 각 문서에 대해 $k$개의 주제에 대한 가중치 $\theta$가 존재한다. 문서 내의 각 단어 $w_n$은 $k$개의 주제에 대한 가중치 $z_n$을 가지는데, $z_n$은 $\theta$에 의한 다항 분포로 선택된다. 마지막으로 실제 단어 $w_n$이 $z_n$에 기반하여 선택된다.

잠재 변수 $\alpha$, $\beta$ 가 주어졌을 때 $\theta$, $z={z_1, \dots ,z_N}$, $\mathbf  {w}$에 대한 [결합 분포](https://ko.wikipedia.org/wiki/%EA%B2%B0%ED%95%A9_%EB%B6%84%ED%8F%AC) 는 다음과 같이 구해진다.


여기서 $z_n$과 $\theta$를 모두 더하여 문서의 [주변 분포](https://ko.wikipedia.org/wiki/%EC%A3%BC%EB%B3%80_%EB%B6%84%ED%8F%AC) (marginal distribution)를 구할 수 있다. 이 때 [디 피네치의 정리](https://ko.wikipedia.org/w/index.php?title=%EB%94%94_%ED%94%BC%EB%84%A4%EC%B9%98%EC%9D%98_%EC%A0%95%EB%A6%AC&action=edit&redlink=1) (de Finetti's theorem)에 의해 단어가 문서 안에서 교환성을 가지는 것을 확인할 수 있다.


마지막으로 각각의 문서에 대한 주변 분포를 모두 곱하여 전집의 확률을 구할 수 있다.









[TFIDF]: https://ko.wikipedia.org/wiki/TF-IDF
[BM25]: https://inyl.github.io/search_engine/2017/04/01/bm25.html
[LDA]: https://ko.wikipedia.org/wiki/잠재_디리클레_할당

[LSA]: https://en.wikipedia.org/wiki/Latent_semantic_analysis


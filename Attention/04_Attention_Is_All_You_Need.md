# 04. Attention Is All You Need

## 논문 정보

- 원본 파일: `attention_papers_pdf/04_Attention_Is_All_You_Need.pdf`
- 제목: Attention Is All You Need
- 저자: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
- 발표: NeurIPS 2017
- 주제: Transformer, self-attention, encoder-decoder, sequence transduction
- 핵심 키워드: scaled dot-product attention, multi-head attention, positional encoding, position-wise FFN, residual connection, layer normalization

## 한눈에 보는 요약

이 논문은 RNN이나 CNN 없이 attention만으로 sequence-to-sequence 문제를 풀 수 있다는 것을 보인 Transformer의 원 논문이다. 이전의 기계번역 모델은 보통 RNN encoder-decoder 구조를 사용했고, 여기에 attention을 보조적으로 붙였다. 반면 Transformer는 recurrent computation 자체를 제거하고, 입력 토큰들 사이의 관계를 self-attention으로 직접 계산한다.

핵심 아이디어는 간단하다. 각 토큰을 하나의 벡터로 보고, 어떤 토큰이 다른 어떤 토큰을 얼마나 참고해야 하는지를 모든 토큰 쌍에 대해 계산한다. 이 관련도 점수에 softmax를 적용해 가중치를 만들고, 그 가중치로 value 벡터들을 weighted sum한다. 이 연산을 여러 head로 병렬 수행하면 서로 다른 의미 관계, 위치 관계, 문법 관계를 동시에 포착할 수 있다.

Transformer는 여전히 encoder-decoder 구조를 따른다. Encoder는 source sentence를 문맥화된 표현들의 sequence로 변환하고, decoder는 target sentence를 왼쪽에서 오른쪽으로 생성한다. 다른 점은 encoder와 decoder 내부의 주된 연산이 RNN hidden state 갱신이 아니라 multi-head attention과 position-wise feed-forward network라는 점이다.

이 리뷰는 논문의 실험 결과보다, 각 구성 요소가 이론적으로 무엇을 의미하며 내부 계산이 어떤 순서로 전개되는지에 초점을 둔다.

## 연구 배경과 문제의식

### 기존 sequence-to-sequence 모델의 흐름

기계번역 같은 sequence transduction 문제는 입력 sequence를 받아 출력 sequence를 생성하는 문제다.

```math
\begin{aligned}
\text{source sentence:}\quad &x_1,x_2,\ldots,x_n,\\
\text{target sentence:}\quad &y_1,y_2,\ldots,y_m.
\end{aligned}
```

초기 neural machine translation에서는 encoder-decoder 구조가 널리 쓰였다.

- Encoder는 입력 문장 $x_1,\ldots,x_n$을 읽어 hidden representation을 만든다.
- Decoder는 encoder가 만든 representation을 조건으로 출력 문장 $y_1,\ldots,y_m$을 생성한다.
- Decoder는 보통 autoregressive하게 동작한다. 즉, $y_i$를 예측할 때 이전 target token $y_1,\ldots,y_{i-1}$을 조건으로 사용한다.

전통적인 RNN 기반 encoder-decoder에서는 시간 순서대로 hidden state를 갱신한다.

```math
h_t=\mathrm{RNN}(h_{t-1},x_t)
```

이 구조의 장점은 sequence 순서를 자연스럽게 처리한다는 점이다. 하지만 치명적인 약점도 있다.

- $h_t$를 계산하려면 반드시 $h_{t-1}$이 먼저 계산되어야 한다.
- 같은 문장 안의 토큰들을 병렬로 처리하기 어렵다.
- 먼 위치의 단어 사이 정보가 여러 recurrent step을 지나야 하므로 장거리 의존성 학습이 어렵다.
- 긴 sequence에서는 학습 시간과 메모리 효율이 나빠진다.

### 이 논문의 문제의식

논문이 던지는 질문은 다음과 같다.

```text
정말 sequence를 이해하려면 반드시 왼쪽에서 오른쪽으로 하나씩 읽어야 하는가?
```

Transformer의 대답은 아니다. 각 위치의 표현을 만들 때 sequence 전체 위치를 한 번에 참고하면 된다. 즉, recurrence 대신 attention으로 모든 위치 사이의 관계를 직접 연결한다.

RNN에서는 위치 $i$와 위치 $j$가 멀수록 정보가 여러 step을 거쳐야 한다. Transformer의 self-attention에서는 한 layer 안에서 모든 위치가 모든 위치를 직접 볼 수 있다.

```text
RNN:
x_1 -> h_1 -> h_2 -> h_3 -> ... -> h_n

Self-attention:
x_i -> directly attends to x_1, x_2, ..., x_n
```

이 차이가 Transformer의 병렬성, 장거리 의존성 처리, 확장성의 출발점이다.

## 핵심 아이디어와 방법

### Transformer를 한 문장으로 설명하면

Transformer는 각 token representation을 반복적으로 갱신하는 모델이다. 각 layer에서 다음 두 종류의 계산을 수행한다.

1. Attention: 토큰들 사이의 정보를 섞는다.
2. FFN: 각 토큰 위치의 feature를 비선형적으로 변환한다.

이 둘을 반복하면 각 token vector는 점점 더 넓은 문맥을 반영한 표현이 된다.

```text
token embedding
-> positional encoding 추가
-> self-attention으로 토큰 간 정보 교환
-> FFN으로 위치별 feature 변환
-> 여러 layer 반복
-> decoder에서 다음 token 확률 계산
```

### Attention이란 무엇인가

Attention은 필요한 정보를 선택적으로 읽는 연산이다. 수식적으로는 query와 key의 관련도를 계산하고, 그 관련도를 value의 가중합에 사용하는 과정이다.

세 가지 벡터가 중요하다.

- Query $Q$: 지금 정보를 찾는 쪽의 질문 벡터
- Key $K$: 각 후보 정보가 어떤 특징을 갖는지 나타내는 색인 벡터
- Value $V$: 실제로 가져올 내용 벡터

직관적으로는 다음과 같다.

```text
Query: "나는 지금 무엇을 찾고 있는가?"
Key:   "각 토큰은 어떤 정보로 검색될 수 있는가?"
Value: "검색되었을 때 실제로 전달할 정보는 무엇인가?"
```

Attention은 query와 key를 비교해 점수를 만들고, 그 점수를 확률처럼 정규화한 뒤, value들을 평균낸다. 단순 평균이 아니라 관련도가 높은 value에 더 큰 가중치를 주는 weighted sum이다.

### Scaled dot-product attention

논문에서 사용하는 attention은 scaled dot-product attention이다.

```math
\mathrm{Attention}(Q,K,V)
=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
```

<p align="center"><img src="https://github.com/user-attachments/assets/7606371f-a76f-43a3-9b39-521121740595" alt="Scaled dot-product attention과 multi-head attention" width="760"></p>
<p align="center"><sub>Figure 2 — scaled dot-product attention과 multi-head attention 구조</sub></p>

Encoder-decoder attention에서는 $Q$가 decoder layer에서, $K$와 $V$가 encoder output에서 온다. Encoder self-attention은 $Q$, $K$, $V$가 모두 이전 encoder layer에서 오고, decoder self-attention은 masking된 decoder state를 사용한다. 원논문 Figure 2의 왼쪽은 `MatMul -> Scale -> Mask (opt.) -> SoftMax -> MatMul` 계산을, 오른쪽은 여러 head를 `Concat -> Linear`로 결합하는 구조를 보여 준다.

각 기호의 의미는 다음과 같다.

- $Q$: query matrix
- $K$: key matrix
- $V$: value matrix
- $d_k$: key/query 벡터의 차원
- $QK^\top$: query와 key 사이의 dot product score matrix
- $\sqrt{d_k}$: score 크기를 안정화하기 위한 scaling factor
- $\mathrm{softmax}(\cdots)$: 각 query가 key들을 얼마나 볼지 나타내는 attention weight
- $\mathrm{softmax}(\cdots)V$: attention weight로 value를 weighted sum한 결과

한 query vector $q$와 여러 key/value가 있을 때는 다음처럼 볼 수 있다.

```math
\begin{aligned}
\mathrm{score}_j &= \frac{q\cdot k_j}{\sqrt{d_k}},\\
\alpha_j &= \frac{\exp(\mathrm{score}_j)}
{\sum_l\exp(\mathrm{score}_l)},\\
\mathrm{output} &= \sum_j\alpha_jv_j.
\end{aligned}
```

즉, attention output은 value 벡터들의 convex combination에 가깝다. softmax weight가 모두 0 이상이고 합이 1이기 때문이다.

### 왜 $\sqrt{d_k}$로 나누는가

query와 key의 각 성분이 평균 0, 분산 1인 독립 변수라고 생각하면 dot product는 다음과 같다.

```math
q\cdot k=\sum_{i=1}^{d_k}q_i k_i
```

이 값의 분산은 대략 $d_k$에 비례한다. $d_k$가 커질수록 dot product 값의 절댓값이 커지고, softmax 입력이 너무 커진다. 그러면 softmax가 한쪽으로 지나치게 뾰족해지고 gradient가 작아진다.

따라서 $\sqrt{d_k}$로 나누면 score의 scale을 안정화할 수 있다.

```math
\begin{aligned}
\text{raw score:}\quad &q\cdot k,\\
\text{scaled score:}\quad &\frac{q\cdot k}{\sqrt{d_k}}.
\end{aligned}
```

이것이 scaled dot-product attention의 핵심이다.

## 모델/알고리즘 구조

### 전체 구조

Transformer는 encoder stack과 decoder stack으로 구성된다.

```text
source tokens
-> source embeddings + positional encodings
-> encoder layer x 6
-> encoder memory

target tokens shifted right
-> target embeddings + positional encodings
-> decoder layer x 6
-> linear projection
-> softmax
-> next token probabilities
```

논문의 base model 설정은 다음과 같다.

| 항목 | 값 |
| --- | ---: |
| encoder layer 수 $N$ | 6 |
| decoder layer 수 $N$ | 6 |
| model dimension $d_{\mathrm{model}}$ | 512 |
| attention head 수 $h$ | 8 |
| head별 key dimension $d_k$ | 64 |
| head별 value dimension $d_v$ | 64 |
| FFN inner dimension $d_{\mathrm{ff}}$ | 2048 |
| dropout | 0.1 |
| label smoothing | 0.1 |

중요한 설계 원칙은 모든 sub-layer의 입력과 출력 차원을 $d_{\mathrm{model}}=512$로 맞춘다는 점이다. 그래야 residual connection을 쉽게 적용할 수 있다.

```math
\mathrm{output}
=\mathrm{LayerNorm}\!\left(x+\mathrm{Sublayer}(x)\right)
```

### Encoder란 무엇인가

Encoder는 입력 sequence를 받아 각 위치별 문맥 표현을 만드는 모듈이다.

입력 문장이 다음과 같다고 하자.

```math
x_1,x_2,\ldots,x_n
```

Encoder는 이를 다음과 같은 continuous representation sequence로 바꾼다.

```math
z_1,z_2,\ldots,z_n
```

각 $z_i$는 단순히 $x_i$만 표현하지 않는다. 여러 self-attention layer를 지나면서 문장 전체 정보를 반영한다. 즉, $z_i$는 위치 $i$의 토큰을 중심으로 하되 전체 문맥을 압축한 벡터다.

Transformer encoder layer 하나는 두 sub-layer로 구성된다.

1. Multi-head self-attention
2. Position-wise feed-forward network

각 sub-layer 뒤에는 residual connection과 layer normalization이 붙는다.

```math
\begin{aligned}
H' &= \mathrm{LayerNorm}\!\left(
H+\mathrm{MultiHeadSelfAttention}(H)\right),\\
H'' &= \mathrm{LayerNorm}\!\left(H'+\mathrm{FFN}(H')\right).
\end{aligned}
```

<p align="center"><img src="https://github.com/user-attachments/assets/67ada7f7-5719-4c0a-8430-4c675a52373b" alt="Encoder layer와 6-layer encoder stack의 연결" width="820"></p>
<p align="center"><sub>보조 도식 — post-norm encoder layer와 6-layer stack의 연결</sub></p>

그림은 원논문의 post-norm 순서를 기준으로 한 encoder layer 하나를 펼친 것이다. 첫 번째 `Add & Norm`에는 layer 입력 $H^{(l-1)}$이 residual 경로로 더해지고, 두 번째 `Add & Norm`에는 attention sub-layer 출력 $U^{(l)}$이 residual 경로로 더해진다. 이 layer를 $N=6$회 반복한 최종 출력 $H^{(6)}$이 decoder가 참조하는 encoder memory $Z$가 된다.

여기서 $H$는 현재 layer에 들어온 전체 sequence representation matrix다.

### Decoder란 무엇인가

Decoder는 출력 sequence를 생성하는 모듈이다. 기계번역에서는 target sentence를 한 단어씩 만든다.

```math
p(y_i\mid y_1,\ldots,y_{i-1},x)
```

여기서 중요한 점은 decoder가 미래 target token을 보면 안 된다는 것이다. 예를 들어 세 번째 단어를 예측할 때 네 번째 정답 단어를 보면 안 된다. 그래서 decoder self-attention에는 causal mask가 들어간다.

Transformer decoder layer 하나는 세 sub-layer로 구성된다.

1. Masked multi-head self-attention
2. Encoder-decoder attention, 또는 cross-attention
3. Position-wise feed-forward network

계산 흐름은 다음과 같다.

```math
\begin{aligned}
T' &=
\mathrm{LayerNorm}\!\left(T+\mathrm{MaskedSelfAttention}(T)\right),\\
T'' &=
\mathrm{LayerNorm}\!\left(
T'+\mathrm{CrossAttention}(\mathrm{query}=T',
\mathrm{key}=Z,\mathrm{value}=Z)\right),\\
T''' &=
\mathrm{LayerNorm}\!\left(T''+\mathrm{FFN}(T'')\right).
\end{aligned}
```

여기서 $T$는 decoder 쪽 target representation이고, $Z$는 encoder output이다.

### Encoder-decoder 구조의 의미

Encoder-decoder 구조는 source sequence와 target sequence의 길이가 다를 수 있는 문제에 적합하다.

```math
\begin{aligned}
\text{source length }n:\quad &x_1,\ldots,x_n,\\
\text{target length }m:\quad &y_1,\ldots,y_m.
\end{aligned}
```

Encoder는 source 쪽을 충분히 읽어 memory $Z$를 만든다. Decoder는 그 memory를 참고하면서 target을 생성한다.

Transformer에서 encoder-decoder attention은 다음 역할을 한다.

- Query는 decoder의 현재 target 위치 표현에서 온다.
- Key와 value는 encoder output, 즉 source memory에서 온다.
- 따라서 target의 각 위치는 source의 모든 위치를 참고할 수 있다.

수식적으로는 다음과 같다.

```math
\begin{aligned}
Q &= TW_Q,\\
K &= ZW_K,\\
V &= ZW_V,\\
\mathrm{CrossAttention}(T,Z)
&=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
\end{aligned}
```

이 구조는 Bahdanau attention이나 Luong attention의 아이디어를 Transformer 방식으로 일반화한 것이다. Decoder가 현재 target을 만들기 위해 source memory에서 필요한 정보를 검색한다는 점은 같다.

## Attention 계산 전개

### Shape로 보는 self-attention

batch size를 $B$, sequence length를 $n$, model dimension을 $d_{\mathrm{model}}$이라고 하자. Encoder의 어떤 layer 입력을 $H$라고 하면 shape는 다음과 같다.

```text
H: [B, n, d_model]
```

base Transformer에서는 $d_{\mathrm{model}}=512$, head 수 $h=8$, head별 차원 $d_k=d_v=64$다.

각 head마다 세 projection matrix를 학습한다.

```math
\begin{aligned}
\mathrm{shape}(W_Q)&=[d_{\mathrm{model}},d_k],\\
\mathrm{shape}(W_K)&=[d_{\mathrm{model}},d_k],\\
\mathrm{shape}(W_V)&=[d_{\mathrm{model}},d_v].
\end{aligned}
```

입력 $H$를 query, key, value로 선형 변환한다.

```math
\begin{aligned}
Q&=HW_Q,
&\mathrm{shape}(Q)&=[B,n,d_k],\\
K&=HW_K,
&\mathrm{shape}(K)&=[B,n,d_k],\\
V&=HW_V,
&\mathrm{shape}(V)&=[B,n,d_v].
\end{aligned}
```

그 다음 모든 query와 모든 key의 dot product를 계산한다.

```math
\begin{aligned}
S&=\frac{QK^\top}{\sqrt{d_k}},\\
\mathrm{shape}(S)&=[B,n,n].
\end{aligned}
```

$S[i,j]$는 위치 $i$가 위치 $j$를 얼마나 참고할지에 대한 score다.

softmax는 보통 마지막 key dimension 방향으로 적용한다.

```math
\begin{aligned}
A&=\mathrm{softmax}(S),\\
\mathrm{shape}(A)&=[B,n,n].
\end{aligned}
```

$A[i,:]$는 위치 $i$가 모든 위치 $1,\ldots,n$을 보는 attention distribution이다.

마지막으로 value를 weighted sum한다.

```math
\begin{aligned}
O&=AV,\\
\mathrm{shape}(O)&=[B,n,d_v].
\end{aligned}
```

$O[i]$는 위치 $i$가 sequence 전체에서 필요한 정보를 모아 만든 새 표현이다.

### 작은 예시로 보는 attention

문장이 세 토큰으로 이루어졌다고 하자.

```text
tokens: [I, love, NLP]
```

어떤 layer에서 `love` 위치의 query가 세 key와 dot product를 만든다.

```math
\begin{aligned}
\mathrm{score}(\mathrm{love}\to\mathrm{I})&=0.2,\\
\mathrm{score}(\mathrm{love}\to\mathrm{love})&=1.5,\\
\mathrm{score}(\mathrm{love}\to\mathrm{NLP})&=1.1.
\end{aligned}
```

softmax를 적용하면 예를 들어 다음과 같은 가중치가 된다.

```math
\alpha=\begin{bmatrix}0.14&0.51&0.35\end{bmatrix}
```

그러면 `love` 위치의 새 표현은 다음과 같다.

```math
\mathrm{new}_{\mathrm{love}}
=0.14V_{\mathrm{I}}+0.51V_{\mathrm{love}}+0.35V_{\mathrm{NLP}}
```

즉, self-attention은 각 위치의 표현을 자기 자신과 주변 위치들의 value 조합으로 갱신한다. 이 과정을 모든 위치에 대해 동시에 수행한다.

### Multi-head attention

단일 attention head만 쓰면 하나의 방식으로만 토큰 관계를 본다. 하지만 문장 안에는 여러 종류의 관계가 동시에 존재한다.

- 주어와 동사의 관계
- 명사와 수식어의 관계
- 대명사와 선행사의 관계
- 가까운 위치 관계
- 먼 위치 관계
- 번역에서 source-target 정렬 관계

Multi-head attention은 query, key, value를 여러 개의 subspace로 projection해서 attention을 병렬로 수행한다.

```math
\mathrm{head}_i
=\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V)
```

각 head의 결과를 concatenate한 뒤 다시 선형 변환한다.

```math
\mathrm{MultiHead}(Q,K,V)
=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W^O
```

<p align="center"><img src="https://github.com/user-attachments/assets/020097b1-4659-4a86-a8d9-302652cfd500" alt="QKV projection과 multi-head attention 결합" width="860"></p>
<p align="center"><sub>보조 도식 — Q/K/V projection, head별 attention, concat, output projection의 흐름</sub></p>

그림은 같은 입력 $H^{(l-1)}$에서 head별 $Q_i$, $K_i$, $V_i$를 병렬 생성하고, scaled dot-product attention을 계산한 뒤 8개 head를 concatenate하여 $W^O$로 다시 $d_{\mathrm{model}}=512$ 차원에 projection하는 전 과정을 나타낸다. 함수와 parameter 표기는 원논문 Section 3.2.2를 따른다.

base model에서는 다음과 같다.

```math
\begin{aligned}
h&=8,\\
d_{\mathrm{model}}&=512,\\
d_k=d_v&=\frac{512}{8}=64.
\end{aligned}
```

각 head는 64차원 공간에서 attention을 계산한다. 8개 head를 합치면 다시 512차원이 된다.

```text
head_1: [B, n, 64]
...
head_8: [B, n, 64]
Concat: [B, n, 512]
W^O:    [512, 512]
output: [B, n, 512]
```

Multi-head attention의 이론적 의미는 하나의 큰 attention을 여러 관점의 attention으로 분해하는 것이다. 각 head는 서로 다른 representation subspace에서 관련도를 계산하므로, 모델이 다양한 관계를 동시에 표현할 수 있다.

### Masked self-attention

Decoder self-attention에서는 미래 token을 보면 안 된다. 이를 위해 score matrix에 mask를 적용한다.

target 길이가 4라면 허용되는 attention 구조는 다음과 같다.

```text
position 1 can attend to: 1
position 2 can attend to: 1, 2
position 3 can attend to: 1, 2, 3
position 4 can attend to: 1, 2, 3, 4
```

미래 위치는 softmax 전에 $-\infty$로 만든다.

```math
S_{\mathrm{masked}}[i,j]=
\begin{cases}
-\infty,&j>i,\\
S[i,j],&\text{otherwise}.
\end{cases}
```

softmax를 적용하면 $-\infty$ 위치의 확률은 0이 된다.

```math
\mathrm{softmax}(-\infty)=0
```

이렇게 하면 학습 중에도 decoder는 미래 정답을 훔쳐보지 않고, 실제 생성 시점과 같은 조건에서 다음 token을 예측한다.

### Padding mask

문장을 batch로 묶으면 길이가 다르기 때문에 padding token이 들어간다.

```text
sentence A: [I, love, NLP, <pad>, <pad>]
sentence B: [Attention, is, all, you, need]
```

`<pad>`는 실제 단어가 아니므로 attention 대상에서 제외해야 한다. 이를 위해 padding 위치도 score 단계에서 $-\infty$로 masking한다.

```math
\begin{aligned}
\text{attention score to pad position} &= -\infty,\\
\text{attention weight to pad position} &= 0.
\end{aligned}
```

masking은 attention이 단순한 weighted sum이면서도 실제 sequence 구조를 지키게 해주는 중요한 장치다.

## FFN이란 무엇인가

### Position-wise feed-forward network의 역할

Transformer layer에는 attention 뒤에 FFN이 있다. 논문에서는 position-wise fully connected feed-forward network라고 부른다.

수식은 다음과 같다.

```math
\mathrm{FFN}(x)
=\max(0,xW_1+b_1)W_2+b_2
```

<p align="center"><img src="https://github.com/user-attachments/assets/3547a805-746f-4cd9-9cd0-c67641b701dc" alt="Position-wise feed-forward network" width="860"></p>
<p align="center"><sub>보조 도식 — 각 token에 동일한 512→2048→512 FFN을 독립적으로 적용하는 과정</sub></p>

Sequence 전체를 행렬로 계산할 때 feature dimension은 `512 -> 2048 -> 512`로 변한다. 동일한 `W_1`, `b_1`, `W_2`, `b_2`가 각 position에 서로 독립적으로 적용되며, FFN에서는 attention처럼 서로 다른 position 사이의 값을 곱하거나 가중합하지 않는다.

여기서 $\max(0,\ldots)$는 ReLU activation이다.

base model의 차원은 다음과 같다.

```text
x:       [d_model] = [512]
W_1:     [512, 2048]
b_1:     [2048]
hidden:  [2048]
W_2:     [2048, 512]
b_2:     [512]
output:  [512]
```

sequence 전체로 보면 다음 shape가 된다.

```text
H:        [B, n, 512]
H W_1:    [B, n, 2048]
ReLU:     [B, n, 2048]
... W_2:  [B, n, 512]
```

### Attention과 FFN의 차이

Attention과 FFN은 하는 일이 다르다.

Attention은 위치 사이의 정보를 섞는다.

```text
position i output = weighted sum of values from positions 1..n
```

FFN은 각 위치를 독립적으로 변환한다.

```text
position i output = MLP(position i vector)
```

즉, attention은 sequence dimension을 따라 정보를 섞고, FFN은 feature dimension을 따라 정보를 변환한다.

```text
Attention:
[token 1, token 2, token 3] 사이의 관계를 계산

FFN:
각 token vector 내부의 feature를 비선형 변환
```

두 연산을 함께 쓰면 다음과 같은 역할 분담이 생긴다.

- Self-attention: 다른 토큰에서 필요한 정보를 가져온다.
- FFN: 가져온 정보를 바탕으로 각 위치의 표현을 더 복잡하게 가공한다.

논문은 FFN을 kernel size 1인 convolution 두 개로 볼 수도 있다고 설명한다. 모든 위치에 같은 가중치 $W_1$, $W_2$를 적용하기 때문이다.

### 왜 FFN이 필요한가

Attention만 있으면 각 위치의 output은 value들의 가중합이다. 물론 projection과 multi-head 구조가 있지만, attention의 핵심은 정보를 섞는 것이다. 비선형 feature 변환 능력을 충분히 주려면 각 위치별 MLP가 필요하다.

FFN은 다음 기능을 담당한다.

- feature 차원을 512에서 2048로 확장해 더 넓은 표현 공간을 사용한다.
- ReLU로 비선형성을 추가한다.
- 다시 512차원으로 줄여 다음 layer와 residual 구조에 맞춘다.
- 모든 위치에 동일한 변환을 적용해 parameter sharing을 유지한다.

요약하면, attention이 "어디에서 정보를 가져올지"를 정한다면 FFN은 "가져온 정보를 어떻게 해석하고 변환할지"를 담당한다.

## Positional Encoding

### 왜 위치 정보가 필요한가

Transformer는 RNN도 CNN도 사용하지 않는다. 따라서 기본 self-attention만 보면 token 순서에 대한 정보가 없다.

예를 들어 단순 self-attention은 다음 두 입력을 구분하기 어렵다.

```text
The dog bit the man
The man bit the dog
```

단어 집합은 같지만 순서가 다르므로 의미가 완전히 달라진다. 그래서 Transformer는 token embedding에 positional encoding을 더한다.

```math
\mathrm{input\ representation}
=\mathrm{token\ embedding}+\mathrm{positional\ encoding}
```

원논문 Section 3.5의 표현을 따르면 positional encoding은 encoder stack과 decoder stack의 가장 아래에서 input embedding에 더해진다. 두 행렬의 마지막 차원은 모두 $d_{\mathrm{model}}$이므로 원소별 덧셈이 가능하다.

```math
\begin{aligned}
\mathrm{shape}(X)&=[n,d_{\mathrm{model}}],\\
\mathrm{shape}(\mathrm{PE})&=[n,d_{\mathrm{model}}],\\
H&=X+\mathrm{PE},\\
\mathrm{shape}(H)&=[n,d_{\mathrm{model}}].
\end{aligned}
```

여기서 $H=X+\mathrm{PE}$는 계산 구조를 설명하기 위한 표기다. 원논문은 이 관계를 별도의 기호 $H$로 정의하지 않고, positional encodings를 input embeddings에 더한다고 서술한다.

### Sinusoidal positional encoding

논문에서는 서로 다른 주파수를 갖는 sine과 cosine 함수로 fixed positional encoding을 구성한다. 아래 수식은 원논문에 사용된 함수와 기호를 그대로 일반 텍스트로 옮긴 것이다.

```math
\begin{aligned}
\mathrm{PE}_{(\mathrm{pos},2i)}
&=\sin\!\left(\frac{\mathrm{pos}}
{10000^{2i/d_{\mathrm{model}}}}\right),\\
\mathrm{PE}_{(\mathrm{pos},2i+1)}
&=\cos\!\left(\frac{\mathrm{pos}}
{10000^{2i/d_{\mathrm{model}}}}\right).
\end{aligned}
```

<p align="center"><img src="https://github.com/user-attachments/assets/687d765e-0703-47cc-917b-a65238a20f71" alt="Sinusoidal positional encoding 정의와 합산" width="860"></p>
<p align="center"><sub>보조 도식 — sinusoidal 정의, 차원 pair, embedding 합산 구조</sub></p>

위쪽은 Section 3.5의 두 sinusoidal 함수와 $d_{\mathrm{model}}=512$일 때 $2i$, $2i+1$ 차원 pair를 나타낸다. 아래쪽은 embedding과 positional encoding을 원소별로 더해 encoder의 첫 hidden state를 만드는 흐름이다.

### 원논문의 기호 표

| 원논문 표기 | 정확한 역할 | $d_{\mathrm{model}}=512$에서의 값 또는 범위 |
|---|---|---|
| $\mathrm{pos}$ | sequence 안에서 token이 놓인 position | $0,1,\ldots,n-1$ |
| $i$ | 원논문에서 dimension이라고 부르는 인덱스. 구현 관점에서는 sine/cosine 차원 쌍의 인덱스 | $0,1,\ldots,255$ |
| $2i$ | sine 함수를 사용하는 짝수 dimension | $0,2,4,\ldots,510$ |
| $2i+1$ | cosine 함수를 사용하는 홀수 dimension | $1,3,5,\ldots,511$ |
| $d_{\mathrm{model}}$ | embedding과 positional encoding이 공유하는 model dimension | $512$ |
| $\mathrm{PE}_{(\mathrm{pos},2i)}$ | position $\mathrm{pos}$의 짝수 dimension 값 | $[-1,1]$ |
| $\mathrm{PE}_{(\mathrm{pos},2i+1)}$ | position $\mathrm{pos}$의 홀수 dimension 값 | $[-1,1]$ |
| $\mathrm{PE}_{\mathrm{pos}}$ | position $\mathrm{pos}$에서의 positional encoding vector 전체 | $[512]$ |
| $\mathrm{PE}$ | 모든 position의 positional encoding을 쌓은 행렬 | $[n,512]$ |

$i$ 하나는 두 개의 model dimension을 만든다.

```text
i = 0   -> dimension 0: sine, dimension 1: cosine
i = 1   -> dimension 2: sine, dimension 3: cosine
i = 2   -> dimension 4: sine, dimension 5: cosine
...
i = 255 -> dimension 510: sine, dimension 511: cosine
```

따라서 $d_{\mathrm{model}}=512$이면 총 256개의 sine/cosine 쌍이 만들어지고, 이들을 이어 붙여 position 하나당 512차원 vector를 구성한다.

### 분모와 주파수가 결정되는 과정

두 함수가 공유하는 핵심은 다음 분모다.

```math
10000^{2i/d_{\mathrm{model}}}
```

$i$가 증가하면 이 값이 커진다. 같은 $\mathrm{pos}$를 더 큰 값으로 나누므로 sine과 cosine에 입력되는 각도가 더 천천히 변한다.

```text
작은 i -> 작은 분모 -> 빠른 진동 -> 짧은 파장
큰 i   -> 큰 분모 -> 느린 진동 -> 긴 파장
```

원논문은 각 dimension의 파장이 다음 범위에서 등비수열을 이룬다고 설명한다.

```math
2\pi\ \longrightarrow\ 10000\cdot2\pi
```

즉 낮은 dimension은 가까운 position 변화에 민감하고, 높은 dimension은 긴 범위에서 천천히 변한다. 한 position은 이처럼 서로 다른 시간 척도를 가진 신호들을 동시에 갖게 된다.

### 실제 값 계산: $d_{\mathrm{model}}=512$

$\mathrm{pos}=1$, $i=0$인 첫 번째 차원 쌍은 다음과 같이 계산된다.

```math
\begin{aligned}
10000^{2\cdot0/512} &= 1,\\
\mathrm{PE}_{(1,0)} &= \sin(1/1)=0.841471,\\
\mathrm{PE}_{(1,1)} &= \cos(1/1)=0.540302.
\end{aligned}
```

$\mathrm{pos}=1$, $i=1$인 두 번째 차원 쌍은 다음과 같다.

```math
\begin{aligned}
10000^{2\cdot1/512} &= 10000^{2/512},\\
\mathrm{PE}_{(1,2)}
&=\sin\!\left(\frac{1}{10000^{2/512}}\right)=0.821856,\\
\mathrm{PE}_{(1,3)}
&=\cos\!\left(\frac{1}{10000^{2/512}}\right)=0.569695.
\end{aligned}
```

처음 네 position과 처음 여덟 dimension을 실제로 계산하면 다음 표가 된다.

| $\mathrm{pos}$ | $\mathrm{PE}_{(\mathrm{pos},0)}$ | $\mathrm{PE}_{(\mathrm{pos},1)}$ | $\mathrm{PE}_{(\mathrm{pos},2)}$ | $\mathrm{PE}_{(\mathrm{pos},3)}$ | $\mathrm{PE}_{(\mathrm{pos},4)}$ | $\mathrm{PE}_{(\mathrm{pos},5)}$ | $\mathrm{PE}_{(\mathrm{pos},6)}$ | $\mathrm{PE}_{(\mathrm{pos},7)}$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 |
| 1 | 0.841471 | 0.540302 | 0.821856 | 0.569695 | 0.801962 | 0.597375 | 0.781887 | 0.623420 |
| 2 | 0.909297 | -0.416147 | 0.936415 | -0.350895 | 0.958144 | -0.286285 | 0.974888 | -0.222695 |
| 3 | 0.141120 | -0.989992 | 0.245085 | -0.969501 | 0.342782 | -0.939415 | 0.433643 | -0.901085 |

$\mathrm{pos}=0$에서는 모든 sine dimension이 $\sin(0)=0$, 모든 cosine dimension이 $\cos(0)=1$이 된다. 따라서 첫 position의 vector가 전부 0인 것은 아니다.

### 전체 PE 행렬에서 나타나는 패턴

<p align="center"><img src="https://github.com/user-attachments/assets/5c9ad5a1-cbfc-44d7-bd18-d784ba4d43bb" alt="Sinusoidal positional encoding 행렬과 주파수별 곡선" width="860"></p>
<p align="center"><sub>보조 도식 — position×dimension 행렬과 차원에 따른 sinusoidal 주파수 변화</sub></p>

위 그림은 원논문의 함수를 $d_{\mathrm{model}}=512$, $\mathrm{pos}=0,\ldots,127$에 직접 적용한 결과다.

- (a)의 가로축은 dimension $0,\ldots,511$, 세로축은 $\mathrm{pos}=0,\ldots,127$이다.
- 색 범위는 $-1$에서 $1$이다.
- 낮은 dimension에서는 position이 변할 때 값이 빠르게 진동한다.
- 높은 dimension에서는 파장이 길어져 128개 position 안에서 값이 거의 변하지 않을 수도 있다.
- 짝수 dimension과 홀수 dimension은 같은 주파수의 sine/cosine 쌍이므로 위상이 90도 다르다.

### 왜 sine/cosine인가

논문이 제시하는 핵심 가설은 고정된 offset $k$에 대해 $\mathrm{PE}_{(\mathrm{pos}+k)}$를 $\mathrm{PE}_{\mathrm{pos}}$의 선형함수로 표현할 수 있다는 것이다.

한 차원 쌍의 각도 증가율을 설명하기 위해 다음 기호를 보조적으로 정의해 보자. 이 $\omega_i$ 표기는 원논문의 수식을 전개하기 위한 리뷰상의 보조 표기다.

```math
\omega_i=\frac{1}{10000^{2i/d_{\mathrm{model}}}}
```

그러면 한 차원 쌍은 다음과 같이 쓸 수 있다.

```math
\begin{aligned}
\mathrm{PE}_{(\mathrm{pos},2i)}
&=\sin(\mathrm{pos}\cdot\omega_i),\\
\mathrm{PE}_{(\mathrm{pos},2i+1)}
&=\cos(\mathrm{pos}\cdot\omega_i).
\end{aligned}
```

offset $k$만큼 이동하면 삼각함수의 덧셈 정리에 의해 다음 계산이 성립한다.

```math
\begin{aligned}
\sin((\mathrm{pos}+k)\omega_i)
&=\sin(\mathrm{pos}\,\omega_i)\cos(k\omega_i)
+\cos(\mathrm{pos}\,\omega_i)\sin(k\omega_i),\\
\cos((\mathrm{pos}+k)\omega_i)
&=\cos(\mathrm{pos}\,\omega_i)\cos(k\omega_i)
-\sin(\mathrm{pos}\,\omega_i)\sin(k\omega_i).
\end{aligned}
```

이를 2차원 vector와 행렬로 정리하면 다음과 같다.

```math
\begin{aligned}
\begin{bmatrix}
\sin((\mathrm{pos}+k)\omega_i)\\
\cos((\mathrm{pos}+k)\omega_i)
\end{bmatrix}
&=
\begin{bmatrix}
\cos(k\omega_i)&\sin(k\omega_i)\\
-\sin(k\omega_i)&\cos(k\omega_i)
\end{bmatrix}
\begin{bmatrix}
\sin(\mathrm{pos}\,\omega_i)\\
\cos(\mathrm{pos}\,\omega_i)
\end{bmatrix}
\end{aligned}
```

오른쪽의 회전행렬은 $\mathrm{pos}$와 무관하고 offset $k$와 frequency $\omega_i$에만 의존한다. 따라서 고정된 상대 거리 $k$만큼 이동한 positional encoding은 현재 positional encoding에 일정한 선형변환을 적용한 것으로 나타낼 수 있다. 이것이 원논문의 다음 문장을 계산으로 풀어 쓴 의미다.

```text
for any fixed offset k,
PE_(pos+k) can be represented as a linear function of PE_pos
```

이 구조가 상대 위치를 자동으로 보장하거나 모델이 반드시 상대 거리를 학습한다는 뜻은 아니다. 논문은 모델이 상대 위치에 따라 attend하는 방법을 쉽게 학습할 수 있을 것이라는 가설로 이를 제시한다.

### Fixed encoding을 선택한 이유

논문은 learned positional embedding도 실험했고 두 방식이 거의 동일한 결과를 냈다고 보고한다. 최종적으로 sinusoidal version을 선택한 이유는 학습 중 경험한 sequence length보다 긴 길이에도 함숫값을 계산하여 extrapolate할 가능성이 있기 때문이다.

다만 이는 긴 길이에서 성능이 자동으로 유지된다는 보장은 아니다. 위치값 자체를 생성할 수 있다는 것과, 모델이 학습 범위를 넘어 안정적으로 동작한다는 것은 구분해야 한다.

## Residual Connection과 Layer Normalization

### Residual connection

Transformer의 각 sub-layer는 다음 형태를 사용한다.

```math
\mathrm{LayerNorm}\!\left(x+\mathrm{Sublayer}(x)\right)
```

$x+\mathrm{Sublayer}(x)$가 residual connection이다. 이는 입력을 그대로 다음 단계로 보낼 수 있는 경로를 만든다.

Residual connection의 장점은 다음과 같다.

- 깊은 네트워크에서 gradient 흐름을 돕는다.
- sub-layer가 전체 표현을 새로 만들 필요 없이 필요한 변화량만 학습할 수 있다.
- attention이나 FFN이 초기 학습에서 불안정해도 원래 정보가 보존된다.

### Layer normalization

Layer normalization은 각 token vector 내부 feature 차원에 대해 평균과 분산을 정규화한다.

개념적으로는 다음과 같다.

```math
\begin{aligned}
\mu &= \mathrm{mean}(x),\\
\sigma &= \mathrm{std}(x),\\
\mathrm{LayerNorm}(x)
&=\gamma\frac{x-\mu}{\sigma}+\beta.
\end{aligned}
```

Transformer에서는 각 sub-layer 출력에 layer normalization을 적용해 학습을 안정화한다.

원 논문의 구조는 현재 많이 쓰이는 pre-norm Transformer와 달리 post-norm 형태다.

```math
\begin{aligned}
\text{post-norm:}\quad
&\mathrm{LayerNorm}\!\left(x+\mathrm{Sublayer}(x)\right),\\
\text{pre-norm:}\quad
&x+\mathrm{Sublayer}\!\left(\mathrm{LayerNorm}(x)\right).
\end{aligned}
```

원 논문은 post-norm을 사용한다.

<p align="center"><img src="https://github.com/Lahecka051/Documents/releases/download/docs-media-v1/04_add_norm_orthogonal.png" alt="Residual connection과 post-norm Add and Norm 계산" width="820"></p>
<p align="center"><sub>보조 도식 — residual connection과 post-norm Add &amp; Norm 순서</sub></p>

그림의 위쪽 경로는 sub-layer 계산과 dropout을, 아래쪽 우회 경로는 원래 입력 $x$를 그대로 전달하는 residual connection을 나타낸다. 두 경로를 원소별로 더한 뒤 LayerNorm을 적용하므로 원논문의 순서는 `Sublayer -> Dropout -> Add -> LayerNorm`이다. 모든 항의 shape가 $[n,512]$로 같아야 덧셈이 가능하다.

## 내부 계산 흐름 전체 정리

### 1단계: token embedding

source token id가 들어오면 embedding lookup으로 벡터를 얻는다.

<p align="center"><img src="https://github.com/Lahecka051/Documents/releases/download/docs-media-v1/04_tokenization_embedding_orthogonal.png" alt="원문 문자열에서 token ID와 embedding 행렬을 얻는 과정" width="820"></p>
<p align="center"><sub>보조 도식 — 문자열을 subword token ID와 embedding 행렬로 변환하는 과정</sub></p>

Tokenizer는 문자열을 subword token sequence로 분해하고 vocabulary의 정수 ID로 변환한다. 이 ID sequence 자체는 정수 벡터이며 실수 행렬이 아니다. 각 ID를 embedding table $E\in\mathbb{R}^{V_{\mathrm{vocab}}\times d_{\mathrm{model}}}$의 행 index로 사용해 조회할 때 비로소 $X\in\mathbb{R}^{n\times512}$인 실수 벡터 sequence가 만들어진다. 원논문의 English-German 설정은 source와 target이 약 37,000개 token의 BPE vocabulary를 공유한다.

```text
source ids: [B, n]
embedding matrix: [vocab_size, d_model]
source embedding X: [B, n, d_model]
```

논문은 embedding에 $\sqrt{d_{\mathrm{model}}}$을 곱한다.

```math
X=\mathrm{Embedding}(\mathrm{source\_ids})\sqrt{d_{\mathrm{model}}}
```

이후 positional encoding을 더한다.

```math
H_0=X+\mathrm{PE}
```

<p align="center"><img src="https://github.com/Lahecka051/Documents/releases/download/docs-media-v1/04_encoder_input_orthogonal.png" alt="Embedding scaling과 positional encoding으로 만드는 encoder 입력" width="820"></p>
<p align="center"><sub>보조 도식 — embedding scaling과 positional encoding 합산으로 만드는 encoder 입력</sub></p>

그림의 $X$는 embedding lookup 결과다. 원논문은 이 값에 $\sqrt{d_{\mathrm{model}}}$을 곱한 뒤 같은 shape의 positional encoding $PE$를 원소별로 더한다. 따라서 첫 encoder layer 입력 $H^{(0)}$의 sequence length $n$과 model dimension 512는 변하지 않는다.

### 2단계: encoder self-attention

각 encoder layer에서 먼저 self-attention을 계산한다.

```math
\begin{aligned}
Q&=HW_Q,\\
K&=HW_K,\\
V&=HW_V,\\
S&=\frac{QK^\top}{\sqrt{d_k}},\\
A&=\mathrm{softmax}(S),\\
O&=AV.
\end{aligned}
```

multi-head이면 이 과정을 head마다 수행한다.

```math
\begin{aligned}
\mathrm{head}_i
&=\mathrm{Attention}(HW_i^Q,HW_i^K,HW_i^V),\\
\mathrm{MHA}
&=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_8)W^O.
\end{aligned}
```

residual과 normalization을 적용한다.

```math
H_{\mathrm{attn}}
=\mathrm{LayerNorm}\!\left(H+\mathrm{Dropout}(\mathrm{MHA})\right)
```

### 3단계: encoder FFN

각 위치에 같은 FFN을 적용한다.

```math
\begin{aligned}
F&=\mathrm{ReLU}(H_{\mathrm{attn}}W_1+b_1)W_2+b_2,\\
H_{\mathrm{next}}
&=\mathrm{LayerNorm}\!\left(H_{\mathrm{attn}}+\mathrm{Dropout}(F)\right).
\end{aligned}
```

이것을 6번 반복하면 encoder output $Z$가 나온다.

```math
\begin{aligned}
Z&=\mathrm{Encoder}(\mathrm{source}),\\
\mathrm{shape}(Z)&=[B,n,d_{\mathrm{model}}].
\end{aligned}
```

### 4단계: decoder masked self-attention

target sequence는 한 칸 오른쪽으로 shift된 상태로 들어간다.

```text
decoder input: <bos>, y_1, y_2, ..., y_{m-1}
prediction:    y_1, y_2, y_3, ..., y_m
```

decoder self-attention에서는 causal mask를 적용한다.

```math
\begin{aligned}
S&=\frac{QK^\top}{\sqrt{d_k}},\\
S[j>i]&=-\infty,\\
A&=\mathrm{softmax}(S),\\
O&=AV.
\end{aligned}
```

이렇게 하면 위치 $i$는 $i$보다 뒤의 target token을 볼 수 없다.

### 5단계: encoder-decoder attention

decoder는 masked self-attention을 거친 뒤 source memory를 조회한다.

```math
\begin{aligned}
Q&=\mathrm{decoder\_state}W_Q,\\
K&=\mathrm{encoder\_output}W_K,\\
V&=\mathrm{encoder\_output}W_V,\\
O&=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
\end{aligned}
```

여기서 query는 target 쪽에서 오고, key/value는 source 쪽에서 온다. 따라서 target 위치마다 source sequence 전체를 다르게 참고할 수 있다.

### 6단계: decoder FFN

cross-attention 결과에 다시 position-wise FFN을 적용한다.

```math
D_{\mathrm{next}}
=\mathrm{LayerNorm}\!\left(D_{\mathrm{cross}}+\mathrm{FFN}(D_{\mathrm{cross}})\right)
```

decoder layer도 6번 반복한다.

### 7단계: output softmax

마지막 decoder output을 vocabulary 크기로 projection한다.

```math
\begin{aligned}
\mathrm{logits}&=D_{\mathrm{final}}W_{\mathrm{vocab}}^\top+b,\\
\mathrm{probs}&=\mathrm{softmax}(\mathrm{logits}).
\end{aligned}
```

각 위치에서 다음 token 확률 분포를 얻는다.

```math
\mathrm{shape}(\mathrm{probs})=[B,m,\mathrm{vocab\_size}]
```

학습에서는 정답 target token에 대한 cross-entropy loss를 최소화한다. 논문은 label smoothing도 사용한다.

## 학습 설정과 실험 결과

### 학습 데이터

논문은 주로 WMT 2014 기계번역 태스크에서 성능을 평가한다.

- English-German: 약 4.5M sentence pairs
- English-French: 약 36M sentence pairs
- English-German에는 BPE 기반 약 37k shared vocabulary 사용
- English-French에는 약 32k word-piece vocabulary 사용

### Optimizer와 learning rate schedule

Optimizer는 Adam을 사용한다.

```math
\begin{aligned}
\beta_1&=0.9,\\
\beta_2&=0.98,\\
\epsilon&=10^{-9}.
\end{aligned}
```

Learning rate schedule은 다음과 같다.

```math
\mathrm{lrate}
=d_{\mathrm{model}}^{-0.5}
\min\!\left(
\mathrm{step\_num}^{-0.5},
\mathrm{step\_num}\cdot\mathrm{warmup\_steps}^{-1.5}
\right)
```

$\mathrm{warmup\_steps}=4000$이다.

이 schedule은 처음에는 learning rate를 선형 증가시키고, warmup 이후에는 step 수의 inverse square root에 비례해 감소시킨다.

### 주요 결과

논문의 큰 주장은 Transformer가 RNN/CNN 기반 모델보다 더 빠르게 학습되면서도 높은 번역 품질을 얻는다는 것이다.

대표 결과는 다음과 같다.

| 모델 | EN-DE BLEU | EN-FR BLEU |
| --- | ---: | ---: |
| Transformer base | 27.3 | 38.1 |
| Transformer big | 28.4 | 41.8 |

English-German에서는 당시 최고 수준 모델과 ensemble을 넘어서는 성능을 보였고, English-French에서도 single model 기준 강한 성능을 보였다. 중요한 점은 성능뿐 아니라 학습 비용이 크게 낮았다는 것이다.

## Self-attention의 이론적 장점

논문은 self-attention을 RNN, CNN과 비교하면서 세 가지 기준을 본다.

1. layer당 계산 복잡도
2. 병렬화 가능성
3. 장거리 의존성을 연결하는 path length

요약하면 다음과 같다.

| layer type | layer당 복잡도 | sequential operations | maximum path length |
| --- | ---: | ---: | ---: |
| self-attention | $O(n^2d)$ | $O(1)$ | $O(1)$ |
| recurrent | $O(nd^2)$ | $O(n)$ | $O(n)$ |
| convolutional | $O(knd^2)$ | $O(1)$ | $O(\log_k n)$ |

여기서:

- $n$: sequence length
- $d$: representation dimension
- $k$: convolution kernel size

Self-attention의 장점은 모든 위치가 한 layer 안에서 직접 연결된다는 것이다. 따라서 최대 path length가 $O(1)$이다. 이는 장거리 의존성 학습에 유리하다.

반면 단점도 있다. 모든 위치 쌍을 비교하므로 sequence length에 대해 $O(n^2)$ 비용이 든다. 짧거나 중간 길이 문장에서는 문제가 작지만, 아주 긴 문서, 이미지, 비디오, 고해상도 feature map에서는 큰 병목이 된다. 이후 Longformer, Reformer, Linformer, Performer, BigBird, FlashAttention 같은 연구는 대부분 이 비용 문제를 다룬다.

## 장점과 기여

이 논문의 가장 큰 기여는 attention을 보조 장치가 아니라 sequence modeling의 중심 연산으로 끌어올렸다는 점이다.

주요 기여는 다음과 같다.

- RNN과 CNN 없이 attention만으로 강한 sequence transduction 모델을 만들었다.
- Self-attention을 통해 sequence 내부 모든 위치의 관계를 직접 계산했다.
- Multi-head attention으로 서로 다른 representation subspace의 관계를 병렬로 학습했다.
- Positional encoding으로 recurrence 없이 순서 정보를 주입했다.
- Encoder-decoder attention을 query-key-value 구조로 깔끔하게 정식화했다.
- Attention과 FFN을 교대로 쌓는 단순하고 확장 가능한 block 구조를 제시했다.
- 병렬 학습 효율을 크게 개선했다.

현재 관점에서 보면 Transformer block의 기본 공식은 다음 한 줄로 요약된다.

```text
context mixing by attention + feature transformation by FFN
```

이 구조가 이후 BERT, GPT, T5, ViT, DETR 등 거의 모든 현대 대형 모델의 기본 단위가 되었다.

## 한계와 비판적 관점

첫 번째 한계는 self-attention의 $O(n^2)$ 비용이다. 모든 token pair를 비교하기 때문에 sequence length가 길어질수록 메모리와 계산량이 빠르게 증가한다. 이 논문 자체도 긴 입력에 대해서는 restricted attention을 future work로 언급한다.

두 번째 한계는 위치 정보가 외부에서 주입된다는 점이다. RNN이나 CNN은 구조 자체에 순서 또는 locality bias가 있다. Transformer는 positional encoding 없이는 순서를 모른다. 이 때문에 위치 표현 방식은 이후 중요한 연구 주제가 되었다. 예를 들어 relative positional encoding, RoPE, ALiBi 같은 방법들이 등장했다.

세 번째 한계는 data와 compute가 충분할 때 특히 강하다는 점이다. Transformer는 inductive bias가 상대적으로 약하고 유연한 구조이므로, 작은 데이터에서는 RNN/CNN의 구조적 bias가 더 도움이 되는 경우도 있다.

네 번째로 attention weight를 해석할 때 주의가 필요하다. 논문은 attention head가 장거리 의존성이나 문법 구조와 관련된 패턴을 보인다고 설명하지만, attention weight가 항상 인간이 생각하는 설명과 일치한다고 단정할 수는 없다.

마지막으로 decoder는 여전히 autoregressive하다. 학습은 병렬화가 가능하지만, 생성 시에는 $y_1$을 만든 뒤 $y_2$를 만들고, 그 다음 $y_3$을 만드는 순차성이 남아 있다. 논문도 generation을 덜 sequential하게 만드는 것을 향후 과제로 언급한다.

## 후속 논문과의 연결점

이 논문 이후 attention 연구는 크게 두 방향으로 확장된다.

첫 번째는 position과 context length 문제다.

- Relative Position Representations: 절대 위치보다 상대 거리 정보를 attention에 반영한다.
- Transformer-XL: segment recurrence와 relative position으로 긴 문맥을 다룬다.
- RoPE, ALiBi, xPos, YaRN, LongRoPE: 길이 extrapolation과 위치 표현을 개선한다.

두 번째는 attention 효율 문제다.

- Sparse Transformer, Longformer, BigBird: 모든 위치를 보지 않고 sparse pattern을 사용한다.
- Linformer, Performer, Nyströmformer: attention matrix를 근사하거나 linear attention으로 바꾼다.
- FlashAttention: attention의 수학은 유지하되 GPU memory access를 최적화한다.
- Multi-query attention, grouped-query attention: decoder inference에서 key/value cache 비용을 줄인다.

Computer vision 쪽으로도 흐름이 이어진다.

- Non-local neural networks: feature map 위치 사이의 장거리 관계를 attention처럼 계산한다.
- ViT: 이미지를 patch sequence로 보고 Transformer encoder를 적용한다.
- DETR: object detection을 set prediction과 Transformer encoder-decoder 문제로 재정의한다.

## 개인 학습/연구 메모

이 논문을 공부할 때 가장 중요한 것은 Transformer를 하나의 마법 같은 구조로 외우지 않는 것이다. 내부 계산은 매우 규칙적이다.

핵심 흐름은 다음과 같다.

```text
1. token을 vector로 바꾼다.
2. 위치 정보를 더한다.
3. 각 위치에서 Q, K, V를 만든다.
4. QK^T로 위치 간 관련도를 계산한다.
5. sqrt(d_k)로 scale을 맞춘다.
6. mask가 필요한 위치는 -inf로 막는다.
7. softmax로 attention weight를 만든다.
8. attention weight로 V를 weighted sum한다.
9. 여러 head 결과를 합치고 선형 변환한다.
10. residual과 layer norm을 적용한다.
11. FFN으로 각 위치의 feature를 비선형 변환한다.
12. 이 과정을 encoder와 decoder에서 여러 번 반복한다.
13. decoder output을 vocabulary softmax로 바꿔 다음 token을 예측한다.
```

Attention은 "어떤 정보를 볼 것인가"의 문제이고, FFN은 "본 정보를 어떻게 가공할 것인가"의 문제다. Encoder는 source를 읽어 memory를 만들고, decoder는 target을 생성하면서 self-attention으로 이전 target 문맥을 보고 cross-attention으로 source memory를 조회한다.

이 관점으로 보면 Transformer의 세 가지 attention도 자연스럽게 구분된다.

```text
encoder self-attention:
source token이 source token들을 본다.

decoder masked self-attention:
target token이 과거 target token들을 본다.

encoder-decoder attention:
target token이 source token들을 본다.
```

결국 Transformer는 sequence를 순서대로 읽는 모델이라기보다, sequence 안의 모든 위치가 서로를 검색하고 정보를 교환하는 모델이다. 이 전환이 "Attention Is All You Need"라는 제목의 의미다.


# 11. A Length-Extrapolatable Transformer

## 논문 정보

- 제목: **A Length-Extrapolatable Transformer**
- 저자: Yutao Sun, Li Dong, Barun Patra, Shuming Ma, Shaohan Huang, Alon Benhaim, Vishrav Chaudhary, Xia Song, Furu Wei
- 발표: 2022년 arXiv 공개, ACL 2023
- 원본 PDF: [11_xPos_Length_Extrapolatable_Transformer.pdf](./11_xPos_Length_Extrapolatable_Transformer.pdf)
- 핵심 키워드: `xPos`, `LEX Transformer`, attention resolution, blockwise causal attention, length extrapolation

## 한눈에 보는 요약

이 논문은 Transformer의 길이 외삽을 단순히 “position index를 더 크게 넣어도 되는가”가 아니라, **긴 거리에서도 attention이 위치 차이를 얼마나 안정적으로 구분하는가**의 문제로 본다. 이를 정량화하기 위해 `attention resolution`을 정의하고, resolution을 높이는 두 장치를 제안한다.

1. **xPos**: RoPE 회전에 query/key의 reciprocal exponential scale을 추가한다.
2. **Blockwise Causal Attention(BCA)**: 추론 시 학습 길이 안의 local block만 보게 해 위치 분해능을 보존한다.

논문에서 `xPos`는 위치 표현이고, `LEX Transformer`는 xPos와 BCA를 결합한 전체 모델이다.

```math
q_m' = R_m q_m\odot \hat\zeta^{m},
\qquad
k_n' = R_n k_n\odot \hat\zeta^{-n}.
```

$Q$와 $K$에 서로 역수인 scale을 적용하므로 dot product에는 절대 위치 각각이 아니라 $m-n$에 따른 scale ratio가 남는다. RoPE의 상대 위치 성질을 유지하면서, 장거리에서 크게 진동하는 고주파 성분을 감쇠한다.

![xPos의 reciprocal scaling, attention resolution, blockwise causal attention](https://github.com/user-attachments/assets/a89fe11f-c946-45ac-a889-4cb4d757802c)

## 연구 배경과 문제의식

### 위치 표현에 필요한 세 가지 성질

논문은 Transformer의 position modeling을 세 축으로 정리한다.

#### 1. Order variance

Token 순서를 바꾸면 의미와 출력도 달라져야 한다.

```math
f(P_\pi(X))\ne P_\pi(f(X)).
```

Position 정보가 전혀 없는 self-attention은 입력 permutation에 equivariant하므로 sequence order를 스스로 알 수 없다.

#### 2. Translation invariance

문장 앞뒤에 padding이 붙어 절대 위치가 이동해도 유효 token의 표현은 같아야 한다. 입력 $X$와 mask $M$ 앞뒤에 padding을 붙인 경우를 다음과 같이 적는다.

```math
X_{pad}=[0]^i\oplus X\oplus[0]^j,
\qquad
M_{pad}=[0]^i\oplus M\oplus[0]^j.
```

올바른 mask가 있다면 유효 구간 출력은 보존되어야 한다.

```math
f(X,M)
=
f(X_{pad},M_{pad})[i:i+n].
```

Relative position 방식은 이 성질을 자연스럽게 만족한다.

#### 3. Length extrapolation

학습 최대 길이 $L$보다 긴 입력에서도 위치 분포가 붕괴하지 않아야 한다. 더 나아가 context가 늘면 다음 token prediction에 사용할 정보가 많아지므로 perplexity가 내려가는 것이 이상적이다.

논문은 기존 방법을 대략 다음처럼 평가한다.

| 방법 | Translation invariance | Length extrapolation |
| --- | --- | --- |
| Absolute sinusoidal/learned | 약함 | 약함 |
| RoPE/T5 relative | 강함 | 약함 |
| ALiBi | 강함 | 가능하지만 일반 성능 절충 |
| LEX Transformer | 강함 | 강함을 목표 |

### RoPE의 문제를 “진동”으로 본다

RoPE는 각 2차원 pair를 위치에 따라 회전한다.

```math
q_m'=R(m\theta)q_m,
\qquad
k_n'=R(n\theta)k_n.
```

Dot product에는 상대 위치 $m-n$이 나타난다.

```math
q_m'^{\top}k_n'
=
q_m^\top R((n-m)\theta)k_n.
```

하지만 cosine과 sine은 주기적이다. 상대 거리가 커지면 일부 주파수 성분은 여러 번 회전하고 score expectation이 심하게 진동한다. 학습 길이 안에서는 모델이 이 패턴에 적응할 수 있지만, 길이 밖에서는 가까운/먼 위치의 순서가 뒤섞일 수 있다.

ALiBi는 단조 거리 감쇠로 이 문제를 피하지만, 먼 token을 지속적으로 감점해 일반적인 장거리 dependency 표현을 희생할 수 있다. 논문은 **RoPE의 회전 표현력은 유지하고 불안정한 장거리 진동만 줄이려 한다.**

## Attention resolution

### 정의

거리 $n$인 query-key 쌍의 attention logit 기대값을 $s[n]$이라 하자. 논문은 다음 값을 attention resolution로 정의한다.

```math
R(s)
=
\frac{
\sum_{i=0}^{N}e^{s[i]}
\left(e^{s[i]}-e^{s[i+1]}\right)
}{
\left(\sum_{i=0}^{N}e^{s[i]}\right)^2
}.
```

이 식은 두 직관을 결합한다.

- $s[i]>s[i+1]$: 거리가 멀수록 score가 낮아지는 monotonicity
- $e^{s[i]}$: softmax에서 실제 영향이 큰 logit 구간에 더 큰 가중치

Resolution가 크면 가까운 context와 먼 context가 attention 확률에서 더 잘 구분된다. 반대로 score가 거리와 무관하거나 심하게 oscillate하면 resolution이 낮아진다.

### 무엇을 측정하고 무엇을 측정하지 않는가

Attention resolution는 task accuracy나 장거리 reasoning 능력 자체가 아니다. 모델의 **평균적인 distance discrimination**을 나타내는 보조 지표다.

같은 resolution를 가진 두 모델도 content selectivity는 다를 수 있다. 또한 언어의 모든 attention이 거리에 따라 단조 감소해야 하는 것도 아니다. 따라서 논문의 지표는 좋은 설계 방향을 제공하지만 완전한 품질 척도는 아니다.

## xPos의 수식 전개

### 복소수 표현에서 출발

2차원 real pair를 하나의 complex dimension으로 보면 RoPE는 다음과 같다.

```math
f_q(q,n)=qe^{in\theta},
\qquad
f_k(k,n)=ke^{in\theta}.
```

논문은 상대 위치 성질을 유지하는 더 일반적인 꼴을 찾는다.

```math
\langle f_q(q,n+r),f_k(k,n)\rangle
=
\langle f_q(q,r),f_k(k,0)\rangle.
```

한 해는 query와 key에 반대 부호의 complex exponent를 주는 것이다.

```math
f_q(q,n)=qe^{\xi n+i\theta n},
\qquad
f_k(k,n)=ke^{-\xi n+i\theta n}.
```

논문의 real-vector 구현에서는 $\zeta=e^\xi$에 해당하는 scale을 사용한다. Query는 $\zeta^n$, key는 $\zeta^{-n}$을 곱하므로 dot product의 scale은 상대 위치의 함수가 된다.

```math
\zeta^m\zeta^{-n}=\zeta^{m-n}.
```

### 장거리 score expectation

RoPE 주파수를 $\theta_i=10000^{-2i/d}$라 하고 각 차원별 scale을 $\zeta_i$라 하자. 논문은 거리 $n$에서 위치 함수의 성격을 다음처럼 본다.

```math
g_\zeta[n]
=
\sum_{i=0}^{d/2-1}
\cos(n\theta_i)\zeta_i^n.
```

RoPE는 $\zeta_i=1$인 특수한 경우다. $0<\zeta_i<1$이면 장거리에서 해당 cosine 진동의 amplitude가 감소한다.

### 차원별 scale 설계

큰 $\theta_i$, 즉 고주파 차원이 장거리에서 더 빨리 oscillate한다. 논문은 이러한 차원에 더 강한 decay를 주고, 저주파의 안정적인 차원은 더 보존한다.

```math
\tilde\zeta_i
=
\frac{i/(d/2)+\gamma}{1+\gamma},
\qquad 0<\tilde\zeta_i\le1.
```

작은 $i$의 고주파 차원은 $\zeta_i$가 작아 더 빨리 감쇠하고, 큰 $i$의 저주파 차원은 1에 가까워 장거리 정보를 유지한다. 최종 $\hat\zeta$는 attention resolution와 fp16 표현 범위를 함께 고려해 정한다.

### 2차원 pair의 실제 변환

Pair $[x_{2i},x_{2i+1}]$에 대해 RoPE 회전은 다음과 같다.

```math
\operatorname{rot}(x)
=
[-x_1,x_0,-x_3,x_2,\ldots].
```

위치 $n$의 query와 key는 다음과 같이 계산할 수 있다.

```math
Q_n'
=
\left(Q_n\odot\cos(n\theta)
+\operatorname{rot}(Q_n)\odot\sin(n\theta)\right)
\odot\hat\zeta^n,
```

```math
K_n'
=
\left(K_n\odot\cos(n\theta)
+\operatorname{rot}(K_n)\odot\sin(n\theta)\right)
\odot\hat\zeta^{-n}.
```

즉 xPos는 RoPE를 없애지 않는다. **회전 뒤에 reciprocal scale을 추가한 RoPE의 일반화**다.

## 왜 reciprocal scale이 상대 위치 성질을 보존하는가

Query와 key에 같은 scale을 곱하면 dot product에 $\zeta^{m+n}$이 남아 절대 위치 합에 의존한다. xPos는 역수를 사용해 $m-n$만 남긴다.

```math
\begin{aligned}
\text{same-direction scaling:}\quad
&q_m\zeta^m,\ k_n\zeta^n,\\
&\text{dot scale}=\zeta^{m+n}
\quad\text{(absolute-position dependent)},\\[4pt]
\text{reciprocal scaling:}\quad
&q_m\zeta^m,\ k_n\zeta^{-n},\\
&\text{dot scale}=\zeta^{m-n}
\quad\text{(relative-position dependent)}.
\end{aligned}
```

이 설계는 translation invariance를 유지하는 핵심이다.

다만 causal attention에서 보통 $m\ge n$이므로 $0<\zeta<1$일 때 $\zeta^{m-n}$가 decay가 된다. Query/key index convention을 반대로 구현하면 exponent 부호도 함께 바꿔야 한다.

## Blockwise Causal Attention

### 왜 위치 표현만으로 충분하지 않다고 보는가

xPos가 장거리 score를 안정화해도 모델은 학습 때 길이 $L$의 attention matrix만 보았다. 추론에서 갑자기 훨씬 많은 key가 softmax 경쟁에 들어오면 hidden-state와 attention 분포가 달라진다.

BCA는 추론 시 한 query가 보는 거리 범위를 학습 범위 안으로 제한한다.

### 구조

학습 길이를 $L$이라 하면 query를 $L/2$ 길이의 block으로 나눈다. 각 query block은 다음 두 block의 key/value에 접근한다.

```text
previous key/value block: L/2
current  key/value block: L/2
total visible window:     L
```

예를 들어 $L=1024$라면 512-token query block이 이전 512 token과 현재 512 token을 본다. 다음 block으로 갈 때 이전 K/V를 재사용하므로 cache-friendly하다.

```text
block 0 query -> keys block 0
block 1 query -> keys block 0 + block 1
block 2 query -> keys block 1 + block 2
block 3 query -> keys block 2 + block 3
```

학습에서는 일반 causal attention을 쓰고, 긴 sequence 추론에서만 BCA를 적용한다. 따라서 architecture retraining 없이 inference policy로 추가할 수 있다.

### 정보가 window보다 멀리 전달되는 방식

현재 block은 바로 이전 block의 hidden state를 본다. 이전 block의 hidden state는 그 이전 context를 이미 여러 layer에서 요약했을 수 있다. 이런 recurrent-like 전달로 raw token의 직접 attention 범위보다 더 먼 정보가 전파된다.

하지만 이는 모든 먼 token에 직접 접근하는 full attention과 같지 않다. 중간 representation을 통한 압축이므로 exact long-range retrieval에는 손실이 있을 수 있다.

## Tensor shape와 구현

```text
Q, K, V                  [B, H, N, d_h]
cos, sin                 [N, d_h]
zeta_power               [N, d_h]
rotated/scaled Q, K      [B, H, N, d_h]
block attention score    [B, H, L/2, L]
```

논문 Algorithm 1에 대응하는 단순화된 구현은 다음과 같다.

```python
def xpos(q, k, cos, sin, scale):
    # scale[p, i] = zeta[i] ** p
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot * scale, k_rot / scale

q, k = xpos(q, k, cos, sin, scale)
scores = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim)
scores = scores.masked_fill(blockwise_causal_mask, float("-inf"))
out = scores.softmax(dim=-1) @ v
```

### 수치 안정성

$\zeta_i^n$은 매우 작아지고 $\zeta_i^{-n}$은 매우 커질 수 있다. 논문은 8192 길이에서 fp16로 표현 가능한 범위를 고려해 $\gamma$를 정한다.

실제 구현에서는 다음을 확인해야 한다.

- scale을 fp32로 생성한 뒤 target dtype으로 변환할지
- exponent range가 fp16/bf16에서 overflow/underflow하지 않는지
- reciprocal을 직접 계산할지 log-space에서 만들지
- KV cache에 저장할 key가 xPos 적용 전인지 후인지
- decode 위치 offset과 block local index가 일치하는지

고정 xPos에서는 이미 변환된 key를 cache할 수 있다. 그러나 scale schedule을 런타임에 바꾸는 동적 방법과 결합하면 과거 key의 변환도 달라질 수 있으므로 주의해야 한다.

## 전체 계산 흐름

1. Hidden state에서 $Q$, $K$, $V$를 projection한다.
2. 각 position과 RoPE pair의 $\cos(n\theta_i)$, $\sin(n\theta_i)$를 준비한다.
3. $Q$와 $K$를 RoPE 방식으로 회전한다.
4. $Q$에는 $\hat\zeta^n$, $K$에는 $\hat\zeta^{-n}$을 곱한다.
5. 학습 길이 안에서는 일반 causal attention을 수행한다.
6. 긴 추론에서는 query를 $L/2$ block으로 나눈다.
7. 각 query block이 현재와 이전 K/V block만 보도록 mask한다.
8. Softmax와 value aggregation은 일반 attention과 동일하다.

## 실험 설정

- 24 Transformer layers
- hidden dimension 1024
- 16 attention heads
- 학습 최대 길이 1024
- Pile의 Books3, OpenWebText2, Stack Exchange, PubMed Abstracts, Wikipedia, PG-19 등 subset
- 16 V100 GPUs
- global batch size 512, 약 0.5M tokens
- Adam, learning rate $3\times10^{-4}$, polynomial decay
- 긴 arXiv 문서의 앞 4k token으로 256~4096 길이 perplexity 평가

비교 대상은 absolute Transformer, ALiBi, RoFormer(RoPE), LEX Transformer다. 2048/4096 평가에서는 각 position 방법에 BCA를 적용한 결과를 기본 표에 사용하고, 별도 ablation에서 BCA 유무를 나눈다.

## 주요 실험 결과

### 길이에 따른 perplexity

모든 모델은 길이 1024로 학습했다.

| 모델 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Absolute Transformer | 46.34 | 36.39 | 29.94 | 132.63 | 1283.79 |
| ALiBi | 37.66 | 29.92 | 24.99 | 23.14 | 24.26 |
| RoPE | 38.09 | 30.38 | 25.52 | 73.60 | 294.45 |
| LEX Transformer | **34.30** | **27.55** | **23.31** | **21.60** | **20.73** |

LEX만 평가 길이가 256에서 4096으로 늘어날수록 perplexity가 일관되게 내려간다. Absolute와 RoPE는 extrapolation 구간에서 폭발하고, ALiBi는 2048까지 좋아진 뒤 4096에서 소폭 악화된다.

### Attention resolution 측정

| 모델 | 길이 1024 | 길이 2048 |
| --- | ---: | ---: |
| Absolute Transformer | 0.87 | 0.28 |
| ALiBi | 0.81 | 0.88 |
| RoPE | 0.91 | 0.08 |
| LEX | **0.98** | **1.08** |
| LEX without BCA | 0.98 | 0.54 |

길이 2048에서 RoPE resolution가 0.08로 크게 낮아진 반면 LEX+BCA는 1.08을 유지한다. BCA를 빼면 0.54로 내려가 xPos와 attention mask가 상호 보완적임을 보여준다.

### 회전의 필요성

| 위치 변환 | validation PPL |
| --- | ---: |
| RoPE | 17.74 |
| xPos | **17.54** |
| xPos에서 rotation 제거 | 33.68 |

Exponential scaling만 남기면 성능이 크게 무너진다. xPos의 성능은 decay 하나가 아니라 **RoPE 회전과 decay의 결합**에서 나온다.

### BCA ablation

| 방법 | 2048 | 4096 |
| --- | ---: | ---: |
| RoPE | 73.60 | 294.45 |
| RoPE + BCA | 25.57 | 25.65 |
| ALiBi | 23.14 | 24.26 |
| ALiBi + BCA | 24.60 | 25.37 |
| xPos | 22.56 | 28.43 |
| xPos + BCA | **21.60** | **20.73** |

BCA는 RoPE의 폭발을 크게 막지만, ALiBi에는 오히려 약간 불리하다. ALiBi 자체가 soft window처럼 동작하므로 hard block을 추가하면 유용한 먼 context를 더 제한할 수 있다. xPos는 BCA와 결합할 때 가장 안정적이다.

## 계산 복잡도와 메모리

### xPos 자체

RoPE와 마찬가지로 element-wise 회전과 scale을 추가하므로 복잡도는 $O(Nd)$이고, dense attention의 $O(N^2d)$를 바꾸지 않는다. Scale tensor를 cache하면 추가 overhead는 작다.

### BCA

전체 길이를 $N$, 학습 window를 $L$이라 하면 각 token은 최대 $L$개의 key를 본다.

```math
\text{attention cost}
\approx O(NLd)
```

$L$을 고정하고 $N$만 늘리는 추론에서는 full attention의 $O(N^2d)$보다 유리하다. 하지만 논문은 BCA를 주로 extrapolation resolution 장치로 사용하며, 현대 sparse kernel에서 실제 속도 이득은 block layout과 구현에 달려 있다.

## 기존 방법과 비교

| 방법 | 핵심 조작 | 장거리 안정화 방식 | pretrained RoPE checkpoint에 즉시 적용 |
| --- | --- | --- | --- |
| RoPE | Q/K 회전 | 별도 장치 없음 | 기준 방식 |
| ALiBi | logit 선형 bias | 단조 recency penalty | 보통 처음부터 재학습 |
| xPos | 회전 + reciprocal scale | 불안정 주파수 decay | 원 논문은 from-scratch 학습 |
| xPos + BCA | 위 변환 + hard block | score와 visible range 모두 제어 | 추론 mask 변경 필요 |
| Position Interpolation | position index 압축 | 학습 phase 범위 안에 매핑 | 짧은 fine-tuning으로 적용 |

xPos는 후속 PI/YaRN처럼 기존 대형 RoPE 모델을 최소 fine-tuning으로 확장하는 실험이 중심이 아니다. 같은 문제군에 있지만 사용 조건이 다르다.

## 장점과 핵심 기여

### 1. Length extrapolation을 측정할 이론적 언어를 제시했다

Attention resolution는 완전한 척도는 아니지만, “긴 거리에서 위치 구분력이 왜 사라지는가”를 분석 가능한 수식으로 만들었다.

### 2. RoPE의 상대 위치 성질을 보존한다

Query/key reciprocal scaling 덕분에 scale이 $m-n$의 함수로 남는다.

### 3. 고주파와 저주파의 역할을 분리했다

불안정한 고주파는 더 강하게 감쇠하고 안정적인 저주파는 보존한다. 이 관점은 이후 YaRN과 LongRoPE의 dimension-wise scaling으로 이어진다.

### 4. 위치 표현과 attention pattern을 함께 설계했다

Position encoding만 바꾸면 모든 extrapolation 문제가 해결된다고 가정하지 않고, 추론 mask가 resolution에 미치는 영향을 실험했다.

### 5. 긴 길이에서 실제 perplexity 감소를 보였다

단순히 폭발을 막는 수준이 아니라 1024에서 4096으로 갈수록 LEX perplexity가 23.31에서 20.73으로 내려갔다.

## 한계와 비판적 관점

### 1. Attention resolution의 단조성 가정

언어에서는 멀리 있는 특정 token이 가까운 불필요 token보다 중요할 수 있다. 평균 score가 거리와 함께 감소해야 한다는 가정은 유용한 prior이지만 모든 head와 task의 이상적 behavior는 아니다.

### 2. BCA는 full attention과 동등하지 않다

이전 block보다 먼 raw K/V에 직접 접근하지 못한다. 장거리 정보가 layer별 hidden state에 성공적으로 압축돼야 하므로 needle retrieval 같은 task에서 제한될 수 있다.

### 3. Train-test attention mismatch

학습은 full causal attention, 긴 추론은 blockwise attention이다. 실험에서는 잘 작동했지만 mask distribution이 달라지는 구조적 mismatch가 있다.

### 4. 수치 범위와 구현 복잡성

$\zeta^n$과 $\zeta^{-n}$은 긴 위치에서 underflow/overflow 위험이 있다. RoPE보다 dtype, cache, scale schedule에 더 주의해야 한다.

### 5. 평가 길이와 모델 규모가 제한적이다

핵심 표는 길이 1024 학습, 4096 평가와 약 24-layer 모델이다. 이후 수십만 token context의 pretrained LLM에 대한 직접 증거와는 구분해야 한다.

### 6. 매우 긴 context의 시스템 비용은 별도다

xPos만 사용하면 dense attention과 KV cache 증가가 그대로다. BCA는 비용을 줄일 수 있지만 global retrieval 능력과 절충한다.

## 자주 헷갈리는 지점

### xPos와 LEX Transformer는 같은가

아니다. xPos는 extrapolatable position embedding이고, LEX Transformer는 xPos와 blockwise causal attention을 포함한 전체 설계다.

### xPos는 RoPE를 대체하는가

RoPE 회전을 유지하고 scale을 추가한다. RoPE의 일반화로 보는 편이 정확하다.

### Query와 key에 왜 서로 반대 scale을 쓰는가

같은 방향으로 scale하면 dot product가 $m+n$에 의존한다. Reciprocal scale은 $m-n$을 남겨 translation invariance를 보존한다.

### Value도 scale하거나 회전하는가

아니다. 위치 변환은 attention score를 만드는 Q/K에 적용한다. V aggregation은 일반 attention과 같다.

### BCA는 학습 때도 쓰는가

원 논문은 짧은 pre-training에서는 일반 causal attention을 쓰고, 긴 추론에서 BCA를 사용한다.

### xPos가 임의 길이를 보장하는가

아니다. 논문 범위에서 4배 extrapolation을 강하게 보였다. 수치 범위, block 전달, task 복잡성 때문에 무한 길이 보장은 없다.

## 온디바이스 및 비전 관점

### 온디바이스

xPos의 element-wise scale은 작지만, 긴 context의 KV cache는 그대로 크다. BCA를 고정 window kernel과 결합하면 메모리 사용을 제한할 수 있어 온디바이스에 더 직접적인 이점이 있다. 다만 block 경계를 넘는 retrieval 품질을 별도로 검증해야 한다.

### Vision

1D token 순서에서 정의한 $m-n$과 BCA를 image patch에 그대로 옮기면 raster order artifact가 생긴다. 2D RoPE pair나 row/column별 reciprocal scale, local window attention으로 재설계하는 편이 자연스럽다. Swin 계열의 shifted window는 BCA와 유사하게 local computation과 block 간 전달을 결합한다.

## 후속 논문과의 연결점

- Position Interpolation은 RoPE의 모든 position을 동일 비율로 압축한다.
- YaRN은 xPos가 강조한 주파수별 차이를 더 직접적으로 사용해 high/transition/low frequency band를 나눈다.
- LongRoPE는 차원별 optimal scale을 heuristic이 아니라 search로 찾는다.

xPos의 가장 중요한 유산은 특정 scale 공식만이 아니라, **RoPE 차원마다 장거리에서 다른 안정성을 보이며 position resolution을 보존해야 한다**는 문제 설정이다.

## 개인 학습/연구 메모

xPos는 다음 세 줄로 기억할 수 있다.

```text
RoPE rotation preserves relative phase.
Reciprocal scaling damps unstable long-distance frequencies.
Blockwise attention keeps each direct comparison inside the trained range.
```

즉 LEX의 length extrapolation은 하나의 마법 같은 position formula가 아니라, **회전 표현 + 거리별 감쇠 + 추론 window**의 결합 결과다.

# 10. Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation

## 논문 정보

- 제목: **Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation**
- 저자: Ofir Press, Noah A. Smith, Mike Lewis
- 발표: ICLR 2022
- 원본 PDF: [10_ALiBi_Train_Short_Test_Long.pdf](./10_ALiBi_Train_Short_Test_Long.pdf)
- 핵심 키워드: `ALiBi`, linear attention bias, length extrapolation, recency prior

## 한눈에 보는 요약

ALiBi(Attention with Linear Biases)는 token embedding에 position embedding을 더하지 않는다. 대신 query와 key의 dot product 뒤에 **상대 거리에 비례하는 음수 선형 bias**를 더한다.

```math
S_{i,j}^{(h)}
=
\frac{q_i^{(h)\top}k_j^{(h)}}{\sqrt{d_h}}
-m_h(i-j),
\qquad j\le i
```

여기서 $m_h>0$는 head마다 다른 고정 slope다. 가까운 과거는 거의 감점하지 않고 먼 과거는 더 크게 감점한다. 큰 slope의 head는 local context에, 작은 slope의 head는 더 긴 범위에 상대적으로 열려 있다.

논문의 핵심 주장은 단순히 “긴 입력을 계산할 수 있다”가 아니다. **짧은 sequence로 학습해 비용을 줄인 뒤, 학습보다 긴 sequence에서 perplexity가 무너지지 않도록 한다**는 것이다. 1.3B 모델에서 길이 1024로 학습한 ALiBi가 길이 2048로 학습한 sinusoidal baseline과 비슷하거나 더 좋은 perplexity를 내면서 학습 시간과 메모리를 각각 약 11% 줄였다.

<p align="center"><img src="https://github.com/user-attachments/assets/1915794d-5ea2-42a5-aa5f-34de385e38c5" alt="ALiBi linear distance bias" width="680"></p>
<p align="center"><sub>Figure 3 — attention score에 head별 선형 거리 bias를 더하는 ALiBi</sub></p>

## 연구 배경과 문제의식

### Transformer의 계산은 길이에 독립적이지만 위치 표현은 그렇지 않다

Self-attention 자체는 입력 길이 $N$이 달라져도 같은 연산을 수행한다.

```text
Q: [B, H, N, d_h]
K: [B, H, N, d_h]
score: [B, H, N, N]
```

그런데 absolute position embedding이나 학습 범위에 맞춰진 위치 함수는 학습 때 보지 못한 position에서 분포가 바뀔 수 있다. 따라서 “수식상 더 긴 배열을 받을 수 있다”와 “더 긴 길이에서 잘 일반화한다”는 전혀 다른 주장이다.

논문은 extrapolation을 다음과 같이 정의한다.

```text
training subsequence length: L
validation subsequence length: L_valid
length extrapolation: L_valid > L에서도 성능을 유지하거나 개선
```

### 왜 짧게 학습하고 길게 평가하려는가

Dense attention의 주요 메모리 항은 $N\times N$ attention matrix다. 학습 길이를 두 배로 늘리면 attention score 원소 수는 네 배가 된다. 같은 token budget이라도 긴 sequence는 batching과 activation memory 측면에서 불리하다.

RNN은 짧은 truncated sequence로 학습하고 더 긴 history에 적용하는 관행이 자연스러웠다. 반면 Transformer는 학습과 추론의 최대 길이를 같게 두는 경우가 많았다. 이 논문은 그 결합을 끊어 학습 비용을 절약하려 한다.

### 기존 위치 방법의 관찰

논문은 WikiText-103에서 네 방법을 같은 언어 모델 조건으로 비교한다.

- sinusoidal embedding은 $L$을 조금만 넘어도 perplexity가 급격히 악화된다.
- RoPE는 sinusoidal보다 낫지만 충분히 긴 길이에서는 악화된다.
- T5 relative bias는 extrapolation이 더 좋지만 당시 구현에서는 느리고 메모리 부담이 컸다.
- ALiBi는 sinusoidal 수준의 속도를 유지하면서 extrapolation을 얻는다.

이 비교에서 중요한 점은 extrapolation 실패를 Transformer 전체의 필연적 한계가 아니라 **position representation의 inductive bias 문제**로 분리했다는 데 있다.

## 핵심 아이디어

### Position vector 대신 logit bias

일반적인 causal self-attention은 다음과 같다.

```math
A_i
=
\mathrm{softmax}\left(
\frac{q_iK_{\le i}^{\top}}{\sqrt{d_h}}
\right),
\qquad
o_i=A_iV_{\le i}.
```

ALiBi는 softmax 직전 score에 head별 거리 bias를 더한다.

```math
A_i^{(h)}
=
\mathrm{softmax}\left(
\frac{q_i^{(h)}K_{\le i}^{(h)\top}}{\sqrt{d_h}}
+m_h[-(i-1),\ldots,-2,-1,0]
\right).
```

query가 위치 $i$에 있을 때 가장 최근 key $j=i$의 bias는 0이다. $j=i-1$은 $-m_h$, $j=i-2$는 $-2m_h$가 된다.

```text
query position i = 4
key position       0      1      2      3      4
distance i-j       4      3      2      1      0
bias             -4m    -3m    -2m    -m      0
```

Content score가 같다면 가까운 key가 더 높은 확률을 받는다. 하지만 content score가 충분히 크면 먼 token도 선택할 수 있으므로 hard window가 아니라 **soft recency prior**다.

### Head마다 다른 slope를 쓰는 이유

모든 head에 같은 slope를 쓰면 모든 head가 비슷한 거리 범위를 선호하게 된다. ALiBi는 slope를 기하급수적으로 배치해 multi-head attention에 여러 시간 척도를 준다.

8개 head의 논문 설정은 다음과 같다.

```math
m_h\in
\left\{
2^{-1},2^{-2},\ldots,2^{-8}
\right\}.
```

16개 head에서는 중간 값을 기하 평균으로 보간해 다음 수열을 사용한다.

```math
2^{-0.5},2^{-1},2^{-1.5},\ldots,2^{-8}.
```

일반화하면 논문이 설명한 기하수열은 첫 항과 공비가 모두 $2^{-8/n}$인 형태다. slope는 학습하지 않고 고정한다.

```text
large m_h  -> distance penalty grows quickly -> local head
small m_h  -> distance penalty grows slowly  -> long-range head
```

논문은 slope를 학습 가능하게 만들었을 때 extrapolation이 좋아지지 않았고 학습 속도도 약 3% 느려졌다고 보고한다. 학습 데이터의 길이 분포에 slope가 과적합되면 학습 범위 밖의 단조 prior가 약해질 수 있다는 해석이 가능하다.

### Bias는 왜 softmax 전에 들어가는가

Softmax에 들어가는 logit 차이는 확률의 비율을 결정한다.

```math
\frac{A_{i,j_1}}{A_{i,j_2}}
=
\exp\left(
S_{i,j_1}-S_{i,j_2}
\right).
```

content score가 같고 $j_1$이 $j_2$보다 $\Delta$만큼 멀다면 ALiBi가 만드는 비율은 다음과 같다.

```math
\frac{A_{i,j_1}}{A_{i,j_2}}
=
e^{-m_h\Delta}.
```

즉 logit에서는 선형 감점이지만 확률 비율에서는 거리에 대한 지수 감쇠가 된다. 이 때문에 논문은 단순한 bias로 강한 recency inductive bias를 얻는다.

## Tensor shape와 실제 계산

입력 hidden state를 $X\in\mathbb{R}^{B\times N\times d_{model}}$라 하자.

```text
X                         [B, N, d_model]
Q, K, V after projection [B, H, N, d_h]
content score            [B, H, N, N]
distance matrix          [N, N]
head slopes              [H, 1, 1]
ALiBi bias               [H, N, N]
causal mask              [N, N]
attention probability    [B, H, N, N]
output                    [B, H, N, d_h]
```

상대 거리 행렬은 query index에서 key index를 뺀다.

```math
D_{i,j}=i-j.
```

Causal 영역에서는 $D_{i,j}\ge 0$이고, 미래 위치 $j>i$는 mask로 $-\infty$를 받는다.

```math
B_{h,i,j}=-m_hD_{i,j}.
```

최종 logit은 다음과 같다.

```math
Z
=
\frac{QK^\top}{\sqrt{d_h}}
+B+M_{causal}.
```

논문에서 명시하듯 ALiBi bias 자체는 $\sqrt{d_h}$로 나누지 않는다. 구현에서 content score를 scale한 뒤 bias를 더하는 순서를 보존해야 한다.

### PyTorch 형태의 구현

```python
def alibi_bias(num_heads, seq_len, slopes, device):
    q_pos = torch.arange(seq_len, device=device)[:, None]
    k_pos = torch.arange(seq_len, device=device)[None, :]
    distance = q_pos - k_pos                    # [N, N]
    bias = -slopes[:, None, None] * distance    # [H, N, N]
    return bias

scores = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim)
scores = scores + alibi[None, :, :, :]
scores = scores.masked_fill(causal_mask, float("-inf"))
attn = scores.softmax(dim=-1)
out = attn @ v
```

실전에서는 causal mask와 ALiBi bias를 미리 합쳐 하나의 additive mask로 캐시할 수 있다. 논문 구현도 이 방식으로 별도의 network operation을 사실상 추가하지 않는다.

### Autoregressive KV cache

decode step의 현재 절대 위치를 $i$라 하고 cache의 key position이 $j$라면 bias는 그대로 $-m_h(i-j)$다.

```text
prefill: [H, N, N] bias or fused causal construction
decode:  [H, 1, cache_len] bias
```

cache를 잘라 sliding window로 사용할 때는 “cache tensor의 local index”가 아니라 실제 query-key 상대 거리를 기준으로 해야 한다. 다만 일정한 offset을 모든 key logit에 동일하게 더하는 경우 softmax에서 상쇄되므로 구현을 단순화할 여지도 있다.

## 전체 알고리즘 흐름

1. Token embedding만 입력에 넣고 별도의 absolute position embedding은 더하지 않는다.
2. 각 layer와 head에서 $Q$, $K$, $V$를 projection한다.
3. $QK^\top/\sqrt{d_h}$로 content score를 계산한다.
4. query-key 거리 행렬에 head별 slope를 곱해 음수 bias를 만든다.
5. ALiBi bias와 causal mask를 score에 더한다.
6. key 축으로 softmax를 적용한다.
7. 확률로 $V$를 가중합한다.
8. 나머지 output projection, residual, FFN은 일반 Transformer와 동일하다.

## 작은 수치 예시

한 head에서 $m=0.5$, query 위치가 $i=3$이고 네 key의 content score가 다음과 같다고 하자.

```text
content score: [1.0, 0.2, 0.8, 0.4]
distance:      [3,   2,   1,   0]
ALiBi bias:    [-1.5,-1.0,-0.5, 0.0]
final logit:   [-0.5,-0.8, 0.3, 0.4]
```

가장 오래된 token은 원래 content score가 가장 컸지만 거리 penalty 때문에 최종 우선순위가 바뀐다. 반대로 첫 token의 content score가 훨씬 크다면 여전히 선택될 수 있다. 이것이 hard locality와 다른 점이다.

## 실험 설정

### WikiText-103

- 약 1.03억 token의 English Wikipedia corpus
- 16 layers, model dimension 1024, 8 heads
- FFN inner dimension 4096
- 약 247M parameters
- 다양한 $L\in\{64,128,256,512,1024,1536,2048,3072\}$에서 학습
- non-overlapping evaluation을 기본으로 하고 일부 비교에는 sliding-window evaluation 사용

### Toronto BookCorpus

WikiText-103에서 고른 동일 slope를 책 domain에 그대로 적용해 hyperparameter transfer를 검증한다. 논문의 주장은 특정 corpus에서 slope를 재탐색하지 않아도 된다는 것이다.

### CC100 + RoBERTa corpus

- 총 461GB corpus
- 25 layers, dimension 2048, 16 heads, FFN 8192
- 1.3B parameters
- 128 V100 GPUs에서 50k updates

이 대규모 설정은 작은 benchmark에서만 잘 되는 inductive bias인지, 실제 대형 LM에서도 비용 이득이 유지되는지를 확인한다.

## 주요 실험 결과

### WikiText-103 length extrapolation

길이 512로 학습한 모델의 validation perplexity는 다음 흐름을 보인다.

| 평가 길이 | Sinusoidal | RoPE | T5 bias | ALiBi |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 20.05 | 20.07 | 19.65 | 19.73 |
| 1,012 | 43.54 | 21.37 | 18.79 | 18.73 |
| 3,512 | 178.97 | 35.54 | 22.91 | 18.40 |
| 10,512 | 341.53 | 60.77 | 52.95 | 18.32 |

ALiBi는 평가 context가 길어질수록 오히려 더 많은 유용한 history를 받아 perplexity가 내려가며, 아주 긴 범위에서도 급격한 붕괴가 없다. 반면 sinusoidal과 RoPE는 학습 범위를 넘은 뒤 빠르게 악화된다.

길이 512로 학습한 ALiBi는 길이 3072에서 validation perplexity 18.40을 기록해, 길이 3072로 직접 학습한 sinusoidal 모델의 18.67보다 좋았다. 학습 속도는 28.3k WPS 대 15.3k WPS로 약 1.84배였다.

### 같은 길이에서도 성능이 좋아지는가

ALiBi의 효과가 오직 extrapolation 때만 나타난 것은 아니다.

| 학습=평가 길이 | Sinusoidal | RoPE | T5 bias | ALiBi |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 20.05 | 20.07 | 19.65 | 19.73 |
| 1,024 | 19.34 | 19.33 | 18.80 | 18.66 |
| 3,072 | 18.67 | 18.57 | 18.01 | 17.60 |

작은/중간 규모 corpus에서는 recency prior 자체가 언어 모델링에 유용했다. 다만 1.3B 대규모 corpus에서는 같은 길이의 sinusoidal과 비슷한 수준이어서, 이 추가 이득이 모든 scale에서 보장되지는 않는다.

### 속도와 메모리

WikiText-103의 batched training에서 길이 1024 기준 결과는 다음과 같다.

| 방법 | 학습 속도 | 평가 속도 | 학습 메모리 |
| --- | ---: | ---: | ---: |
| Sinusoidal | 26.0k WPS | 77.8k WPS | 19.2GB |
| RoPE | 17.7k WPS | 39.4k WPS | 22.8GB |
| T5 bias | 13.0k WPS | 20.2k WPS | 20.9GB |
| ALiBi | 25.8k WPS | 76.4k WPS | 19.3GB |

이 표는 당시 코드와 하드웨어의 구현 결과이므로 현대 fused kernel의 절대 비교로 일반화하면 안 된다. 논문이 입증한 더 안정적인 결론은 ALiBi 자체가 작은 additive bias이며, **같은 길이에서는 sinusoidal에 가까운 비용**이라는 점이다.

### 1.3B 모델

| 모델 | 학습 길이 | 평가 길이 | 메모리 | validation PPL |
| --- | ---: | ---: | ---: | ---: |
| Sinusoidal | 2048 | 2048 | 29.3GB | 9.01 |
| ALiBi | 1024 | 2048 | 26.2GB | 8.92 |

ALiBi 모델은 더 짧게 학습했기 때문에 3.1GB 적은 메모리를 사용했고, 같은 perplexity 지점에 평균 약 11% 빨리 도달했다.

### 성능의 최적 길이는 무한히 늘어나지 않는다

CC100+RoBERTa 실험에서 길이 512 모델은 약 1012, 길이 1024 모델은 약 2024에서 가장 좋은 perplexity를 보였다. 이후에도 강하게 유지되지만 계속 개선되지는 않았다.

따라서 ALiBi를 “임의 길이에 완벽히 extrapolate한다”로 읽으면 과장이다. 논문이 보여준 것은 학습 길이의 약 두 배 부근에서 가장 좋은 활용을 보이고, 10k 수준까지 catastrophic failure 없이 유지된다는 것이다.

## 계산 복잡도와 메모리

ALiBi는 attention의 점근 복잡도를 바꾸지 않는다.

```math
\text{time}=O(N^2d),
\qquad
\text{score memory}=O(HN^2).
```

위치 embedding parameter는 없고 bias 생성 비용은 작다. 하지만 bias를 `[H, N, N]` tensor로 materialize하면 추가 메모리가 생긴다. 현대 구현에서는 다음 방식이 바람직하다.

- causal mask 생성 시 head slope를 함께 fuse한다.
- FlashAttention류 kernel의 bias/slope interface를 사용한다.
- decode에서는 `[H, 1, cache_len]`만 계산한다.
- 정적 최대 길이 bias 전체를 무조건 저장하지 않는다.

ALiBi가 학습 메모리를 줄인다는 표현은 bias 자체가 attention을 선형화해서가 아니다. **더 짧은 $L$로 학습해도 긴 평가 길이에 쓸 수 있기 때문에** 전체 학습 메모리가 줄어드는 것이다.

## 기존 위치 표현과 비교

| 방법 | 위치 정보가 들어가는 곳 | 학습 parameter | 길이 밖 동작 | 주요 성격 |
| --- | --- | ---: | --- | --- |
| Learned absolute | 입력 embedding | 있음 | 새 index를 정의하기 어려움 | 절대 위치 lookup |
| Sinusoidal | 입력 embedding | 없음 | 계산은 가능하나 실험상 취약 | 고정 absolute signal |
| RoPE | Q/K | 없음 | phase OOD 가능 | dot product에 상대 회전 |
| T5 bias | attention logit | 있음 | bucket 밖은 clip | 학습된 상대 거리 bias |
| ALiBi | attention logit | 없음 | 선형식이 그대로 연장 | 고정 recency prior |

ALiBi의 장점은 위치를 정밀하게 “인코딩”한다기보다 attention이 거리와 함께 **어떻게 감쇠해야 하는지**를 직접 규정한다는 데 있다.

## 장점과 핵심 기여

### 1. 아이디어와 구현이 매우 단순하다

입력 embedding과 Q/K 회전 규칙을 건드리지 않고 additive mask만 바꾸면 된다. 기존 Transformer의 대부분을 그대로 유지한다.

### 2. Length extrapolation을 비용 문제와 연결했다

긴 입력 성능을 위치 표현의 정확도만으로 보지 않고, 짧게 학습해 시간과 메모리를 줄이는 실용적 목표로 정식화했다.

### 3. Head별 slope가 여러 거리 척도를 만든다

학습 parameter 없이 local-to-global head 분화를 유도한다.

### 4. 강한 실험 통제

동일 architecture와 corpus에서 sinusoidal, RoPE, T5 bias를 비교했고, 247M부터 1.3B까지 scale을 넓혔다.

### 5. 후속 장문 연구의 중요한 기준점

이후 xPos, PI, YaRN, LongRoPE 등은 ALiBi를 length extrapolation baseline 또는 대안적 positional design으로 자주 비교한다.

## 한계와 비판적 관점

### 1. 단조 선형 prior가 모든 관계에 맞지는 않는다

언어에는 먼 위치의 제목, 정의, 주어처럼 거리가 멀어도 중요한 token이 있다. Content score가 bias를 이길 수 있지만, 긴 거리에 체계적 불이익을 주는 것은 명확한 가정이다.

### 2. Bidirectional attention으로의 확장은 자명하지 않다

Causal LM에서는 과거 방향 $i-j\ge0$만 다룬다. Encoder에서는 $|i-j|$를 쓸지, 방향별 slope를 둘지 결정해야 하며 원 논문의 핵심 실험 범위를 벗어난다.

### 3. Pretrained RoPE 모델의 plug-in context 확장법은 아니다

ALiBi는 처음부터 해당 bias로 학습할 때의 방법이다. 이미 RoPE로 학습된 LLaMA류 모델에 bias만 추가해 원래 능력을 보존한다는 보장은 없다. PI/YaRN/LongRoPE와 목표 조건이 다르다.

### 4. Dense attention 비용은 그대로다

10k token에서 perplexity가 좋아도 실제 serving의 $O(N^2)$ prefill과 $O(N)$ decode/KV cache 비용은 별도 문제다.

### 5. “Extrapolation” 측정은 평가 프로토콜에 민감하다

Non-overlapping perplexity는 각 segment 초반 token이 적은 context만 받는 `early token curse`의 영향을 크게 받는다. 긴 segment는 평균적으로 더 많은 context를 제공하므로 성능이 좋아질 수 있다. 논문도 sliding-window 분석을 추가해 이 효과를 논의한다.

### 6. Head slope가 정말 서로 다른 기능을 보장하지는 않는다

Slope는 attention 분포에 prior를 줄 뿐, 각 head가 의미적으로 local syntax와 long-range topic을 분담한다고 증명하지는 않는다.

## 자주 헷갈리는 지점

### ALiBi는 relative position embedding인가

넓은 의미에서는 그렇다. 하지만 relative position **vector**를 lookup하지 않고 상대 거리의 scalar 함수를 logit에 넣는다.

### Bias가 음수면 먼 token을 전혀 못 보는가

아니다. Softmax 전 감점일 뿐이다. Content score가 크면 먼 token도 높은 attention을 받을 수 있다.

### Slope는 학습하는가

원 논문에서는 고정한다. 학습 가능한 slope 실험은 extrapolation이 약했고 속도도 느렸다.

### $\sqrt{d_h}$ scale을 bias에도 적용하는가

원 논문은 적용하지 않는다. 먼저 content dot product를 scale하고 그 뒤 ALiBi bias를 더한다.

### ALiBi가 attention을 선형 복잡도로 만드는가

아니다. 이름의 `Linear`는 거리에 대해 bias가 선형이라는 뜻이다. Dense attention의 $N^2$ 계산은 그대로다.

### 더 긴 길이에서 perplexity가 낮아지면 위치 extrapolation이 완벽한가

아니다. 긴 segment가 더 많은 history를 제공하는 이득과 위치 분포의 안정성이 함께 반영된다. retrieval, reasoning, generation 안정성은 별도로 평가해야 한다.

## 온디바이스 및 비전 관점

### 온디바이스 언어 모델

ALiBi는 parameter와 lookup table이 없고 연산이 단순해 kernel fusion에 유리하다. 하지만 long-context on-device serving에서는 위치 bias보다 KV cache가 더 큰 병목이다. GQA/MQA, KV quantization, sliding window와 함께 설계해야 한다.

### Vision Transformer

1D causal distance $i-j$를 image patch의 raster index에 그대로 적용하면 행 경계에서 잘못된 이웃 관계가 생긴다. Vision에서는 보통 2D 거리로 바꾸는 편이 자연스럽다.

```math
B_{(r_i,c_i),(r_j,c_j)}^{(h)}
=
-m_{h,r}|r_i-r_j|-m_{h,c}|c_i-c_j|.
```

다만 image attention은 bidirectional이므로 causal recency라는 원래 inductive bias와 의미가 달라진다. ALiBi의 단순식을 그대로 가져오기보다 task의 공간 구조에 맞춰야 한다.

## 후속 논문과의 연결점

- **xPos/LEX**는 extrapolation을 attention resolution과 장거리 진동의 관점에서 분석하고 RoPE에 reciprocal decay를 추가한다.
- **Position Interpolation**은 기존 RoPE LLM을 재학습하지 않고 position index를 압축해 context를 늘린다.
- **YaRN**은 RoPE 주파수별로 interpolation 강도를 달리하고 attention temperature를 보정한다.
- **LongRoPE**는 차원별 scaling factor를 탐색하고 단계적 확장으로 million-token context를 목표로 한다.

ALiBi는 이들과 달리 “기존 RoPE checkpoint의 확장”보다 **처음부터 extrapolation-friendly한 position mechanism으로 학습**하는 쪽에 가깝다.

## 개인 학습/연구 메모

ALiBi를 기억할 때는 다음 식 하나면 충분하다.

```math
\boxed{
\text{content similarity}
-
\text{head-specific slope}\times\text{distance}
}
```

논문의 진짜 기여는 bias 한 줄보다도, 위치 표현이 학습 길이와 추론 길이의 결합을 끊어 **train short, test long**이라는 비용 절감 전략을 가능하게 할 수 있음을 명확히 보여준 데 있다.

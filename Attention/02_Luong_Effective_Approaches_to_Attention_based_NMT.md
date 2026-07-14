# 02. Effective Approaches to Attention-based Neural Machine Translation

## 논문 정보

- 원본 파일: `02_Luong_Effective_Approaches_to_Attention_based_NMT.pdf`
- 제목: Effective Approaches to Attention-based Neural Machine Translation
- 저자: Minh-Thang Luong, Hieu Pham, Christopher D. Manning
- 발표: EMNLP 2015
- 주제: NMT attention을 global/local 범위와 여러 score 함수로 체계화하고 input-feeding을 제안
- 핵심 키워드: Luong attention, global attention, local attention, dot/general/concat score, predictive alignment, input-feeding

## 한눈에 보는 요약

Bahdanau attention이 soft alignment를 NMT에 성공적으로 도입한 뒤, 이 논문은 attention 설계 공간을 보다 단순하고 비교 가능한 형태로 정리한다.

논문이 묻는 질문은 세 가지다.

1. 매 target step에서 source 전체를 볼 것인가, 일부 window만 볼 것인가?
2. Decoder state와 source state의 관련도를 어떤 score 함수로 계산할 것인가?
3. 이전 step의 alignment 정보를 다음 step에 어떻게 전달할 것인가?

첫 번째 질문에 대해 두 계열을 제안한다.

```text
global attention:
all source positions are scored

local attention:
only a window around an aligned position is scored
```

두 번째 질문에서는 dot, general, concat score를 비교한다.

```math
\begin{aligned}
\text{dot:}\quad & h_t^\top h_s,\\
\text{general:}\quad & h_t^\top W_a h_s,\\
\text{concat:}\quad & v_a^\top\tanh\!\left(W_a[h_t;h_s]\right).
\end{aligned}
```

세 번째 질문에는 input-feeding으로 답한다. 현재 attention 결과 $\tilde h_t$를 다음 decoder step의 입력에 다시 넣어 과거 alignment 선택을 recurrent state가 기억하게 한다.

실험에서 local-p attention, general score, input-feeding의 조합이 강한 성능을 보인다. English-German에서 non-attentional baseline 대비 single model 기준 약 `+5 BLEU`를 얻고, 8-model ensemble과 unknown replacement로 당시 강한 결과를 기록한다.

이 논문의 중요성은 특정 score 하나가 아니라 attention을 `범위`, `score`, `history 연결`이라는 설계 축으로 분해해 후속 연구가 비교할 수 있는 vocabulary를 만든 데 있다.

## Bahdanau attention 이후의 문제

Bahdanau 모델은 다음 구조를 사용한다.

```text
previous decoder state s_(i-1)
-> additive alignment with encoder annotations
-> context c_i
-> current decoder state s_i
```

Luong 모델은 계산 순서를 단순화한다.

```text
current decoder hidden state h_t
-> attention over source states
-> context c_t
-> attentional hidden state h_t_tilde
-> output y_t
```

즉 decoder LSTM이 현재 hidden state를 먼저 만든 뒤 그 state를 query로 사용한다.

논문은 Bahdanau의 한 가지 attention 형태를 그대로 반복하기보다 다음 선택지를 실험한다.

- Source 전체 대 source 일부 window
- Content-based score 대 location-only score
- Parameter 없는 dot score 대 learned bilinear/MLP score
- 독립적인 step별 attention 대 input-feeding으로 연결된 attention

## 공통 NMT 구조

### Encoder와 decoder

Source sentence $x_1,\ldots,x_S$를 encoder LSTM이 읽어 source hidden states를 만든다.

```math
\bar h_1,\bar h_2,\ldots,\bar h_S
```

Decoder는 target history를 읽어 현재 hidden state $h_t$를 만든다.

```math
h_t=\operatorname{LSTM}(h_{t-1},\operatorname{input}_t)
```

Attention은 $h_t$를 source states와 비교해 context $c_t$를 만든다.

### Attentional hidden state

Context와 current target hidden state를 concatenate하고 projection과 $\tanh$를 적용한다.

```math
\tilde h_t=\tanh\!\left(W_c[c_t;h_t]\right)
```

Output distribution은 attentional hidden state에서 계산한다.

```math
p(y_t\mid y_{<t},x)=\operatorname{softmax}(W_s\tilde h_t)
```

$\tilde h_t$는 decoder language-model state $h_t$와 source retrieval result $c_t$를 결합한 표현이다.

## Global attention

### 모든 source position을 본다

Global attention은 target step $t$마다 모든 encoder state를 scoring한다.

```math
a_t(s)
=\frac{\exp(\operatorname{score}(h_t,\bar h_s))}
{\sum_{s'}\exp(\operatorname{score}(h_t,\bar h_{s'}))}
```

Context는 source states의 weighted sum이다.

```math
c_t=\sum_s a_t(s)\bar h_s
```

Alignment vector 길이는 source length $S$와 같다.

```text
a_t: [S]
```

Bahdanau soft attention과 같은 전역 검색이지만 query와 score 함수, decoder 결합 순서가 다르다.

### Content-based와 location-based

Content-based attention은 $h_t$와 각 $\bar h_s$를 직접 비교한다.

```math
a_t(s)=\operatorname{softmax}_s\!\left(\operatorname{score}(h_t,\bar h_s)\right)
```

논문은 초기 실험으로 source content를 보지 않는 location-based 형태도 둔다.

```math
a_t=\operatorname{softmax}(W_a h_t)
```

이 식은 target hidden state만으로 source position distribution을 예측한다. Source length가 고정된다는 제약이 있고 source content에 따라 alignment를 바꿀 수 없으므로 content-based 방식보다 일반성이 낮다.

## Alignment score 함수

### Dot

```math
\operatorname{score}(h_t,\bar h_s)=h_t^\top\bar h_s
```

장점은 parameter가 없고 계산이 빠르다는 점이다. 두 vector dimension이 같아야 한다.

```math
h_t,\bar h_s\in\mathbb{R}^d
```

현대 scaled dot-product attention의 직접적인 전신이다. 이 논문에서는 아직 $\sqrt{d}$ scaling을 사용하지 않는다.

### General

```math
\operatorname{score}(h_t,\bar h_s)=h_t^\top W_a\bar h_s
```

Source state를 learned matrix로 변환한 뒤 query와 dot product한다.

```math
\begin{aligned}
\operatorname{key}_s &= W_a\bar h_s,\\
\operatorname{score} &= h_t^\top\operatorname{key}_s.
\end{aligned}
```

Dot score보다 표현력이 높지만 $W_a$ parameter와 projection 비용이 추가된다.

### Concat

```math
\operatorname{score}(h_t,\bar h_s)
=v_a^\top\tanh\!\left(W_a[h_t;\bar h_s]\right)
```

Query와 source state를 concatenate해 MLP로 scalar를 만든다. Bahdanau additive score와 같은 MLP 계열이다.

논문 구현에서는 concat이 예상만큼 좋은 perplexity를 내지 못했다. 저자들은 $W_a$의 source 부분을 identity로 단순화한 구현이 원인일 수 있다고 적는다. 따라서 이 결과를 concat score 일반의 열등함으로 해석하면 안 된다.

### 세 score의 관계

| Score | Parameter | 차원 조건 | 특징 |
| --- | ---: | --- | --- |
| dot | 없음 | query/key dimension 동일 | 가장 단순하고 빠름 |
| general | $W_a$ | projection으로 조정 가능 | learned bilinear matching |
| concat | $W_a$, $v_a$ | 자유로움 | nonlinear MLP matching |

Dot/general은 matrix multiplication으로 효율적으로 계산하기 쉽다. Concat/additive는 query-key pair마다 hidden activation을 만들어야 한다.

## Local attention

### 왜 local window를 쓰는가

Global attention은 target step마다 source 전체를 본다.

```math
\begin{aligned}
\text{cost per step:}\quad &O(S),\\
\text{total alignment pairs:}\quad &O(TS).
\end{aligned}
```

긴 source에서 모든 position을 볼 필요가 없고 번역 alignment가 대체로 국소적이라는 가정을 사용하면 일부 window만 scoring할 수 있다.

![Luong의 global attention과 local attention](https://github.com/user-attachments/assets/49f9eca7-d89c-4c20-afc8-ae8a585592aa)

Local attention은 먼저 현재 target word가 대응할 source 중심 위치 $p_t$를 정하고, 그 주변 $[p_t-D,p_t+D]$만 본다.

```math
\begin{aligned}
\operatorname{window} &= [p_t-D,p_t+D],\\
\operatorname{window\ size} &= 2D+1.
\end{aligned}
```

### Local-m: monotonic alignment

가장 단순한 형태는 source와 target이 대략 같은 속도로 진행한다고 가정한다.

```math
p_t=t
```

현재 target index를 window center로 사용한다. Parameter가 없지만 source-target length와 어순 차이에 약하다.

### Local-p: predictive alignment

Decoder state에서 연속적인 source position을 예측한다.

```math
p_t=S\cdot\operatorname{sigmoid}\!\left(v_p^\top\tanh(W_ph_t)\right)
```

$S$는 source length다. Sigmoid 출력이 $[0,1]$이므로 $p_t$는 $[0,S]$ 범위에 놓인다.

$p_t$는 실수이며 source position $s$는 정수다. Window 안의 content alignment에 Gaussian prior를 곱한다.

```math
\begin{aligned}
a_t(s)
&=\operatorname{align}(h_t,\bar h_s)
\exp\!\left(-\frac{(s-p_t)^2}{2\sigma^2}\right),\\
\sigma&=\frac{D}{2}.
\end{aligned}
```

Window 중심에 가까운 position을 선호하면서 content score로 세부 alignment를 결정한다.

### Soft attention과 hard attention의 절충

Local attention은 다음 두 극단의 중간이다.

```text
global soft attention:
all positions, differentiable

hard attention:
one sampled position, non-differentiable

local attention:
differentiable center/window, soft weights inside window
```

Local-p는 $p_t$를 continuous function으로 예측하고 Gaussian을 사용하므로 거의 모든 곳에서 미분 가능하다.

## Input-feeding

### 독립적인 alignment decision의 문제

각 target step의 attention이 현재 $h_t$만 사용하면 이전에 어느 source word를 번역했는지 직접 알기 어렵다.

기계번역에서는 coverage가 중요하다.

- Source word를 빠뜨리지 않아야 한다.
- 같은 source phrase를 반복 번역하지 않아야 한다.
- 이전 alignment와 다음 alignment가 일관되어야 한다.

### 이전 attentional vector를 다음 입력에 넣는다

Input-feeding은 이전 step의 $\tilde h_{t-1}$를 다음 decoder input과 concatenate한다.

```math
\operatorname{decoder\_input}_t
=\operatorname{concat}\!\left(\operatorname{embedding}(y_{t-1}),\tilde h_{t-1}\right)
```

그 결과 계산 경로는 다음과 같다.

```text
h_t
-> a_t
-> c_t
-> h_t_tilde
-> input of step t+1
```

Attention history가 decoder recurrence에 들어가므로 모델이 이전 alignment 선택을 암묵적으로 추적할 수 있다.

논문은 이를 horizontal/vertical 방향으로 깊어진 network라고 설명한다.

### Bahdanau 방식과의 관계

Bahdanau decoder도 context $c_i$를 state update에 넣으므로 이전 attention 정보가 다음 state로 전달된다. Luong input-feeding은 attention 결과를 명시적으로 다음 입력에 concatenate해 stacked recurrent architecture에도 쉽게 적용할 수 있게 한다.

## Tensor shape와 vectorized 계산

다음 shape를 사용하자.

```text
H_enc: [B, S, d]
h_t:   [B, d]
```

Dot global attention은 다음과 같다.

```math
\begin{aligned}
\operatorname{scores}
&=H_{\mathrm{enc}}\,h_t[:,:, \mathrm{None}],
&\operatorname{shape}&:[B,S,1]\to[B,S],\\
a_t
&=\operatorname{softmax}(\operatorname{scores},
\operatorname{dim}=\operatorname{source\_position}),
&\operatorname{shape}&=[B,S],\\
c_t
&=a_t[:,\mathrm{None},:]\,H_{\mathrm{enc}},
&\operatorname{shape}&=[B,d].
\end{aligned}
```

General score는 encoder states 또는 query를 projection한다.

```math
\operatorname{scores}
=(H_{\mathrm{enc}}W_a^\top)\,h_t[:,:, \mathrm{None}]
```

Local attention에서는 window gather가 추가된다.

```text
H_window: [B, 2D+1, d]
scores:   [B, 2D+1]
```

Batch마다 $p_t$가 다를 수 있으므로 window index clipping과 gather 구현에 주의해야 한다.

## 전체 알고리즘 흐름

### Global attention

```text
1. decoder LSTM computes h_t
2. score h_t against all source states
3. softmax -> a_t
4. weighted sum -> c_t
5. h_t_tilde = tanh(W_c [c_t; h_t])
6. predict y_t from h_t_tilde
7. optionally feed h_t_tilde into next step
```

### Local-p attention

```text
1. decoder computes h_t
2. predict continuous source center p_t
3. select source window [p_t-D, p_t+D]
4. compute content scores inside window
5. multiply Gaussian prior centered at p_t
6. normalize and compute c_t
7. build h_t_tilde and predict y_t
8. input-feed h_t_tilde to next step
```

## 계산 복잡도

Target length를 $T$, source length를 $S$, window radius를 $D$라고 하자.

| 방식 | score pair 수 | 특징 |
| --- | ---: | --- |
| Global | $O(TS)$ | source 전체 검색 |
| Local | $O(T(2D+1))$ | window 크기가 고정이면 source length에 선형 비의존 |

Local attention은 alignment 계산을 줄이지만 다음 비용은 남는다.

- Encoder/decoder LSTM의 sequential dependency
- Window gather와 batch별 index 처리
- $p_t$ 예측 오류로 필요한 source position이 window 밖에 놓일 위험

## 실험 설정

### 데이터와 model

- WMT 2014 English-German 4.5M sentence pair
- English 116M word, German 110M word
- 각 언어 vocabulary 50k, 나머지는 `<unk>`
- Length 50 초과 pair 제거
- 4-layer stacked LSTM
- Layer당 1,000 cell
- Embedding dimension 1,000
- Minibatch 128
- Plain SGD, 초기 learning rate 1
- Gradient norm 5로 clipping
- Dropout 0.2
- Local window radius $D=10$
- Tesla K40 한 장에서 약 1k target word/s
- 모델당 7-10일 학습

평가는 tokenized BLEU와 WMT NIST BLEU를 구분해 보고한다.

## English-German 결과

### Progressive ablation

| 시스템 | Perplexity | BLEU |
| --- | ---: | ---: |
| Base | 10.6 | 11.3 |
| + source reverse | 9.9 | 12.6 |
| + dropout | 8.1 | 14.0 |
| + global location attention | 7.3 | 16.8 |
| + input-feeding | 6.4 | 18.1 |
| + local-p general attention | 5.9 | 19.0 |
| + unknown replacement | - | 20.9 |
| 8-model ensemble + unknown replacement | - | 23.0 |

Attention 전 non-attentional baseline 중 reverse+dropout이 14.0 BLEU다. Local-p single model은 19.0으로 `+5.0 BLEU` 높다.

```text
global attention gain: +2.8 BLEU
input-feeding gain:    +1.3 BLEU
local-p gain:          +0.9 BLEU
```

Unknown replacement는 attention alignment를 사용해 `<unk>`를 source word 또는 dictionary translation으로 바꾸며 추가 `+1.9 BLEU`를 준다. 이는 attention이 rare word 위치를 찾는 데 실제로 유용했음을 보여 준다.

### 당시 시스템과 비교

- WMT14 English-German tokenized BLEU: ensemble 23.0
- 기존 Jean et al. ensemble: 21.6
- WMT15 newstest2015 NIST BLEU: 25.9
- 당시 top NMT + 5-gram reranker: 24.9

Ensemble 결과에는 8개 model과 unknown replacement가 포함되므로 attention architecture 단독 효과와 ensemble 효과를 분리해서 읽어야 한다.

## German-English 결과

| 시스템 | Perplexity | BLEU |
| --- | ---: | ---: |
| Base with source reverse | 14.3 | 16.9 |
| + global location | 12.7 | 19.1 |
| + input-feeding | 10.9 | 20.1 |
| + global dot + dropout | 9.7 | 22.8 |
| + unknown replacement | - | 24.9 |

Attention `+2.2`, input-feeding `+1.0`, dot score와 dropout `+2.7`, unknown replacement `+2.1 BLEU`의 progressive gain을 보고한다.

다만 당시 phrase-based SOTA 29.2에는 미치지 못한다. English-German 한 방향의 강한 결과를 모든 translation direction의 우위로 일반화하면 안 된다.

## Attention architecture 분석

| 모델 | Perplexity | BLEU before UNK replace | BLEU after |
| --- | ---: | ---: | ---: |
| global location | 6.4 | 18.1 | 19.3 |
| global dot | 6.1 | 18.6 | 20.5 |
| global general | 6.1 | 17.3 | 19.1 |
| local-m general | 6.2 | 18.6 | 20.4 |
| local-p dot | 6.6 | 18.0 | 19.6 |
| local-p general | 5.9 | 19.0 | 20.9 |

이 실험에서는 다음 경향이 나타난다.

- Global에서는 dot score가 좋다.
- Local에서는 general score가 좋다.
- Predictive local-p + general 조합이 가장 좋다.
- Location-only score는 unknown replacement gain도 작아 alignment 품질이 낮다.
- Local-m dot은 안정적으로 학습되지 않았다.

모든 가능한 조합을 동일 자원으로 평가한 것은 아니므로 절대적인 ranking보다는 설계 trade-off의 초기 증거로 읽어야 한다.

## 긴 sentence와 alignment 품질

Attention model은 non-attentional model보다 모든 length bucket에서 높은 BLEU를 보이고 긴 sentence에서도 성능 저하가 작다.

논문은 attention alignment를 gold alignment와 비교해 AER도 측정한다. 낮을수록 좋다.

| 방식 | AER |
| --- | ---: |
| global location | 0.39 |
| local-m general | 0.34 |
| local-p general | 0.36 |
| attention ensemble | 0.34 |
| Berkeley Aligner | 0.32 |

Local attention은 global location보다 좋은 alignment를 보이지만 Berkeley Aligner에는 약간 못 미친다. 또한 ensemble의 translation BLEU가 높다고 AER가 반드시 더 낮지는 않다.

```text
better translation score
!=
better word alignment score
```

이는 attention weight를 전통적 alignment와 완전히 동일시하면 안 된다는 중요한 관찰이다.

## 장점과 핵심 기여

### 1. Attention 설계 공간을 체계화했다

범위(global/local), score(dot/general/concat), history(input-feeding)를 독립 축으로 분해했다.

### 2. Local differentiable attention을 제안했다

Hard sampling 없이도 검색 window를 줄이는 방법을 보여 줬다.

### 3. Dot-product attention의 실용성을 입증했다

간단한 dot score가 global attention에서 강한 결과를 내며 이후 Transformer의 핵심 score로 이어진다.

### 4. Input-feeding으로 alignment history를 연결했다

이전 attention decision을 다음 decoder step에 전달해 coverage를 암묵적으로 학습한다.

### 5. Architecture별 정량 분석을 제공했다

BLEU뿐 아니라 perplexity, sentence length, AER, unknown replacement 효과를 비교했다.

## 한계와 비판적 관점

### 1. Local window의 inductive bias가 강하다

번역 alignment가 대체로 monotonic/local하다는 가정에 의존한다. 장거리 reorder가 window 밖에 있으면 중요한 source word를 볼 수 없다.

### 2. $p_t$ 하나가 multi-modal alignment를 표현하지 못한다

한 target word가 멀리 떨어진 여러 source region을 동시에 봐야 해도 Gaussian center는 하나다.

### 3. RNN sequential bottleneck은 남는다

Attention score 범위를 줄여도 decoder step은 병렬화되지 않는다.

### 4. Score 비교가 완전한 factorial experiment는 아니다

자원 제약으로 global/local과 모든 score 조합을 동일하게 평가하지 못했다. Concat 구현도 원형을 단순화했다.

### 5. Input-feeding은 명시적 coverage guarantee가 아니다

과거 alignment를 state가 기억할 기회를 줄 뿐 source word를 정확히 한 번씩 번역하도록 강제하지 않는다.

### 6. Ensemble과 post-processing 효과가 크다

최고 BLEU에는 8-model ensemble과 unknown replacement가 포함된다. Single-model architecture improvement와 배치해 해석해야 한다.

## Bahdanau, Luong, Transformer 비교

| 항목 | Bahdanau | Luong | Transformer |
| --- | --- | --- | --- |
| Query | 이전 decoder state | 현재 decoder hidden state | projected token representation Q |
| Memory | BiRNN annotation | encoder top-layer states | projected K/V |
| 대표 score | additive MLP | dot/general/concat | scaled dot product |
| 범위 | global | global 또는 local | 보통 global, mask/sparse 변형 가능 |
| 계산 | target step별 sequential | target step별 sequential | sequence query를 matrix로 병렬 계산 |
| History | context가 state update에 포함 | input-feeding | layer/residual representation에 포함 |

Transformer의 scaled dot-product는 Luong dot score에 dimension scaling과 multi-head projection을 결합한 것으로 볼 수 있다.

```math
\begin{aligned}
\text{Luong dot:}\quad &h_t^\top h_s,\\
\text{Transformer:}\quad &\frac{q_i^\top k_j}{\sqrt{d_k}}.
\end{aligned}
```

## 구현 체크리스트

```text
1. Global softmax 축이 source position인가?
2. Source padding score를 -inf로 mask하는가?
3. Dot score에서 query/key dimension이 같은가?
4. General matrix 방향과 transpose가 맞는가?
5. Local window가 sentence boundary를 넘을 때 clipping하는가?
6. p_t가 [0,S] 범위의 연속값으로 계산되는가?
7. Gaussian sigma가 논문 설정 D/2와 맞는가?
8. Window 밖 position weight가 정확히 0인가?
9. Input-feeding에서 이전 h_t_tilde가 다음 입력에 연결되는가?
10. Beam search에서 beam별 h_t_tilde history를 따로 유지하는가?
11. Alignment visualization의 target/source 축을 확인했는가?
12. BLEU가 tokenized인지 NIST 방식인지 구분했는가?
```

## 개인 학습/연구 메모

이 논문을 기억할 때는 다음 세 축으로 정리하면 된다.

```text
WHERE to look:
global vs local

HOW to score:
dot vs general vs concat

HOW to remember past attention:
input-feeding
```

Luong attention은 attention을 하나의 고정 공식이 아니라 조합 가능한 design space로 만든 논문이다. 이후 연구의 global/local, dot/additive, coverage라는 용어가 여기서 선명해졌다.

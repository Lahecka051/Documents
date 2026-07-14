# 01. Neural Machine Translation by Jointly Learning to Align and Translate

## 논문 정보

- 원본 파일: `01_Bahdanau_NMT_Jointly_Learning_to_Align_and_Translate.pdf`
- 제목: Neural Machine Translation by Jointly Learning to Align and Translate
- 저자: Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
- 발표: ICLR 2015
- 주제: encoder-decoder NMT에 학습 가능한 soft alignment를 결합해 target word마다 다른 source context를 조회하는 방법
- 핵심 키워드: additive attention, soft alignment, bidirectional RNN, context vector, joint learning, neural machine translation

## 한눈에 보는 요약

기존 RNN encoder-decoder는 source sentence 전체를 하나의 fixed-length vector에 압축한 뒤 decoder에 전달했다. 문장이 길어질수록 이 vector가 모든 정보를 보존해야 하는 병목이 심해진다.

이 논문은 source를 하나의 vector로 압축하지 않고 위치별 annotation sequence로 유지한다.

```text
source tokens x_1, ..., x_Tx
-> bidirectional encoder
-> annotations h_1, ..., h_Tx
```

Decoder가 target word $y_i$를 생성할 때마다 이전 decoder state $s_{i-1}$와 모든 source annotation $h_j$를 비교한다. 비교 점수를 softmax해 alignment weight $\alpha_{i,j}$를 만들고, annotation의 weighted sum으로 현재 target step 전용 context $c_i$를 계산한다.

```math
\begin{aligned}
e_{i,j} &= \operatorname{alignment\_score}(s_{i-1}, h_j),\\
\alpha_{i,:} &= \operatorname{softmax}(e_{i,:}),\\
c_i &= \sum_j \alpha_{i,j}h_j.
\end{aligned}
```

이후 decoder는 $c_i$를 사용해 state를 갱신하고 다음 target word를 예측한다. 즉 decoder는 번역하는 동안 source sentence를 매 step 다시 검색한다.

이 논문의 가장 큰 기여는 attention을 별도 정렬 알고리즘이나 latent discrete variable로 두지 않고, 번역 loss만으로 encoder, decoder, alignment model을 end-to-end 공동 학습했다는 점이다. 현대 Transformer attention과 계산 형태는 다르지만, query에 따라 memory를 검색하고 weighted sum하는 핵심 원리는 여기서 명확하게 정립되었다.

## 연구 배경과 문제의식

### 기존 encoder-decoder의 fixed-length bottleneck

초기 neural machine translation은 encoder RNN이 source sentence를 순서대로 읽고 마지막 hidden state 또는 하나의 context vector $c$로 요약했다.

```math
\begin{aligned}
h_t &= f(x_t,h_{t-1}),\\
c &= q(h_1,\ldots,h_{T_x}).
\end{aligned}
```

Decoder는 같은 $c$를 모든 target step에서 사용했다.

```math
p(y_i \mid y_{<i},x)=g(y_{i-1},s_i,c)
```

이 구조에서는 source length가 5든 50이든 정보 통로의 크기는 항상 같다.

```text
variable-length source
-> one fixed-dimensional vector
-> variable-length target
```

짧은 sentence에서는 충분할 수 있지만 긴 sentence에서는 다음 문제가 생긴다.

- Source의 모든 lexical, syntactic, semantic 정보를 하나의 vector에 보존해야 한다.
- Decoder가 특정 source word를 다시 직접 조회할 수 없다.
- 먼 source 정보는 여러 recurrent step을 거치며 압축된다.
- Target step이 달라도 decoder가 보는 source summary는 동일하다.

논문의 문제의식은 명확하다.

```text
왜 모든 target word가 동일한 source summary만 봐야 하는가?
```

### 전통적 word alignment와의 차이

통계적 기계번역에서는 source-target word alignment가 중요한 구조였다. 그러나 alignment를 별도 모델로 학습하거나 discrete latent variable로 처리하면 neural model 전체와 공동 최적화하기 어렵다.

이 논문은 hard alignment 하나를 고르지 않는다. 모든 source position에 확률적인 weight를 두고 weighted sum을 사용한다.

```text
hard alignment:
target i -> one source position j

soft alignment:
target i -> distribution over all source positions
```

Softmax weight는 미분 가능하므로 translation negative log-likelihood의 gradient가 alignment network까지 전달된다.

### 논문이 해결하려는 핵심 문제

기존 모델:

```text
source sentence -> c -> every target step
```

제안 모델:

```text
source sentence -> h_1, ..., h_Tx
target step i -> alignment alpha_i -> context c_i
```

Context를 source sentence당 하나가 아니라 target step마다 하나씩 만드는 것이 핵심 변화다.

## 모델 구조

### 전체 흐름

모델은 세 부분으로 나눌 수 있다.

1. Bidirectional RNN encoder가 source position별 annotation을 만든다.
2. Alignment model이 이전 decoder state와 각 annotation의 관련도를 계산한다.
3. Decoder가 context vector와 이전 target word를 사용해 다음 word를 예측한다.

```text
source x_1...x_Tx
-> BiRNN annotations h_1...h_Tx

for target step i:
    s_(i-1) and each h_j
    -> energy e_(i,j)
    -> softmax alpha_(i,j)
    -> context c_i
    -> decoder state s_i
    -> p(y_i)
```

<p align="center"><img src="https://github.com/user-attachments/assets/fb1cbf8a-72e6-425b-8d7c-040047e5518f" alt="Bahdanau additive attention 계산 흐름" width="860"></p>
<p align="center"><sub>보조 도식 — decoder state에서 alignment score, source weight, context vector를 계산하는 흐름</sub></p>

### Bidirectional encoder

Source position $j$의 표현이 왼쪽 문맥뿐 아니라 오른쪽 문맥도 포함하도록 forward RNN과 backward RNN을 사용한다.

```math
\begin{aligned}
h_j^{\mathrm{forward}} &= \operatorname{forward\_RNN}(x_j,h_{j-1}^{\mathrm{forward}}),\\
h_j^{\mathrm{backward}} &= \operatorname{backward\_RNN}(x_j,h_{j+1}^{\mathrm{backward}}).
\end{aligned}
```

두 state를 concatenate해 annotation을 만든다.

```math
h_j=\operatorname{concat}\!\left(h_j^{\mathrm{forward}},h_j^{\mathrm{backward}}\right)
```

따라서 $h_j$는 단순한 $x_j$ embedding이 아니라 position $j$ 주변의 양방향 문맥을 반영한다.

```text
h_j = source position j centered contextual representation
```

이 논문은 attention만 제안한 것이 아니라 attention이 조회할 memory를 BiRNN annotation sequence로 구성했다.

### Decoder state update

Target step $i$에서 decoder state는 이전 state, 이전 target word, 현재 context에 의존한다.

```math
s_i=f(s_{i-1},y_{i-1},c_i)
```

다음 target word의 조건부 확률은 다음 형태다.

```math
p(y_i\mid y_{<i},x)=g(y_{i-1},s_i,c_i)
```

논문 구현은 gated hidden unit과 maxout output layer를 사용한다. 중요한 구조적 사실은 $c_i$가 state update와 word prediction에 매 step 들어간다는 점이다.

## Additive alignment의 수식

### Energy score

Target step $i$가 source position $j$를 얼마나 참고할지를 나타내는 unnormalized score를 $e_{i,j}$라 하자.

```math
e_{i,j}=a(s_{i-1},h_j)
```

논문의 alignment model $a$는 작은 feed-forward network다. 현대 표기로 쓰면 다음과 같은 additive score로 표현할 수 있다.

```math
e_{i,j}=v_a^\top\tanh\!\left(W_s s_{i-1}+W_hh_j+b_a\right)
```

Decoder state와 encoder annotation을 각각 선형 변환한 뒤 더하고, $\tanh$와 learned vector $v_a$로 scalar score를 만든다.

이를 additive attention이라고 부르는 이유는 두 representation을 dot product하기보다 projected state를 더한 뒤 비선형 network로 scoring하기 때문이다.

### Soft alignment weight

각 target step에서 source position 방향으로 softmax를 적용한다.

```math
\alpha_{i,j}
=\frac{\exp(e_{i,j})}{\sum_k \exp(e_{i,k})}
```

따라서 고정된 $i$에 대해 다음이 성립한다.

```math
\begin{aligned}
\alpha_{i,j} &\ge 0,\\
\sum_j\alpha_{i,j} &= 1.
\end{aligned}
```

$\alpha_{i,j}$는 target word $y_i$를 결정할 때 source position $j$가 얼마나 중요한지를 나타낸다.

### Context vector

Context는 모든 source annotation의 weighted sum이다.

```math
c_i=\sum_j\alpha_{i,j}h_j
```

Soft alignment distribution 아래 annotation의 expectation으로 볼 수 있다.

```math
c_i=\mathbb{E}_{j\sim\alpha_i}[h_j]
```

Hard alignment가 position 하나를 선택한다면 soft attention은 여러 position의 정보를 동시에 섞을 수 있다. Article과 noun, multi-word expression, source-target length mismatch처럼 one-to-one alignment가 부자연스러운 경우에 유리하다.

### 왜 $s_{i-1}$을 사용하는가

Alignment score는 현재 state $s_i$가 아니라 직전 state $s_{i-1}$을 query로 사용한다.

```text
s_(i-1) -> alignment -> c_i -> s_i
```

현재 state를 만들기 위해 현재 context가 필요하므로 순환 의존성을 피하는 자연스러운 계산 순서다. 후속 Luong attention은 decoder hidden state를 먼저 만든 뒤 attention을 적용하는 다른 순서를 사용한다.

## Tensor shape로 보는 계산

다음 기호를 사용하자.

```text
B: batch size
Tx: source length
Ty: target length
d_h: annotation dimension
d_s: decoder state dimension
d_a: alignment hidden dimension
```

Encoder output과 decoder state shape는 다음과 같다.

```text
H: [B, Tx, d_h]
s_(i-1): [B, d_s]
```

Additive score를 vectorized하면 다음과 같다.

```math
\begin{aligned}
\operatorname{projected\_state} &= s_{i-1}W_s,
&\operatorname{shape} &= [B,d_a],\\
\operatorname{projected\_memory} &= HW_h,
&\operatorname{shape} &= [B,T_x,d_a].
\end{aligned}
```

State를 source length 축으로 broadcast해 더한다.

```math
\begin{aligned}
E_{\mathrm{hidden}}
&=\tanh\!\left(
\operatorname{projected\_memory}
+\operatorname{projected\_state}[:,\mathrm{None},:]
\right),\\
\operatorname{shape}(E_{\mathrm{hidden}})&=[B,T_x,d_a].
\end{aligned}
```

$v_a$와 내적해 source position별 scalar score를 만든다.

```math
\begin{aligned}
e_i&=E_{\mathrm{hidden}}v_a,\\
\operatorname{shape}(e_i)&=[B,T_x].
\end{aligned}
```

Softmax와 context 계산은 다음과 같다.

```math
\begin{aligned}
\alpha_i
&=\operatorname{softmax}(e_i,\operatorname{dim}=\operatorname{source\_position}),\\
\operatorname{shape}(\alpha_i)&=[B,T_x],\\
c_i&=\alpha_i[:,\mathrm{None},:]\,H,\\
\operatorname{shape}(c_i)&=[B,d_h].
\end{aligned}
```

전체 target sequence의 alignment를 모으면 다음 matrix가 된다.

```text
Alpha: [B, Ty, Tx]
```

`Alpha[b,i,j]`는 sample `b`의 target position `i`가 source position `j`를 보는 weight다.

## 작은 예시로 보는 alignment

Source가 다음 세 token이라고 하자.

```text
source: [the, red, car]
```

French target에서 `voiture`를 생성할 때 energy가 다음과 같다고 가정하자.

```math
e=\begin{bmatrix}0.2&0.7&2.1\end{bmatrix}
```

Softmax 후 weight는 대략 다음처럼 된다.

```math
\alpha=\begin{bmatrix}0.11&0.18&0.71\end{bmatrix}
```

Context는 다음 weighted sum이다.

```math
c_i=0.11h_{\mathrm{the}}+0.18h_{\mathrm{red}}+0.71h_{\mathrm{car}}
```

`car` annotation이 가장 크게 반영되지만 article과 adjective 정보도 완전히 버리지 않는다. 다음 target `rouge`를 생성할 때는 query state가 바뀌므로 alignment distribution도 달라진다.

```text
target voiture: attend mostly to car
target rouge:   attend mostly to red
```

이처럼 같은 source memory를 target step마다 다른 방식으로 읽는다.

## 학습과 미분 가능성

### End-to-end objective

모델은 target sentence의 negative log-likelihood를 최소화한다.

```math
\mathcal{L}=-\sum_i\log p(y_i\mid y_{<i},x)
```

Context $c_i$는 $\alpha_{i,j}$의 differentiable weighted sum이고, $\alpha$는 energy score의 softmax다. 따라서 gradient가 다음 경로를 따라 흐른다.

```text
translation loss
-> output probability
-> decoder state and context
-> alignment weights
-> alignment network
-> encoder annotations
```

별도 alignment label 없이 번역 supervision만으로 정렬 패턴이 학습된다.

### Soft attention의 의미

논문은 alignment를 discrete latent variable로 sampling하지 않는다. 모든 source position을 동시에 고려하는 expectation을 사용한다.

```text
soft attention:
deterministic + differentiable
```

이 점이 REINFORCE 같은 고분산 gradient estimator 없이 alignment model을 공동 학습할 수 있게 한다.

## 계산 복잡도와 병목

Target length가 $T_y$, source length가 $T_x$라면 모든 target-source pair에 score를 계산한다.

```math
\begin{aligned}
\text{alignment score complexity:}\quad &O(T_yT_xd_a),\\
\text{alignment matrix memory:}\quad &O(T_yT_x)\ \text{if stored}.
\end{aligned}
```

RNN decoder는 target step을 순차적으로 처리하므로 병렬화가 제한된다.

```text
for i = 1...Ty:
    score all source positions
    compute c_i
    update s_i
```

Fixed-vector bottleneck은 완화하지만 recurrent dependency는 남는다. Transformer는 이후 모든 target query의 attention을 matrix multiplication으로 병렬화한다.

## 실험 설정

### 데이터

논문은 WMT 2014 English-to-French translation을 사용한다. 원 parallel corpora는 약 850M word이고 data selection 후 348M word로 줄인다.

- Validation: news-test-2012 + news-test-2013
- Test: news-test-2014, 3,003 sentence
- Vocabulary: 각 언어의 상위 30,000 word
- Vocabulary 밖 word: `[UNK]`
- Monolingual data: neural model에는 별도 사용하지 않음

현대 subword tokenization 이전의 word-level 설정이므로 unknown word 비율이 성능 해석에 큰 영향을 준다.

### 비교 모델

두 architecture를 maximum training sentence length 30과 50으로 각각 학습한다.

```text
RNNencdec-30
RNNencdec-50
RNNsearch-30
RNNsearch-50
```

`RNNencdec`는 fixed context encoder-decoder이고 `RNNsearch`가 attention model이다.

주요 설정은 다음과 같다.

- Encoder/decoder hidden unit: 1,000
- RNNsearch encoder: forward 1,000 + backward 1,000
- Minibatch: 80 sentence
- Optimizer: Adadelta 기반 minibatch SGD
- Training time: 모델당 약 5일
- Decoding: beam search

## 실험 결과

### BLEU

| 모델 | 전체 test BLEU | source/reference에 UNK가 없는 subset |
| --- | ---: | ---: |
| RNNencdec-30 | 13.93 | 24.19 |
| RNNsearch-30 | 21.50 | 31.44 |
| RNNencdec-50 | 17.82 | 26.71 |
| RNNsearch-50 | 26.75 | 34.16 |
| RNNsearch-50, longer training | 28.45 | 36.15 |
| Moses phrase-based SMT | 33.30 | 35.63 |

Attention을 사용한 RNNsearch가 같은 maximum length의 RNNencdec를 크게 앞선다.

```text
length 30: +7.57 BLEU on all test sentences
length 50: +8.93 BLEU on all test sentences
```

UNK가 없는 subset에서 오래 학습한 RNNsearch-50은 Moses보다 `0.52 BLEU` 높다. 다만 전체 test에서는 Moses가 여전히 `4.85 BLEU` 높다. Word-level 30k vocabulary의 `[UNK]` 문제가 neural model 성능을 크게 제한했음을 보여 준다.

Moses는 추가 418M-word monolingual corpus를 사용했다는 차이도 있어 완전히 동일한 조건의 비교는 아니다.

### 긴 sentence에 대한 효과

논문의 Figure 2에서 fixed-vector RNNencdec는 sentence length가 길어질수록 BLEU가 급격히 하락한다. RNNsearch는 더 완만하게 감소하며, RNNsearch-50은 길이 50 이상에서도 상대적으로 안정적이다.

중요한 비교는 다음이다.

```text
RNNsearch-30 > RNNencdec-50
```

단순히 긴 sentence로 학습하는 것보다 target step별 source 조회 구조가 더 큰 이득을 준다는 증거다.

### 학습된 alignment

<p align="center"><img src="https://github.com/user-attachments/assets/cf77d1d1-a746-43d0-8f13-db68990b8f62" alt="RNNsearch가 학습한 source-target soft alignment" width="680"></p>
<p align="center"><sub>원 논문 Figure 3 — RNNsearch-50이 학습한 영어-프랑스어 soft alignment</sub></p>

Alignment matrix에는 대체로 diagonal pattern이 나타나지만 영어-프랑스어 어순 차이도 학습한다. 논문은 `European Economic Area`가 `zone économique européenne`로 순서가 바뀌는 사례를 보여 준다.

Soft alignment는 다음과 같은 장점이 있다.

- 하나의 target word가 여러 source word를 함께 참고할 수 있다.
- Article gender처럼 인접 noun 정보가 필요한 경우를 자연스럽게 처리한다.
- Source와 target phrase length가 달라도 NULL alignment를 명시적으로 만들 필요가 없다.

그러나 attention heatmap이 사람이 기대한 정렬과 비슷하다는 사실만으로 weight가 완전한 인과 설명이라고 결론 내릴 수는 없다. 이는 모델의 정보 routing pattern을 보여 주는 유용한 진단 도구로 보는 편이 안전하다.

## 장점과 핵심 기여

### 1. Fixed-length bottleneck을 직접 완화했다

Source 정보를 하나의 vector가 아니라 위치별 memory로 유지하고 필요할 때 선택적으로 조회한다.

### 2. Alignment와 translation을 공동 학습했다

별도 word alignment label 없이 translation loss만으로 attention network가 학습된다.

### 3. Query-dependent context를 도입했다

Target step마다 query state가 달라지고, 그에 따라 source summary도 달라진다.

```text
one source sentence
-> many target-specific context vectors
```

### 4. Soft alignment를 시각화할 수 있다

Alignment matrix를 통해 model이 target word별로 source의 어느 위치를 참고하는지 관찰할 수 있다.

### 5. 긴 sentence 성능을 크게 개선했다

실험은 attention의 이득이 단순한 parameter 증가가 아니라 sequence bottleneck 완화와 관련 있음을 보여 준다.

## 한계와 비판적 관점

### 1. RNN의 순차 계산은 그대로다

Encoder의 forward/backward RNN과 autoregressive decoder 모두 순차 dependency를 가진다. Attention은 memory access를 개선하지만 parallelism 문제는 해결하지 않는다.

### 2. Pairwise alignment 비용이 든다

Target step마다 모든 source annotation을 scoring하므로 $O(T_xT_y)$ pair를 계산한다. 긴 sequence에서는 비용이 커진다.

### 3. Word-level vocabulary가 성능을 제한한다

30k shortlist 밖 word를 모두 `[UNK]`로 처리한다. 전체 test와 no-UNK subset의 큰 BLEU 차이가 이를 보여 준다.

### 4. Context는 여전히 weighted average다

여러 annotation을 하나의 $c_i$로 합치므로 서로 다른 source 위치의 세부 정보가 평균 과정에서 섞일 수 있다. Multi-head attention은 이후 여러 독립적인 weighted sum을 동시에 사용해 이 한계를 완화한다.

### 5. Soft alignment가 항상 sparse하거나 해석 가능한 것은 아니다

Softmax가 모든 source position에 weight를 배분하므로 distribution이 넓게 퍼질 수 있다. Alignment weight와 언어학적 word alignment가 반드시 일치하는 것도 아니다.

### 6. 실험 비교에는 시대적 제약이 있다

Single test set, word-level vocabulary, 제한된 model scale과 decoding 설정의 결과다. 오늘날 architecture의 절대 성능으로 읽기보다 fixed bottleneck에 대한 구조적 증거로 읽어야 한다.

## Luong attention 및 Transformer와의 연결

Bahdanau attention의 기본 식은 다음 후속 연구의 출발점이 된다.

```text
query = decoder state
keys/values = encoder annotations
weights = softmax(score(query, keys))
context = weighted sum(values)
```

Luong attention은 다음을 체계적으로 비교한다.

- Additive MLP 대신 dot/general/concat score
- 모든 source position을 보는 global attention
- 일부 window만 보는 local attention
- Attentional vector를 다음 step에 전달하는 input-feeding

Transformer에서는 RNN state 대신 token representation matrix에서 Q/K/V를 만들고 attention을 병렬화한다.

```text
Bahdanau:
one decoder query at a time -> encoder memory

Transformer cross-attention:
all decoder queries in a layer -> encoder memory matrix
```

계산 구현은 달라졌지만 `query-dependent differentiable memory lookup`이라는 핵심은 동일하다.

## 구현 체크리스트

```text
1. Encoder annotation이 forward/backward state를 올바르게 concatenate하는가?
2. Alignment query가 s_i가 아니라 논문 순서대로 s_(i-1)인가?
3. Source padding position을 softmax 전에 mask하는가?
4. Softmax 축이 source length 축인가?
5. 각 target step에서 alpha 합이 1인가?
6. Context shape가 decoder 입력과 맞는가?
7. Teacher forcing에서 y_(i-1) shift가 올바른가?
8. Beam search 중 beam별 decoder state와 alignment가 분리되는가?
9. Alignment matrix의 target/source 축을 뒤집어 해석하지 않는가?
10. BLEU 비교에서 UNK 처리와 tokenization 조건을 확인했는가?
```

## 개인 학습/연구 메모

이 논문의 핵심은 단순히 weighted sum을 발명했다는 데 있지 않다. Sequence 전체를 하나의 고정 표현으로 먼저 압축해야 한다는 encoder-decoder의 가정을 바꾸었다.

```text
compression-first modeling
-> query-dependent retrieval
```

기억해야 할 세 식은 다음이다.

```math
\begin{aligned}
e_{i,j} &= v_a^\top\tanh\!\left(W_s s_{i-1}+W_hh_j\right),\\
\alpha_{i,:} &= \operatorname{softmax}(e_{i,:}),\\
c_i &= \sum_j\alpha_{i,j}h_j.
\end{aligned}
```

이 세 줄이 additive attention의 score, normalization, retrieval을 모두 나타낸다.

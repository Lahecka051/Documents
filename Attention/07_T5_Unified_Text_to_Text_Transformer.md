# 07. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

## 논문 정보

- 원본 파일: `Attention/07_T5_Unified_Text_to_Text_Transformer.pdf`
- 제목: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
- 저자: Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu
- 발표: JMLR 2020
- 모델명: T5, Text-to-Text Transfer Transformer
- 주제: transfer learning, text-to-text formulation, span corruption, C4, relative position bias, scaling
- 핵심 키워드: task prefix, encoder-decoder, sentinel token, denoising pre-training, relative position bucket

## 한눈에 보는 요약

T5는 새로운 attention 연산 하나를 제안하는 논문이라기보다 NLP transfer learning의 여러 선택지를 통제된 조건에서 비교한 대규모 경험 연구다.

논문은 다음 질문을 순서대로 탐색한다.

1. 모든 NLP task를 하나의 입출력 형식으로 만들 수 있는가?
2. encoder-decoder, decoder-only, prefix LM 중 어떤 구조가 좋은가?
3. causal language modeling과 여러 denoising objective 중 무엇이 좋은가?
4. 어떤 pre-training corpus와 fine-tuning 전략이 좋은가?
5. 추가 compute를 model size, training length, ensemble 중 어디에 쓰는 것이 좋은가?

T5의 답은 다음과 같다.

```text
all tasks -> text-to-text
architecture -> encoder-decoder Transformer
pre-training -> span corruption denoising
data -> large cleaned web corpus C4
position -> learned relative attention bias
scaling -> larger models and more pre-training data
```

모든 task가 text input과 text output으로 변환되므로 pre-training, classification, regression, translation, summarization이 같은 token-level maximum likelihood loss를 사용한다.

```text
input text + task prefix
-> encoder
-> autoregressive decoder
-> output text
```

최종 T5 계열은 60M, 220M, 770M, 3B, 11B parameter로 확장되었고, 11B 모델은 당시 24개 평가 task 중 18개에서 state of the art를 달성했다.

![T5의 text-to-text, span corruption, relative position bucket](https://github.com/user-attachments/assets/bd28765a-d638-4ea7-bf72-8098ce2e9a62)

## 연구 배경과 문제의식

### Transfer learning 연구의 비교가 어려웠다

T5 이전에도 BERT, GPT, XLNet, MASS 등 다양한 pre-training 방법이 존재했다. 그러나 각 연구는 다음 요소를 동시에 바꾸는 경우가 많았다.

- model architecture
- pre-training objective
- corpus와 filtering
- vocabulary
- training budget
- fine-tuning method
- downstream benchmark

성능이 올라도 어느 요소가 실제 원인인지 분리하기 어려웠다. T5 논문은 합리적인 baseline을 정한 뒤 한 번에 한 요소만 바꾸는 coordinate-ascent식 실험을 수행했다.

이 방법도 모든 interaction을 탐색하지는 못한다. 특정 objective가 다른 model size에서 더 좋을 수 있지만 모든 조합을 평가하기에는 계산량이 너무 크다. 논문은 이 한계를 명시한다.

### Task별 head와 loss가 만드는 복잡성

전통적인 multi-task NLP는 task마다 출력 head와 loss가 달랐다.

```text
classification -> class logits + cross entropy
regression -> scalar + MSE
span QA -> start/end logits
translation -> autoregressive token loss
summarization -> autoregressive token loss
```

T5는 모든 출력을 text token sequence로 바꿔 같은 decoder와 같은 cross entropy를 사용한다.

## 핵심 아이디어 1: Text-to-text framework

### Task prefix

모델이 수행할 작업을 자연어 prefix로 입력 앞에 붙인다.

```text
translation
input:  translate English to German: That is good.
target: Das ist gut.

summarization
input:  summarize: <article>
target: <summary>

classification
input:  mnli premise: ... hypothesis: ...
target: entailment

regression-like task
input:  stsb sentence1: ... sentence2: ...
target: 3.8
```

task prefix는 별도 task ID embedding이 아니라 vocabulary에 존재하는 text token이다. wording은 hyperparameter가 될 수 있지만 논문에서는 prefix 문구 변화의 영향이 제한적이었다고 보고한다.

### Classification도 생성이다

MNLI처럼 세 class를 가진 문제는 label word를 생성한다.

```text
valid targets:
entailment
neutral
contradiction
```

모델이 `hamburger`처럼 허용되지 않은 문자열을 출력하면 오답으로 처리한다. 논문의 학습된 모델에서는 이런 현상을 관찰하지 않았다고 한다.

### Regression도 문자열로 바꾼다

STS-B score는 1에서 5 사이이며 주로 0.2 간격으로 annotation되어 있다. 논문은 값을 가장 가까운 0.2 단위로 반올림하고 literal string으로 출력한다.

```text
2.57 -> "2.6"
```

결과적으로 연속 회귀를 21개 가능한 문자열을 생성하는 문제로 바꾼 셈이다. 출력이 범위 밖이거나 숫자로 변환되지 않으면 오답이다.

### Text-to-text가 통일하는 것

모든 task는 다음 확률을 최대화한다.

$$
\log P_\theta(y\mid x)
=
\sum_{t=1}^{T_y}
\log P_\theta(y_t\mid y_{<t},x)
$$

따라서 다음 요소가 공통화된다.

- model interface
- output vocabulary
- teacher forcing
- token-level cross entropy
- decoding procedure
- checkpoint format
- evaluation pipeline의 큰 틀

### 통일되지 않는 것

text-to-text라고 해서 모든 task가 완전히 동일해지는 것은 아니다.

- task별 input preprocessing은 필요하다.
- label string mapping이 필요하다.
- metric은 task마다 다르다.
- output length와 decoding 설정이 다르다.
- task mixing ratio가 필요하다.

즉, T5는 **학습 인터페이스를 통일**하지만 데이터 의미와 평가지표까지 없애지는 않는다.

## 핵심 아이디어 2: Encoder-decoder architecture

### Baseline 구조

T5 baseline은 원래 Transformer의 encoder-decoder 구조를 따른다.

```text
input tokens
-> bidirectional encoder self-attention
-> encoder representations
-> decoder cross-attention
-> causal decoder self-attention
-> output tokens
```

baseline의 주요 크기는 다음과 같다.

| 항목 | 값 |
|---|---:|
| encoder block 수 | 12 |
| decoder block 수 | 12 |
| $d_{model}$ | 768 |
| $d_{ff}$ | 3072 |
| attention head 수 | 12 |
| head key/value dimension | 64 |
| parameter 수 | 약 220M |
| dropout | 0.1 |

### 원래 Transformer와 다른 세부

T5 구현은 몇 가지 차이가 있다.

1. LayerNorm을 sublayer 입력에 적용하는 pre-normalization 형태다.
2. LayerNorm은 activation을 rescale하지만 additive bias를 사용하지 않는다.
3. relative position representation은 vector가 아니라 attention logit에 더하는 scalar bias다.
4. input embedding과 output softmax weight를 공유한다.

논문은 이 세부 변경을 transfer-learning 비교 요인과 독립적인 baseline 설계로 취급하며 각각의 ablation은 수행하지 않았다.

### 왜 encoder-decoder가 유리했는가

encoder는 입력 전체를 fully visible attention으로 읽고 decoder는 target을 causal하게 생성한다. decoder-only LM은 input과 target을 한 sequence로 이어 붙이므로 input representation도 왼쪽 문맥만 본다.

```text
decoder-only LM input representation:
x_i sees x_1 ... x_i

encoder representation:
x_i sees all input tokens
```

prefix LM은 input prefix 내부를 bidirectional하게 볼 수 있어 이 문제를 줄인다. 그래도 실험에서는 explicit encoder-decoder attention을 가진 구조가 가장 좋았다.

### Architecture comparison

논문은 parameter 수 `P`와 compute `M`을 기준으로 여러 구조를 비교했다.

| Architecture | Objective | Params | Cost | GLUE | SuperGLUE |
|---|---|---:|---:|---:|---:|
| encoder-decoder | denoising | 2P | M | 83.28 | 71.36 |
| shared encoder-decoder | denoising | P | M | 82.81 | 70.73 |
| shallow encoder-decoder | denoising | P | M/2 | 80.88 | 68.42 |
| decoder-only LM | denoising | P | M | 74.70 | 55.02 |
| prefix LM | denoising | P | M | 81.82 | 68.11 |

encoder와 decoder parameter를 공유하면 parameter는 절반이지만 성능 하락이 작았다. 반면 layer 수를 절반으로 줄이면 성능이 더 크게 떨어졌다.

또한 같은 구조에서는 denoising objective가 causal LM objective보다 downstream 성능이 일관되게 좋았다.

## 핵심 아이디어 3: Relative position bias

### Vector를 더하지 않고 logit에 scalar를 더한다

head `h`의 query position `i`와 key position `j` 사이 attention logit을 다음처럼 둔다.

$$
e_{i,j}^{(h)}
=
\frac{q_i^{(h)\top}k_j^{(h)}}{\sqrt{d_h}}
+b^{(h)}_{\operatorname{bucket}(j-i)}
$$

여기서 `b`는 relative position bucket마다 학습되는 scalar다.

Shaw 방식처럼 key/value vector에 relative embedding을 더하거나 Transformer-XL처럼 네 항으로 score를 분해하지 않는다. 단순히 attention logit에 bias 하나를 더한다.

### 32개 bucket

논문 설정은 다음과 같다.

- bucket 수: 32
- 가까운 offset: 더 세밀하게 구분
- 먼 offset: logarithmic하게 더 넓은 범위를 한 bucket으로 묶음
- 최대 거리: 128
- 128보다 먼 offset: 동일한 마지막 bucket 사용

개념적인 bucket 함수는 다음과 같다.

```python
def relative_position_bucket(relative_position,
                             num_buckets=32,
                             max_distance=128,
                             bidirectional=True):
    # 실제 구현은 sign과 causal 여부를 먼저 처리한다.
    n = abs(relative_position)

    max_exact = num_buckets // 2
    if n < max_exact:
        bucket = n
    else:
        bucket = max_exact + log_scale(n, max_distance)

    return min(bucket, num_buckets - 1)
```

실제 encoder에서는 양방향 offset의 부호를 나누기 위해 bucket 절반을 각 방향에 배정한다. decoder causal self-attention에서는 미래가 mask되므로 과거 방향만 사용한다.

### Layer와 head 사이 공유

논문은 효율을 위해 relative position bias parameter를 모든 layer에서 공유한다. 같은 layer 안에서는 head마다 서로 다른 bias table을 사용한다.

```text
shared across layers
different across heads
```

한 layer는 128보다 먼 정확한 거리를 구분하지 못하지만, 여러 layer가 local 정보를 조합하면서 더 긴 범위에 민감해질 수 있다고 논문은 설명한다.

### Transformer-XL과 비교

| 방법 | position이 들어가는 방식 | 거리 표현 |
|---|---|---|
| Transformer-XL | content/position score 네 항 | sinusoidal relative vector |
| T5 | attention logit에 scalar bias | learned log buckets |

T5 방식은 매우 단순하고 parameter가 적다. 반면 마지막 bucket 밖의 모든 거리를 동일하게 취급하므로 정확한 장거리 차이를 직접 표현하지 않는다.

## 핵심 아이디어 4: Span corruption

### 기본 절차

T5의 최종 pre-training objective는 연속 token span을 가리고, 가려진 span만 decoder target으로 생성한다.

원문을 다음처럼 두자.

```text
Thank you for inviting me to your party last week.
```

`for inviting`과 `last`를 가리면 encoder input은 다음과 같다.

```text
Thank you <extra_id_0> me to your party <extra_id_1> week.
```

decoder target은 가려진 span을 sentinel 순서대로 연결한다.

```text
<extra_id_0> for inviting <extra_id_1> last <extra_id_2>
```

마지막 sentinel은 target 종료 경계를 명확히 한다.

### Sentinel token의 역할

각 span은 example 내부에서 고유한 sentinel ID를 가진다.

```text
<extra_id_0> -> first missing span
<extra_id_1> -> second missing span
<extra_id_2> -> final boundary
```

같은 mask token 하나만 반복하면 decoder가 어떤 복원 span이 입력의 어느 위치에 대응하는지 모호해질 수 있다. 고유 sentinel은 위치 대응과 순서를 유지한다.

### 왜 원문 전체를 복원하지 않는가

BERT-style encoder-decoder objective가 원문 전체를 target으로 예측하면 decoder sequence가 길다. T5는 가려진 token만 예측해 decoder 계산량을 줄인다.

```text
full reconstruction target length ~= original length
T5 target length ~= corrupted tokens + sentinels
```

15%만 가린다면 target은 원문보다 훨씬 짧다. encoder input도 연속 span을 sentinel 하나로 압축하므로 약간 짧아진다.

### 선택된 hyperparameter

최종 설정은 다음과 같다.

- corruption rate: 15%
- mean corrupted span length: 3
- contiguous random spans
- target: corrupted spans only

500 token sequence라면 평균적으로 다음과 같다.

```text
corrupted tokens = 500 * 0.15 = 75
mean span length = 3
number of spans ~= 75 / 3 = 25
```

### Objective ablation

큰 방향에서는 denoising이 prefix LM과 deshuffling보다 좋았다.

| Objective | GLUE | SuperGLUE |
|---|---:|---:|
| prefix language modeling | 80.69 | 65.27 |
| BERT-style denoising | 82.96 | 69.85 |
| deshuffling | 73.17 | 58.47 |

그러나 denoising 내부의 여러 변형은 성능 차이가 작았다. 논문은 이 경우 sequence length가 짧아 계산 효율이 좋은 objective를 선택하는 것이 합리적이라고 결론낸다.

평균 span length 3은 i.i.d. corruption보다 대부분의 non-translation benchmark에서 약간 좋았고 target도 짧았다.

## C4: Colossal Clean Crawled Corpus

### 왜 새 corpus가 필요했는가

Common Crawl은 매월 대규모 web text를 제공하지만 그대로 사용하면 다음이 많다.

- menu와 boilerplate
- error message
- 중복 문장
- code와 placeholder
- 비자연어 text
- task와 무관하거나 유해한 내용

논문은 Common Crawl 한 시점의 scrape에서 English text를 heuristic하게 정제해 C4를 만들었다.

### 주요 filtering rule

- 문장 부호로 끝나는 line만 유지
- 3문장 미만 page 제거
- 5단어 미만 line 제거
- language detector가 English probability 0.99 미만이면 제거
- JavaScript, lorem ipsum, curly brace가 포함된 page 제거
- 일부 policy boilerplate line 제거
- 반복되는 세 문장 span을 deduplicate
- 특정 bad-word list가 포함된 page 제거

최종 C4는 약 745GB였다. 같은 scrape에서 heuristic filtering을 제거한 버전은 약 6.1TB였다.

### Filtering의 장점과 위험

unfiltered C4는 baseline보다 대부분의 task에서 나빴다. 이는 단순한 data quantity보다 quality control이 중요함을 보여준다.

그러나 heuristic filtering은 중립적이지 않다.

- bad-word filter가 정당한 문맥까지 제거할 수 있다.
- 언어 감지가 code-switching과 비표준 영어를 배제할 수 있다.
- punctuation rule이 대화체와 목록형 문서를 줄일 수 있다.
- web corpus의 사회적 편향이 남는다.

논문은 C4의 크기와 효용을 강조했지만 이후 연구에서는 contamination, 개인정보, 편향, filtering transparency도 중요한 평가 항목이 되었다.

### Data repetition 실험

논문은 작은 C4 subset을 여러 번 반복해 pre-training하면 성능이 나빠질 수 있음을 보였다. 같은 token budget이라도 충분히 다양한 데이터가 중요하다.

```text
more steps on repeated small data
!=
more unique pre-training data
```

## Vocabulary와 sequence 구성

### SentencePiece

논문은 32,000 wordpiece vocabulary를 사용한다. English뿐 아니라 German, French, Romanian translation도 다루기 위해 vocabulary 학습 corpus를 다음처럼 섞었다.

```text
English C4 : German : French : Romanian
10         : 1      : 1      : 1
```

input과 output vocabulary 및 embedding/softmax weight를 공유한다.

### Packing

짧은 example 여러 개를 length 512 sequence 하나에 pack해 padding 낭비를 줄인다. 서로 다른 example 사이 attention과 loss가 섞이지 않도록 segment mask가 필요하다.

```text
packed row:
[example A][example B][example C][padding]
```

단순 concatenation만 하고 attention mask를 분리하지 않으면 example leakage가 생긴다.

## Baseline 학습 설정

| 항목 | 설정 |
|---|---:|
| maximum sequence length | 512 |
| batch | 128 sequences |
| packed token 수 | 약 65,536 |
| pre-training steps | $2^{19}=524,288$ |
| baseline token budget | 약 $2^{35}$, 34B tokens |
| optimizer | AdaFactor |
| warmup | 10,000 steps |
| learning rate | warmup 후 inverse square root decay |
| fine-tuning learning rate | 0.001 constant |
| baseline decoding | greedy |

최종 T5 모델은 baseline보다 훨씬 길게, 총 1T token 이상으로 학습되었다. translation과 summarization의 최종 결과에는 beam width 4, length penalty 0.6을 사용했다.

## Model scaling

최종 공개 model size는 다음과 같다.

| 모델 | Parameter 수 |
|---|---:|
| T5-Small | 60M |
| T5-Base | 220M |
| T5-Large | 770M |
| T5-3B | 3B |
| T5-11B | 11B |

논문은 추가 compute를 쓰는 여러 방법을 비교했다.

- 더 오래 pre-train
- 더 큰 model
- ensemble

모두 성능을 높였지만, 작은 모델을 훨씬 오래 학습하는 것보다 큰 모델을 상대적으로 적게 학습하는 편이 더 좋은 경우가 많았다.

### Scale만으로 설명되지 않는 개선

논문은 baseline, baseline을 1T token으로 늘린 모델, T5-Base를 비교했다.

| Model | GLUE | CNN/DM | SQuAD | SuperGLUE |
|---|---:|---:|---:|---:|
| baseline | 83.28 | 19.24 | 80.88 | 71.36 |
| baseline-1T | 84.80 | 19.62 | 83.01 | 73.90 |
| T5-Base | 85.97 | 20.90 | 85.44 | 75.64 |

training token을 늘리는 것만으로도 좋아지지만 T5-Base가 모든 task에서 더 좋다. span corruption, data/mixture 전략 등 non-scaling choice도 기여했다는 근거다.

## 최종 결과

T5-11B의 대표 결과는 다음과 같다.

| Benchmark | T5-11B | 당시 previous best |
|---|---:|---:|
| GLUE average | 90.3 | 89.4 |
| SuperGLUE average | 88.9 | 84.6 |
| SQuAD EM | 91.26 | 90.1 |
| SQuAD F1 | 96.22 | 95.5 |
| CNN/DM ROUGE-2 | 21.55 | 20.30 |

11B가 모든 분야에서 이긴 것은 아니다. WMT translation에서는 당시 state of the art를 넘지 못했다. 논문은 English 중심 C4, backtranslation 미사용, 다른 연구의 추가 cross-lingual training을 원인 후보로 든다.

CNN/Daily Mail에서는 ROUGE가 개선되었지만 ROUGE 증가가 반드시 더 사실적이고 일관된 요약을 뜻하지 않는다는 점도 논문이 지적한다.

## 구현 관점의 전체 pipeline

```text
1. raw task example을 text-to-text 형식으로 변환
2. task prefix 추가
3. SentencePiece tokenization
4. pre-training이면 15% span corruption 적용
5. encoder input과 sentinel target 생성
6. example packing과 attention mask 생성
7. encoder-decoder teacher forcing
8. token cross entropy 계산
9. fine-tuning에서 동일한 interface 유지
10. task에 맞는 greedy/beam decoding
11. output text를 label, score, answer로 후처리
```

### Span corruption pseudocode

```python
def make_t5_example(tokens, noise_density=0.15, mean_span_length=3):
    mask = sample_random_spans(
        length=len(tokens),
        noise_density=noise_density,
        mean_span_length=mean_span_length,
    )

    spans = consecutive_true_regions(mask)
    encoder = []
    target = []
    cursor = 0

    for span_id, (start, end) in enumerate(spans):
        sentinel = extra_id(span_id)
        encoder.extend(tokens[cursor:start])
        encoder.append(sentinel)

        target.append(sentinel)
        target.extend(tokens[start:end])
        cursor = end

    encoder.extend(tokens[cursor:])
    target.append(extra_id(len(spans)))
    return encoder, target
```

실제 구현에서는 span sampling이 원하는 총 noise token 수와 span 수를 정확히 만족하도록 segmentation을 생성한다.

## Tensor shape

```text
B: batch size
S: encoder input length
T: decoder target length
D: model dimension
H: number of heads
Dh: head dimension

encoder hidden: [B, S, D]
encoder self-attention score: [B, H, S, S]
decoder hidden: [B, T, D]
decoder self-attention score: [B, H, T, T]
cross-attention score: [B, H, T, S]

relative bias encoder: [1 or B, H, S, S]
relative bias decoder: [1 or B, H, T, T]
```

relative bias는 token content와 무관하므로 같은 sequence length와 position layout이면 batch 사이에 broadcast하거나 cache할 수 있다.

## 계산 비용

encoder-decoder의 attention cost는 대략 다음 세 부분이다.

$$
O(S^2) + O(T^2) + O(ST)
$$

- encoder self-attention: `S^2`
- decoder self-attention: `T^2`
- cross-attention: `S*T`

span corruption은 `T`를 원문 전체보다 훨씬 짧게 만들어 decoder self-attention과 cross-attention 비용을 줄인다. T5 objective가 단순한 학습 target 설계를 넘어 compute 최적화이기도 한 이유다.

## 구현 체크리스트

```text
1. sentinel token ID 순서가 tokenizer convention과 일치하는가?
2. encoder input과 decoder target에 같은 sentinel이 대응하는가?
3. target 마지막 sentinel이 포함되는가?
4. 15%가 span 수가 아니라 token 수 기준인가?
5. mean span length가 sampling 결과에서 유지되는가?
6. packed example 사이 attention이 차단되는가?
7. padding token loss가 무시되는가?
8. decoder input이 target을 one-step right shift했는가?
9. encoder와 decoder relative bucket의 bidirectional 설정이 다른가?
10. relative bias가 head별이고 layer 간 공유되는 checkpoint인가?
11. distance sign이 key-query convention과 맞는가?
12. label text의 대소문자와 normalization이 평가 코드와 맞는가?
13. generative task의 max output length가 충분한가?
14. beam search length penalty가 task에 맞는가?
```

## 자주 헷갈리는 지점

### “T5는 decoder-only LLM인가?”

아니다. 논문의 T5는 encoder-decoder Transformer다. 입력은 encoder가 bidirectional하게 처리하고 decoder가 cross-attention으로 읽는다.

### “T5 pre-training은 BERT MLM과 같은가?”

둘 다 denoising이지만 출력 방식이 다르다. BERT는 encoder 위치에서 masked token을 분류하고 T5는 decoder가 sentinel로 구분된 missing span을 생성한다.

### “Relative position embedding vector를 Q/K에 더하는가?”

아니다. T5는 bucket별 scalar bias를 attention logit에 직접 더한다.

### “128 token보다 먼 문맥을 못 보는가?”

attention 자체는 전체 sequence를 볼 수 있다. 단지 한 layer의 relative bias가 128 이후 정확한 거리 차이를 구분하지 않고 같은 bucket을 사용한다.

### “모든 task가 완전히 같은 prompt를 쓰는가?”

아니다. task별 prefix와 preprocessing이 있다. 통일되는 것은 text input/output interface와 학습 loss다.

### “T5 성능은 11B scale 하나로만 설명되는가?”

아니다. baseline-1T와 T5-Base 비교에서 non-scaling design choice의 이득도 확인된다.

## 장점과 핵심 기여

### 1. NLP task를 하나의 생성 interface로 통일했다

classification, QA, translation, summarization, regression-like task를 같은 model과 loss로 학습했다.

### 2. Transfer learning 선택지를 체계적으로 비교했다

architecture, objective, data, fine-tuning, multi-task mixing, scaling을 공통 baseline에서 바꾸어 상대적 영향을 분석했다.

### 3. 효율적인 span corruption objective를 정착시켰다

missing span만 생성해 decoder target을 짧게 만들면서 강한 downstream 성능을 유지했다.

### 4. C4를 공개했다

대규모 cleaned web corpus와 데이터 pipeline을 제공해 후속 pre-training 연구의 기반이 되었다.

### 5. 단순한 relative bias를 실용화했다

log-bucket scalar bias는 구현이 쉽고 parameter가 작으며 여러 T5 계열 모델에 널리 사용되었다.

## 한계와 비판적 관점

### 실험이 coordinate ascent에 의존한다

한 번에 한 요소를 바꾸는 방식은 factor interaction을 놓친다. 작은 model에서 좋은 objective가 11B에서도 최선이라는 보장은 없다.

### Text generation은 classification에 비효율적일 수 있다

한 단어 label도 autoregressive decoder를 거친다. task-specific linear head보다 latency와 memory가 크다.

### Encoder-decoder는 배포 비용이 크다

encoder와 decoder stack, cross-attention, autoregressive decoding이 모두 필요하다. on-device classification에는 과도할 수 있다.

### C4 heuristic의 편향과 재현성 문제

web snapshot, filtering list, language detector, deduplication rule이 corpus 분포를 결정한다. 큰 corpus라는 이유만으로 대표성이나 안전성이 보장되지 않는다.

### Benchmark와 metric의 한계

SQuAD와 SuperGLUE 일부 task에서 인간 성능을 넘는 수치가 실제 언어 이해의 완성을 뜻하지 않는다. metric이 machine prediction에 유리하거나 dataset artifact를 반영할 수 있다.

### 11B scale의 접근성

최고 성능은 대규모 TPU와 model/data parallelism에 의존했다. 연구 재현과 실서비스 비용 측면에서 큰 장벽이다.

## 후속 연구와의 연결

T5의 설계는 이후 많은 encoder-decoder model에 직접 이어졌다.

- mT5: multilingual C4와 다국어 text-to-text
- ByT5: subword 대신 byte-level token
- UL2: 여러 denoising mode의 mixture
- FLAN-T5: instruction tuning으로 task generalization 강화
- Switch Transformer: sparse MoE로 parameter scaling
- LongT5: local/global attention과 long-input pre-training

또한 task prefix를 text로 표현하는 방식은 prompt-based learning과 instruction tuning의 중요한 선행 형태다.

```text
task prefix
-> natural-language prompt
-> instruction tuning
```

## 온디바이스 및 비전 관점의 메모

### 작은 장치에서의 T5

T5-Small도 encoder와 decoder가 모두 필요하므로 token 생성 latency가 중요하다. 다음 최적화가 필요할 수 있다.

- encoder output cache
- decoder KV cache
- int8/int4 weight quantization
- beam size 축소 또는 greedy decoding
- maximum target length 제한
- task-specific distillation

classification 하나만 필요하다면 encoder-only distilled model이나 task-specific head가 더 효율적일 수 있다.

### Span corruption과 vision

span corruption의 핵심은 연속 영역을 가리고 compact target으로 복원하는 것이다. image patch나 video tube를 연속 block으로 mask하는 masked autoencoding과 개념적으로 연결된다.

```text
text span <-> image patch block <-> video tube
```

다만 T5는 discrete token을 autoregressive하게 생성한다. vision에서는 pixel/feature regression 또는 discrete visual token 생성 중 어떤 target이 적절한지 별도 설계가 필요하다.

### Relative bucket의 2D 확장

image patch는 row와 column 두 축이 있다. 1D distance bucket 하나보다 다음처럼 축별 bias를 더하는 방식이 자연스럽다.

$$
b_{(r_i-r_j)}^{row}+b_{(c_i-c_j)}^{col}
$$

이는 Swin, ViT 계열의 relative position bias와 연결된다.

## 개인 학습/연구 메모

T5를 기억할 때는 model 이름보다 다음 세 문장을 기억하는 것이 좋다.

```text
1. Every NLP task is conditional text generation.
2. Pre-training predicts only corrupted spans.
3. Relative distance enters as a bucketed scalar logit bias.
```

핵심 식은 다음 두 개다.

$$
\mathcal{L}
=
-\sum_t \log P(y_t\mid y_{<t},x)
$$

$$
e_{i,j}^{(h)}
=
\frac{q_i^{(h)\top}k_j^{(h)}}{\sqrt{d_h}}
+b_{\operatorname{bucket}(j-i)}^{(h)}
$$

T5의 가장 큰 기여는 하나의 복잡한 수식이 아니라 다음 조합을 대규모 실험으로 검증하고 공개한 데 있다.

```text
unified task interface
+ encoder-decoder inductive bias
+ efficient denoising target
+ large diverse corpus
+ systematic scaling
= reusable transfer-learning recipe
```

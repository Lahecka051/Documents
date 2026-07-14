# 20. Big Bird: Transformers for Longer Sequences

## 논문 정보

- 제목: **Big Bird: Transformers for Longer Sequences**
- 저자: Manzil Zaheer et al.
- 발표: NeurIPS 2020
- 핵심 키워드: sparse attention graph, random attention, global tokens, universal approximation, long documents, genomics

## 한눈에 보는 요약

BigBird는 긴 sequence를 위한 sparse attention pattern을 세 종류 edge의 합으로 구성한다.

```text
local window + random links + global links
```

- local edge는 가까운 문맥과 높은 clustering을 보존한다.
- random edge는 graph diameter를 줄이는 장거리 shortcut을 만든다.
- global token은 모든 token과 연결되는 hub가 되어 정보를 수집하고 배포한다.

각 token이 local `w`, random `r`, global `g`개의 연결만 사용하면 복잡도는 `O(n(w+r+g))`이다. `w,r,g`를 고정하면 길이에 대해 선형이다.

BigBird의 특징은 sparse pattern을 실험적 heuristic으로만 제시하지 않고 graph 관점의 이론을 붙였다는 점이다. global star를 포함한 특정 sparse graph가 Transformer의 universal approximation과 Turing completeness 성질을 유지한다고 보인다. 그러나 이 결과는 arbitrary precision, 충분한 depth/width 같은 조건 아래의 표현 가능성이고, 현실의 finite-depth model이 full attention과 같은 품질을 자동으로 낸다는 뜻은 아니다.

![BigBird의 local·random·global sparse attention graph](https://github.com/user-attachments/assets/f308a85a-e393-4b05-a5e7-0bacd44f74c6)

## 문제의식

Dense attention을 graph로 보면 `n`개의 token node가 완전 그래프를 이룬다. 모든 pair가 한 layer에서 직접 연결되지만 edge 수가 `n²`이다. 기존 sparse 방법은 edge를 줄였으나 다음 질문이 남는다.

- local window만으로 먼 정보가 너무 늦게 전달되지 않는가?
- sparse graph가 full Transformer의 표현력을 얼마나 유지하는가?
- 특정 task에만 맞는 pattern이 아니라 여러 domain에서 사용할 수 있는가?

BigBird는 small-world graph와 expander graph의 아이디어를 attention에 가져온다. 현실의 네트워크처럼 local clustering과 소수 장거리 shortcut, hub를 결합하면 적은 edge로도 전체 graph를 빠르게 연결할 수 있다는 관점이다.

## 세 가지 attention edge

### 1. Local window

각 token `i`가 주변 `w/2` 범위를 본다.

```math
L_i=\left\{j:\lvert i-j\rvert\le\frac{w}{2}\right\}
```

텍스트의 인접 단어, DNA의 근접 base처럼 local dependency를 직접 모델링한다. regular pattern이라 block-sparse kernel에도 유리하다.

### 2. Random attention

각 token 또는 block이 sequence 전체에서 `r`개의 random 위치와 연결된다.

```math
R_i\subseteq\{1,\ldots,n\},\qquad R_i\text{는 random subset},\qquad \lvert R_i\rvert=r
```

random shortcut은 local ring의 긴 graph distance를 줄이고, 여러 layer에서 정보가 빠르게 섞이게 한다. 실제 구현은 token 단위보다 block 단위 random pattern을 사용해 kernel regularity를 확보한다.

random은 매 forward마다 완전히 새로 뽑는 dropout과 같지 않다. 재현성과 layout을 위해 head/layer별 pattern을 미리 구성하거나 seed를 고정한다.

### 3. Global attention

소수 global token은 모든 token과 양방향 연결된다.

```text
G token -> all tokens
all tokens -> G token
```

global token이 한 layer에서 문서 전체를 모으고 다음 layer에서 다시 전달할 수 있어 graph diameter를 크게 줄인다. 이론과 empirical ablation 모두에서 가장 중요한 구성이다.

## BigBird-ITC와 BigBird-ETC

논문은 global token을 구성하는 두 방식을 구분한다.

### Internal Transformer Construction (ITC)

기존 input token 중 일부를 global로 지정한다. `[CLS]` 같은 special token이나 task token을 사용할 수 있다. sequence length를 추가하지 않고 pretrained BERT 구조와 맞추기 쉽다.

### Extended Transformer Construction (ETC)

원래 sequence 밖에 별도 global token을 추가한다. global memory의 역할이 명확하고 task 구조를 담기 쉽지만 input representation과 parameter가 확장된다.

두 방식 모두 local/random/global 연결의 기본 원리는 같다.

## Tensor와 block-sparse 구조

Dense attention의 score는 `[B,H,N,N]`이다. BigBird는 sequence를 block size `b`로 나누고 query block마다 다음 key block만 모은다.

```text
selected blocks = local neighbor blocks
                + random blocks
                + global blocks

score shape per query block:
[B,H,b,(num_selected_blocks*b + g)]
```

논리 mask만 sparse하게 만들고 dense score를 계산하면 이득이 없다. 실제 구현은 selected key/value block을 gather하거나 block-sparse matmul을 사용해야 한다.

## 계산 복잡도

Token 기준 edge 수를 단순화하면

```math
\begin{aligned}
\text{local edges}&:\ nw,\\
\text{random edges}&:\ nr,\\
\text{global edges}&:\ 2ng,\\
\text{total}&:\ O\!\left(n(w+r+g)\right).
\end{aligned}
```

global token끼리의 `g²` 항도 있지만 `g << n`이면 작다. 실제 비용은 block rounding, duplicated edge, padding, gather에 의해 이론치보다 커질 수 있다.

## Graph 관점의 정보 전달

Local window만 있는 graph에서 sequence 양 끝 사이 최단 경로는 대략 `O(n/w)` layer다. random edge를 추가하면 높은 확률로 diameter가 크게 줄어든다. global hub가 있으면 일반 token A의 정보가

```text
A -> global -> B
```

처럼 매우 짧은 경로로 B에 도달할 수 있다. attention update의 방향과 layer timing을 고려하면 실제 전달에는 여러 sublayer가 필요하지만, local-only보다 훨씬 짧다.

이 세 요소는 역할이 겹치지 않는다. local은 세부 보존, random은 다양한 shortcut, global은 안정적인 전역 bottleneck이다.

## 이론 1: Universal approximation

논문은 sparse graph가 star graph를 포함하고 global token이 모든 token과 연결되는 조건에서 sequence-to-sequence function의 universal approximator가 될 수 있음을 보인다. 직관적으로 global token이 전체 input을 모으고 필요한 계산을 거쳐 각 위치에 결과를 전달할 수 있기 때문이다.

중요한 해석의 경계는 다음과 같다.

- 존재 정리이지 학습이 그 parameter를 쉽게 찾는다는 보장이 아니다.
- 필요한 depth, width, precision이 실제 model budget 안에 있다는 보장이 아니다.
- global bottleneck을 통한 표현은 full attention의 직접 pair interaction과 다르다.

따라서 “sparse attention도 이론상 모든 함수를 표현할 수 있다”는 주장과 “같은 12 layer에서 full attention과 동등하다”는 주장을 혼동하면 안 된다.

## 이론 2: Turing completeness

BigBird는 positional information과 arbitrary precision 같은 표준 이론 조건 아래에서 sparse Transformer가 Turing complete함을 보인다. global token과 sparse routing으로 필요한 memory access와 계산을 모사한다.

이 역시 실제 floating-point network의 reliability나 sample efficiency를 말하는 결과는 아니다. sparse pattern이 원리적으로 Transformer의 계산 능력을 완전히 파괴하지 않는다는 이론적 안전망으로 읽는 것이 적절하다.

## 이론 3: Sparse attention의 한계

논문은 각 row에서 가장 멀리 떨어진 vector를 찾는 종류의 문제를 예로 들어, full attention은 한 layer에서 해결할 수 있지만 선형 edge 수를 가진 sparse attention은 Orthogonal Vectors Conjecture 아래 충분한 layer가 필요하다는 하한을 논의한다.

즉 논문 자체가 다음 trade-off를 인정한다.

```text
fewer edges per layer
<->
some interactions require greater depth
```

Universal approximation 결과가 finite-depth efficiency 차이를 없애지는 않는다.

## Pretraining과 길이 확장

BigBird는 BERT/RoBERTa 계열 checkpoint를 기반으로 sparse attention을 적용하고 긴 sequence로 추가 pretraining한다. local/random/global mask를 사용해 최대 수천 token을 처리한다. Global token은 `[CLS]` 또는 별도 token으로 구성하며 task fine-tuning에서 routing 역할을 맡는다.

Random+window만 사용한 ablation은 512 길이에서도 BERT보다 낮은 성능을 보였고, global token을 추가해야 성능이 회복된다. sparse edge 수만 맞추는 것보다 graph의 hub structure가 중요하다.

## 실험 결과: Question answering

논문의 dev 결과 중 BigBird-ETC를 요약하면 다음과 같다.

| Task | Metric | BigBird-ETC |
| --- | --- | ---: |
| HotpotQA | Joint F1 | 67.8 |
| Natural Questions | Long/Short Answer | 73.9 / 54.9 |
| TriviaQA | F1 | 78.7 |
| WikiHop | Accuracy | 75.9 |

Test 결과에서는 다음 수준을 보고한다.

| Task | BigBird-ETC test |
| --- | ---: |
| HotpotQA Joint F1 | 73.6 |
| Natural Questions Long/Short | 77.8 / 57.9 |
| TriviaQA F1 / verified | 84.5 / 92.4 |
| WikiHop Accuracy | 82.3 |

긴 document evidence를 한 번에 입력하고 global token을 question/task representation으로 사용한 것이 multi-hop QA에 효과적이었다.

## 실험 결과: Summarization

Pegasus에 BigBird encoder를 결합한 결과는 다음과 같다.

| Dataset | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| arXiv | 46.63 | 19.02 | 41.77 |
| PubMed | 46.32 | 20.65 | 42.33 |
| BigPatent | 60.64 | 42.46 | 50.01 |

긴 source를 truncate하지 않고 더 많은 문맥을 encoder에 제공하는 장점이 나타난다. 다만 decoder와 cross-attention 비용은 별도로 남는다.

## 실험 결과: Genomics

DNA sequence는 자연어보다 훨씬 길고 local motif와 장거리 regulation이 함께 존재한다. BigBird는 이 구조에 잘 맞는 실험 사례를 제시한다.

| Task | 결과 |
| --- | ---: |
| MLM BPC | 1.12 |
| BERT 512 baseline MLM BPC | 1.23 |
| Promoter classification F1 | 99.9 |
| Chromatin profile mean AUC/metric | 88.7 |

긴 context를 직접 모델링해 promoter와 chromatin task에서 강한 성능을 보였다. 특히 논문이 NLP 밖의 biological sequence까지 sparse graph를 검증했다는 점이 의미 있다.

## 실험을 해석할 때 주의할 점

BigBird의 성능 개선은 sparse mask만의 효과가 아니다. 더 긴 input, 추가 pretraining, task-specific global token, base model 크기와 training recipe가 함께 작용한다. 공정한 비교에서는 동일한 context와 compute budget을 맞춰야 한다.

또한 random edge가 이론상 expander-like property를 제공해도 block 단위 구현과 작은 head 수에서는 이상적인 random graph와 다르다. seed와 pattern별 variance를 보고해야 한다.

## 장점과 기여

- local, random, global edge를 결합한 범용 sparse attention template을 제시했다.
- `O(n)` edge로 긴 document와 genomic sequence를 처리했다.
- graph diameter와 hub routing의 직관을 모델 설계에 연결했다.
- universal approximation, Turing completeness, sparse 한계 하한을 함께 논의했다.
- QA, summarization, genomics에서 폭넓은 실험을 수행했다.

## 한계와 비판적 관점

### 1. Global bottleneck 의존

Ablation과 이론 모두 global token이 핵심이다. global representation이 너무 적거나 task와 맞지 않으면 정보가 한정된 hub에 몰린다.

### 2. Random pattern의 재현성과 구현

seed, head, layer, block마다 pattern이 달라 결과가 변할 수 있다. irregular memory access는 dense kernel보다 hardware utilization이 낮다.

### 3. Finite-depth 표현력 차이

이론상 universal하더라도 직접 edge가 없는 pair는 여러 layer를 거친다. retrieval처럼 정확한 pair matching이 중요한 task에서 품질 차이가 날 수 있다.

### 4. Global token policy

ITC/ETC 선택과 어떤 token을 global로 둘지 task-specific 설계가 필요하다. 완전히 자동인 long-context solution은 아니다.

### 5. Modern baseline과 재비교 필요

FlashAttention류 exact kernel은 중간 길이에서 dense attention을 크게 가속한다. BigBird의 우위는 매우 긴 길이, memory cap, 지원되는 block-sparse kernel 조건에서 다시 측정해야 한다.

## 관련 방법과 비교

| 방법 | Local | Random | Global | 복잡도 |
| --- | --- | --- | --- | ---: |
| Sparse Transformer | 구조화 local/stride | 없음 | summary 위치 | `O(n sqrt n)` |
| Longformer | window/dilation | 없음 | task token | `O(n(w+g))` |
| BigBird | window | 있음 | ITC/ETC | `O(n(w+r+g))` |

BigBird는 Longformer의 local+global 구성에 random shortcut과 더 강한 이론 분석을 더한 것으로 볼 수 있다.

## 구현 체크리스트

- random block mask가 causal/padding 규칙을 위반하지 않는가?
- local·random·global edge의 중복을 제거하거나 올바르게 처리하는가?
- global attention이 양방향인가?
- seed와 layer/head별 pattern을 재현할 수 있는가?
- dense score 후 mask가 아닌 block-sparse 계산인가?
- input length가 block size로 나누어지지 않을 때 padding이 정확한가?
- global token 수와 random block 수별 quality/latency curve를 측정했는가?

## 온디바이스·비전 관점

Local window와 고정 global token은 정적 graph라 온디바이스에 비교적 적합하지만 random block은 gather pattern을 복잡하게 만든다. 배포 시 random pattern을 고정하고 block size를 NPU tile에 맞추면 compile 가능성을 높일 수 있다.

Vision patch에서는 local edge가 spatial neighborhood를, global token이 class/object query를, random edge가 멀리 떨어진 region shortcut을 담당할 수 있다. 그러나 2D locality를 보존하도록 block flatten 순서를 신중히 정해야 한다. 실제 latency는 sparse operator 지원 여부가 결정한다.

## 최종 평가

BigBird는 긴 sequence attention을 하나의 sparse mask가 아니라 **small-world graph 설계 문제**로 정리한 논문이다. local detail, random shortcut, global hub의 역할을 분리했고, 이론적으로 표현 가능성을 보존하면서 QA·summarization·genomics에서 실용성을 확인했다. 다만 global token과 custom sparse implementation에 크게 의존하며, 이론적 universal property가 finite-depth 품질을 보장하지는 않는다. Longformer와 함께 구조화 sparse attention의 대표 기준점이며, 특히 graph topology가 attention 품질을 결정한다는 사실을 가장 명확하게 드러낸다.

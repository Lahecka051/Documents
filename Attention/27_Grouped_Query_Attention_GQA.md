# 27. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

## 논문 정보

- 제목: **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints**
- 저자: Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai
- 발표: EMNLP 2023
- 핵심 키워드: grouped-query attention, multi-query attention, KV cache, uptraining, checkpoint conversion

## 한눈에 보는 요약

Grouped-Query Attention(GQA)은 MHA와 MQA 사이를 일반화한다. Query head `H`개를 `G`개 group으로 나누고, 같은 group의 query head가 key/value head 하나를 공유한다.

```math
\begin{aligned}
\mathrm{MHA}:&\quad G=H,\\
\mathrm{GQA}:&\quad 1<G<H,\\
\mathrm{MQA}:&\quad G=1.
\end{aligned}
```

KV cache는 MHA 대비 `G/H`로 줄고, MQA보다는 `G`배 크다. `G`를 조절해 품질과 decode bandwidth 사이의 연속적인 trade-off를 선택할 수 있다.

논문의 두 번째 공헌은 이미 학습된 MHA checkpoint를 처음부터 다시 학습하지 않고 GQA/MQA로 바꾸는 **uptraining recipe**다. 같은 group에 속한 MHA key/value projection head를 평균내어 초기화하고, 원 pretraining compute의 약 5%만 추가 학습한다.

T5-XXL에서 GQA-8은 MHA-XXL에 가까운 품질을 유지하면서 inference time `1.51s → 0.28s`로 줄였고, MQA의 `0.24s`와 거의 비슷했다.

<p align="center"><img src="https://github.com/user-attachments/assets/dc36ea6d-769c-4ca3-b60e-c0293edd94c9" alt="MHA GQA MQA head grouping" width="760"></p>
<p align="center"><sub>원 논문 Figure 2 패널 재배치 — MHA·GQA·MQA의 query/KV head 공유 구조</sub></p>

## MQA가 남긴 문제

MQA는 모든 query head가 key/value 하나를 공유해 KV cache와 bandwidth를 크게 줄인다. 그러나 다음 문제가 있다.

- MHA보다 품질이 하락할 수 있다.
- training stability가 나빠질 수 있다.
- 이미 비싼 MHA checkpoint가 있어도 MQA model을 처음부터 새로 학습해야 할 수 있다.

GQA는 KV head 수를 1보다 크게 두어 capacity를 일부 회복하고, checkpoint conversion으로 재학습 비용을 줄인다.

## GQA 수식

Query head 수를 `H`, KV group 수를 `G`, group당 query head 수를 `H/G`라 하자. Query head `h`가 속하는 group index는

```math
g(h)=\left\lfloor\frac{h}{H/G}\right\rfloor
```

처럼 정할 수 있다. 각 head output은

```math
\begin{aligned}
q_h&=W_h^Qx,
&k_g&=W_g^Kx,
&v_g&=W_g^Vx,\\
o_h&=\operatorname{softmax}\!\left(\frac{q_hK_{g(h)}^{\top}}{\sqrt{d_h}}\right)V_{g(h)}.
\end{aligned}
```

이다. Query와 output projection은 head별로 유지되고 K/V만 group 안에서 공유된다.

## Tensor shape와 cache

```text
Q       : [B,N,H,D_h]
K,V     : [B,N,G,D_h]
KV cache: 2 × B × N × G × D_h
```

MHA cache는 `2BNHD_h`, MQA는 `2BND_h`이므로 비율은 다음과 같다.

```math
\frac{\mathrm{GQA}}{\mathrm{MHA}}=\frac{G}{H},
\qquad
\frac{\mathrm{GQA}}{\mathrm{MQA}}=G
```

예를 들어 query head 64개, KV group 8개면 MHA cache의 1/8이고 각 KV head를 query 8개가 공유한다.

## 왜 GQA가 MQA에 가까운 속도를 낼 수 있는가

Decode 전체 시간에는 KV cache read만 있는 것이 아니다. Model weight, projection, FFN, communication도 읽고 계산한다. MQA에서 GQA-8로 cache가 8배 늘어도 전체 memory traffic의 일부만 늘기 때문에 end-to-end latency는 modest하게 증가할 수 있다.

특히 큰 model은 weight와 다른 activation 비용이 크므로 group 1에서 8로 늘릴 때 품질 개선에 비해 시간 손실이 작다. 반대로 아주 긴 context나 큰 batch에서는 KV read 비중이 커져 group 수의 latency 영향도 커질 수 있다.

## Uptraining: MHA checkpoint 변환

### 1. Head grouping

MHA의 `H`개 KV head를 `G`개의 contiguous group으로 나눈다.

### 2. Mean pooling

Group 안의 key/value projection weight를 평균낸다.

```math
\begin{aligned}
W_g^K&=\operatorname*{mean}_{h\in\mathrm{group}\ g}W_h^K,\\
W_g^V&=\operatorname*{mean}_{h\in\mathrm{group}\ g}W_h^V.
\end{aligned}
```

MQA는 모든 `H` head의 평균 하나를 사용한다. Query와 output projection, 나머지 network weight는 checkpoint에서 그대로 가져온다.

### 3. 추가 pretraining

변환 직후에는 서로 다른 KV head 정보를 평균내어 손실이 생기므로 원 pretraining schedule/data로 짧게 적응시킨다. 논문의 기본 비율 `α=0.05`는 original pretraining step의 5%다. T5-XXL 기준 약 600 TPUv3 chip-days로, 처음부터 재학습하는 것보다 훨씬 작지만 절대 비용은 여전히 크다.

## 왜 평균 초기화인가

논문은 다음 초기화를 비교한다.

- group head 평균
- 첫 번째 head 선택
- random initialization

Mean pooling이 가장 안정적이고 좋은 성능을 보였다. 여러 MHA head의 공통 subspace를 보존하면서 scale도 유지하기 때문이다. 첫 head만 고르면 나머지 정보를 버리고, random은 attention path를 사실상 새로 학습해야 한다.

그러나 서로 다른 head가 전문화되어 있다면 단순 평균이 의미를 상쇄할 수 있다. Uptraining이 이 초기 손실을 다시 분화시키는 역할을 한다.

## 적용 범위

논문은 T5.1.1 encoder-decoder model에서 decoder self-attention과 cross-attention에 GQA/MQA를 적용한다. Encoder self-attention에는 적용하지 않는다.

이유는 decoder의 incremental inference와 cross-attention K/V 반복 read가 bandwidth 병목이기 때문이다. Encoder는 input 전체를 병렬 계산해 GQA의 decode 이득이 작다.

## 실험 설정

- T5 Large와 T5 XXL checkpoint
- MQA 및 GQA-8로 변환
- 원 pretraining compute의 5% uptraining
- Summarization: CNN/DailyMail, arXiv, PubMed, MediaSum, MultiNews
- Translation: WMT14 En-De
- QA: TriviaQA

긴 summarization input은 2,048, output은 512로 설정한다. 생성 task를 선택한 것은 KV-cache inference 품질과 속도를 직접 평가하기 위해서다.

## 주요 결과

| 모델 | Inference time/sample | CNN R1 | arXiv R1 | PubMed R1 | MediaSum R1 | MultiNews R1 | WMT BLEU | TriviaQA F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MHA-Large | 0.37s | 46.0 | 42.9 | 44.6 | 46.2 | 35.5 | 27.7 | 78.2 |
| MHA-XXL | 1.51s | **47.2** | **43.8** | **45.6** | 47.5 | **36.4** | 28.4 | **81.9** |
| MQA-XXL | **0.24s** | 46.6 | 43.0 | 45.0 | 46.9 | 36.1 | **28.5** | 81.3 |
| GQA-8-XXL | 0.28s | 47.1 | 43.5 | 45.4 | **47.7** | 36.3 | 28.4 | 81.6 |

GQA-8은 MHA-XXL보다 약 5.4배 빠르고, MQA보다 0.04초만 느리면서 대부분 task에서 MQA보다 품질이 높다. MHA-XXL과도 매우 가깝다.

## Uptraining 비율 ablation

변환 직후 GQA는 이미 합리적인 성능을 보이지만 MQA는 더 큰 하락과 불안정성이 있다. Uptraining을 늘리면 둘 다 개선되며 5% 이후 10%로 갈 때는 diminishing return이 나타난다.

이는 GQA의 group별 평균이 원 MHA 구조를 더 잘 보존한다는 의미다. MQA는 모든 head를 하나로 압축하므로 적응해야 할 구조 변화가 더 크다.

## Group 수 ablation

`G=1` MQA에서 group 수를 늘릴수록 품질은 MHA에 접근한다. Inference time은 처음 1→8 증가에서 modest하게 늘고, 더 많은 group에서 점차 MHA 비용에 가까워진다.

논문은 8 group을 좋은 절충으로 선택했지만 보편적 상수는 아니다.

```text
optimal G depends on:
H, model width, context length, batch size,
hardware bandwidth, tensor-parallel layout, target quality
```

## Tensor parallel 관점

KV head 수가 tensor-parallel shard 수보다 작거나 나누어지지 않으면 일부 device가 KV head를 복제해야 할 수 있다. MQA는 head 하나를 모든 shard에 복제해 compute/memory waste가 생길 수 있다. GQA는 group 수를 shard 수와 맞춰 각 device가 하나 이상의 KV group을 소유하게 만들 수 있다.

따라서 GQA는 quality뿐 아니라 distributed partitioning의 실용성에서도 MQA보다 유연하다.

## 장점과 기여

- KV head 수를 연속적인 hyperparameter로 만들어 MHA와 MQA를 통합했다.
- Cache, bandwidth, capacity trade-off를 간단한 group 수로 제어한다.
- 기존 MHA checkpoint를 mean pooling과 5% uptraining으로 변환하는 recipe를 제시했다.
- T5-XXL에서 MHA 품질과 MQA 속도에 가까운 결과를 보였다.
- 현대 decoder LLM의 사실상 표준 attention 형태가 되었다.

## 한계와 비판적 관점

### 1. 5%도 절대적으로 비싸다

T5-XXL에서 약 600 TPUv3 chip-days다. “처음부터 학습하지 않는다”는 장점은 크지만 작은 팀에게 무료 변환은 아니다.

### 2. Mean pooling이 head specialization을 손상할 수 있다

서로 다른 역할의 KV head를 평균내면 정보가 상쇄될 수 있다. Group 구성도 contiguous head가 최선이라는 보장은 없다.

### 3. 품질 결과의 범위

주로 encoder-decoder T5와 생성 benchmark를 평가한다. Decoder-only 초장문 retrieval, code, multimodal task에는 optimal group이 다를 수 있다.

### 4. Cache는 MQA보다 크다

GQA-8은 MQA의 8배 KV cache다. 매우 긴 context 또는 memory-constrained device에서는 작은 latency 차이보다 capacity 차이가 더 중요할 수 있다.

### 5. 전용 kernel과 layout

Query head를 KV group에 효율적으로 매핑해야 한다. 단순 repeat-interleave로 K/V를 H개로 복제하면 bandwidth 이득이 줄어든다.

## 구현 체크리스트

- `H % G == 0`이고 query-to-group mapping이 정확한가?
- MHA checkpoint의 K/V weight를 올바른 axis로 group mean하는가?
- Bias와 quantization scale도 같은 grouping 규칙으로 변환하는가?
- K/V를 query head 수로 물리적 복제하지 않는가?
- Tensor-parallel shard 수와 KV group 수가 맞는가?
- Prefill, decode, memory capacity를 별도로 측정하는가?
- 변환 직후와 uptraining 후 품질을 모두 기록하는가?

## 온디바이스 관점

GQA는 온디바이스에서 quality와 RAM 사이를 조절하기 좋다. MQA보다 몇 개 KV head를 허용해 품질을 회복하면서 MHA보다 cache와 DRAM traffic을 크게 줄인다. Group 수를 NPU core 수나 vector tile에 맞추면 정적 compile도 쉽다.

실제 선택에서는 최대 context에서 cache가 RAM 안에 들어가는지 먼저 확인하고, 그 범위에서 가장 큰 `G`를 사용해 품질을 확보할 수 있다. 기존 MHA model을 배포용 GQA로 변환할 때는 calibration만으로 충분하지 않고 짧은 uptraining이 필요하다는 점도 중요하다.

## 최종 평가

GQA의 강점은 새로운 복잡한 attention 수식이 아니라 **KV head 수를 시스템 budget에 맞춰 조절하고, 이미 학습된 MHA 자산을 저비용으로 전환하는 방법**을 제공한 데 있다. GQA-8은 T5-XXL에서 MHA 품질에 거의 도달하면서 MQA 속도에 가까웠다. 이후 LLM이 GQA를 널리 채택한 것은 quality·cache·tensor parallelism을 동시에 타협하기 쉬운 구조이기 때문이다.

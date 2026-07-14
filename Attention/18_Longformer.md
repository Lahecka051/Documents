# 18. Longformer: The Long-Document Transformer

## 논문 정보

- 제목: **Longformer: The Long-Document Transformer**
- 저자: Iz Beltagy, Matthew E. Peters, Arman Cohan
- 발표: 2020
- 핵심 키워드: sliding-window attention, dilated attention, global attention, long document, LED

## 한눈에 보는 요약

Longformer는 긴 문서의 모든 token pair를 연결하는 대신 세 종류의 attention을 결합한다.

- **Sliding window**: 각 token이 주변 `w`개 token만 본다.
- **Dilated window**: 일부 head가 간격을 두고 더 넓은 범위를 본다.
- **Global attention**: task에 중요한 소수 token이 전체 sequence와 양방향으로 연결된다.

local attention의 비용은 `O(nw)`, global token이 `g`개일 때 전체 비용은 `O(n(w+g))`이다. `w,g`를 sequence length에 비례하지 않게 유지하면 `n`에 대해 선형이다.

핵심 설계는 단순 local window만 쓰지 않는다는 점이다. local edge는 문서의 세부 문맥을 처리하고, `[CLS]`, question token, answer candidate처럼 task에 따라 선택한 global token이 장거리 정보 routing을 담당한다. Longformer는 이 구조로 RoBERTa의 512-token 제한을 4,096 token까지 확장하고, encoder-decoder 변형인 LED로 16K 길이 summarization도 수행한다.

<p align="center"><img src="https://github.com/user-attachments/assets/f16d9552-2f27-429b-b2d5-582fd3769dc6" alt="Longformer attention patterns" width="680"></p>
<p align="center"><sub>원 논문 Figure 2 패널 재배치 — full·sliding·dilated·global attention pattern 비교</sub></p>

## 문제의식

BERT/RoBERTa의 full self-attention은 길이 `n`에서 `O(n²)`이므로 512 token을 넘어가면 memory와 compute가 빠르게 증가한다. 긴 문서를 512-token chunk로 잘라 독립 처리하면 다음 정보가 손실된다.

- 문서 앞의 정의와 뒤의 결론 연결
- 여러 문단에 흩어진 multi-hop evidence
- 질문과 긴 context 전체의 상호작용
- 장문 summarization에서 문서 전역 구조

Longformer는 convolution처럼 local receptive field를 쌓되, attention의 content-dependent weighting을 유지하고, 소수의 global route로 문서 전체를 연결한다.

## Sliding-window attention

각 token `i`가 좌우 `w/2` 범위의 token만 본다고 하자.

```math
A_i=\left\{j:\lvert i-j\rvert\le\frac{w}{2}\right\}
```

각 query가 `w`개의 key만 보므로 layer당 score 수는 `n×w`다.

```text
Q,K,V          : [B,H,N,D]
local scores   : [B,H,N,W]
local output   : [B,H,N,D]
```

한 layer의 receptive field는 `w`지만 `L`개 layer를 쌓으면 정보가 이웃을 따라 전달되어 대략 `Lw` 범위까지 확장된다. full attention처럼 한 layer에서 문서 끝까지 직접 연결되지는 않지만, 자연어의 많은 dependency가 local하다는 inductive bias를 활용한다.

### 경계 처리

문서 시작과 끝에서는 window 일부가 범위를 벗어난다. padding 위치를 score에서 mask하고, 실제 token이 없는 곳의 softmax 확률이 0이 되도록 해야 한다. causal variant라면 오른쪽 절반도 mask한다.

## Dilated sliding-window attention

일부 head는 연속된 이웃 대신 dilation `d` 간격으로 token을 본다.

```math
A_i^{(d)}=\left\{i+td:-\frac{w}{2}\le t\le\frac{w}{2}\right\}
```

직접 계산하는 key 수는 여전히 `w` 정도지만 한 layer에서 커버하는 물리적 범위는 `d×w`로 커진다. 저자들은 local head와 dilated head를 섞어 근접 detail과 장거리 전달을 함께 확보한다.

모든 head에 큰 dilation을 주면 가까운 token을 건너뛰는 aliasing이 생길 수 있다. 논문 실험에서도 일부 head만 dilation을 쓰는 구성이 더 낫다.

## Global attention

local window만으로 먼 위치 정보가 만나려면 여러 layer가 필요하다. Longformer는 task에 중요한 위치 집합 `G`를 지정해 full connection을 추가한다.

```text
global token i ∈ G:
  i attends to every token j

ordinary token j:
  j attends to every global token i ∈ G
```

즉 연결은 대칭이다. global token만 전체를 보고 일반 token이 global token을 보지 못하면, 수집한 전역 정보가 다시 문서로 퍼지기 어렵다.

Global attention은 local attention과 별도의 projection을 사용한다.

```text
local  : Q,K,V
global : Q_g,K_g,V_g
```

pretrained RoBERTa에서 초기화할 때 global projection은 대응하는 local projection weight를 복사한다. 이후 fine-tuning에서 task-specific global role을 학습한다.

## 어떤 token을 global로 지정하는가

Global attention은 학습으로 위치 집합 자체를 발견하지 않는다. task가 지정한다.

| Task | Global token 예시 |
| --- | --- |
| Classification | `[CLS]` |
| Question answering | question의 모든 token |
| Multiple choice | question·candidate token |
| Coreference | potential mention representation |
| Summarization encoder | task 설정에 따른 special token 또는 제한적 global |

따라서 global-token policy는 hyperparameter가 아니라 model architecture의 일부다. 너무 적으면 정보 bottleneck이 되고, 너무 많으면 비용이 `O(ng)`로 증가한다.

## 전체 복잡도

local window score는 `nw`, global-to-all 및 all-to-global edge는 대략 `2ng`다.

```math
\text{time/memory}\approx O(nw+ng)=O\!\left(n(w+g)\right)
```

`w=512`, `g`가 수십 개라면 4K token에서 full attention의 16M score 대신 수백만 개 edge만 계산한다. 하지만 `g`가 문서 길이에 비례하면 다시 quadratic에 가까워진다.

## 구현 방식

논문은 세 구현을 비교한다.

### Loop implementation

각 token window를 반복 처리하므로 개념은 단순하지만 매우 느리다.

### Chunked implementation

sequence를 overlap chunk로 나누고 dense matrix multiplication을 사용한다. non-dilated window에 적합하고 일반 GEMM을 활용할 수 있지만, mask될 일부 score도 계산해 메모리를 더 쓸 수 있다.

### TVM custom CUDA kernel

dilation과 global attention을 포함한 pattern을 직접 실행한다. 가장 일반적이지만 특정 hardware/compiler에 대한 kernel 최적화가 필요하다.

이 구분은 중요하다. `O(nw)` 알고리즘이라고 해서 Python slicing으로 구현한 코드가 빠른 것은 아니다. window tile이 contiguous하고 fused softmax를 사용할 수 있어야 실제 speedup이 난다.

## Pretraining: RoBERTa를 Longformer로 확장

저자들은 RoBERTa checkpoint에서 시작해 최대 길이를 4,096으로 늘린다.

### Position embedding 초기화

RoBERTa의 512개 learned position embedding을 반복 복사해 4,096 위치를 채운다.

```text
position 0..511      <- original embeddings
position 512..1023   <- repeated copy
...
```

처음에는 서로 다른 absolute position이 같은 vector를 공유하지만, long-document MLM pretraining에서 각 위치가 분화된다.

### Attention 초기화

local projection은 RoBERTa weight를 그대로 사용한다. global projection도 local weight를 복사해 시작한다. window 크기 512를 사용하면 각 token이 보는 key 수가 RoBERTa 512-token attention과 비슷해 layer당 계산 규모를 관리할 수 있다.

### 추가 MLM pretraining

긴 문서 corpus로 약 65K update를 수행한다. MLM validation bits-per-character는 다음처럼 개선된다.

| 모델/시점 | BPC |
| --- | ---: |
| RoBERTa-base | 1.846 |
| 길이 확장 직후 copied initialization | 1.957 |
| 2K update | 1.753 |
| 65K update | **1.705** |

단순 position 복사 직후에는 성능이 나쁘지만, 짧은 adaptation만으로 원본을 넘고 장기 pretraining에서 더 개선된다. context extension에는 weight copy뿐 아니라 적응 학습이 필요하다는 증거다.

## Character-level language modeling

Text8과 Enwik8에서 sequence length를 단계적으로 늘리며 학습한다. 최대 32K character context를 사용하고, window와 dilation을 layer별로 조정한다.

Ablation의 핵심은 다음과 같다.

- 모든 layer에 작은 동일 window를 쓰는 것보다 상위 layer로 갈수록 window를 키우는 편이 좋다.
- 일부 head에 dilation을 주면 receptive field가 커져 성능이 개선된다.
- local structure와 long-range route의 혼합이 중요하다.

논문 당시 두 dataset에서 경쟁력 있는 또는 SOTA 수준의 BPC를 기록하며 local attention이 단순 encoder fine-tuning뿐 아니라 autoregressive modeling에도 작동함을 보였다.

## Long-document downstream 결과

Longformer-base와 RoBERTa-base의 dev 결과를 논문 표에서 요약하면 다음과 같다.

| Task | RoBERTa-base | Longformer-base |
| --- | ---: | ---: |
| WikiHop | 72.4 | **75.0** |
| TriviaQA | 74.3 | **75.2** |
| HotpotQA | 63.5 | **64.4** |
| OntoNotes coreference | 78.4 | **78.6** |
| IMDB | 95.3 | **95.7** |
| Hyperpartisan | 87.4 | **94.8** |

긴 evidence를 직접 입력할 수 있는 WikiHop, HotpotQA, TriviaQA와 장문 classification에서 이득이 나타난다. OntoNotes처럼 긴 context 이득이 제한적인 task에서는 차이도 작다.

### WikiHop global-attention ablation

Global attention과 별도 projection을 제거한 설정은 약 `65.5`로 크게 하락한다. full 구성은 dev/test 약 `73.8/75.0` 수준이다. local window만 길게 쌓는 것보다 task token을 global hub로 만드는 것이 핵심임을 보여준다.

## LED: Longformer Encoder-Decoder

Longformer Encoder-Decoder는 BART의 encoder self-attention을 Longformer sparse attention으로 바꾸고 decoder는 기존 causal full attention을 유지한다.

```text
encoder self-attention : local + global sparse, up to 16K
decoder self-attention : causal full attention
cross-attention         : decoder -> all encoder states
```

BART의 position embedding을 반복 복사해 16K까지 확장하고 summarization에 fine-tune한다. arXiv summarization에서 ROUGE-1/2/L `46.63 / 19.62 / 41.83`을 기록해 긴 원문을 자르지 않고 처리하는 장점을 보였다.

## 장점과 기여

- 구현 가능한 local-window pattern으로 attention 비용을 `O(nw)`로 낮췄다.
- dilation으로 edge 수를 늘리지 않고 receptive field를 확장했다.
- 소수 global token을 통해 task-specific 전역 routing을 제공했다.
- RoBERTa/BART checkpoint를 긴 context 모델로 확장하는 실용적 초기화와 pretraining 절차를 제시했다.
- QA, classification, coreference, summarization까지 폭넓게 검증했다.

## 한계와 비판적 관점

### 1. Global token 선택이 task knowledge를 요구한다

질문 token, `[CLS]`, candidate 등 무엇을 global로 둘지 사람이 정한다. 잘못된 policy는 필요한 long-range path를 막거나 비용을 늘린다.

### 2. 일반 token 간 장거리 관계는 multi-hop이다

global token을 통하지 않는 먼 두 token은 여러 layer의 local window를 따라 정보를 전달한다. finite depth에서는 dense attention과 같은 직접 상호작용이 아니다.

### 3. Position embedding 반복은 추가 학습이 필요하다

초기에는 512 간격 위치가 같은 embedding을 공유한다. 논문 결과에서도 확장 직후 MLM이 악화되므로 adaptation 비용을 무시할 수 없다.

### 4. Sparse kernel 지원이 필요하다

일반 dense library만으로는 dilation과 global edge를 효율적으로 처리하기 어렵다. accelerator별 custom kernel 또는 compiler 지원이 실제 성능을 좌우한다.

### 5. 생성 decode의 병목은 별개다

LED는 긴 encoder input을 효율화하지만 decoder self-attention과 cross-attention 비용은 남는다. token-by-token generation latency가 local encoder 덕분에 같은 비율로 줄지는 않는다.

## 후속 연구와의 연결

- **BigBird**는 local+global에 random edge를 추가하고 graph 이론을 강화한다.
- **LED**는 긴 문서 summarization의 대표 baseline이 된다.
- **Window attention 계열**은 Swin Transformer 등 vision model의 local inductive bias와 연결된다.
- **Modern long-context LLM**은 sliding-window layer와 occasional global/full layer를 혼합하는 설계를 사용한다.

## 구현 체크리스트

- local window 폭이 한쪽 기준인지 전체 기준인지 명확한가?
- padding과 causal boundary mask가 정확한가?
- global connection이 양방향으로 들어가는가?
- local/global projection이 분리되어 있고 올바르게 초기화되는가?
- global token 수가 input마다 달라질 때 batching이 효율적인가?
- dense mask 후 계산하는 것이 아니라 sparse tile만 실행하는가?
- position embedding 확장 후 충분한 adaptation을 했는가?

## 온디바이스·비전 관점

Sliding window는 index pattern이 정적이고 memory access가 규칙적이어서 LSH나 random sparsity보다 온디바이스에 적합하다. 1D 문서뿐 아니라 2D patch grid에서도 local window와 몇 개의 global token을 조합할 수 있다. 다만 2D에서는 flatten 경계가 인접하지 않은 patch를 연결하지 않도록 window partition을 설계해야 한다.

NPU가 window attention primitive를 지원하면 peak memory와 energy를 줄일 수 있다. 지원이 없을 때는 unfold/im2col 과정이 큰 임시 tensor를 만들 수 있으므로 실제 memory trace를 확인해야 한다. global token 수를 고정하면 static shape compilation에도 유리하다.

## 최종 평가

Longformer는 긴 문서의 attention을 **local processing과 global routing의 역할 분담**으로 해결한 실용적인 모델이다. local window는 비용을 선형으로 만들고, dilation은 receptive field를 넓히며, task가 지정한 global token은 multi-hop 지연을 줄인다. 완전한 범용 sparse attention이라기보다 task-aware architecture이고 custom kernel에 의존하지만, pretrained encoder를 긴 context로 확장해 실제 NLP task에서 이득을 보여준 점이 강하다. 이후 long-context 설계에서 반복되는 `local + sparse global` 패턴의 대표적인 기준점이다.

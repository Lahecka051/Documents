# 16. Generating Long Sequences with Sparse Transformers

## 논문 정보

- 원본 파일: `C:\Users\lkm\Documents\Codex\Embbeded_OnDevice_ComputerVision\attention_papers_pdf\16_Sparse_Transformer.pdf`
- 제목: Generating Long Sequences with Sparse Transformers
- 저자: Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever
- 발표: 2019
- 주제: Sparse Transformer의 sparse attention graph
- 핵심 키워드: Sparse Transformer, factorized attention, strided attention, long sequence generation

## 한눈에 보는 요약

Sparse Transformer는 모든 위치 쌍을 보지 않고, 설계된 sparse pattern으로 긴 sequence를 생성한다.

Local attention과 strided attention을 조합해 계산량을 줄이면서도 긴 거리 정보 경로를 만든다.

이미지, 오디오, 텍스트처럼 매우 긴 sequence의 autoregressive modeling을 목표로 한다.

## 연구 배경과 문제의식

표준 self-attention은 `O(n^2)`라 수만 길이 sequence에 적용하기 어렵다.

Full causal attention을 local/strided sparse pattern으로 분해해 긴 sequence 생성 비용을 줄인다.

하지만 모든 token이 모든 과거 token을 직접 볼 필요는 없을 수 있다. 국소 주변과 일정 간격의 먼 위치를 보면 정보 경로를 만들 수 있다.

## 이 논문에서 먼저 잡아야 할 이론적 축

이 논문의 중심축은 `Sparse Transformer의 sparse attention graph`이다.

따라서 리뷰의 초점은 단순히 모델이 성능을 올렸다는 사실이 아니라, 논문이 기존 방법의 어떤 가정을 바꾸었고 그 변화가 수식과 계산 절차에서 어떻게 나타나는지에 둔다.

아래 섹션에서는 핵심 개념을 먼저 분해하고, 그 다음 실제 계산 흐름을 단계별로 따라간다.

## 기존 방식과 무엇이 다른가

표준 attention과 효율 attention의 차이는 `n x n` 행렬을 얼마나 직접 다루는지에 있다.

```text
standard:
S = QK^T                  # [n, n]
P = softmax(S)            # [n, n]
O = P V

efficient:
avoid or approximate S by
- sparse subset
- low-rank projection
- random feature kernel
- landmark approximation
```

이때 중요한 질문은 `P`를 정확히 보존하는가이다. 보존하지 않는다면 그 방법은 속도를 얻는 대신 attention 분포를 바꾼다.

## 핵심 개념 상세 해설

### 효율 attention의 핵심 수식

Sparse Transformer는 attention graph를 직접 줄인다. 각 위치가 모든 과거 위치를 보지 않고 local window와 strided position만 본다.

```text
full causal: S_i = {1, 2, ..., i}
sparse: S_i = local_window(i) union strided_positions(i)
```

한 layer에서 직접 연결되지 않은 위치도 여러 layer를 거치면 정보가 전달될 수 있다. 따라서 sparse pattern의 graph diameter가 중요하다.

## Tensor shape와 자료구조

효율 attention의 shape는 어떤 차원을 줄이느냐에 따라 달라진다. 표준 attention은 `[n, n]` score matrix를 만들지만, sparse/low-rank/kernel 방식은 이 matrix를 직접 만들지 않는다.

```text
standard:
Q: [n, d], K: [n, d], V: [n, d_v]
S = QK^T: [n, n]
O = softmax(S)V: [n, d_v]

linear/low-rank examples:
K' or landmark: [k, d] where k << n
score: [n, k] or implicit kernel aggregate
output: [n, d_v]
```

따라서 이 계열을 읽을 때는 `n x n` 행렬이 어디에서 사라지는지 추적해야 한다. 사라지는 방식이 곧 논문의 이론적 가정이다.

## 수식과 계산 전개

### Sparse attention set

```text
표준 causal attention은 위치 `i`가 모든 `j <= i`를 본다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Local attention block

```text
Sparse attention은 `j in S_i`인 위치만 score를 계산하고 나머지는 mask한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Strided attention

```text
`S_i`가 local window 크기 `w`이면 비용은 대략 `O(nw)`가 된다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Layerwise receptive field

```text
Strided attention은 `i mod stride`가 같은 과거 위치 등을 보게 하여 먼 거리 연결을 만든다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Autoregressive density modeling

```text
여러 sparse layer를 쌓으면 한 위치의 receptive field가 간접적으로 넓어진다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

## 전체 알고리즘 흐름

1. Q/K/V를 만든다.
2. Full `n x n` score를 만들지 않기 위한 sparse set, projection, random feature, landmark를 구성한다.
3. 제한되거나 근사된 attention score 또는 kernel aggregate를 계산한다.
4. Normalization을 적용한다.
5. Value aggregation으로 output representation을 만든다.

## 작은 예시로 보는 직관

길이 10,000 문서에서 full attention은 한 head마다 1억 개 score를 만든다. 하지만 문서 이해에 모든 token pair가 같은 중요도를 갖지는 않는다.

```text
full score count: n^2 = 10000^2 = 100,000,000
local window w=256: n*w = 2,560,000
landmarks k=256: n*k = 2,560,000
```

효율 attention은 이 차이를 이용한다. 다만 줄인 score가 실제 필요한 관계를 충분히 포함해야 한다.

## 계산 복잡도와 병목

- Sparse pattern이 hand-designed라 task와 데이터 구조에 따라 최적이 아닐 수 있다.
- 이 논문에서 말하는 효율 또는 성능 개선이 어느 병목을 줄이는지 분리해서 봐야 한다.
- 수식상 복잡도, 실제 메모리 사용량, GPU 실행 효율, 학습 안정성, 추론 latency는 서로 다른 축이다.

예를 들어 같은 `더 효율적이다`라는 주장도 Linformer에서는 low-rank 근사, FlashAttention에서는 IO 절감, PagedAttention에서는 KV cache memory management, ViT에서는 대규모 pretraining을 통한 inductive bias 보완을 뜻한다.

## 구현할 때 확인할 부분

- 수식에서 normalization이 어느 축에 적용되는지 확인한다.
- Mask, position index, cache index, spatial flatten order처럼 off-by-one 오류가 나기 쉬운 부분을 별도로 검증한다.
- 논문이 exact 계산을 유지하는지, 근사 또는 sparse 제한을 쓰는지 구분한다.
- 학습 시 이득과 추론 시 이득이 같은지 분리해서 본다.

## 자주 헷갈리는 지점

- Attention weight가 항상 사람이 해석하는 원인 설명과 일치한다고 보면 안 된다. 모델 내부의 정보 routing 가중치로 보는 편이 안전하다.
- Score에 추가되는 bias나 position term은 value에 직접 더해지는 정보와 역할이 다르다. Score는 무엇을 볼지, value는 무엇을 가져올지를 정한다.
- Linear complexity라고 해서 항상 실제 wall-clock이 빠른 것은 아니다. GPU kernel, memory layout, batch shape가 중요하다.
- Approximate attention은 exact softmax attention과 같은 결과를 내지 않는다. 성능 비교에서 품질 손실을 함께 봐야 한다.

## 관련 방법과 비교

| 방법 | 줄이는 대상 | Exact 여부 |
| --- | --- | --- |
| Sparse/Longformer/BigBird | attention edge 수 | sparse graph 기준 exact, full attention은 아님 |
| Linformer | sequence dimension rank | 근사 |
| Performer | softmax kernel | random feature 근사 |
| Nyströmformer | attention matrix rank | landmark 근사 |
| FlashAttention | HBM IO와 저장량 | full attention exact |

## 실험 결과를 읽는 관점

효율 attention 실험은 점수와 복잡도를 함께 봐야 한다. 근사 방식은 속도와 메모리를 줄일 수 있지만, attention distribution이 바뀌므로 task 성능이 손상될 수 있다.

또한 이론적 `O(n)` 또는 `O(n log n)` 복잡도가 실제 GPU wall-clock speedup으로 이어지는지 확인해야 한다. Sparse 또는 low-rank 알고리즘은 kernel 구현의 영향을 크게 받는다.

## 장점과 기여

- 기존 방법의 한계를 명확한 이론적 문제로 재정의한다.
- 그 문제를 해결하기 위한 계산 구조 또는 학습/추론 절차를 제안한다.
- 후속 연구가 비교할 수 있는 기준점과 용어를 제공한다.

## 한계와 비판적 관점

- Sparse pattern이 hand-designed라 task와 데이터 구조에 따라 최적이 아닐 수 있다.
- Full attention처럼 임의의 두 위치가 한 layer에서 직접 상호작용하지는 못한다.
- 효율적인 sparse kernel 구현이 없으면 이론적 절감이 실제 속도로 이어지기 어렵다.

## 후속 논문과의 연결점

Longformer와 BigBird는 더 체계적인 sparse attention pattern을 제안하고, FlashAttention 계열은 full attention을 exact하게 빠르게 계산하는 방향을 택한다.

## 개인 학습/연구 메모

Sparse Transformer는 `모든 과거를 직접 보지 말고, 가까운 곳과 규칙적으로 먼 곳만 보자`는 long sequence용 attention graph 설계다.


# 14. LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens

## 논문 정보

- 원본 파일: `C:\Users\lkm\Documents\Codex\Embbeded_OnDevice_ComputerVision\attention_papers_pdf\14_LongRoPE.pdf`
- 제목: LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens
- 저자: Yiran Ding et al.
- 발표: 2024
- 주제: LongRoPE의 초장문 context 확장
- 핵심 키워드: LongRoPE, long context, non-uniform RoPE interpolation, progressive extension

## 한눈에 보는 요약

LongRoPE는 RoPE 기반 LLM의 context window를 매우 긴 길이까지 확장하기 위해 비균일 interpolation과 progressive extension을 사용한다.

차원별, 위치 구간별로 다른 scaling factor를 탐색해 짧은 context 성능과 긴 context 성능을 동시에 유지하려 한다.

단순한 한 번의 scaling이 아니라 단계적으로 context length를 늘리는 절차가 핵심이다.

## 연구 배경과 문제의식

Position Interpolation은 간단하지만 확장 배율이 커질수록 위치 resolution 손실이 심해진다.

차원별/구간별 비균일 interpolation factor와 progressive extension으로 매우 긴 context window를 확장한다.

RoPE 차원마다 장거리와 단거리 정보 기여가 다르므로 uniform scaling은 최적이 아닐 수 있다.

## 이 논문에서 먼저 잡아야 할 이론적 축

이 논문의 중심축은 `LongRoPE의 초장문 context 확장`이다.

따라서 리뷰의 초점은 단순히 모델이 성능을 올렸다는 사실이 아니라, 논문이 기존 방법의 어떤 가정을 바꾸었고 그 변화가 수식과 계산 절차에서 어떻게 나타나는지에 둔다.

아래 섹션에서는 핵심 개념을 먼저 분해하고, 그 다음 실제 계산 흐름을 단계별로 따라간다.

## 기존 방식과 무엇이 다른가

기본 Transformer는 위치 정보를 입력 embedding에 더한 뒤 attention을 한다. 위치 표현 논문들은 이 위치 정보가 들어가는 위치를 바꾼다.

```text
absolute position:
X = token_embedding + position_embedding
S = QK^T

relative/bias/rotary variants:
Q, K = position-aware projection or rotation
S_ij = content_score(i,j) + position_term(i,j)
```

따라서 이 계열을 비교할 때는 `position_term(i,j)`가 table lookup인지, 회전인지, 선형 bias인지, scaling된 position index인지 보면 된다.

## 핵심 개념 상세 해설

### LongRoPE의 RoPE 확장 관점

이 논문은 RoPE의 핵심 성질을 유지하면서 학습 context length 밖에서 score 분포가 깨지는 문제를 다룬다.

RoPE는 긴 position index에 대해 회전 angle이 계속 커지므로, 학습 때 보지 못한 phase 조합이 생긴다. Context extension 방법들은 이 angle을 압축하거나, 차원별로 다르게 조정하거나, scale을 보정한다.

```text
base RoPE angle: p * theta_i
uniform interpolation: (p / s) * theta_i
dimension-wise scaling: p * theta'_i
query/key scaling: scale(p) * R_p q, scale(p)^-1 * R_p k
```

핵심 trade-off는 길이를 늘리면 위치 resolution이 줄어든다는 점이다. 좋은 방법은 짧은 거리 resolution과 긴 거리 안정성을 동시에 보존해야 한다.

## Tensor shape와 자료구조

위치 표현 논문들은 대개 Transformer의 기본 shape는 유지한다. 입력 hidden state `X`는 `[B, n, d_model]`이고, attention head마다 Q/K/V를 만든다.

```text
X: [B, n, d_model]
Q = X W_Q: [B, h, n, d_k]
K = X W_K: [B, h, n, d_k]
V = X W_V: [B, h, n, d_v]
score S: [B, h, n, n]
output O: [B, h, n, d_v]
```

차이는 `S = QK^T / sqrt(d_k)`를 그대로 쓰지 않는다는 점이다. 상대 위치 embedding, relative bias, rotary transformation, linear bias, interpolation scaling 등이 `Q`, `K`, 또는 `S`에 들어간다.

## 수식과 계산 전개

### 비균일 RoPE scale

```text
RoPE의 각 dimension pair에 대해 서로 다른 scale `s_i`를 둘 수 있다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### 차원별 interpolation factor

```text
위치 `p`의 angle은 uniform하게 `p/s`가 아니라 `p/s_i` 또는 구간별 변형된 위치 함수로 계산된다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Progressive extension

```text
검색 또는 최적화 절차로 긴 context perplexity와 짧은 context 품질을 함께 고려하는 scaling 조합을 찾는다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Short-context readjustment

```text
확장된 context에서 fine-tuning한 뒤 다시 더 긴 target context로 확장하는 progressive schedule을 사용한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### 초장문 평가

```text
짧은 context 성능 회복을 위해 원래 scale 또는 짧은 길이 전용 보정도 함께 고려한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

## 전체 알고리즘 흐름

1. Token embedding을 만들고 필요한 경우 absolute/relative position 정보를 준비한다.
2. 각 layer에서 Q/K/V를 projection한다.
3. 논문별 방식에 따라 Q/K를 회전, 위치 bias 추가, relative embedding 결합, position scaling 등을 적용한다.
4. 수정된 score로 softmax attention을 수행한다.
5. Value를 합산하고 residual/FFN을 거쳐 다음 layer로 넘긴다.

## 작은 예시로 보는 직관

문장 `not very good`에서 `good`은 바로 앞의 `very`, 두 칸 앞의 `not`과 관계가 있다. 절대 위치 17, 18, 19보다 중요한 것은 `good` 기준으로 -1, -2 위치라는 상대 관계다.

```text
absolute: position(good) = 19
relative: distance(good, very) = -1
relative: distance(good, not) = -2
```

Relative position, RoPE, ALiBi 계열은 이런 상대 거리 정보를 attention score가 직접 느끼게 만든다.

## 계산 복잡도와 병목

- Scaling factor 탐색과 progressive fine-tuning 비용이 있다.
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
- 긴 position index를 계산할 수 있다는 것과 모델이 그 길이에서 잘 일반화한다는 것은 다르다.
- RoPE scaling은 attention 비용을 줄이지 않는다. 위치 일반화를 개선할 뿐이다.

## 관련 방법과 비교

| 방법 | 위치 정보가 들어가는 곳 | 긴 길이 일반화 관점 |
| --- | --- | --- |
| Absolute embedding | input embedding | 학습 위치 밖 취약 |
| Relative position | attention score 또는 value | 상대 거리 표현에 강함 |
| RoPE | Q/K rotation | 상대 phase를 dot product에 내장 |
| ALiBi | score bias | position table 없이 거리 penalty |
| PI/YaRN/LongRoPE | RoPE position scaling | 기존 RoPE LLM context 확장 |

## 실험 결과를 읽는 관점

위치 표현 논문의 실험은 보통 두 가지를 본다. 첫째, 같은 길이 또는 학습 길이 안에서 성능이 좋아지는가. 둘째, 학습보다 긴 context에서 extrapolation이 되는가.

특히 RoPE scaling 계열에서는 짧은 context 성능 보존과 긴 context perplexity 개선을 동시에 봐야 한다. 긴 길이만 좋아지고 짧은 길이가 무너지면 실용적인 확장이라고 보기 어렵다.

## 장점과 기여

- 기존 방법의 한계를 명확한 이론적 문제로 재정의한다.
- 그 문제를 해결하기 위한 계산 구조 또는 학습/추론 절차를 제안한다.
- 후속 연구가 비교할 수 있는 기준점과 용어를 제공한다.

## 한계와 비판적 관점

- Scaling factor 탐색과 progressive fine-tuning 비용이 있다.
- 매우 긴 context를 실제로 사용하는 serving에서는 attention 비용과 KV cache 비용이 여전히 병목이다.
- 긴 context benchmark의 품질이 실제 복잡한 장거리 reasoning을 충분히 대표하는지는 별도 검증이 필요하다.

## 후속 논문과의 연결점

LongRoPE는 RoPE scaling 계열의 극단적 context extension 사례이며, 효율 attention 및 KV cache 관리 연구와 함께 써야 실용성이 커진다.

## 개인 학습/연구 메모

LongRoPE는 `RoPE 위치 축을 한 번에 균일 압축하지 않고, 차원별/단계별로 조심스럽게 늘린다`고 정리하면 된다.


# 09. RoFormer: Enhanced Transformer with Rotary Position Embedding

## 논문 정보

- 원본 파일: `C:\Users\lkm\Documents\Codex\Embbeded_OnDevice_ComputerVision\attention_papers_pdf\09_RoFormer_RoPE.pdf`
- 제목: RoFormer: Enhanced Transformer with Rotary Position Embedding
- 저자: Jianlin Su et al.
- 발표: 2021
- 주제: Rotary position embedding으로 상대 위치를 dot product 내부에 내장
- 핵심 키워드: RoPE, rotary position embedding, relative position by rotation

## 한눈에 보는 요약

RoPE는 query와 key 벡터를 위치에 따라 회전시켜 dot product가 상대 위치 차이에 의존하게 만든다.

절대 위치 embedding을 입력에 더하는 대신, attention에 들어가는 `Q`, `K`에 위치별 rotation matrix를 적용한다.

현대 LLM의 표준적인 위치 표현 중 하나가 되었다.

## 연구 배경과 문제의식

Absolute position embedding은 학습 길이 밖 position index에 약하고, relative bias는 별도 bias table이나 score 항을 요구한다.

RoPE는 Q/K 벡터를 위치에 따라 회전시킨다. 그러면 두 위치의 dot product가 상대 위치 차이에 자연스럽게 의존한다.

이 방식은 현대 decoder-only LLM에서 널리 쓰이는 위치 표현의 기반이 되었다.

## 이 논문에서 먼저 잡아야 할 이론적 축

이 논문의 중심축은 `Rotary position embedding으로 상대 위치를 dot product 내부에 내장`이다.

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

### RoPE의 회전 성질

RoPE는 위치를 더하지 않고 Q/K를 회전시킨다. 2차원 feature pair마다 위치 `m`에 비례한 angle로 회전한다.

```text
q_m = R_m q
k_n = R_n k
q_m^T k_n = q^T R_m^T R_n k = q^T R_{n-m} k
```

이 식 때문에 dot product가 절대 위치가 아니라 상대 위치 차이 `n - m`을 자연스럽게 포함한다.

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

### 회전 행렬

```text
[x_1, x_2] -> [x_1 cos(m theta) - x_2 sin(m theta),
              x_1 sin(m theta) + x_2 cos(m theta)]
```

위치 `m`에 따라 2D pair가 회전한다.

### 상대 위치 성질

```text
q_m = R_m q
k_n = R_n k
q_m^T k_n = q^T R_{n-m} k
```

Dot product 안에 상대 위치 차이가 들어간다.

### Complex form

```text
(x_{2i} + i x_{2i+1}) * exp(i m theta_i)
```

구현과 직관에서 complex multiplication으로 이해할 수 있다.

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

- 학습 길이를 훨씬 넘는 위치에서는 회전 각도 분포가 학습 때와 달라져 extrapolation 문제가 생긴다.
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

- 학습 길이보다 훨씬 긴 위치에서는 회전 phase가 학습 분포 밖으로 나간다.
- Context 확장에는 position interpolation, YaRN, LongRoPE 같은 보정이 필요하다.
- Attention 계산량 자체를 줄이지 않는다.

## 후속 논문과의 연결점

Position Interpolation, YaRN, LongRoPE, xPos는 모두 RoPE의 길이 extrapolation 한계를 완화하려는 연구로 볼 수 있다.

## 개인 학습/연구 메모

RoPE의 가장 중요한 식은 `R_m^T R_n = R_{n-m}`이다. 이 한 줄이 상대 위치 성질을 만든다.


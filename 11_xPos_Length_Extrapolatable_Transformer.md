# 11. A Length-Extrapolatable Transformer

## 논문 정보

- 원본 파일: `C:\Users\lkm\Documents\Codex\Embbeded_OnDevice_ComputerVision\attention_papers_pdf\11_xPos_Length_Extrapolatable_Transformer.pdf`
- 제목: A Length-Extrapolatable Transformer
- 저자: Yutao Sun et al.
- 발표: 2023
- 주제: xPos의 길이 외삽 이론
- 핵심 키워드: xPos, length extrapolation, rotary position, attention resolution

## 한눈에 보는 요약

xPos는 RoPE의 상대 위치 장점을 유지하면서 긴 길이 extrapolation에서 attention score가 불안정해지는 문제를 완화한다.

핵심은 query와 key에 서로 반대 방향의 scale을 적용해 상대 위치 성질과 attention resolution을 함께 보존하는 것이다.

논문은 길이 외삽에서 attention score의 monotonicity와 resolution이 중요하다고 분석한다.

## 연구 배경과 문제의식

RoPE는 상대 위치를 잘 표현하지만, 학습 길이를 넘어서면 회전 phase가 낯선 영역으로 들어가 attention pattern이 깨질 수 있다.

RoPE의 회전 구조에 query/key scale을 추가해 긴 길이에서 attention score resolution과 monotonicity를 보존하려는 방법이다.

긴 sequence에서 위치가 멀어질수록 attention score의 구분력이 떨어지거나 비정상적으로 진동하면 extrapolation이 어렵다.

## 이 논문에서 먼저 잡아야 할 이론적 축

이 논문의 중심축은 `xPos의 길이 외삽 이론`이다.

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

### xPos의 RoPE 확장 관점

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

### Attention resolution

```text
RoPE는 `q_m = R_m q`, `k_n = R_n k`로 회전한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Position encoding으로 resolution 보정

```text
xPos는 여기에 scale을 넣어 개념적으로 `q_m = s_m R_m q`, `k_n = s_n^{-1} R_n k`처럼 reciprocal 구조를 만든다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Reciprocal Q/K scaling

```text
Dot product는 회전 차이와 scale ratio를 함께 포함해 `q_m^T k_n`이 상대 거리의 함수로 유지된다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Score monotonicity

```text
Scale 설계는 긴 위치에서도 attention score가 너무 커지거나 작아지지 않도록 조절한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### 길이 외삽 평가

```text
결과적으로 긴 sequence에서 가까운 위치와 먼 위치를 구분하는 resolution을 더 오래 유지한다.
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

- RoPE보다 구현과 hyperparameter 설계가 복잡하다.
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

- RoPE보다 구현과 hyperparameter 설계가 복잡하다.
- 매우 긴 context에서는 여전히 attention의 `O(n^2)` 비용 문제가 남는다.
- 모든 task에서 ALiBi나 RoPE scaling보다 항상 우월하다고 보기는 어렵고, 모델/데이터 조건에 따라 달라진다.

## 후속 논문과의 연결점

xPos는 RoPE 이후의 length extrapolation 계열이며, Position Interpolation, YaRN, LongRoPE와 함께 긴 context 위치 표현 연구의 한 축이다.

## 개인 학습/연구 메모

xPos를 볼 때는 `회전 + reciprocal scale`이 score 안정성을 위해 들어간다고 잡으면 된다.


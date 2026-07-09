# 05. Self-Attention with Relative Position Representations

## 논문 정보

- 원본 파일: `C:\Users\lkm\Documents\Codex\Embbeded_OnDevice_ComputerVision\attention_papers_pdf\05_Self_Attention_with_Relative_Position_Representations.pdf`
- 제목: Self-Attention with Relative Position Representations
- 저자: Peter Shaw, Jakob Uszkoreit, Ashish Vaswani
- 발표: NAACL 2018
- 주제: Transformer 위치 정보를 token 속성이 아니라 token 쌍의 관계로 모델링
- 핵심 키워드: relative position, relation-aware self-attention, clipping distance

## 한눈에 보는 요약

이 논문은 Transformer의 absolute positional encoding을 보완해 token 간 상대 거리를 attention 계산 안에 직접 넣는다.

Query-key score와 value aggregation 양쪽에 relative position embedding을 추가한다.

중요한 변화는 위치 정보를 입력 embedding에 한 번 더하는 것이 아니라, attention edge `i -> j`의 relation feature로 모델링한다는 점이다.

## 연구 배경과 문제의식

기본 Transformer는 token embedding에 absolute positional encoding을 더한다. 하지만 self-attention score 자체는 두 위치 사이 상대 거리를 직접 알지 못한다.

언어에서는 절대 위치보다 상대 위치가 중요한 경우가 많다. 예를 들어 현재 단어 바로 앞의 단어, 두 칸 뒤의 단어 같은 관계가 문법적으로 의미가 있다.

Shaw et al.은 self-attention을 fully connected directed graph로 보고, edge `i -> j`마다 relative position representation을 부여한다.

## 이 논문에서 먼저 잡아야 할 이론적 축

이 논문의 중심축은 `Transformer 위치 정보를 token 속성이 아니라 token 쌍의 관계로 모델링`이다.

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

### 상대 위치를 edge feature로 보기

기본 Transformer에서는 위치 정보가 token embedding에 더해진다. Shaw et al.은 위치를 token의 속성으로 보지 않고, token `i`가 token `j`를 볼 때의 edge relation으로 본다.

```text
e_ij = (x_i W_Q)(x_j W_K + a^K_{ij})^T / sqrt(d)
z_i = sum_j alpha_ij (x_j W_V + a^V_{ij})
```

여기서 `a^K_{ij}`와 `a^V_{ij}`는 상대 거리 `j - i`에 의해 결정된다. 즉, 같은 단어라도 어느 위치에서 보느냐에 따라 다른 relation embedding이 붙는다.

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

### 기본 self-attention

```text
e_ij = (x_i W_Q)(x_j W_K)^T / sqrt(d)
```

기본식은 content query와 content key의 내적만 사용한다.

### Relative key 추가

```text
e_ij = (x_i W_Q)(x_j W_K + a^K_ij)^T / sqrt(d)
```

`a^K_ij`는 상대 거리 `j - i`에 의해 선택되는 embedding이다.

### Relative value 추가

```text
z_i = sum_j alpha_ij (x_j W_V + a^V_ij)
```

Value aggregation에도 상대 위치 정보를 넣는다.

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

- 모든 위치 쌍에 relative relation을 고려하므로 구현 복잡도가 증가한다.
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

- 모든 위치 쌍에 relation embedding을 적용하므로 구현이 복잡해진다.
- Clipping 범위 밖의 거리 차이는 구분하지 못한다.
- Attention 비용 자체의 `O(n^2)` 문제는 해결하지 않는다.

## 후속 논문과의 연결점

Transformer-XL은 relative position을 segment recurrence와 결합하고, T5는 logit에 scalar relative position bias를 더하는 단순한 방식을 사용한다.

## 개인 학습/연구 메모

이 논문은 `위치도 token의 속성인가, token 쌍의 관계인가`라는 질문에서 후자를 택한 논문이다.


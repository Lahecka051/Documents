# 08. DeBERTa: Decoding-enhanced BERT with Disentangled Attention

## 논문 정보

- 원본 파일: `C:\Users\lkm\Documents\Codex\Embbeded_OnDevice_ComputerVision\attention_papers_pdf\08_DeBERTa.pdf`
- 제목: DeBERTa: Decoding-enhanced BERT with Disentangled Attention
- 저자: Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen
- 발표: ICLR 2021
- 주제: Content와 position을 분리한 disentangled attention 이론
- 핵심 키워드: disentangled attention, content-position attention, enhanced mask decoder

## 한눈에 보는 요약

DeBERTa는 token content와 position을 하나의 embedding에 더해 섞지 않고, 분리된 벡터로 attention을 계산한다.

Attention score를 content-content, content-position, position-content 관계로 나누어 상대 위치 정보를 더 정교하게 반영한다.

Masked language modeling의 decoder 단계에서는 absolute position 정보도 보강한다.

## 연구 배경과 문제의식

BERT는 token embedding, segment embedding, position embedding을 더한 뒤 self-attention을 수행한다. 이 경우 content 정보와 position 정보가 초기에 섞인다.

DeBERTa는 단어 자체의 의미와 위치 관계를 분리해 attention score를 계산한다. 이는 상대 위치가 token pair 관계라는 관점을 더 강하게 밀어붙인 것이다.

또한 MLM에서 `[MASK]` 위치의 absolute position 정보가 약해지는 문제를 enhanced mask decoder로 보완한다.

## 이 논문에서 먼저 잡아야 할 이론적 축

이 논문의 중심축은 `Content와 position을 분리한 disentangled attention 이론`이다.

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

### Disentangled attention

DeBERTa는 content와 position을 처음부터 더해버리지 않는다. 대신 attention score에서 content-content, content-position, position-content 항을 따로 계산한다.

```text
score(i,j) =
  q_i^c · k_j^c        # content to content
+ q_i^c · k_{i-j}^r    # content to relative position
+ q_{j-i}^r · k_j^c    # relative position to content
```

이렇게 분리하면 `무슨 단어인가`와 `얼마나 떨어져 있는가`가 attention logit에서 별도의 역할을 갖는다.

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

### Disentangled score

```text
A_ij = q_i^c · k_j^c
     + q_i^c · k_{i-j}^r
     + q_{j-i}^r · k_j^c
```

세 항은 각각 content-content, content-position, position-content 관계를 나타낸다.

### Attention weight

```text
alpha_ij = softmax_j(A_ij / sqrt(3d))
```

여러 항을 더하므로 scale을 조정한다.

### MLM decoding

```text
p(x_i | corrupted sequence, position_i)
```

Decoder 단계에서 absolute position을 다시 넣어 masked token 예측을 보강한다.

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

- Attention score 항이 늘어나므로 구현과 계산이 일반 BERT보다 복잡하다.
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

- Score 항이 많아 구현이 복잡하다.
- 상대 위치 bucket 설계에 의존한다.
- Encoder-only NLU 중심이라 autoregressive LLM 추론 효율을 직접 다루지는 않는다.

## 후속 논문과의 연결점

DeBERTa는 상대 위치와 content-position 분리의 중요성을 보여주며, RoPE나 ALiBi처럼 attention logit을 위치에 따라 바꾸는 흐름과 연결된다.

## 개인 학습/연구 메모

BERT가 `content + position`을 더한 뒤 attention한다면, DeBERTa는 `content와 position의 상호작용 항을 따로 계산한다`고 기억하면 된다.


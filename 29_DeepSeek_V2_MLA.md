# 29. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

## 논문 정보

- 원본 파일: `C:\Users\lkm\Documents\Codex\Embbeded_OnDevice_ComputerVision\attention_papers_pdf\29_DeepSeek_V2_MLA.pdf`
- 제목: DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
- 저자: DeepSeek-AI
- 발표: 2024
- 주제: DeepSeek-V2 MLA와 MoE 효율 이론
- 핵심 키워드: DeepSeek-V2, MLA, Multi-head Latent Attention, DeepSeekMoE, KV cache compression

## 한눈에 보는 요약

DeepSeek-V2는 효율적인 대형 언어모델을 위해 Multi-head Latent Attention(MLA)과 DeepSeekMoE를 결합한다.

MLA는 key/value를 직접 모든 head별로 cache하지 않고, 저차원 latent representation을 cache한 뒤 필요할 때 key/value를 복원한다.

목표는 GQA/MQA보다 더 강한 품질을 유지하면서 KV cache를 크게 줄이는 것이다.

## 연구 배경과 문제의식

대형 decoder-only LLM serving에서 KV cache는 sequence length, layer 수, head 수에 비례해 증가한다.

KV cache를 latent representation으로 압축하고 MoE로 활성 parameter를 줄여 경제적인 LLM을 만든다.

MQA/GQA는 K/V head 수를 줄이는 방식이지만, head별 표현력을 일부 희생한다.

## 이 논문에서 먼저 잡아야 할 이론적 축

이 논문의 중심축은 `DeepSeek-V2 MLA와 MoE 효율 이론`이다.

따라서 리뷰의 초점은 단순히 모델이 성능을 올렸다는 사실이 아니라, 논문이 기존 방법의 어떤 가정을 바꾸었고 그 변화가 수식과 계산 절차에서 어떻게 나타나는지에 둔다.

아래 섹션에서는 핵심 개념을 먼저 분해하고, 그 다음 실제 계산 흐름을 단계별로 따라간다.

## 기존 방식과 무엇이 다른가

Serving 최적화 논문들은 Transformer layer의 수식보다 decode-time state를 바꾼다.

```text
training view:
all tokens processed in parallel
attention over full prefix matrix

serving view:
one new token at a time
read old K/V cache
compute attention for current query
append new K/V cache
```

따라서 이 계열에서는 FLOPs보다 KV cache size, memory bandwidth, fragmentation, batching efficiency가 더 중요해질 수 있다.

## 핵심 개념 상세 해설

### KV cache 관점의 핵심

MLA는 full K/V를 저장하지 않고 compressed latent를 저장한다.

```text
c_t^KV = W_DKV h_t
k_t = W_UK c_t^KV + positional key part
v_t = W_UV c_t^KV
cache stores c_t^KV, not all head-wise K/V
```

K/V head 수를 줄이는 MQA/GQA와 달리, MLA는 latent basis에서 head별 K/V를 복원한다. 그래서 cache는 작게 유지하면서 표현력은 더 많이 보존하려 한다.

## Tensor shape와 자료구조

Serving 계열에서는 training-time tensor보다 decode-time KV cache shape가 중요하다. Layer 수를 `L`, query head 수를 `H_q`, KV head 수를 `H_kv`, head dimension을 `d`, 현재 길이를 `n`이라고 하자.

```text
MHA KV cache per layer: [n, H_q, d] for K and [n, H_q, d] for V
MQA KV cache per layer: [n, 1, d] for K and [n, 1, d] for V
GQA KV cache per layer: [n, H_kv, d], where 1 < H_kv < H_q
MLA cache per layer: [n, d_latent] plus positional components
PagedAttention: logical [n, ...] mapped to physical KV blocks
```

즉, attention score 수식보다 cache의 head dimension과 layout이 실제 throughput을 결정한다. 긴 context에서는 `n`이 커지므로 cache 절감 효과가 매우 커진다.

## 수식과 계산 전개

### Latent KV compression

```text
`c_t^KV = W_DKV h_t`로 low-dimensional latent KV를 만든다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Content key 복원

```text
`k_t^C = W_UK c_t^KV`, `v_t^C = W_UV c_t^KV`로 content key/value를 복원한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Value 복원

```text
Query도 compressed latent를 거쳐 head별 query로 up-projection될 수 있다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Decoupled RoPE

```text
RoPE는 content key와 분리된 positional key에 적용해 cache 압축과 위치 encoding 충돌을 줄인다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Decode cache 구성

```text
Decode cache에는 full head별 K/V 대신 `c_t^KV`와 필요한 positional component만 저장한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

## 전체 알고리즘 흐름

1. 현재 token hidden state로 query를 만든다.
2. 과거 token의 K/V cache를 shared, grouped, latent, paged layout에 맞게 읽는다.
3. 현재 query와 과거 key 사이 attention score를 계산한다.
4. 과거 value를 weighted sum해 현재 output을 만든다.
5. 현재 token의 K/V 또는 latent cache를 cache에 append한다.

## 작은 예시로 보는 직관

사용자가 prompt 3000 token을 넣고 모델이 1 token을 생성한다고 하자. 새 token의 query는 과거 3000개 token의 K/V를 모두 읽는다.

```text
decode step t:
q_t: [H_q, d]
K_cache: [t, H_kv, d]
V_cache: [t, H_kv, d]
attention reads K_cache and V_cache every step
```

그래서 `H_kv`를 줄이거나 cache layout을 개선하면 실제 응답 속도가 크게 달라진다.

## 계산 복잡도와 병목

- MLA는 구조가 복잡하고 기존 Transformer 구현과 호환성이 낮다.
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
- MQA/GQA/MLA는 학습 FLOPs보다 decoding cache bandwidth를 겨냥한다.
- KV cache 최적화는 모델 구조와 serving scheduler가 함께 맞아야 효과가 크다.

## 관련 방법과 비교

| 방법 | KV cache 전략 | 장점 |
| --- | --- | --- |
| MHA | head별 K/V 저장 | 표현력 높음, cache 큼 |
| MQA | 모든 query head가 K/V 공유 | cache 최소화 |
| GQA | group별 K/V 공유 | 품질/효율 절충 |
| MLA | latent KV 저장 후 복원 | cache 압축과 head 표현력 절충 |
| PagedAttention | block table로 KV 관리 | fragmentation 감소, prefix sharing |

## 실험 결과를 읽는 관점

Serving 논문의 실험은 perplexity보다 throughput, latency, batch 처리량, GPU memory 사용량이 핵심이다.

MQA/GQA/MLA는 품질 손실과 cache 절감의 trade-off를 보고, PagedAttention은 실제 request workload에서 fragmentation과 scheduling 효율을 본다.

## 장점과 기여

- 기존 방법의 한계를 명확한 이론적 문제로 재정의한다.
- 그 문제를 해결하기 위한 계산 구조 또는 학습/추론 절차를 제안한다.
- 후속 연구가 비교할 수 있는 기준점과 용어를 제공한다.

## 한계와 비판적 관점

- MLA는 구조가 복잡하고 기존 Transformer 구현과 호환성이 낮다.
- 압축 차원 선택이 품질과 cache 절감의 핵심 trade-off다.
- MoE routing까지 포함하면 학습 안정성, load balancing, serving scheduling 문제가 함께 생긴다.

## 후속 논문과의 연결점

MLA는 MQA/GQA 이후 KV cache 효율화의 더 공격적인 방향이며, long-context serving과 MoE inference 최적화와 밀접하게 연결된다.

## 개인 학습/연구 메모

MLA는 `full KV를 저장하지 말고 latent KV를 저장한 뒤 head별 K/V를 필요할 때 복원한다`고 기억하면 된다.


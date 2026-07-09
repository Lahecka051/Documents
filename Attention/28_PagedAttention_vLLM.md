# 28. Efficient Memory Management for Large Language Model Serving with PagedAttention

## 논문 정보

- 원본 파일: `C:\Users\lkm\Documents\Codex\Embbeded_OnDevice_ComputerVision\attention_papers_pdf\28_PagedAttention_vLLM.pdf`
- 제목: Efficient Memory Management for Large Language Model Serving with PagedAttention
- 저자: Woosuk Kwon et al.
- 발표: SOSP 2023
- 주제: PagedAttention의 LLM serving memory management
- 핵심 키워드: PagedAttention, vLLM, KV cache, LLM serving, memory paging

## 한눈에 보는 요약

PagedAttention은 LLM serving에서 KV cache memory를 운영체제의 virtual memory paging처럼 block 단위로 관리한다.

각 sequence의 logical token positions를 physical KV blocks에 매핑해 fragmentation을 줄인다.

vLLM은 이 메모리 관리 방식을 사용해 batching, parallel sampling, throughput을 개선한다.

## 연구 배경과 문제의식

Autoregressive LLM serving에서는 prompt와 generated token마다 key/value cache가 쌓인다.

KV cache를 block/page 단위로 관리해 fragmentation을 줄이고 prefix sharing을 효율화한다.

Sequence마다 길이가 다르고 언제 끝날지 모르기 때문에 contiguous allocation은 내부/외부 fragmentation을 만든다.

## 이 논문에서 먼저 잡아야 할 이론적 축

이 논문의 중심축은 `PagedAttention의 LLM serving memory management`이다.

따라서 리뷰의 초점은 단순히 모델이 성능을 올렸다는 사실이 아니라, 논문이 기존 방법의 어떤 가정을 바꾸었고 그 변화가 수식과 계산 절차에서 어떻게 나타나는지에 둔다.

아래 섹션에서는 핵심 개념을 먼저 분해하고, 그 다음 실제 계산 흐름을 단계별로 따라간다.

## 기존 방식과 무엇이 다른가

PagedAttention은 Transformer attention score를 바꾸지 않는다. 바꾸는 것은 serving 중 KV cache를 저장하고 참조하는 자료구조다.

```text
contiguous KV cache:
request -> one continuous memory region
length changes -> fragmentation / over-reservation

paged KV cache:
request logical blocks -> physical KV blocks
block table -> flexible allocation and sharing
```

이는 운영체제의 virtual memory와 비슷한 발상이다. 논리적으로는 연속된 sequence처럼 보이지만 실제 GPU memory에서는 여러 block에 나뉘어 저장된다.

## 핵심 개념 상세 해설

### KV cache 관점의 핵심

PagedAttention은 KV cache를 contiguous tensor 하나로 보지 않고 block table로 관리한다.

```text
logical block id -> physical KV block id
token position -> block id + offset
shared prefix -> same physical blocks with reference counts
```

이 방식은 sequence 길이가 제각각인 serving workload에서 fragmentation을 줄이고 prefix sharing을 효율화한다.

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

### KV block table

```text
표준 attention은 현재 query가 과거 모든 token의 K/V cache를 읽는다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Logical-to-physical mapping

```text
PagedAttention에서는 logical position `t`의 K/V 위치를 block table로 찾아 physical block offset을 계산한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Paged attention read

```text
Attention kernel은 contiguous sequence가 아니라 block list를 순회하며 K/V를 읽는다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Prefix sharing

```text
여러 candidate가 같은 prefix를 공유하면 같은 physical KV block을 reference count로 공유한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Copy-on-write

```text
Candidate가 서로 다른 token을 생성하는 시점에만 copy-on-write로 새 block을 할당한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

## 전체 알고리즘 흐름

1. 현재 token hidden state로 query를 만든다.
2. 과거 token의 K/V cache를 shared, grouped, latent, paged layout에 맞게 읽는다.
3. 현재 query와 과거 key 사이 attention score를 계산한다.
4. 과거 value를 weighted sum해 현재 output을 만든다.
5. 현재 token의 K/V 또는 latent cache를 cache에 append한다.

## 작은 예시로 보는 직관

두 사용자가 같은 긴 prompt에서 서로 다른 답변 후보를 sampling한다고 하자. Contiguous cache라면 prefix KV를 복사하기 쉽지만 memory 낭비가 크다. PagedAttention은 같은 prefix block을 공유하고, 생성이 갈라지는 순간 새 block만 할당한다.

```text
shared prefix blocks: A B C
sample 1 continuation: D1 E1
sample 2 continuation: D2 E2

physical KV blocks:
A B C shared
D1 E1, D2 E2 allocated separately
```

이것이 copy-on-write가 serving throughput에 주는 핵심 이득이다.

## 계산 복잡도와 병목

- Block table lookup과 non-contiguous memory access를 효율적으로 처리하는 kernel이 필요하다.
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

| 방식 | KV 저장 | 장점 | 문제 |
| --- | --- | --- | --- |
| Contiguous cache | request별 연속 영역 | 주소 계산 단순 | fragmentation과 over-reservation |
| PagedAttention | block table 기반 page | memory 낭비 감소, prefix sharing | block lookup이 필요한 kernel |
| MQA/GQA | K/V head 수 감소 | cache 자체 크기 감소 | 모델 구조 변경 필요 |

## 실험 결과를 읽는 관점

Serving 논문의 실험은 perplexity보다 throughput, latency, batch 처리량, GPU memory 사용량이 핵심이다.

MQA/GQA/MLA는 품질 손실과 cache 절감의 trade-off를 보고, PagedAttention은 실제 request workload에서 fragmentation과 scheduling 효율을 본다.

## 장점과 기여

- 기존 방법의 한계를 명확한 이론적 문제로 재정의한다.
- 그 문제를 해결하기 위한 계산 구조 또는 학습/추론 절차를 제안한다.
- 후속 연구가 비교할 수 있는 기준점과 용어를 제공한다.

## 한계와 비판적 관점

- Block table lookup과 non-contiguous memory access를 효율적으로 처리하는 kernel이 필요하다.
- Block size가 너무 크면 fragmentation이 늘고, 너무 작으면 metadata overhead가 증가한다.
- Attention 계산량 자체를 줄이지는 않는다.

## 후속 논문과의 연결점

PagedAttention은 vLLM의 핵심이며, MQA/GQA, FlashAttention decode kernel, speculative decoding과 함께 LLM serving 최적화의 중요한 축이다.

## 개인 학습/연구 메모

PagedAttention은 `KV cache를 연속 배열이 아니라 page/block table로 관리한다`는 시스템 아이디어다.


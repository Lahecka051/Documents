# 25. FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision

## 논문 정보

- 원본 파일: `C:\Users\lkm\Documents\Codex\Embbeded_OnDevice_ComputerVision\attention_papers_pdf\25_FlashAttention_3.pdf`
- 제목: FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision
- 저자: Jay Shah et al.
- 발표: 2024
- 주제: FlashAttention-3의 Hopper 비동기/저정밀 attention
- 핵심 키워드: FlashAttention-3, Hopper GPU, asynchronous pipeline, FP8 attention

## 한눈에 보는 요약

FlashAttention-3는 Hopper GPU 특성을 활용해 attention kernel을 더 빠르게 만든다.

Tensor Core matmul과 softmax 관련 연산을 asynchronous하게 overlap하고, FP8 low-precision 경로도 다룬다.

핵심은 GPU 하드웨어 실행 모델에 맞춘 pipeline과 warp specialization이다.

## 연구 배경과 문제의식

Hopper GPU는 Tensor Memory Accelerator, asynchronous transaction, FP8 Tensor Core 등 새로운 기능을 제공한다.

Hopper GPU의 asynchronous pipeline과 FP8을 활용해 exact 또는 고정확 attention을 더 빠르게 계산한다.

Attention은 GEMM, softmax, masking, normalization이 섞인 연산이라 단순 matmul보다 pipeline 설계가 어렵다.

## 이 논문에서 먼저 잡아야 할 이론적 축

이 논문의 중심축은 `FlashAttention-3의 Hopper 비동기/저정밀 attention`이다.

따라서 리뷰의 초점은 단순히 모델이 성능을 올렸다는 사실이 아니라, 논문이 기존 방법의 어떤 가정을 바꾸었고 그 변화가 수식과 계산 절차에서 어떻게 나타나는지에 둔다.

아래 섹션에서는 핵심 개념을 먼저 분해하고, 그 다음 실제 계산 흐름을 단계별로 따라간다.

## 기존 방식과 무엇이 다른가

FlashAttention 계열은 Linformer나 Performer처럼 attention을 근사하지 않는다. 표준 attention과 같은 수학적 결과를 내되, 중간 matrix를 저장하지 않는 실행 알고리즘을 설계한다.

```text
approximate efficient attention:
change attention math to reduce pairwise interactions

FlashAttention:
keep exact softmax attention
change memory access pattern and tiling
```

따라서 FlashAttention의 핵심 이론은 모델링 이론이라기보다 IO complexity와 online softmax 알고리즘이다.

## 핵심 개념 상세 해설

### Online softmax

FlashAttention의 핵심은 softmax를 tile 단위로 나누어도 정확히 계산할 수 있다는 점이다. 각 row에 대해 현재까지 본 score의 maximum `m`과 exp sum `l`을 유지한다.

```text
m_new = max(m_old, max(S_tile))
l_new = exp(m_old - m_new) * l_old
      + sum(exp(S_tile - m_new))

acc_new = exp(m_old - m_new) * acc_old
        + exp(S_tile - m_new) @ V_tile
output = acc_final / l_final
```

이 update를 쓰면 tile을 여러 번 나누어 보더라도 전체 row에 softmax를 한 것과 같은 결과가 나온다.

### IO-aware attention

표준 구현은 `S`와 `P`를 큰 tensor로 HBM에 저장한다. FlashAttention은 Q/K/V tile을 SRAM에 올려 계산하고, 중간 attention probability를 저장하지 않는다.

FlashAttention-2는 work partitioning을 개선하고, FlashAttention-3는 Hopper GPU의 asynchronous pipeline과 low precision을 활용한다. 하지만 세 논문의 공통점은 attention 수학을 바꾸지 않는다는 것이다.

## Tensor shape와 자료구조

FlashAttention 계열은 tensor shape를 줄이는 것이 아니라 중간 행렬을 저장하지 않는다. 논리적으로는 `[B, h, n, n]` attention을 계산하지만, 실제 메모리에 전체 matrix를 materialize하지 않는다.

```text
logical computation:
S = QK^T: [B, h, n, n]
P = softmax(S): [B, h, n, n]
O = P V: [B, h, n, d_v]

FlashAttention execution:
Q block: [B_q, d]
K/V block: [B_k, d]
running max m: [B_q]
running normalizer l: [B_q]
running output acc: [B_q, d_v]
```

여기서 핵심 tensor는 `m`, `l`, `acc`다. 이 세 값을 tile마다 업데이트하면 전체 softmax matrix 없이도 정확한 output을 만들 수 있다.

## 수식과 계산 전개

### Hopper execution model

```text
기본 attention 수식은 `O = softmax(QK^T / sqrt(d))V`다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Asynchronous pipeline

```text
Tile 단위 online softmax 통계 `m`, `l`은 유지한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Warp specialization

```text
Pipeline stage 1에서 다음 Q/K tile load 또는 QK matmul을 준비하는 동안 stage 2에서 softmax 및 PV accumulation을 수행한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### FP8 scaling

```text
FP8에서는 Q/K/V quantization과 scale 관리가 output error에 큰 영향을 주므로 accumulation precision을 조절한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Accuracy management

```text
Asynchrony는 연산 순서를 바꾸지만 최종 softmax normalization의 수학적 의미는 유지한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

## 전체 알고리즘 흐름

1. Q/K/V를 tile 단위로 나눈다.
2. Q tile 하나에 대해 K/V tile들을 순회한다.
3. 각 tile score를 계산하고 running max와 normalizer를 update한다.
4. Value contribution을 output accumulator에 더한다.
5. 모든 tile을 본 뒤 accumulator를 normalizer로 나누어 output을 쓴다.

## 작은 예시로 보는 직관

길이 8192 attention에서 `8192 x 8192` probability matrix를 저장하면 매우 큰 메모리가 필요하다. FlashAttention은 한 번에 예를 들어 128개 query와 128개 key tile만 SRAM에서 처리한다.

```text
process Q[0:128] with K[0:128], K[128:256], ...
keep running max/sum/output for Q[0:128]
never store full P[8192, 8192]
```

결과는 full softmax attention과 같지만, 중간 matrix를 저장하지 않는다.

## 계산 복잡도와 병목

- Hopper 같은 특정 GPU 세대에 최적화되어 portability가 제한될 수 있다.
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
- FlashAttention은 sparse attention이나 linear attention이 아니다. Exact attention을 더 효율적으로 계산하는 kernel이다.
- 메모리를 덜 쓴다고 해서 이론적 `O(n^2)` pairwise score 성격이 사라지는 것은 아니다.

## 관련 방법과 비교

| 방법 | 수학적 attention | 줄이는 병목 | 품질 영향 |
| --- | --- | --- | --- |
| 표준 attention | Exact | `S`, `P` materialization으로 HBM IO 큼 | 기준 |
| Linear/low-rank attention | 근사 또는 변형 | `n x n` pairwise 계산 | 근사 오차 가능 |
| FlashAttention | Exact | HBM read/write와 peak memory | 동일 precision에서 동일 결과 |
| FlashAttention-2/3 | Exact 또는 저정밀 경로 | 병렬화, pipeline, hardware utilization | 저정밀 사용 시 검증 필요 |

## 실험 결과를 읽는 관점

FlashAttention 계열은 모델 품질 실험보다 kernel throughput, memory footprint, sequence length별 속도 비교가 중요하다. Exact attention이므로 같은 precision 조건에서는 모델 output이 표준 attention과 같아야 한다.

따라서 실험을 볼 때는 `얼마나 빠른가`, `얼마나 긴 sequence를 메모리에 올릴 수 있는가`, `backward까지 포함했는가`, `어떤 GPU에서 측정했는가`를 확인해야 한다.

## 장점과 기여

- 기존 방법의 한계를 명확한 이론적 문제로 재정의한다.
- 그 문제를 해결하기 위한 계산 구조 또는 학습/추론 절차를 제안한다.
- 후속 연구가 비교할 수 있는 기준점과 용어를 제공한다.

## 한계와 비판적 관점

- Hopper 같은 특정 GPU 세대에 최적화되어 portability가 제한될 수 있다.
- FP8 사용 시 model, scale, sequence length에 따라 정확도 검증이 필요하다.
- Kernel 수준 복잡도가 높아 일반 연구자가 직접 수정하기 어렵다.

## 후속 논문과의 연결점

FlashAttention 계열은 exact attention의 시스템 최적화 방향이고, PagedAttention/vLLM은 serving memory management 방향에서 KV cache 문제를 다룬다.

## 개인 학습/연구 메모

FlashAttention-3는 `수학은 attention 그대로, 실행은 Hopper GPU pipeline에 맞게 재배치`라고 기억하면 된다.


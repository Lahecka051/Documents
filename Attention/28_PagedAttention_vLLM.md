# 28. Efficient Memory Management for Large Language Model Serving with PagedAttention

## 논문 정보

- 제목: **Efficient Memory Management for Large Language Model Serving with PagedAttention**
- 저자: Woosuk Kwon et al.
- 발표: SOSP 2023
- 핵심 키워드: PagedAttention, vLLM, KV cache, virtual memory, block table, copy-on-write, continuous batching

## 한눈에 보는 요약

PagedAttention은 attention 수식을 근사하거나 KV head 수를 줄이지 않는다. 대신 LLM serving에서 request마다 길이가 동적으로 늘어나는 KV cache를 운영체제의 virtual memory page처럼 **고정 크기 block**으로 관리한다.

기존 시스템은 request 하나의 KV cache를 contiguous memory에 최대 길이만큼 미리 예약한다. 실제 output 길이를 모르므로 reserved space, internal fragmentation, external fragmentation이 생기고 batch에 넣을 수 있는 request 수가 줄어든다.

vLLM은 각 sequence의 logical KV block을 GPU의 임의 physical block에 매핑하는 block table을 유지한다. 새 token이 block을 채울 때만 physical block을 할당하므로 낭비가 마지막 block 하나 이내로 제한된다. 여러 sample이나 beam이 같은 prefix를 가질 때는 physical block을 공유하고 수정 시 copy-on-write한다.

PagedAttention kernel은 block table을 따라 non-contiguous K/V를 읽으면서 exact attention을 계산한다. 논문 평가에서 vLLM은 FasterTransformer와 Orca 대비 같은 latency 수준에서 `2~4×` throughput을 달성했다.

<p align="center"><img src="https://github.com/user-attachments/assets/b00d8c8a-e2f9-4429-92de-946ccda34f47" alt="PagedAttention logical to physical block mapping" width="820"></p>
<p align="center"><sub>원 논문 Figure 6 — logical KV block·block table·physical block 매핑</sub></p>

## LLM serving에서 KV cache가 왜 큰가

Autoregressive generation은 이전 token의 K/V를 layer마다 저장한다. Request 하나의 cache byte는 대략

```math
2\,\mathrm{num\_layers}\,\mathrm{num\_kv\_heads}\,\mathrm{head\_dim}
\,\mathrm{sequence\_length}\,\mathrm{bytes\_per\_element}
```

이다. 논문 예시의 OPT-13B는 최대 길이 2,048에서 request 하나의 KV cache가 최대 약 `1.6 GB`다. GPU memory가 수십 GB여도 동시에 수용할 수 있는 request는 많지 않다.

Serving throughput을 높이려면 여러 request를 batch해야 하지만, batch size가 KV cache capacity에 제한된다. GPU FLOPs가 빨라도 cache를 비효율적으로 배치하면 compute unit이 놀게 된다.

## 기존 contiguous allocation의 세 가지 낭비

### Reserved memory

Output 길이를 모르므로 최대 길이까지 미리 잡는다. 아직 생성되지 않은 미래 token space가 request가 끝날 때까지 다른 request에 쓰이지 못한다.

### Internal fragmentation

실제 sequence가 allocation보다 짧으면 chunk 내부 unused space가 남는다.

### External fragmentation

Free memory 총량은 충분해도 contiguous 큰 chunk가 없어 새 request를 배치하지 못한다.

논문 측정에서 기존 방식의 effective KV memory utilization이 최저 `20.4%`까지 떨어졌다. 나머지가 모두 영구 손실은 아니더라도 serving 중 concurrent batch를 제한한다.

## Virtual memory 비유

PagedAttention의 대응은 다음과 같다.

| 운영체제 | vLLM KV cache |
| --- | --- |
| Virtual address | Sequence의 logical token position |
| Page | Logical KV block |
| Physical frame | GPU physical KV block |
| Page table | Request별 block table |
| Copy-on-write | Shared prefix block 분기 |

한 request의 logical block 0,1,2가 physical block 7,1,9처럼 떨어져 있어도 된다. Model은 logical 순서를 보고, kernel이 block table로 실제 주소를 변환한다.

## Block 구조

Block 하나는 고정 token 수 `B_s`의 K/V를 모든 layer 또는 layer별 cache layout에 저장한다. 새 request의 prompt를 block으로 채우고, generation이 진행돼 마지막 logical block이 가득 차면 free physical block을 하나 더 할당한다.

```text
logical blocks:  [0] [1] [2]
physical blocks: [7] [1] [9]
block table:      0->7, 1->1, 2->9
```

Request가 끝나면 reference count가 0인 physical block을 free pool로 즉시 반환한다. Physical pool이 fixed-size block으로만 구성되어 external fragmentation이 사실상 사라진다.

## PagedAttention kernel

Decode query `q_t`는 모든 이전 logical block의 K/V를 봐야 한다. Kernel은 request block table을 읽어 각 logical block의 physical address를 찾고 attention을 계산한다.

```text
for logical block b in sequence:
    p = block_table[request, b]
    K_block, V_block = physical_kv_pool[p]
    scores = q_t @ K_block^T
    update online softmax and output
```

Non-contiguous block을 하나의 contiguous tensor로 복사한 뒤 attention하면 overhead가 크다. PagedAttention은 address translation과 block read, attention을 fuse한다. Warp가 block 단위로 coalesced read하도록 memory layout을 설계한다.

Attention 결과는 standard KV cache를 쓴 것과 같다. 차이는 K/V의 물리적 배치다.

## Block size trade-off

작은 block은 마지막 block 낭비가 적고 sharing granularity가 세밀하지만 다음 overhead가 늘어난다.

- block table entry 수
- address lookup와 branch
- 작은 memory transaction
- allocator와 scheduling metadata

큰 block은 kernel 효율이 좋지만 internal fragmentation과 copy-on-write 복사량이 늘고, 짧은 sequence에는 불리하다.

논문은 ShareGPT에서 16~128 token block이 좋았고, 짧은 Alpaca workload에서는 16/32가 좋았다. 기본 block size로 16을 선택한다.

## Prefix sharing

### Parallel sampling

같은 prompt에서 여러 output을 sample하면 prompt KV는 동일하다. 각 sequence의 logical prompt block이 같은 physical block을 가리키게 하고 reference count를 늘린다. Generation 뒤의 다른 suffix만 새 block을 할당한다.

### Copy-on-write

공유 중인 마지막 block에 새 token을 쓰려 할 때 reference count가 1보다 크면 physical block 하나를 복사해 해당 sequence만 새 block을 가리키게 한다. OS의 fork 후 copy-on-write와 같다.

### Beam search

Beam은 매 step 후보를 복제·선택하므로 prefix sharing topology가 계속 변한다. Block reference count를 update하면 대부분 cache를 복사하지 않고 pointer mapping만 바꿀 수 있다. 분기된 마지막 block만 copy-on-write한다.

논문은 Parallel sampling에서 dataset에 따라 `6.1~30.5%`, beam search에서 `37.6~66.3%`의 cache memory sharing 절감을 보고한다.

## Scheduler와 continuous batching

vLLM은 iteration마다 진행 가능한 sequence를 골라 새 token block을 할당하고 하나의 batch로 실행한다. Request가 서로 다른 시점에 도착하고 끝나도 iteration boundary에서 batch 구성을 바꾼다.

Physical block이 부족하면 sequence 단위로 preemption한다. 논문은 두 recovery 방법을 지원한다.

- **Swapping**: KV block을 CPU memory로 이동했다가 복원
- **Recomputation**: prompt와 이미 생성한 token으로 KV를 다시 계산

작은 block에서는 recomputation이 swap보다 유리할 수 있고, block이 커질수록 swap의 상대 비용이 낮아진다. 모든 block이 함께 쓰인다는 application semantics를 이용해 sequence 단위 all-or-nothing eviction을 한다.

## 시스템 구성

vLLM은 다음 요소를 co-design한다.

```text
request scheduler
GPU/CPU block allocator
per-request block table
PagedAttention CUDA kernel
block write / read fusion
copy-on-write block-copy kernel
distributed worker coordination
```

PagedAttention kernel 하나만 기존 server에 넣는다고 같은 throughput이 나오는 것이 아니다. Memory manager가 더 큰 batch를 허용하고 scheduler가 그 capacity를 채워야 end-to-end 이득이 난다.

## 실험 결과: Serving throughput

OPT와 LLaMA 계열, ShareGPT/Alpaca trace, 여러 model·GPU 구성에서 평가한다. vLLM은 동일한 normalized latency 수준에서 FasterTransformer와 Orca 대비 대체로 `2~4×` 높은 throughput을 보였다.

ShareGPT에서 Orca 대비 sustain 가능한 request rate가 약 `1.7~2.7×` 늘었고, OPT-13B에서는 일부 설정에서 같은 latency로 약 `2.2×` 더 많은 request를 처리했다. 긴 sequence, 큰 model, beam/parallel sampling처럼 sharing이 많은 조건에서 이득이 커졌다.

## Memory sharing 결과

| Decoding 방식 | Alpaca memory saving | ShareGPT memory saving |
| --- | ---: | ---: |
| Parallel sampling | 6.1~9.8% | 16.2~30.5% |
| Beam search | 37.6~55.2% | 44.3~66.3% |

Beam 후보가 긴 prefix를 공유하므로 savings가 특히 크다. 기존 시스템의 반복 KV copy도 block table remapping으로 줄어든다.

## Kernel overhead

Indirect address와 branch 때문에 PagedAttention kernel 자체는 contiguous optimized attention보다 약 `20~26%` latency가 높을 수 있다. 그럼에도 전체 시스템은 memory utilization이 높아져 더 큰 batch를 실행하므로 throughput이 크게 개선된다.

이 결과는 중요한 시스템 교훈이다.

```text
single-request kernel latency가 약간 나빠져도
memory capacity로 batch를 크게 만들면
service throughput은 좋아질 수 있다.
```

Latency 최적화와 throughput 최적화의 목적함수가 다르다.

## 장점과 기여

- Dynamic KV cache 문제를 virtual-memory abstraction으로 명확히 재정의했다.
- Contiguous allocation의 reserved/internal/external fragmentation을 제거했다.
- Block table 기반 exact attention kernel을 구현했다.
- Reference count와 copy-on-write로 sampling·beam·prefix sharing을 지원했다.
- Scheduler, allocator, distributed execution까지 co-design해 `2~4×` throughput을 보였다.

## 한계와 비판적 관점

### 1. Address indirection overhead

Block table lookup과 non-contiguous read 때문에 kernel 자체는 더 느릴 수 있다. Batch가 memory-bound가 아니면 이득이 작거나 손해가 날 수 있다.

### 2. Block size가 workload-dependent

짧은 output에는 큰 block이 낭비고, 긴 output에는 지나치게 작은 block이 metadata와 kernel overhead를 늘린다. 고정 16이 모든 workload에 최적은 아니다.

### 3. Scheduler 복잡도

Preemption, swap/recompute, beam reference count, distributed block id를 정확히 동기화해야 한다. Memory leak나 stale mapping은 잘못된 generation으로 이어질 수 있다.

### 4. KV cache 크기 자체는 줄이지 않는다

Fragmentation을 없애지만 token당 K/V element 수는 모델 architecture와 같다. MQA/GQA/MLA 또는 quantization과 결합해야 cache의 본질적 크기가 줄어든다.

### 5. Single-request latency 목적에는 제한

동시 request가 적고 cache가 충분하면 paging overhead만 남을 수 있다. 논문의 주 목표는 high-throughput online serving이다.

## FlashAttention과 차이

| 방법 | 해결하는 memory | 핵심 문제 |
| --- | --- | --- |
| FlashAttention | 한 attention 연산의 N² temporary | HBM IO와 activation materialization |
| PagedAttention | request 사이 장기 KV cache | fragmentation, allocation, sharing |

둘은 이름은 비슷하지만 다른 계층을 최적화하며 함께 사용할 수 있다.

## 구현 체크리스트

- Logical block 순서와 physical block table mapping이 일치하는가?
- 마지막 partial block의 valid token mask가 정확한가?
- Reference count가 fork/free/copy-on-write마다 원자적으로 갱신되는가?
- 공유 block을 쓰기 전에 copy-on-write가 수행되는가?
- Block read와 attention이 fuse되어 불필요한 contiguous copy가 없는가?
- Preemption 후 swap/recompute 결과가 원 sequence와 같은가?
- Kernel latency뿐 아니라 batch capacity와 request throughput을 측정하는가?

## 온디바이스 관점

단일 사용자 온디바이스 LLM에서는 multi-request throughput보다 fragmentation 이득이 작을 수 있다. 그래도 여러 대화 session, speculative branch, beam search, multimodal prefix cache를 동시에 유지한다면 paged KV pool과 prefix sharing이 유용하다.

Mobile OS의 unified memory와 NPU 전용 memory 사이 copy가 비싸므로 block swapping보다 recomputation이 유리할 수 있다. Hardware가 scatter-gather DMA나 indirect block read를 지원하는지도 중요하다. 작은 RAM에서는 GQA/MLA와 paging을 함께 써야 한다.

## 최종 평가

PagedAttention의 핵심은 attention 수학이 아니라 **동적으로 성장하고 공유되는 KV cache를 어떤 주소 공간으로 표현할 것인가**다. Virtual page, block table, reference count, copy-on-write를 GPU serving에 맞게 재해석해 낭비를 마지막 block 하나로 제한하고 batch capacity를 크게 늘렸다. Kernel 단독으로는 indirection overhead가 있지만 시스템 전체 throughput은 `2~4×` 개선된다. LLM serving이 model architecture만이 아니라 memory management와 scheduling의 문제임을 확립한 대표 논문이다.

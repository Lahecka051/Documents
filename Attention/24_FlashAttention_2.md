# 24. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

## 논문 정보

- 제목: **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**
- 저자: Tri Dao
- 발표: 2023
- 핵심 키워드: exact attention, GPU occupancy, work partitioning, warp communication, online softmax

## 한눈에 보는 요약

FlashAttention-2(FA2)는 FlashAttention의 IO-aware tiling과 정확한 softmax를 유지하면서 GPU 작업 배치를 다시 설계한다. 1세대가 `[N,N]` HBM materialization을 없애 memory 병목을 해결했다면, 2세대는 그 kernel이 A100 peak의 25~40%만 사용하던 이유를 분석해 다음 세 부분을 개선한다.

1. online softmax를 재배열해 느린 non-matmul FP32 연산을 줄인다.
2. batch와 head뿐 아니라 sequence의 query block 방향으로 thread block을 병렬화한다.
3. 한 thread block 안에서 warp가 Q block을 나눠 갖게 해 shared-memory communication을 줄인다.

출력은 표준 attention과 동일하고 memory도 sequence 길이에 선형이다. A100에서 FlashAttention 대비 약 `2×`, theoretical peak의 `50~73%`를 달성했으며 GPT-style model 학습에서 GPU당 최대 `225 TFLOPs/s`, 약 72% model FLOPs utilization을 보고한다.

<p align="center"><img src="https://github.com/user-attachments/assets/ba17f3e6-5c67-4d6d-9456-e9bd4ccc871c" alt="FlashAttention-2 worker parallelism" width="820"></p>
<p align="center"><sub>원 논문 Figure 2 — attention block을 worker에 분할하는 FlashAttention-2</sub></p>

## 왜 FlashAttention-1도 충분히 빠르지 않았는가

FA1은 HBM IO를 크게 줄였지만 fused kernel 내부에서 다음 병목이 남았다.

- batch×head 수가 GPU SM 수보다 작으면 thread block이 부족해 occupancy가 낮다.
- warp 사이에서 partial output을 합치느라 shared memory read/write가 많다.
- softmax의 max, exp, rescale 같은 FP32 scalar/vector operation이 Tensor Core GEMM보다 훨씬 느리다.

A100의 theoretical throughput 예를 들면 FP16/BF16 matmul은 약 `312 TFLOPs/s`, FP32 non-matmul은 약 `19.5 TFLOPs/s`다. 같은 FLOP 하나라도 후자가 약 16배 비싼 셈이다. 따라서 총 FLOP 중 작은 비율인 softmax/rescale가 전체 kernel을 막을 수 있다.

## 기본 attention과 online softmax

FA2도 다음 exact output을 계산한다.

```math
\begin{aligned}
S&=\frac{QK^{\top}}{\sqrt d},\\
P&=\operatorname{softmax}(S),\\
O&=PV.
\end{aligned}
```

Q/K/V tile을 SRAM에 올리고 row별 running maximum `m`과 sum `l`을 유지한다. 새 key block `j`에 대해

```math
\begin{aligned}
m_{\mathrm{new}}&=\max\!\left(m_{\mathrm{old}},\operatorname{rowmax}(S_{ij})\right),\\
l_{\mathrm{new}}&=\exp\!\left(m_{\mathrm{old}}-m_{\mathrm{new}}\right)l_{\mathrm{old}}
+\operatorname{rowsum}\!\left(\exp(S_{ij}-m_{\mathrm{new}})\right).
\end{aligned}
```

로 exact softmax를 합친다는 원리는 같다.

## 개선 1: Non-matmul FLOPs 줄이기

FA1은 매 block마다 normalized output `O`를 유지하면서 기존 accumulator를 `l`과 max 변화에 따라 여러 번 rescale한다. FA2는 unnormalized output accumulator를 더 오래 유지하고 normalization을 뒤로 미뤄 rescale 연산을 줄인다.

개념적으로는

```math
\begin{aligned}
\tilde O_{\mathrm{new}}
&=\exp\!\left(m_{\mathrm{old}}-m_{\mathrm{new}}\right)\tilde O_{\mathrm{old}}
+\exp\!\left(S_{ij}-m_{\mathrm{new}}\right)V_j,\\
O&=\frac{\tilde O_{\mathrm{final}}}{l_{\mathrm{final}}}.
\end{aligned}
```

형태다. Matmul FLOPs는 거의 그대로지만 FP32 multiply/divide와 memory movement가 줄어든다. 이 변경은 approximation이 아니라 algebraic 재배열이다.

## 개선 2: Sequence 방향 thread-block parallelism

### FA1의 병렬 단위

주로 `(batch, head)`마다 하나의 thread block을 배치한다. Batch가 작거나 head가 적으면 활성 block 수가 GPU SM 수보다 적어 많은 SM이 빈다.

```text
parallel blocks ≈ batch × heads
```

### FA2의 병렬 단위

각 head의 query sequence도 여러 row block으로 나눠 독립 thread block에 할당한다.

```math
\text{parallel blocks}\approx\text{batch}\times\text{heads}\times\left\lceil\frac{N}{B_r}\right\rceil
```

각 Q block의 output row는 서로 독립이므로 forward에서 synchronization 없이 병렬화할 수 있다. 특히 long sequence, small batch, few heads setting에서 occupancy가 크게 높아진다.

Backward에서도 sequence 축 병렬화를 강화한다. `dQ`와 `dK,dV`의 write conflict를 피하도록 work를 분배하고 필요한 reduction을 schedule한다.

## 개선 3: Thread block 내부 warp partitioning

FA1은 warp들이 K/V column 방향으로 일을 나누고 각자 partial output을 계산한 뒤 shared memory에서 합치는 `split-K`에 가까운 구성을 사용했다. Partial output communication이 비싸다.

FA2는 warp들이 Q row를 나눠 갖는다.

```text
all warps share K_j, V_j tile
warp 0 owns subset of Q_i rows and O_i rows
warp 1 owns another subset
...
```

각 warp가 자신의 output row를 끝까지 유지하므로 warp 간 output reduction이 사라진다. Q tile도 가능한 한 register에 유지하고 K/V만 shared memory를 통해 공유한다. 이 변경이 shared-memory read/write와 synchronization을 줄인다.

## Causal attention 최적화

Query block row index와 key block column index를 비교하면 세 영역이 생긴다.

```text
key block entirely in future -> skip tile GEMM
diagonal boundary            -> apply causal mask
key block entirely in past   -> no element mask
```

미래 절반 block을 아예 계산하지 않으므로 causal attention은 non-causal보다 이론 FLOPs가 절반에 가깝다. FA2는 diagonal 한 block에만 fine-grained mask를 적용하고 나머지를 skip/normal GEMM으로 처리해 약 `1.7~1.8×` speedup을 얻는다.

## Forward algorithm의 shape

```text
Q_i          : [B_r,d]
K_j,V_j      : [B_c,d]
S_ij         : [B_r,B_c]
m_i,l_i      : [B_r]
O_tilde_i    : [B_r,d]
```

FA2에서는 Q block 하나를 thread block이 담당해 모든 K/V block을 순회한다. 이는 FA1의 loop order와 work ownership을 조정한 것으로, output row가 하나의 block에 귀속되어 parallel write conflict를 피한다.

## 정확성·memory 복잡도

- Online softmax recurrence는 exact하다.
- 표준 scaled dot-product attention과 수치 오차 범위에서 같은 출력을 낸다.
- `[N,N]` score/probability는 HBM에 저장하지 않는다.
- 추가 memory는 row logsumexp와 output 등 `O(N)`이다.
- FLOPs는 dense attention과 같은 `O(N²d)`다.

즉 FA2의 성능 향상은 model architecture나 근사 품질과 관계없이 kernel scheduling에서 나온다.

## 성능 결과: Attention kernel

Sequence length 512~16K, head dimension 64/128, causal/non-causal 조건에서 A100 성능을 측정했다. FA2는 최대 약 `230 TFLOPs/s`, A100 FP16/BF16 theoretical peak의 `73%`에 도달한다. FA1 대비 대체로 약 `2×` 빠르며, batch×head가 작고 sequence가 길어 sequence parallelism이 필요한 구간에서 이득이 크다.

Head dimension 128은 register와 SRAM pressure가 커 tile 선택이 제한된다. Causal mask 여부에 따라서도 effective FLOP 계산과 speed가 달라진다. 하나의 숫자보다 shape별 kernel curve를 봐야 한다.

## End-to-end GPT training

8×A100 환경의 논문 표를 요약하면 다음과 같다.

| 모델 | Context | Baseline | FlashAttention | FlashAttention-2 |
| --- | ---: | ---: | ---: | ---: |
| GPT3-1.3B | 2K | 142 | 189 | **196 TFLOPs/s** |
| GPT3-1.3B | 8K | 72 | 170 | **220 TFLOPs/s** |
| GPT3-2.7B | 2K | 149 | 189 | **205 TFLOPs/s** |
| GPT3-2.7B | 8K | 80 | 175 | **225 TFLOPs/s** |

8K context에서 baseline 대비 최대 약 `2.8×`, FA1 대비 약 `1.3×` end-to-end speedup이다. Kernel 단독 `2×`가 전체 모델에서 `1.3×`가 되는 이유는 MLP, communication, optimizer 등 attention 밖의 시간이 남기 때문이다.

최대 `225 TFLOPs/s/GPU`는 저자 정의에서 약 `72% model FLOPs utilization`에 해당한다.

## H100 초기 결과와 논문의 경계

FA2를 H100에서 기존 instruction 방식으로 실행했을 때 최대 약 `335 TFLOPs/s`를 보고한다. 논문은 Hopper의 TMA와 4세대 Tensor Core를 직접 활용하면 추가 `1.5~2×` 가능하다고 예상했으며, 이것이 FlashAttention-3의 출발점이 된다.

따라서 FA2는 hardware-independent 최종 algorithm이라기보다 A100/Ampere의 execution model에 맞춘 scheduling 개선이다.

## 장점과 기여

- FA1의 정확성과 linear activation memory를 유지했다.
- FLOP 종류별 hardware throughput 차이를 고려해 non-matmul 연산을 줄였다.
- sequence block 병렬화로 small batch/few-head의 occupancy를 개선했다.
- warp별 Q ownership으로 shared-memory communication을 줄였다.
- attention kernel뿐 아니라 GPT training에서 높은 model utilization을 보였다.

## 한계와 비판적 관점

### 1. Dense quadratic compute

FA1과 마찬가지로 memory는 줄지만 `N²d` GEMM은 남는다. context가 극단적으로 길어지면 compute 한계가 지배한다.

### 2. GPU architecture 특화

Warp, shared memory, Tensor Core throughput 비율에 맞춘 설계다. 다른 GPU 세대나 NPU에서는 optimal work partitioning이 다르다.

### 3. Shape별 tuning

Head dimension, sequence length, causal 여부에 따라 tile과 warp 수를 선택해야 한다. 범용 kernel 하나로 peak를 내기 어렵다.

### 4. Register pressure와 occupancy trade-off

Q/O를 warp register에 오래 유지하면 communication은 줄지만 register 사용량이 늘어 resident block 수를 제한할 수 있다.

### 5. End-to-end 이득의 상한

Attention이 전체 시간의 일부일 때 kernel speedup은 Amdahl's law에 제한된다. Distributed communication과 MLP가 병목이면 FA2 추가 개선 폭이 작다.

## FA1과 FA2 비교

| 축 | FlashAttention | FlashAttention-2 |
| --- | --- | --- |
| 핵심 문제 | HBM IO와 N² activation | occupancy와 on-chip work partition |
| 정확성 | exact | exact |
| 추가 memory | `O(N)` | `O(N)` |
| 병렬 단위 | 주로 batch×head | batch×head×Q blocks |
| Warp work | K/V 방향 분할·reduction | Q row 소유·K/V 공유 |
| 성능 | A100 peak 25~40% | 50~73% |

## 구현 체크리스트

- normalized output과 unnormalized accumulator 수식이 섞이지 않았는가?
- final `O/l` normalization이 정확한가?
- Q block별 thread block이 같은 output을 중복 write하지 않는가?
- K/V shared load와 Q/O register ownership이 warp layout과 맞는가?
- causal future tile이 실제 GEMM 전에 skip되는가?
- head dimension별 register spill과 occupancy를 profiler로 확인했는가?
- kernel TFLOPs와 실제 tokens/s를 구분해 보고했는가?

## 온디바이스 관점

FA2의 직접 CUDA 구현은 mobile NPU로 옮길 수 없지만 원칙은 적용된다. Local scratchpad에 어떤 tile을 남기고, output row를 어떤 processing element가 소유하며, expensive scalar operation을 얼마나 줄이는지가 중요하다. 정적 sequence shape에서는 compile-time tile scheduling이 가능하다.

다만 edge accelerator는 exp/div throughput, SRAM 크기, thread/warp model이 다르다. GPU용 work partition을 복사하기보다 device의 matrix engine과 vector unit을 동시에 사용하도록 algorithm을 다시 배치해야 한다.

## 최종 평가

FlashAttention-2는 새로운 attention 수식을 제안하지 않는다. 대신 exact IO-aware attention을 **GPU에서 실제 peak에 가깝게 실행하려면 누가 어떤 output을 소유하고, 어느 축을 병렬화하며, 어떤 FLOP를 줄여야 하는가**를 해결한다. FA1이 memory hierarchy를 attention algorithm의 일부로 만들었다면, FA2는 occupancy와 warp communication까지 algorithm 설계에 포함시켰다. Long-context training의 사실상 표준 kernel로 발전한 이유를 잘 보여주는 시스템 논문이다.

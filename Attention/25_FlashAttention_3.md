# 25. FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision

## 논문 정보

- 제목: **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision**
- 저자: Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao
- 발표: 2024
- 핵심 키워드: Hopper GPU, TMA, WGMMA, warp specialization, asynchronous pipeline, FP8, incoherent processing

## 한눈에 보는 요약

FlashAttention-3(FA3)는 Hopper H100의 새로운 비동기 하드웨어를 직접 활용하도록 exact attention kernel을 재설계한다. FlashAttention-2가 H100에서 약 35% utilization에 머무는 이유를 data movement, Tensor Core GEMM, softmax가 충분히 겹치지 않는 scheduling 문제로 본다.

세 가지 핵심 기법은 다음과 같다.

1. **Producer–consumer warp specialization**: 일부 warp는 TMA로 Q/K/V tile을 옮기고, 다른 warpgroup은 WGMMA로 행렬 곱을 수행한다.
2. **GEMM–softmax overlap**: 한 warpgroup이 score block softmax를 계산하는 동안 다른 warpgroup이 다음 GEMM을 수행하는 ping-pong pipeline을 만든다.
3. **정확한 FP8 attention**: block-wise quantization과 random orthogonal mixing(incoherent processing)으로 outlier를 완화한다.

FP16 forward는 FlashAttention-2 대비 `1.5~2.0×`, 최대 `740 TFLOPs/s`와 H100 peak의 약 75%를 달성한다. FP8은 약 `1.2 PFLOPs/s`에 도달하며, 제안한 정확도 기법으로 baseline FP8 attention보다 numerical error를 `2.6×` 줄인다.

<p align="center"><img src="https://github.com/user-attachments/assets/4f1a2975-4432-45c4-a789-8324a123381c" alt="FlashAttention-3 warpgroup ping-pong scheduling" width="820"></p>
<p align="center"><sub>원 논문 Figure 1 원본·확대 패널 — warpgroup ping-pong scheduling과 GEMM-softmax 중첩</sub></p>

## Hopper에서 병목이 바뀌었다

H100은 matrix multiplication용 Tensor Core가 매우 빨라졌다. 논문이 제시한 예에서 FP16 matmul peak는 약 `989 TFLOPs/s`지만 exponential 같은 special function throughput은 약 `3.9 TFLOPs/s`다. FP8 Tensor Core는 matmul이 더 빨라지지만 exp unit은 같은 속도이므로 softmax 상대 비용이 더 커진다.

또한 Hopper에는 다음 기능이 있다.

- **TMA (Tensor Memory Accelerator)**: 전용 unit이 global memory와 shared memory 사이 tile copy를 비동기 수행한다.
- **WGMMA**: warpgroup 단위 asynchronous Tensor Core matrix multiplication.
- **setmaxnreg**: producer와 consumer warpgroup 사이 register quota를 동적으로 재분배한다.

기존처럼 모든 warp가 load→GEMM→softmax를 순차 수행하면 이 하드웨어의 동시성을 활용하지 못한다.

## 기본 attention과 정확성

FA3도 다음 표준 attention을 계산한다.

```math
\begin{aligned}
S&=\alpha QK^{\top},\\
P&=\operatorname{softmax}(S),\\
O&=PV.
\end{aligned}
```

Tiling, online softmax, backward recomputation 원리는 이전 FlashAttention과 같다. FP16/BF16 mode에서는 algorithmic approximation이 없다. FP8 mode에서는 Q/K/V quantization으로 수치 근사가 추가되지만 attention graph 자체는 dense하다.

## Producer–consumer warp specialization

한 CTA(thread block)의 warp를 역할별로 나눈다.

```text
producer warp:
  TMA 명령으로 Q/K/V tile을 GMEM(HBM) -> SMEM에 비동기 load

consumer warpgroups:
  WGMMA로 QK^T, PV 계산
  online softmax와 accumulator update
```

TMA load를 issue한 producer는 copy 완료를 기다리며 CUDA core를 점유하지 않는다. Consumer는 이전 buffer를 계산하는 동안 producer가 다음 buffer를 채운다. Circular shared-memory buffer와 barrier로 tile의 ready/free 상태를 관리한다.

Producer는 register가 거의 필요 없고 consumer는 accumulator 때문에 많이 필요하다. `setmaxnreg`로 producer quota를 줄여 consumer에 더 많은 register를 배정한다.

## Ping-pong scheduling

단일 warpgroup 안에서는 `QK^T → softmax → PV` dependency가 있어 완전히 겹치기 어렵다. FA3는 두 consumer warpgroup이 서로 다른 tile stage를 처리하게 한다.

```text
time t:
  warpgroup A -> score block의 softmax
  warpgroup B -> 다음/다른 block의 WGMMA

time t+1:
  warpgroup A -> WGMMA
  warpgroup B -> softmax
```

두 group이 역할을 교대해 matrix engine이 softmax 동안 놀지 않게 한다. 실제 scheduler는 그림처럼 완벽히 교대하지 않지만, 논문 예에서 FP16 head dimension 128 forward가 약 `570 → 620~640 TFLOPs/s`로 개선되었다.

## Intra-warpgroup GEMM–softmax pipeline

Warpgroup 하나 안에서도 iteration 간 dependency 일부를 분해한다. Key block `j`의 softmax를 계산하는 동안 다음 stage의 WGMMA를 issue하도록 2-stage pipeline을 구성한다.

온라인 softmax는 block maximum과 exponential sum 때문에 순차 dependency가 있지만, 모든 instruction이 같은 데이터에 의존하는 것은 아니다. WGMMA가 asynchronous하게 진행되는 동안 CUDA core가 max/exp/rescale을 수행하도록 instruction ordering을 바꾼다.

논문은 compiler scheduling과 register lifetime이 실제 overlap에 큰 영향을 준다고 지적한다. 이 단계는 단순 수식 변경보다 low-level instruction pipeline 설계에 가깝다.

## TMA와 WGMMA의 memory layout

첫 번째 GEMM은 `Q K^T`, 두 번째는 `P V`다. FP16과 FP8 WGMMA는 operand layout 제약이 다르고, 두 GEMM을 한 kernel에서 연속 실행하려면 첫 GEMM의 accumulator와 두 번째 GEMM input layout을 맞춰야 한다.

특히 FP8 WGMMA의 operand는 k-major layout 제약이 있어 V tile 또는 softmax output의 transpose가 필요하다. FA3는 LDSM/STSM instruction을 사용한 in-kernel transpose와 register/shared-memory layout 변환을 설계한다.

이 부분은 “FP8로 cast하면 2배 빨라진다”가 성립하지 않는 이유다. Quantization format뿐 아니라 Tensor Core가 요구하는 물리적 tile layout을 맞춰야 한다.

## FP8의 정확도 문제

FP8 E4M3는 mantissa가 3 bit이므로 FP16보다 quantization error가 크다. Attention은 Q/K의 outlier가 dot product와 softmax에 증폭될 수 있고, V error도 output에 직접 들어간다.

### Block-wise quantization

Tensor 전체에 scale 하나를 쓰지 않고 작은 block마다 scale을 정한다.

```math
x_{\mathrm{FP8}}=\operatorname{round}\!\left(\frac{x}{\mathrm{scale}_{\mathrm{block}}}\right)
```

Outlier가 있는 block만 큰 scale을 사용하므로 다른 block의 작은 값이 해상도를 잃지 않는다. Scale metadata와 conversion 비용은 늘지만 error가 줄어든다.

### Incoherent processing

Q와 K에 같은 random orthogonal matrix `M`을 곱한 뒤 quantize한다.

```math
\begin{aligned}
Q'&=QM,
&K'&=KM,\\
Q'{K'}^{\top}&=QMM^{\top}K^{\top}=QK^{\top}.
\end{aligned}
```

Exact arithmetic에서 dot product는 변하지 않는다. 그러나 각 coordinate가 여러 원래 coordinate의 random mixture가 되어 sparse outlier가 dimension 전체에 퍼지고 magnitude distribution이 균일해진다. 그 결과 FP8 quantization error가 감소한다.

M은 Hadamard transform과 random sign처럼 빠른 orthogonal transform으로 구현할 수 있다. 수학적으로 score를 보존하지만 finite-precision transform cost와 오차는 남는다.

### FP32 softmax state

Score accumulator, row maximum, exponential sum 등 민감한 중간 값을 FP32로 유지한다. 논문은 이 때문에 FP16 FA2/FA3가 중간 softmax를 낮은 precision으로 저장하는 일반 baseline보다 오히려 낮은 error를 보일 수 있다고 설명한다.

## 전체 pipeline

```text
TMA producer loads K_j,V_j into circular SMEM buffer
              ↓ async barrier
consumer WG A issues WGMMA for Q_i K_j^T
consumer WG B performs softmax on previous score tile
              ↓
WGMMA computes P_ij V_j while next load/GEMM overlaps
              ↓
online max/sum rescales O accumulator
              ↓
only final O and logsumexp written to HBM
```

FA3의 “algorithm”에는 수식뿐 아니라 warp role, buffer state, instruction issue 순서가 포함된다.

## 성능 결과

### FP16/BF16

H100에서 FA2 대비 forward `1.5~2.0×`, backward `1.5~1.75×` speedup을 보고한다. Forward 최대 성능은 약 `740 TFLOPs/s`, H100 theoretical peak의 약 `75%`다.

### FP8

Forward에서 약 `1.2 PFLOPs/s`에 도달한다. Head dimension 64에서는 경쟁 구현보다 앞서고, 128/256에서는 대체로 비슷한 수준이라고 설명한다. Small sequence와 causal setting에서는 FP8 kernel이 persistent/load-balancing 최적화가 덜 되어 FP16만큼 일관된 우위를 보이지 않는다.

### Pipeline ablation

고정 설정 `{batch=4, seqlen=8448, heads=16, hdim=128}`에서 결과는 다음과 같다.

| 구성 | 시간 | 처리량 |
| --- | ---: | ---: |
| Pipelining 없음, warp specialization 있음 | 4.021 ms | 582 TFLOPs/s |
| Pipelining 있음, warp specialization 없음 | 4.105 ms | 570 TFLOPs/s |
| 둘 다 적용 | 더 짧음 | **661 TFLOPs/s** |

두 기법이 상호보완적이며 각각만으로는 peak에 도달하지 못함을 보여준다.

### Numerical error

FP16 FlashAttention-2/3는 FP32 softmax intermediate 덕분에 표준 구현보다 RMSE가 약 `1.7×` 낮았다. FP8 FA3는 block quantization과 incoherent processing을 결합해 per-tensor scaling baseline보다 error를 `2.6×` 줄였다.

이는 속도 수치만큼 중요하다. FP8 peak를 얻더라도 softmax output error가 크면 모델 품질을 보존할 수 없기 때문이다.

## 장점과 기여

- Hopper의 TMA/WGMMA를 attention pipeline에 직접 통합했다.
- producer–consumer specialization으로 data movement와 compute를 겹쳤다.
- ping-pong 및 intra-warpgroup scheduling으로 softmax를 GEMM 아래 숨겼다.
- FP8 attention의 layout 문제와 numerical error를 함께 해결했다.
- FP16 740 TFLOPs/s, FP8 1.2 PFLOPs/s라는 높은 utilization을 보였다.

## 한계와 비판적 관점

### 1. Hopper 특화

TMA, WGMMA, register reallocation에 의존하므로 Ampere나 다른 accelerator에서 동일 kernel을 사용할 수 없다. Hardware 세대마다 재설계가 필요하다.

### 2. 복잡한 correctness surface

Async barrier, circular buffer, warpgroup synchronization, layout transpose, causal boundary, online softmax가 한 kernel에 결합된다. Race condition과 silent numerical error를 검증하기 어렵다.

### 3. FP8은 exact가 아니다

FP16 mode의 attention 수학은 exact지만 FP8 Q/K/V quantization은 근사다. RMSE 개선이 모든 model·distribution의 downstream quality 보존을 보장하지 않는다.

### 4. Dense quadratic compute

더 높은 hardware utilization은 같은 `N²d` FLOPs를 더 빨리 수행하는 것이다. context가 커질수록 총 compute는 여전히 quadratic이다.

### 5. Training 중심

논문은 forward/backward throughput에 초점을 둔다. One-token decode는 GEMM shape가 작고 KV cache bandwidth가 지배하므로 같은 pipeline 이득이 그대로 적용되지 않는다.

## 세 세대 비교

| 세대 | 주 병목 | 핵심 해법 | 대표 hardware |
| --- | --- | --- | --- |
| FlashAttention | HBM IO, N² activation | tiling, online softmax, recompute | A100 포함 GPU |
| FlashAttention-2 | occupancy, warp communication | sequence parallelism, Q-row ownership | Ampere/A100 |
| FlashAttention-3 | async overlap, FP8 accuracy | TMA/WGMMA pipeline, block quant, mixing | Hopper/H100 |

## 구현 체크리스트

- TMA producer와 WGMMA consumer의 buffer barrier가 정확한가?
- Ping-pong warpgroup이 같은 tile을 동시에 overwrite하지 않는가?
- Online softmax의 max/sum/O accumulator는 FP32인가?
- FP8 block scale의 granularity와 metadata layout이 일치하는가?
- Q와 K에 정확히 같은 orthogonal transform을 적용하는가?
- `M M^T = I`가 구현 precision에서 충분히 유지되는가?
- Shape별 register spill, occupancy, Tensor Core utilization을 확인했는가?

## 온디바이스 관점

FA3 코드는 H100 전용이지만, 핵심 원리는 차세대 NPU에도 적용된다. DMA engine, matrix engine, vector exp unit이 독립적이라면 producer/consumer pipeline으로 겹쳐야 한다. 저정밀 attention에서는 per-block scale과 outlier smoothing이 특히 유용하다.

반면 mobile chip은 scratchpad와 synchronization primitive가 더 제한적이고 FP8을 지원하지 않을 수 있다. INT8/INT4에서 orthogonal mixing cost가 이득을 상쇄하는지, 전력과 latency를 함께 측정해야 한다.

## 최종 평가

FlashAttention-3는 attention 최적화가 이제 수학식이나 CUDA tiling만의 문제가 아니라 **비동기 hardware unit의 scheduling과 low-precision numerical design**의 문제임을 보여준다. TMA가 데이터를 옮기는 동안 WGMMA와 softmax를 겹치고, FP8 outlier를 block scale과 orthogonal mixing으로 제어해 H100 성능을 끌어낸다. 범용성은 낮고 구현 난도가 높지만, hardware capability를 algorithm 구조에 흡수한 대표적인 co-design 연구다.

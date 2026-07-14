# 84. TVM: An Automated End-to-End Optimizing Compiler for Deep Learning

## 논문 정보

- 제목: TVM: An Automated End-to-End Optimizing Compiler for Deep Learning
- 저자: Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Meghan Cowan, Haichen Shen, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy
- 소속: University of Washington, AWS, Shanghai Jiao Tong University, UC Davis, Cornell
- 발표: OSDI 2018
- arXiv: [1802.04799](https://arxiv.org/abs/1802.04799)
- 원본 파일: `84_TVM.pdf`

## 한눈에 보는 요약

TVM은 TensorFlow, MXNet, PyTorch 같은 framework의 model을 받아 CPU, GPU, mobile GPU, FPGA accelerator에 맞는 deployable code를 생성하는 end-to-end deep learning compiler다. 핵심은 graph-level 최적화와 operator-level schedule 최적화를 한 stack 안에서 연결한 것이다.

기존 framework는 graph를 최적화하더라도 실제 operator 실행은 vendor library에 맡기는 경우가 많았다. 이 구조에서는 새로운 fused operator, depthwise convolution, custom accelerator intrinsic이 생길 때 library 구현이 없으면 성능이 급격히 낮아진다. TVM은 tensor expression으로 operator의 계산 의미를 쓰고, schedule로 실행 방법을 분리한다. 그리고 tile, loop order, thread binding, memory scope, tensorization, latency hiding 조합을 ML 기반 cost model로 탐색한다.

논문의 전체 flow는 다음과 같다.

```text
Framework model
  -> computational graph
  -> graph rewrite와 fusion
  -> fused tensor operator
  -> tensor expression + schedule space
  -> ML cost model 기반 search
  -> target-specific loop code
  -> LLVM/CUDA/Metal/OpenCL/accelerator backend
  -> deployable module
```

대표 결과는 다음과 같다.

- NVIDIA Titan X에서 기존 framework 대비 end-to-end 1.6-3.8배 speedup
- ARM Cortex-A53에서 TensorFlow Lite보다 빠른 ResNet/MobileNet/DQN code 생성
- Mali-T860MP4에서 ARM Compute Library보다 1.2-1.6배 speedup
- FPGA VDLA의 offload 가능한 convolution layer에서 Cortex-A9 대비 40배 speedup
- explicit latency hiding으로 FPGA accelerator peak compute utilization을 70%에서 88%로 향상

이 수치는 target, framework version, batch, datatype, model이 서로 다르므로 하나의 통합 benchmark처럼 섞어 읽으면 안 된다.

## 문제 설정: portability와 performance를 함께 얻기

hardware는 CPU, GPU, tensor accelerator마다 다르다.

- CPU: cache hierarchy, vector instruction, hardware prefetch
- GPU: SIMT thread hierarchy, shared memory, warp scheduling
- accelerator: explicit SRAM, DMA, tensor instruction, software-managed pipeline

같은 convolution도 target에 따라 최적 tile, layout, parallel axis, fusion이 다르다. vendor library는 특정 target에서 매우 빠르지만 다음 한계가 있다.

1. 새로운 operator마다 hand-tuned kernel이 필요하다.
2. fusion pattern과 layout, dtype, intrinsic 조합이 늘면 구현 수가 폭발한다.
3. embedded와 custom accelerator 지원에 큰 engineering cost가 든다.
4. graph rewrite가 library에 없는 fused operator를 만들면 최적화가 끊긴다.

TVM의 목표는 operator library를 미리 모두 작성하는 대신, high-level 계산 정의에서 target code를 자동 생성해 performance portability를 얻는 것이다.

## TVM의 세 계층

### 1. Graph-level optimizer

framework model을 computational graph로 가져와 operator 사이의 전역 관계를 최적화한다.

- operator fusion
- constant folding
- static memory planning
- data layout transformation

graph node는 tensor operation, edge는 data dependency다. 입력 shape가 고정된 inference workload에서는 shape specialization도 가능하다.

### 2. Tensor expression과 schedule

각 fused operator의 수학적 계산을 declarative하게 쓰고, loop와 memory 실행 방법은 schedule로 분리한다. 하나의 expression에 여러 schedule이 가능하다.

### 3. ML-based automated optimizer

schedule template의 tile, unroll, vectorization, thread binding 같은 knob 조합은 수십억 개가 될 수 있다. cost model이 lowered loop program의 성능을 예측하고, explorer가 유망 후보만 실제 device에서 측정한다. 측정 data는 다시 cost model을 개선한다.

## Graph-level optimization

### Computational graph representation

TVM graph는 multi-dimensional tensor를 node 사이에 전달한다. LLVM 같은 저수준 IR보다 operator 의미와 shape가 높게 보이므로 다음 변환이 쉽다.

- compile-time constant subgraph 계산
- tensor lifetime에 따른 buffer 재사용
- target-friendly layout 삽입
- producer-consumer fusion

반면 graph IR만으로는 loop tile, shared memory load, tensor instruction mapping을 결정하기 어렵다. 그래서 operator-level IR이 별도로 필요하다.

### Operator fusion rule

논문은 operator를 네 종류로 분류한다.

1. injective: element-wise add처럼 input element와 output element가 일대일 대응
2. reduction: sum처럼 여러 element를 축약
3. complex-out-fusable: conv2d처럼 계산은 복잡하지만 output element-wise op를 붙일 수 있음
4. opaque: sort처럼 내부를 fusion하기 어려움

generic rule은 다음과 같다.

- 여러 injective op는 하나의 injective kernel로 fuse
- reduction 앞의 injective producer를 reduction에 fuse 가능
- conv2d 뒤의 bias, BN, ReLU 같은 element-wise op를 output에 fuse 가능
- opaque op는 fusion boundary

Figure 4의 Titan X 측정에서 conv+BN+ReLU, depthwise conv+BN+ReLU, RNN/LSTM cell fusion은 workload에 따라 약 1.2-2배 speedup을 보였다. 핵심 원인은 intermediate tensor를 global memory에 쓰고 다시 읽는 일을 줄인 것이다.

### Data layout transformation

같은 tensor도 row-major, column-major, blocked layout으로 저장할 수 있다. 4x4 matrix primitive를 가진 accelerator라면 channel과 spatial 축을 4 단위 tile로 배치하는 편이 유리할 수 있다.

TVM은 operator마다 선호 layout을 정하고 producer와 consumer layout이 다를 때 transform을 넣는다. 좋은 graph optimizer는 transform 자체의 비용까지 포함해 전체 graph를 봐야 한다. 한 operator만 빠르게 만들기 위해 layout을 바꿨다가 다음 operator에서 transpose를 반복하면 end-to-end는 느려질 수 있다.

## Tensor expression과 compute/schedule 분리

transposed matrix multiplication 예시는 다음과 같다.

$$
C[y,x]=\sum_{k=0}^{h-1}A[k,y]B[k,x],
$$

여기서 $A\in\mathbb{R}^{h\times m}$, $B\in\mathbb{R}^{h\times n}$, $C\in\mathbb{R}^{m\times n}$다.

논문식 TVM 표현을 단순화하면 다음과 같다.

```python
m, n, h = var("m"), var("n"), var("h")
A = placeholder((h, m), name="A")
B = placeholder((h, n), name="B")
k = reduce_axis((0, h), name="k")
C = compute(
    (m, n),
    lambda y, x: sum(A[k, y] * B[k, x], axis=k),
)
```

이 코드는 output shape와 element rule만 정한다. loop order, tile, vectorization, cache 위치는 정하지 않는다. schedule primitive를 적용하면 논리적으로 같은 다양한 구현을 만든다.

```python
s = create_schedule(C.op)
yo, yi = s[C].split(C.axis[0], factor=8)
xo, xi = s[C].split(C.axis[1], factor=8)
ko, ki = s[C].split(k, factor=8)
s[C].reorder(yo, xo, ko, yi, xi, ki)
s[C].parallel(yo)
s[C].vectorize(xi)
```

실제 최적 schedule은 target별로 다르다. 이 분리가 TVM의 확장성 기반이다.

## Hardware-aware schedule primitive

### Loop transformation

split, fuse, reorder, tile, unroll, vectorize로 locality와 parallelism을 만든다. CPU에서는 cache와 SIMD width에 맞춘 tile이 중요하다.

### Thread binding과 cooperative fetching

GPU에서는 loop axis를 block/thread hierarchy에 bind한다. 여러 thread가 함께 필요한 tile을 global memory에서 shared memory로 가져오고 barrier 뒤에 재사용한다.

논문 Figure 7의 Titan X matrix multiplication에서 cooperative shared-memory fetching이 없는 TVM보다 있는 TVM이 훨씬 빠르고 cuBLAS에 가까운 성능을 보였다. shared memory scope와 synchronization을 schedule IR에 표현해야 compiler가 올바른 barrier를 삽입할 수 있다.

### Special memory scope

GPU shared memory, accelerator input buffer, accumulator buffer처럼 address space와 access rule이 다른 memory를 표시한다. 이는 단순 cache hint가 아니라 code generation rule을 바꾸는 정보다.

### Tensorization

tensorization은 SIMD vectorization을 multi-dimensional tensor intrinsic으로 확장한 개념이다. 예를 들어 target에 8x8 GEMM instruction이 있다면 다음 세 부분을 선언한다.

- intrinsic이 계산하는 수학적 behavior
- accumulator reset
- compute/update의 lowering rule

그 후 schedule의 특정 loop nest가 intrinsic pattern과 일치하면 `tensorize`로 micro-kernel call로 치환한다.

```python
CL = cache_write(C, "acc_buffer")
AL = cache_read(A, "inp_buffer")
BL = cache_read(B, "inp_buffer")
yo, xo, ko, yi, xi, ki = tile(C, 8, 8, 8)
tensorize(yi, gemm8x8_intrinsic)
```

hardware intrinsic과 schedule을 분리하므로 새 accelerator의 tensor instruction을 compiler 전체에 hard-code하지 않아도 된다. 논문은 mobile CPU의 1-2 bit bit-serial matrix-vector micro-kernel을 tensor intrinsic으로 연결해 non-tensorized version보다 최대 1.5배 speedup을 얻었다.

## Explicit memory latency hiding

CPU는 hardware prefetch와 SMT, GPU는 많은 warp context switching으로 memory latency를 숨긴다. TPU-like accelerator는 control을 단순화하고 load, execute, store pipeline을 분리한 decoupled access-execute 구조를 쓸 수 있다. 이때 dependency token을 software가 명시적으로 넣어야 한다.

TVM은 virtual thread schedule을 제공한다.

1. programmer는 여러 virtual thread가 독립적으로 load와 compute를 하는 것처럼 high-level schedule을 작성한다.
2. compiler는 read-after-write와 write-after-read dependency를 분석한다.
3. load/execute queue 사이 push/pop synchronization을 삽입한다.
4. virtual thread operation을 하나의 instruction stream으로 interleave한다.
5. hardware는 dependency가 허용하는 load-compute overlap을 실행한다.

FPGA ResNet layer 실험에서 latency hiding이 없을 때 peak compute utilization은 70%, 있을 때 88%였다. 이 결과는 같은 arithmetic unit이라도 compiler schedule이 utilization을 크게 바꿀 수 있음을 보여 준다.

## ML 기반 schedule search

### 왜 brute-force가 어려운가

tile 크기, loop 순서, unroll factor, vector width, memory scope, thread mapping 조합은 쉽게 수십억 개가 된다. 모든 후보를 compile하고 device에서 측정하는 black-box autotuning은 너무 느리다. 반대로 hand-written analytic cost model은 현대 hardware의 cache, pipeline, compiler effect를 정확히 반영하기 어렵고 target마다 다시 작성해야 한다.

TVM은 둘 사이를 택한다.

| 방식 | measurement cost | model bias | hardware 정보 필요 | 과거 data 학습 |
|---|---:|---:|---:|---:|
| black-box autotuning | 높음 | 없음 | 아니오 | 아니오 |
| predefined cost model | 없음 | 높음 | 예 | 아니오 |
| ML cost model | 낮음 | 낮음 | 아니오 | 예 |

### Cost model 입력

lowered loop AST에서 다음 feature를 추출한다.

- loop level별 buffer access count
- reuse ratio
- touched memory size
- vectorize, unroll, parallel annotation one-hot

기본 model은 XGBoost gradient tree boosting이다. objective는 absolute runtime을 정확히 맞추는 것보다 candidate의 상대 순위를 맞추는 ranking objective다. explorer는 top candidate만 실제 측정하면 되므로 이 선택이 합리적이다.

논문은 TreeRNN으로 AST를 직접 요약하는 model도 비교했지만 predictive quality는 비슷하고 XGBoost가 prediction은 약 2배 빠르며 training cost도 낮아 기본값으로 선택했다. 평균 prediction 시간은 0.67 ms로 실제 device 측정보다 수천 배 빠르다.

### Exploration loop

```python
history = []
cost_model = initialize()
states = random_schedules(template)

while budget_remains():
    candidates = parallel_simulated_annealing(states, cost_model)
    batch = select_top_predicted(candidates)
    measured = rpc_compile_run_profile(batch, target_devices)
    history.extend(measured)
    cost_model.fit(history, ranking_objective=True)
    states = continue_from_previous_search_states()

return best_measured_schedule(history)
```

초기 data가 없으면 random candidate를 측정한다. 이후 parallel simulated annealing이 cost model이 예측한 낮은 cost 쪽으로 이동한다. Figure 12의 ResNet-18 conv2d, Titan X 실험에서 ML model은 random search와 black-box genetic algorithm보다 적은 trial로 cuDNN 이상의 configuration을 찾았다.

### Distributed device pool

host에서 cross-compile하고 RPC로 Raspberry Pi, Mali GPU, NVIDIA GPU, FPGA board에 module을 upload해 실행한다. compile, deploy, profile을 자동화하므로 embedded device tuning에서 수동 작업을 줄인다. 측정 기반 search이므로 실제 target clock, driver, compiler 상태가 결과에 반영된다.

## End-to-end build와 runtime

논문 당시 API를 개념적으로 정리하면 다음과 같다.

```python
graph, params = frontend.from_framework(model)
target = target_spec("cuda")
graph = graph_rewrite(graph, target)
lib = compile_fused_operators(graph, target, tuned_schedules)
module = runtime.create(graph, lib, device)
module.set_input(**params)
module.run(data=input_tensor)
output = module.get_output(0)
```

deployable module은 optimized graph, generated operator library, parameter를 포함한다. runtime은 C++, Java, Python binding을 제공했다. 중요한 것은 compile-time tuning cost와 runtime inference cost를 분리하는 것이다. autotuning이 오래 걸려도 동일 shape와 target에 재사용된다면 deployment inference에는 직접 포함되지 않는다.

## 평가 조건을 분리해서 읽기

### Server-class GPU

- target: NVIDIA Titan X
- baseline: MXNet 1.1, TensorFlow 1.7, TensorFlow XLA
- library: cuDNN 7, cuBLAS 8
- workload: ResNet-18, MobileNet, LSTM LM, DQN, DCGAN
- 결과: end-to-end 1.6-3.8배 speedup

DQN의 3.8배는 4x4 conv, stride 2 같은 비정형 operator가 cuDNN에서 충분히 최적화되지 않은 영향이 크다. ResNet처럼 표준 convolution에서는 차이가 더 작다. 따라서 3.8배를 모든 network의 일반 speedup으로 보면 안 된다.

single-kernel 비교에서는 ResNet-18의 12개 conv와 MobileNet의 9개 depthwise conv shape를 명시했다. TensorComprehensions는 operator마다 10 generation x 100 population x 2 seed, 총 2000 trial의 best kernel을 사용했다. TVM은 대부분 layer에서 cuDNN/MXNet kernel과 경쟁하거나 앞섰고, 특히 당시 새 operator였던 depthwise conv에서 강했다.

### Embedded CPU

- target: quad-core ARM Cortex-A53 1.2 GHz
- baseline: TensorFlow Lite commit 7558b085
- workload: ResNet-18, MobileNet, DQN
- 결과: graph optimization 포함 TVM이 end-to-end baseline보다 빠름

low-precision 별도 실험은 ResNet의 2-bit activation, 1-bit weight convolution을 Caffe2 commit 39e07f7과 비교했다. baseline이 single-threaded이므로 TVM도 single-thread와 multi-thread를 따로 제시했다. 1x1 stride 2처럼 baseline이 덜 최적화된 layer에서 큰 차이가 났다.

### Embedded GPU

- board: Firefly-RK3399
- GPU: ARM Mali-T860MP4
- baseline: ARM Compute Library 18.03
- datatype: FP32와 FP16
- workload: ResNet-18, MobileNet, DQN
- 결과: 1.2-1.6배 end-to-end speedup

DCGAN과 LSTM은 baseline이 지원하지 않아 이 비교에서 제외됐다. unsupported workload를 TVM만 실행한 결과와 baseline 비교에 섞지 않았다.

### FPGA accelerator

- board: PYNQ
- host CPU: dual-core ARM Cortex-A9 667 MHz
- FPGA: Artix-7 fabric
- matrix-vector unit: 16x16, 200 MHz
- operand/accumulator: INT8 product, INT32 accumulation
- theoretical peak: 약 102.4 GOPS/s
- on-chip memory: activation 32 kB, parameter 32 kB, microcode 32 kB, register file 128 kB
- backend 추가 코드: Python 약 2k LoC

ResNet 첫 convolution은 channel depth가 얕아 FPGA offload가 비효율적이어서 CPU에 남겼다. residual과 activation도 VDLA가 지원하지 않아 CPU에서 실행했다. 나머지 convolution의 speedup은 Cortex-A9 대비 40배였지만 전체 model은 CPU section 때문에 Amdahl's law에 제한됐다.

이 사례는 accelerator peak만 빠르면 충분하지 않고 operator coverage, transfer, heterogeneous partition이 중요하다는 증거다.

## 별도 손계산: fusion과 tiling

이 절은 논문 측정값이 아니라 memory traffic 구조를 이해하기 위한 계산이다.

### Conv-BN-ReLU fusion

conv output이 FP32 $[1,128,56,56]$라고 하자. element 수는

$$
128\times56\times56=401{,}408
$$

이고 tensor 크기는 약

$$
401{,}408\times4=1.53\text{ MiB}
$$

다. unfused하게 conv output을 쓰고 BN이 읽고 쓴 뒤 ReLU가 읽고 쓰면 intermediate만 단순히 5회 traffic, 약 7.66 MiB가 될 수 있다. fused kernel이 accumulator에서 BN과 ReLU까지 적용하고 최종 output만 한 번 쓰면 intermediate global traffic을 크게 줄일 수 있다. 실제 cache hit와 inplace 실행에 따라 값은 달라지므로 이것은 upper-level 설명용 계산이다.

### 1024x1024 GEMM 8x8 tile

$C=A^\top B$의 한 $8\times8$ output tile을 reduction tile 8로 계산하면 매 $k$ tile마다 $A$와 $B$에서 각각 64 element를 shared/local buffer로 가져오고 512 MAC을 수행한다. FP16이면 input traffic은 $128\times2=256$ bytes, arithmetic intensity는 output write를 제외하고 2 MAC/byte다. 같은 tile을 더 큰 register block이나 여러 output tile에서 재사용하면 intensity가 높아진다. tile 선택이 memory reuse와 register pressure 사이 trade-off인 이유다.

### Peak와 utilization

VDLA theoretical peak 102.4 GOPS/s에서 utilization 70%면 약 71.7 GOPS/s, 88%면 약 90.1 GOPS/s다. 이 값은 논문의 utilization percentage에 theoretical peak를 곱한 별도 계산이며 measured end-to-end throughput이 아니다. load/store와 CPU section을 포함하면 전체는 더 낮다.

## 논문 증거와 별도 해석의 경계

### 논문이 직접 보고한 것

- target별 framework/library version과 hardware 조건
- graph fusion의 Titan X speedup
- ML cost model의 trial efficiency와 0.67 ms prediction
- Titan X 1.6-3.8배, Mali 1.2-1.6배 end-to-end speedup
- FPGA latency hiding utilization 70%에서 88%
- offloaded convolution 40배 speedup과 CPU bottleneck

### 이 리뷰의 별도 계산

- conv-BN-ReLU intermediate 1.53 MiB와 traffic upper bound
- 8x8 GEMM tile의 단순 MAC/byte
- utilization을 theoretical peak에 곱한 71.7/90.1 GOPS/s

별도 계산은 compiler transformation의 방향을 설명하기 위한 것이며 원문 benchmark 표가 아니다.

## 온디바이스 관점

### TVM이 특히 유리한 경우

- vendor library가 잘 다루지 않는 depthwise, low-precision, custom fused operator가 많다.
- target device에서 직접 autotuning할 수 있다.
- input shape가 고정되어 specialization과 static memory plan이 가능하다.
- CPU/GPU/NPU가 섞인 heterogeneous target에 operator backend를 확장할 수 있다.
- application 배포 전에 tuning cost를 amortize할 수 있다.

### 주의할 점

- paper의 API와 compiler 내부는 현재 TVM과 다를 수 있다. 논문의 개념과 최신 사용법을 구분해야 한다.
- tuning 결과는 device model, driver, clock, thermal state, OS load에 민감하다.
- search 중 측정 noise가 cost model을 오염시킬 수 있다.
- dynamic shape가 많으면 shape별 tuning record가 필요하거나 generic schedule을 써야 한다.
- operator 하나의 best kernel이 graph 전체의 best layout과 일치하지 않을 수 있다.
- accelerator offload coverage가 낮으면 Amdahl's law와 transfer overhead가 전체 성능을 제한한다.
- compile time과 binary size, tuning database 관리 비용도 deployment system에 포함해야 한다.

### 모바일 평가 계획

1. target SoC, OS, runtime, clock policy를 고정한다.
2. baseline vendor delegate와 TVM module의 dtype, preprocessing, thread 수를 맞춘다.
3. graph fusion과 operator tuning을 각각 on/off해 contribution을 분리한다.
4. warm-up 후 p50/p95 latency와 peak memory를 측정한다.
5. operator trace에서 layout transform, memcpy, fallback을 찾는다.
6. batch=1과 실제 input resolution으로 tune한다.
7. tuning trial 수, 총 tuning 시간, best-record 재현성을 기록한다.
8. sustained run에서 thermal throttling 후 latency를 다시 측정한다.

## 장점과 기여

1. graph와 operator optimization을 end-to-end로 연결했다.
2. compute와 schedule을 분리해 target-specific code generation 공간을 만들었다.
3. GPU cooperative fetching, tensorization, explicit latency hiding을 schedule primitive로 표현했다.
4. ML cost model과 실제 hardware measurement를 결합했다.
5. server GPU뿐 아니라 ARM CPU, Mali GPU, FPGA를 같은 stack에서 검증했다.
6. vendor library에 없는 새 operator와 custom intrinsic을 빠르게 지원할 수 있음을 보였다.
7. compiler가 accelerator utilization과 operator coverage를 결정한다는 점을 실증했다.

## 한계와 비판적 관점

1. baseline과 software version이 2018년 기준이라 최신 compiler/library와 절대 성능 비교는 유효하지 않다.
2. speedup 범위는 workload별 baseline maturity 차이의 영향을 크게 받는다.
3. autotuning은 free가 아니며 device pool과 compile/measurement budget이 필요하다.
4. schedule template에 developer knowledge가 들어가므로 완전히 search-space free인 자동화는 아니다.
5. fixed shape specialization은 dynamic NLP/VLM workload에 record explosion을 만들 수 있다.
6. cost model은 measurement distribution 밖의 새 architecture에서 다시 학습해야 한다.
7. FPGA 전체 speedup은 unsupported CPU section 때문에 제한됐으며 end-to-end 40배가 아니다.
8. energy와 power 평가는 latency만큼 상세하지 않다.

## 구현 및 재현 체크리스트

- [ ] framework graph의 input/output shape와 dtype을 고정했는가
- [ ] constant folding과 static memory planning을 적용했는가
- [ ] fusion boundary와 opaque op를 확인했는가
- [ ] producer-consumer layout transform 수를 기록했는가
- [ ] tensor expression의 reduction axis가 정확한가
- [ ] schedule이 logical equivalence를 보존하는가
- [ ] shared/local memory scope와 barrier가 올바른가
- [ ] tensor intrinsic의 behavior/reset/update가 정의됐는가
- [ ] virtual thread dependency가 RAW/WAR를 모두 보존하는가
- [ ] tuning measurement 전에 warm-up과 synchronization을 했는가
- [ ] target device에서 직접 측정했는가
- [ ] search budget과 trial 수를 baseline 사이에 공정하게 맞췄는가
- [ ] graph-level과 operator-level contribution을 ablation했는가
- [ ] unsupported operator의 CPU fallback과 copy를 포함했는가
- [ ] software version, driver, thread, batch, dtype을 기록했는가
- [ ] p50/p95, peak memory, power, thermal을 함께 측정했는가

## 후속 연구와의 연결

TVM의 compute/schedule 분리와 autotuning은 이후 Ansor, MetaSchedule, auto-scheduler 연구로 발전했다. tensorization과 multi-level IR은 MLIR, TensorIR, accelerator compiler 설계와 연결된다. hardware-aware NAS와 Once-for-All은 architecture를 선택할 때 TVM 같은 measured latency/cost model을 사용할 수 있다.

온디바이스 연구에서 중요한 연결은 다음과 같다.

- MobileNet 계열의 depthwise operator를 target별로 tuning
- ToMe의 gather/scatter를 지원 가능한 fused kernel로 lowering
- SmoothQuant W8A8 graph의 scale fusion과 INT8 BMM code generation
- OFA sub-network 후보의 실제 device latency table 생성
- VLM에서 vision encoder, projector, decoder를 heterogeneous backend에 partition

## 최종 평가

TVM의 핵심 공헌은 deep learning deployment를 model export 문제가 아니라 compiler optimization 문제로 재정의했다는 점이다. 같은 graph와 같은 MAC 수라도 fusion, layout, tile, memory scope, tensor intrinsic, latency hiding에 따라 실제 성능은 크게 달라진다. 이를 사람이 모든 target에 hand-code하는 대신 schedule search와 measured cost model로 자동화했다.

이 논문을 현재 온디바이스 프로젝트에 적용할 때는 2018년 speedup 숫자보다 실험 방법을 가져오는 것이 중요하다. target device에서 직접 측정하고, graph rewrite와 kernel schedule을 분리하며, unsupported operator와 copy를 end-to-end에 포함해야 한다. 그런 방식으로 평가하면 FLOPs가 아닌 실제 device latency와 memory를 기준으로 model과 compiler를 함께 설계할 수 있다.

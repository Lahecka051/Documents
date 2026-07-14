# 83. Efficient Processing of Deep Neural Networks: A Tutorial and Survey

## 논문 정보

- 제목: Efficient Processing of Deep Neural Networks: A Tutorial and Survey
- 저자: Vivienne Sze, Yu-Hsin Chen, Tien-Ju Yang, Joel Emer
- 소속: MIT, NVIDIA
- 발표: Proceedings of the IEEE, 2017
- arXiv: [1703.09039](https://arxiv.org/abs/1703.09039)
- 원본 파일: `83_Efficient_Processing_of_DNNs.pdf`

## 한눈에 보는 요약

이 논문은 DNN을 빠르게 계산하는 방법을 단순한 MAC 수 감소가 아니라 compute, memory hierarchy, dataflow, numerical precision, sparsity, algorithm-hardware co-design의 통합 문제로 설명하는 tutorial survey다. 2017년 논문이라 Transformer와 최신 NPU를 직접 다루지는 않지만, 온디바이스 AI의 병목을 이해하는 기본 언어는 여전히 유효하다.

가장 중요한 메시지는 다음과 같다.

> DNN energy와 throughput은 연산 수만으로 결정되지 않으며, 어떤 데이터를 어느 memory level에서 몇 번 이동하고 재사용하는지가 핵심이다.

논문 Figure 22의 normalized energy model에서는 ALU와 RF access를 1로 두었을 때 PE 간 NoC가 2, 100-500 kB buffer가 6, DRAM access가 200이다. 이 값은 모든 현대 chip에 그대로 적용되는 절대 법칙이 아니라 survey가 인용한 특정 energy model의 정규화 수치다. 그러나 data movement가 MAC보다 훨씬 비쌀 수 있다는 방향성은 지금도 중요하다.

논문의 주요 범위는 다음과 같다.

- CNN/FC의 tensor shape와 MAC 구조
- CPU/GPU의 temporal architecture와 matrix multiplication mapping
- accelerator의 spatial architecture와 local memory hierarchy
- weight stationary, output stationary, no local reuse, row stationary dataflow
- FFT, Winograd, Strassen 같은 compute transform
- reduced precision, sparsity, pruning, compact architecture, distillation
- near-data processing과 새로운 memory technology
- accuracy, latency, throughput, energy, area를 함께 보는 benchmark 원칙

## 왜 embedded inference가 중심인가

training은 backward intermediate를 저장해야 하고 gradient 때문에 precision 요구도 더 높다. 논문은 제한된 compute, memory, energy 안에서 sensor 근처에서 실행되는 inference를 중심으로 본다. edge inference의 이유는 다음과 같다.

- raw video를 cloud로 보내는 communication cost 감소
- autonomous vehicle, drone, robot의 latency 요구
- network 연결 의존성 감소
- privacy와 security 개선

이 관점은 오늘날의 on-device VLM에도 그대로 적용된다. vision encoder와 LLM의 MAC 수뿐 아니라 camera frame 이동, activation buffer, KV cache, accelerator-CPU 왕복을 모두 고려해야 한다.

## CNN 계산을 tensor shape로 읽기

### Convolution 기호

논문은 convolution/fully connected layer를 다음 기호로 정리한다.

| 기호 | 의미 |
|---|---|
| $N$ | batch size |
| $M$ | 3D filter 수, output channel 수 |
| $C$ | input/filter channel 수 |
| $H,W$ | input feature map 높이와 너비 |
| $R,S$ | filter 높이와 너비 |
| $E,F$ | output feature map 높이와 너비 |
| $U$ | stride |

padding을 생략한 원문의 식은 다음과 같다.

$$
O[z,u,x,y]=B[u]+
\sum_{k=0}^{C-1}\sum_{i=0}^{S-1}\sum_{j=0}^{R-1}
I[z,k,Ux+i,Uy+j]W[u,k,i,j].
$$

출력 크기는

$$
E=\frac{H-R+U}{U},\qquad
F=\frac{W-S+U}{U}
$$

로 표기한다. 실제 framework에서는 padding과 dilation을 추가해야 한다.

### Tensor shape와 MAC 수

- input activation $I$: $[N,C,H,W]$
- weight $W$: $[M,C,R,S]$
- output activation $O$: $[N,M,E,F]$
- MAC 수:

$$
NMEFCRS.
$$

- weight 수:

$$
MCRS.
$$

FC는 $R=H$, $S=W$, $E=F=1$, $U=1$인 특수한 경우로 볼 수 있다. FC는 spatial 위치마다 같은 weight를 재사용하는 convolutional reuse가 없고, batch가 1이면 weight reuse도 제한되어 대개 weight bandwidth에 민감하다.

## 대표 network의 비용

논문은 당시의 대표 모델을 비교한다.

| 모델 | 총 weight | image당 MAC |
|---|---:|---:|
| LeNet-5 | 60k | 341k |
| AlexNet | 61M | 724M |
| OverFeat | 146M | 2.8G |
| VGG-16 | 138M | 15.5G |
| GoogLeNet | 7M | 1.43G |
| ResNet-50 | 25.5M | 3.9G |

이 표는 weight 수와 MAC 수가 같은 방향으로 움직이지 않음을 보여 준다. VGG-16은 AlexNet보다 weight가 약 2.3배지만 MAC은 약 21배다. GoogLeNet은 깊지만 1x1 bottleneck 덕분에 weight를 크게 줄인다. 따라서 model size 하나만으로 latency나 energy를 판단할 수 없다.

## Temporal architecture와 spatial architecture

### Temporal architecture

CPU와 GPU는 여러 ALU가 중앙의 control과 memory hierarchy를 공유한다. SIMD/SIMT로 같은 instruction을 많은 data에 적용하지만, ALU끼리 직접 partial sum을 전달하기보다 register/cache를 통해 data를 가져온다. convolution은 im2col 또는 Toeplitz 형태로 GEMM에 mapping해 BLAS/cuDNN 같은 최적화 library를 활용할 수 있다.

장점은 범용성, mature compiler, 큰 ecosystem이다. 단점은 convolution의 overlapping input을 matrix에 복제하면 storage와 bandwidth가 늘 수 있고, cache behavior와 launch overhead에 따라 실제 성능이 달라진다는 것이다.

### Spatial architecture

ASIC/FPGA accelerator는 ALU와 작은 local memory를 묶은 processing element, PE를 배열로 배치하고 data를 이웃 PE에 전달한다. global buffer, NoC, RF/scratchpad를 명시적으로 사용해 reuse한다. 어느 data를 stationary하게 두고 어느 data를 이동시킬지가 dataflow다.

spatial architecture의 핵심은 MAC unit 개수보다 다음 질문이다.

- weight, input activation, partial sum 중 무엇을 RF에 오래 둘 것인가
- PE 간 multicast와 reduction을 어떻게 할 것인가
- layer shape가 바뀌어도 utilization을 유지하는가
- local buffer 용량 안에서 tile을 어떻게 잡는가
- DRAM round trip을 몇 번 줄이는가

## Memory hierarchy와 data movement energy

### 원문이 제시한 normalized 값

Figure 22는 ALU 또는 RF access를 1의 reference로 두고 다음 값을 제시한다.

| operation/storage level | normalized energy |
|---|---:|
| ALU MAC | 1 |
| 0.5-1.0 kB RF | 1 |
| PE 간 NoC | 2 |
| 100-500 kB global buffer | 6 |
| DRAM | 200 |

이 표의 200배는 특정 technology와 cited model에 기반한 정규화 값이다. 현대 LPDDR, SRAM, precision, voltage에 그대로 대입하면 안 된다. 논문이 전달하는 일반 원리는 memory가 멀고 클수록 access energy가 커지며, DRAM에서 한 번 읽은 data를 가까운 memory에서 최대한 재사용해야 한다는 것이다.

### MAC 하나의 worst-case traffic

논문 Figure 21은 MAC 하나에 weight read, input activation read, partial sum read, updated partial sum write가 필요하다고 설명한다. reuse가 전혀 없고 모두 DRAM을 통과하면 MAC당 3 read와 1 write다. AlexNet 724M MAC에 단순히 4를 곱하면 약 2,896M, 논문 표현으로 거의 3,000M DRAM access다.

local hierarchy에서 weight/input을 재사용하고 partial sum을 local accumulate하면 논문은 AlexNet의 3,000M DRAM access를 best-case 61M까지 낮출 수 있다고 설명한다. CONV layer의 DRAM read는 최대 500배 줄일 수 있다고 보고한다. 이 61M은 AlexNet weight 수와 같은 order지만 모든 실제 accelerator가 자동으로 달성하는 값은 아니다.

## 세 종류의 reuse

### Convolutional reuse

sliding window가 겹치므로 같은 input pixel과 filter weight가 인접 output 계산에 반복 사용된다. CONV layer에서만 나타난다. 예를 들어 stride 1의 3x3 filter는 내부 input pixel 하나가 여러 output window에 들어간다.

### Feature map reuse

같은 input feature map에 여러 output filter를 적용한다. input activation은 output channel $M$개에 걸쳐 재사용된다. CONV와 FC 모두 가능하다.

### Filter reuse

같은 weight를 batch의 여러 input에 적용한다. batch size가 1인 on-device inference에서는 이 reuse가 거의 사라진다. server benchmark의 large batch throughput이 모바일 batch=1 latency를 대변하지 못하는 이유 중 하나다.

## Dataflow taxonomy

### Weight stationary, WS

weight를 PE의 RF에 두고 가능한 많은 MAC에서 재사용한다. input activation은 PE array로 broadcast되고 partial sum은 spatial reduction 또는 buffer 이동을 한다.

장점:

- weight RF access를 최대화
- convolutional reuse와 filter reuse에 유리

약점:

- input과 partial sum traffic이 커질 수 있음
- batch=1 FC처럼 weight reuse가 약한 layer에서는 이점 제한

### Output stationary, OS

partial sum을 PE에 유지하고 output 하나가 완성될 때까지 accumulate한다. input과 weight가 PE로 stream된다. partial sum read/write를 localize하는 데 유리하다.

논문은 parallel output region에 따라 OSA, OSB, OSC variant를 구분한다. OSA는 같은 output channel의 여러 spatial activation을 병렬로 처리해 CONV에 맞고, OSC는 여러 channel의 output을 병렬로 처리해 FC에 맞는다.

### No local reuse, NLR

PE의 RF를 최소화하고 그 area를 큰 global buffer에 사용한다. DRAM access는 줄일 수 있지만 대부분 access가 global buffer와 NoC에서 발생한다. RF보다 global buffer access가 비싸기 때문에 DRAM만 적다고 전체 energy가 최소인 것은 아니다.

### Row stationary, RS

filter row, input row, partial sum을 모두 RF/PE array에서 재사용하도록 1D convolution을 PE에 mapping한다. 특정 data type 하나가 아니라 전체 access energy를 함께 줄이는 것이 목표다.

1. PE 하나가 filter의 한 row와 input row로 1D convolution을 수행한다.
2. 여러 PE를 세로로 묶어 filter row들의 partial sum을 합쳐 2D convolution을 만든다.
3. filter row는 가로로, input row는 대각선으로, partial sum은 세로로 재사용한다.
4. channel, filter, feature map 차원은 interleaving과 concatenation으로 같은 PE에 mapping한다.

## Eyeriss 사례

Eyeriss는 row-stationary dataflow의 대표 accelerator다. 논문이 요약한 구조는 다음과 같다.

- 12x14, 총 168 PE
- 108 kB global buffer
- ReLU와 feature map compression unit
- off-chip DRAM과 64-bit bidirectional bus
- layer shape에 따라 replication과 folding
- 필요한 PE에만 data를 보내는 multicast network

AlexNet layer 3-5의 2D convolution은 13x3 PE mapping을 네 번 replicate할 수 있다. layer 2의 27x5 mapping은 14x5와 13x5로 fold한다. 사용하지 않는 PE는 clock gating한다. 이 예시는 theoretical PE count보다 mapping utilization이 중요하다는 점을 보여 준다.

### Dataflow energy 비교의 원문 조건

논문은 dataflow 간 공정한 비교를 위해 simulation에서 다음을 맞춘다.

- PE 수 256
- 같은 total hardware area
- PE당 RF 약 0.5-1.0 kB
- global buffer 약 100-500 kB
- AlexNet layer configuration
- batch size 16
- memory level별 energy 포함

그 결과 CONV layer에서 RS가 다른 dataflow보다 총 energy를 1.4-2.5배 낮췄고, FC에서는 batch 16에서 약 1.3배 낮췄다. WS는 weight, OS는 partial sum access에서 각각 강하지만 RS는 전체 data type의 합을 최적화했다.

또한 AlexNet energy breakdown에서 CONV가 약 80%, FC가 약 20%를 차지했다. CONV에서는 RF access가, FC에서는 DRAM이 지배했다. 이는 FC가 weight 수가 많더라도 전체 model energy는 CONV가 더 클 수 있음을 보여 준다.

## Mapper는 accelerator의 compiler다

DNN shape와 hardware resource는 inference 전에 알려진다. 따라서 mapper는 다음 입력으로 offline mapping을 찾을 수 있다.

- layer의 $N,M,C,H,W,R,S,E,F$
- PE 수와 array shape
- RF/global buffer 용량
- NoC multicast/reduction 능력
- dataflow rule

출력은 tile, replication, folding, loop order, buffer placement다. 일반 processor에서 compiler가 program을 ISA에 mapping하듯, DNN accelerator에서는 mapper가 tensor shape를 dataflow와 microarchitecture에 mapping한다. 이 관점은 뒤의 TVM 논문과 직접 연결된다.

## CPU/GPU compute transform

### GEMM mapping

FC는 weight matrix $[M,CHW]$와 input matrix $[CHW,N]$의 multiplication으로 나타낼 수 있다. CONV는 overlapping window를 펼친 Toeplitz/im2col matrix로 GEMM에 mapping할 수 있다. library가 효율적인 GEMM을 제공한다는 장점이 있지만 input duplication 또는 복잡한 indirect access가 생긴다.

### FFT

frequency domain에서 convolution을 multiplication으로 바꿔 큰 filter의 multiplication 수를 줄인다. filter FFT는 미리 계산하고, input FFT는 여러 output channel에 재사용할 수 있다. 그러나 complex coefficient, 추가 storage/bandwidth, 작은 filter에서 낮은 이득, sparsity 활용 어려움이 있다.

### Strassen과 Winograd

Strassen은 matrix multiplication exponent를 $O(N^3)$에서 $O(N^{2.807})$로 낮춘다. Winograd는 3x3 filter에서 multiplication 수를 약 2.25배 줄일 수 있다. 대신 addition, transform storage, numerical stability, shape-specific implementation 비용이 생긴다.

논문은 실제 library가 shape에 따라 알고리즘을 선택한다고 설명한다. 큰 filter에는 FFT, 3x3 이하에는 Winograd가 유리할 수 있다. 한 algorithm이 모든 layer에 최선이 아니라는 점이 중요하다.

## Reduced precision

bitwidth를 낮추면 다음을 동시에 줄일 수 있다.

- weight와 activation storage
- memory bandwidth
- multiplier area와 energy
- 한 vector register에 넣는 element 수 증가

논문 Table III의 AlexNet top-5 accuracy loss 예시는 다음과 같다.

| 방법 | weight bit | activation bit | 32-bit float 대비 loss |
|---|---:|---:|---:|
| dynamic fixed point, fine-tuning 없음 | 8 | 10 | 0.4% |
| dynamic fixed point, fine-tuning | 8 | 8 | 0.6% |
| BinaryConnect | 1 | FP32 | 19.2% |
| Binary Weight Network | 1 | FP32 | 0.8% |
| TTQ | 2 | FP32 | 0.6% |
| XNOR-Net | 1 | 1 | 11% |
| BNN | 1 | 1 | 29.8% |
| Deep Compression | conv 8, FC 4 | 16 | 0% |

첫/마지막 layer를 높은 precision으로 유지하거나 scale factor를 사용하는 등 세부 recipe가 결과를 크게 바꾼다. bitwidth만 보고 accuracy를 비교하면 안 된다.

## Sparsity와 pruning

### Activation sparsity

ReLU는 음수를 0으로 만들어 AlexNet feature map에서 layer에 따라 약 19-63% sparsity를 만든다. 논문이 인용한 구현 결과는 다음과 같다.

- 16-bit nonzero와 최대 31개 zero run을 표현하는 run-length coding으로 activation external bandwidth 2.1배 감소
- weight까지 포함한 전체 external bandwidth 1.5배 감소
- zero activation의 weight read와 MAC을 skip해 energy 45% 감소
- zero cycle 자체를 skip해 throughput 1.37배 증가

이 값들은 서로 다른 cited design의 결과이며 하나의 device에서 동시에 보장되는 누적 speedup이 아니다.

### Weight pruning

magnitude pruning과 fine-tuning으로 AlexNet weight를 9배, MAC을 3배 줄인 사례를 설명한다. weight reduction은 FC에서 9.9배, CONV에서 2.7배였다. 그러나 weight 수는 energy proxy로 불충분하다. 논문이 인용한 energy-aware pruning은 memory hierarchy access와 MAC, sparsity를 직접 모델링해 AlexNet energy를 3.7배 줄였고 magnitude-based pruning보다 1.74배 효율적이었다. GoogLeNet에서도 1.6배 energy 감소를 보고했다.

### Unstructured와 structured sparsity

unstructured sparsity는 압축률이 높지만 index metadata, irregular gather, load imbalance가 생긴다. CSR은 sparse row를 따라 input vector 일부를 반복해서 읽을 수 있고, CSC는 input element 하나를 읽어 여러 output을 update한다. output 크기와 matrix shape에 따라 traffic이 달라진다.

structured pruning은 row, column, channel, filter 단위로 제거해 SIMD와 dense kernel에 맞추고 index overhead를 amortize한다. 그룹이 클수록 hardware 효율은 좋아지지만 accuracy loss가 커질 수 있다.

## Compact architecture와 weight 수의 함정

큰 filter를 작은 filter sequence로 바꾸거나 1x1 bottleneck, depthwise separable convolution을 사용하면 MAC과 weight를 줄일 수 있다. 논문은 SqueezeNet이 AlexNet과 비슷한 accuracy에서 weight를 50배 줄였지만 energy는 오히려 더 클 수 있다고 지적한다. 이유는 weight 수가 data movement와 activation reuse를 반영하지 않기 때문이다.

이 교훈은 현재의 mobile Transformer에도 동일하다. parameter가 작은 model이라도 high-resolution activation, reshape, transpose, attention score 때문에 더 느리거나 더 많은 energy를 쓸 수 있다.

## Near-data processing

논문은 DRAM과 compute를 가까이 가져오는 방법도 조사한다.

- eDRAM: off-chip capacitance를 피하고 높은 on-chip density 제공
- 3D stacked memory/HBM/HMC: TSV로 bandwidth 증가와 access energy 감소
- processing-in-memory: memory array 안이나 근처에서 MAC 수행
- near-sensor processing: image sensor 가까이에서 feature 추출

원문은 eDRAM이 SRAM보다 2.85배 높은 density, DDR3 DRAM보다 321배 높은 energy efficiency를 보인 cited design을 소개하고, 3D memory가 2D DRAM보다 order-of-magnitude bandwidth와 최대 5배 access energy 절감을 제공할 수 있다고 정리한다. 이 수치 역시 특정 technology 사례이며 일반적인 최신 chip 상수가 아니다.

analog/mixed-signal 방식은 ADC/DAC overhead, device non-ideality, 낮은 precision을 함께 고려해야 한다. compute-in-memory의 TOPS/W만 보고 system energy를 판단하면 안 되는 이유다.

## 별도 손계산: 56x56 convolution

이 절은 원문 표의 측정값이 아니라 tensor shape를 이해하기 위한 계산이다.

$N=1$, input $[1,64,56,56]$, output channel $M=128$, kernel $3\times3$, stride 1, same padding이라고 하자. 출력은 $[1,128,56,56]$다.

MAC 수는

$$
56\times56\times128\times64\times3\times3
=231{,}211{,}008
$$

이다. tensor element 수는 다음과 같다.

- input: $56\times56\times64=200{,}704$
- weight: $128\times64\times3\times3=73{,}728$
- output: $56\times56\times128=401{,}408$

FP16 tensor를 이상적으로 한 번씩만 DRAM에서 읽고 output을 한 번 쓰면

$$
(200{,}704+73{,}728+401{,}408)\times2
\approx1.29\text{ MiB}
$$

다. 반대로 MAC마다 input, weight, partial sum read와 partial sum write가 모두 2 bytes DRAM traffic이라고 비현실적으로 가정하면

$$
231{,}211{,}008\times4\times2
\approx1.85\text{ GB}
$$

다. 실제 traffic은 이 두 극단 사이이며 tile, cache/RF capacity, output accumulation, compression에 따라 달라진다. 이 차이가 data reuse의 가치다.

### Arithmetic intensity

위의 ideal one-read 근사에서 MAC/byte는

$$
231.2\text{M}/1.35\text{M bytes}\approx171\text{ MAC/byte}
$$

다. 하지만 im2col로 input을 여러 번 materialize하거나 tile이 작아 weight를 반복 load하면 effective intensity가 낮아진다. 따라서 network-level MAC 수만이 아니라 measured byte traffic을 함께 봐야 한다.

## 논문 원문 수치와 별도 계산의 구분

### 논문 또는 논문이 인용한 직접 수치

- Figure 22: ALU/RF 1, NoC 2, buffer 6, DRAM 200 normalized energy
- AlexNet: 724M MAC, worst-case 약 3,000M DRAM access
- ideal reuse 시 AlexNet DRAM access 61M, CONV read 최대 500배 감소
- RS dataflow: AlexNet CONV batch 16에서 다른 dataflow보다 1.4-2.5배 낮은 energy
- RS dataflow: FC batch 16에서 약 1.3배 낮은 energy
- AlexNet energy: CONV 약 80%, FC 약 20%
- Eyeriss benchmark: 65nm, 12.25 mm², 192 kB on-chip memory, 168 multipliers

### 이 리뷰의 별도 계산

- 56x56 convolution의 231.2M MAC
- FP16 ideal traffic 1.29 MiB와 no-reuse 극단 1.85 GB
- 약 171 MAC/byte의 ideal arithmetic intensity

별도 계산은 dataflow를 이해하기 위한 bound이며 실제 chip 측정이 아니다.

## Benchmark를 읽는 법

논문은 accelerator 비교에서 최소한 다음을 명시해야 한다고 강조한다.

- process technology, voltage, core area
- on-chip memory와 multiplier 수
- measured인지 simulated인지, simulation이면 synthesis/PnR 여부
- DNN model과 dataset accuracy
- dense/sparse 여부, weight/activation bitwidth
- 지원 layer 범위
- batch size, runtime, throughput
- power, energy, off-chip access
- test image 수

Eyeriss 예시 표는 65nm LP TSMC 1.0V, 12.25 mm², 192 kB, 168 multiplier, measured result를 명시한다. AlexNet batch 4 runtime 115.3 ms, power 278 mW, off-chip traffic 3.85 MB/image이며 VGG-16 batch 3 runtime 4309.4 ms, power 236 mW, traffic 107.03 MB/image다.

TOPS/W 하나만 비교하면 precision, sparsity, supported layers, DRAM 제외 여부, batch, utilization 차이를 숨긴다. 이 survey의 benchmark checklist가 중요한 이유다.

## 온디바이스 시스템으로 번역하기

### CNN뿐 아니라 Transformer에도 적용되는 원리

| survey 개념 | Transformer/VLM 대응 |
|---|---|
| weight reuse | linear/GEMM weight tile reuse |
| feature map reuse | Q/K/V와 visual token reuse |
| partial sum stationary | GEMM accumulator register 유지 |
| DRAM access 최소화 | activation/KV cache/weight paging 최소화 |
| dataflow mapping | attention/MLP tile과 accelerator schedule |
| structured sparsity | head/channel/token block pruning |
| compact architecture | local attention, token reduction, shared encoder |

attention의 $QK^\top$는 sequence length가 길면 activation와 score traffic이 커진다. LLM decode는 weight와 KV cache bandwidth가 지배할 수 있다. vision high resolution에서는 patch/token activation이 커진다. 병목은 layer마다 달라지므로 하나의 dataflow나 precision이 전체 model에 최적이지 않다.

### 실제 측정 계획

1. target device와 batch=1, input resolution을 고정한다.
2. layer별 MAC, weight bytes, input/output activation bytes를 계산한다.
3. profiler로 DRAM read/write, cache miss, accelerator utilization을 측정한다.
4. operator fusion 전후 intermediate tensor materialization을 비교한다.
5. dense와 sparse model에서 index overhead와 fallback을 확인한다.
6. p50/p95 latency, peak memory, power, temperature를 함께 기록한다.
7. 5-10분 지속 실행 후 thermal throttling 상태를 다시 측정한다.
8. accuracy와 latency를 동일 precision, 동일 preprocessing으로 비교한다.

## 장점과 기여

1. DNN algorithm과 hardware를 하나의 공통 tensor/data-movement 언어로 설명했다.
2. MAC 수가 energy의 충분한 proxy가 아님을 구체적 hierarchy 값과 dataflow 비교로 보였다.
3. WS, OS, NLR, RS를 같은 area 조건에서 비교했다.
4. precision, sparsity, compact architecture, near-memory를 함께 다뤘다.
5. accelerator 논문을 비교할 때 필요한 benchmark metric을 체계화했다.
6. mapper/compiler가 hardware efficiency의 핵심임을 강조했다.

## 한계와 비판적 관점

1. 2017년 survey라 Transformer, modern mobile NPU, chiplet, current HBM을 직접 다루지 않는다.
2. energy ratio는 technology-dependent라 현대 device에 숫자를 그대로 적용할 수 없다.
3. 주요 dataflow 비교가 AlexNet, batch 16 중심이다. batch=1 edge workload와 차이가 있다.
4. simulation 결과와 silicon measurement를 구분해서 읽어야 한다.
5. operator fusion, compiler autotuning, dynamic shape에 대한 논의는 이후 TVM류 연구보다 제한적이다.
6. sparsity speedup은 hardware가 compressed format과 load balance를 지원할 때만 실현된다.
7. accuracy loss 표는 오래된 model/dataset 기반이므로 최신 model의 quantization tolerance와 다를 수 있다.

## 구현 및 논문 읽기 체크리스트

- [ ] $N,M,C,H,W,R,S,E,F$ shape를 정확히 적었는가
- [ ] MAC와 FLOP 정의를 혼용하지 않았는가
- [ ] weight bytes와 activation bytes를 분리했는가
- [ ] partial sum precision과 storage 위치를 확인했는가
- [ ] batch size가 filter reuse에 미치는 영향을 고려했는가
- [ ] DRAM, global buffer, NoC, RF traffic을 구분했는가
- [ ] dataflow가 어떤 data type을 stationary하게 두는지 설명했는가
- [ ] PE utilization과 folding/replication을 확인했는가
- [ ] measured result와 simulated result를 구분했는가
- [ ] process, voltage, precision, sparsity 조건을 맞췄는가
- [ ] off-chip energy가 power 수치에 포함되는지 확인했는가
- [ ] unsupported layer의 CPU execution을 포함했는가
- [ ] compression metadata와 index overhead를 포함했는가
- [ ] accuracy, latency, energy, area를 동시에 비교했는가
- [ ] sustained thermal condition에서 재측정했는가

## 후속 논문과의 연결

이 survey의 mapper 관점은 TVM, TensorRT, XLA, MLIR 같은 compiler 연구로 이어진다. row-stationary와 hierarchy 분석은 Eyeriss 계열 accelerator와 dataflow exploration의 기반이다. reduced precision은 integer-only quantization, SmoothQuant, AWQ로 이어지고, pruning과 compact architecture는 MobileNet, Once-for-All, hardware-aware NAS와 연결된다.

특히 다음 읽기 순서가 자연스럽다.

1. 이 survey로 shape, reuse, hierarchy 언어를 익힌다.
2. TVM으로 graph/operator scheduling과 autotuning을 본다.
3. Once-for-All로 device별 architecture specialization을 본다.
4. quantization과 token compression을 같은 latency/memory framework에서 평가한다.

## 최종 평가

이 논문의 핵심은 200배라는 특정 숫자가 아니라 data movement를 1급 설계 변수로 올려놓았다는 점이다. 좋은 on-device model은 MAC가 적은 model이 아니라, target hardware의 memory hierarchy와 operator support 안에서 reuse가 잘 되고 높은 utilization을 내는 model이다.

따라서 이 survey를 읽고 얻어야 할 습관은 매 layer마다 tensor shape, MAC, weight, activation, partial sum, data movement를 따로 적는 것이다. 그 위에 실제 p50/p95 latency, peak memory, power, temperature를 측정해야 한다. 이 습관이 있으면 최신 CNN, ViT, VLM의 efficiency 주장도 FLOPs 표를 넘어 시스템 수준에서 검증할 수 있다.

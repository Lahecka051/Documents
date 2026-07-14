# 40. MobileNetV4: Universal Models for the Mobile Ecosystem

## 논문 정보

- 제목: **MobileNetV4: Universal Models for the Mobile Ecosystem**
- 저자: Danfeng Qin, Chas Leichner, Manolis Delakis, Marco Fornoni, Shixin Luo, Fan Yang, Weijun Wang, Colby Banbury, Chengxi Ye, Berkin Akin, Vaibhav Aggarwal, Tenghui Zhu, Daniele Moro, Andrew Howard
- 소속: Google
- 발표: ECCV 2024
- arXiv: [https://arxiv.org/abs/2404.10518](https://arxiv.org/abs/2404.10518)
- PDF 기준 버전: arXiv v2, 2024-09-29
- 원본 파일: `40_MobileNetV4.pdf`
- 핵심 키워드: Universal Inverted Bottleneck, UIB, Mobile MQA, hardware-aware NAS, roofline model, multi-hardware Pareto efficiency, distillation

## 한눈에 보는 요약

MobileNetV4(MNv4)는 특정 한 phone CPU에서만 빠른 model이 아니라 CPU, GPU, DSP, Apple Neural Engine, Google EdgeTPU 전반에서 Pareto-efficient한 **universal mobile backbone family**를 목표로 한다. 이를 위해 네 층의 기여를 결합한다.

1. **UIB(Universal Inverted Bottleneck)**: MobileNetV2 inverted bottleneck 앞과 중간의 depthwise convolution을 optional로 만들어 IB, ConvNeXt-like, FFN, ExtraDW를 하나의 search block으로 통합한다.
2. **Mobile MQA**: Query는 multi-head로 유지하지만 K/V를 head 간 공유하고, 필요하면 K/V spatial resolution만 줄인다.
3. **Device-aware NAS와 roofline 분석**: MAC만 최소화하지 않고 compute와 memory bandwidth bottleneck이 다른 hardware 전체를 고려한다.
4. **Offline distillation**: 다양한 augmentation dataset과 class-balanced JFT subset을 동적으로 섞어 architecture 변경 없이 정확도를 올린다.

```text
MNv4-Conv
  = FusedIB stem + NAS가 선택한 UIB instantiations + simple head

MNv4-Hybrid
  = MNv4-Conv의 later low-resolution stages에 Mobile MQA를 interleave
```

Paper의 regular ImageNet-1K 결과에서 MNv4-Conv-S는 `73.8%`, `3.8M`, `0.2G MACs`이며 Pixel 6 CPU `2.4 ms`, Pixel 8 EdgeTPU `0.7 ms`다. MNv4-Conv-M은 `79.9%`, `1.0G`, Pixel 6 CPU `11.4 ms`; MNv4-Hybrid-L은 `83.4%`, `7.2G`, Pixel 8 EdgeTPU `3.8 ms`다. Enhanced distillation 후 Table 9의 Hybrid-L은 `86.6%`; abstract와 conclusion은 JFT pretraining까지 결합한 Hybrid-Large가 `87.0%`와 Pixel 8 EdgeTPU `3.8 ms`를 달성한다고 보고한다.

가장 중요한 교훈은 "MAC가 작으면 모든 device에서 빠르다"가 아니다. Layer별 operational intensity, weight/activation bytes, operator 지원, transpose와 layout cost가 device별 latency 순위를 바꾼다.

## 연구 배경과 문제의식

### 하나의 hardware에서 빠른 model이 다른 곳에서는 느릴 수 있다

Mobile ecosystem은 균일하지 않다.

- ARM CPU는 compute가 제한되고 XNNPACK kernel 품질의 영향을 받는다.
- Mobile GPU는 FP16 throughput은 높지만 memory traffic과 graph launch에 민감하다.
- DSP/NPU는 INT8에서 매우 빠르지만 지원 operator가 제한적이다.
- Apple Neural Engine과 EdgeTPU는 서로 다른 compiler와 preferred dataflow를 갖는다.

FastViT처럼 ANE에서 잘 동작하는 model이 Android GPU/CPU에서는 느릴 수 있고, SE/LayerNorm/GELU가 일부 DSP에서 지원되지 않을 수 있다. Group convolution은 MAC가 작아도 multi-path memory access 때문에 실제로 느릴 수 있다.

MobileNetV4는 architecture를 특정 latency table 하나에 과적합하기보다, 여러 hardware의 bottleneck을 설명할 수 있는 공통 search space와 cost reasoning을 구축한다.

### MAC-only 비교의 한계

MAC는 arithmetic 양만 세고 다음을 보지 않는다.

- Weight와 activation을 몇 byte 읽고 쓰는가?
- Operator 사이 intermediate가 materialize되는가?
- Transpose/reshape가 copy인가 view인가?
- Kernel이 accelerator에 존재하는가?
- 작은 GEMM이 peak throughput을 활용하는가?

MobileNetV4의 roofline 분석과 Mobile MQA Einsum 최적화는 이 문제를 model-level과 operator-level에서 각각 다룬다.

## Roofline model과 hardware-independent efficiency

### Operational intensity

Layer `i`의 operational intensity를 논문은 다음처럼 본다.

```math
I_i=
\frac{\mathrm{LayerMACs}_i}
{\mathrm{WeightBytes}_i+\mathrm{ActivationBytes}_i}
```

값이 높으면 byte당 계산이 많고, 낮으면 data movement 비중이 크다.

### Layer와 model latency 근사

Hardware의 peak arithmetic throughput을 `PeakMACs`, memory bandwidth를 `PeakMemBW`라 하자.

```math
\mathrm{MACTime}_i
=\frac{\mathrm{LayerMACs}_i}{\mathrm{PeakMACs}}
```

```math
\mathrm{MemTime}_i
=\frac{\mathrm{WeightBytes}_i+\mathrm{ActivationBytes}_i}
{\mathrm{PeakMemBW}}
```

Compute와 memory transfer가 대략 overlap한다고 보고 느린 쪽이 layer time을 결정한다.

```math
\mathrm{ModelTime}
=\sum_i\max(\mathrm{MACTime}_i,\mathrm{MemTime}_i)
```

### Ridge point

```math
\mathrm{RP}=
\frac{\mathrm{PeakMACs}}{\mathrm{PeakMemBW}}
```

RP는 peak compute를 사용하려면 필요한 최소 MACs/byte다. Low-RP CPU는 compute-bound가 되기 쉬워 MAC 감소가 중요하다. High-RP accelerator는 compute가 bandwidth보다 훨씬 빠르므로 memory-bound가 되기 쉽고, MAC를 조금 늘려도 latency가 거의 변하지 않는 대신 bytes와 irregular operator가 중요하다.

논문은 RP를 `0-500 MACs/byte`로 sweep한다. `RP=0`은 memory time을 무시해 total MAC만 보는 일반적인 proxy와 대응한다. MNv4가 이 범위에서 이전 MobileNet보다 대부분 Pareto frontier에 있다는 분석을 제시한다.

### 이 model의 한계

Roofline은 좋은 사고 도구지만 실제 latency simulator는 아니다.

- Cache hierarchy와 tensor reuse를 단순화한다.
- Kernel launch, compiler fusion, transpose, padding을 직접 모델링하지 않는다.
- Software implementation이 성능에 영향을 주지 않는다고 가정한다.
- WeightBytes와 ActivationBytes의 정확한 lifetime/traffic 추정에 민감하다.

저자도 pruning처럼 복잡한 memory access가 roofline에서는 실제보다 좋아 보일 수 있다고 지적한다. 따라서 RP sweep은 search-space 방향을 잡고 여러 device 실측으로 검증하는 용도다.

## Universal Inverted Bottleneck

### UIB의 일반형

Input channel `C_in`, expanded channel `E`, output channel `C_out`인 UIB는 다음 순서다.

```text
input
 -> optional start depthwise K1
 -> 1×1 pointwise expansion: C_in -> E
 -> optional middle depthwise K2
 -> 1×1 pointwise projection: E -> C_out
 -> shape이 맞으면 residual
```

두 optional depthwise의 on/off 조합으로 네 block을 표현한다.

| Start DW | Middle DW | UIB instantiation | 해석 |
| :---: | :---: | --- | --- |
| - | ✓ | MobileNet IB | Expanded feature에서 spatial mixing |
| ✓ | - | ConvNeXt-like | Expansion 전에 저렴하게 spatial mixing |
| - | - | FFN | 두 `1×1` channel-mixing layer |
| ✓ | ✓ | ExtraDW | 두 spatial mixing으로 depth/receptive field 증가 |

Original MobileNetV2 block을 포함하면서 stage와 device에 따라 spatial mixing 위치와 횟수를 NAS가 선택할 수 있다. Search supernet에서는 expansion/projection weight를 공유하고 depthwise만 option으로 두어 instantiation 간 parameter의 `>95%`를 공유한다.

### FusedIB

`k×k` FusedIB는 `k×k` full Conv2D 뒤 `1×1` Conv2D를 사용한다. High-resolution stem에서는 depthwise+pointwise를 분리하는 것보다 full convolution이 arithmetic intensity와 hardware utilization 면에서 유리할 수 있다. 모든 MNv4 stem에 FusedIB가 사용된다.

### Parameter/MAC 식

Stride 1, same spatial `H×W`, bias/normalization 제외 시 UIB weight는

```math
\#W=
\mathbf{1}_{DW1}k_1^2C_{in}
+C_{in}E
+\mathbf{1}_{DW2}k_2^2E
+EC_{out}
```

이고 MAC은 대략 `HW × #W`다. Stride 2에서는 어느 depthwise에서 downsample하는지에 따라 각 항의 spatial 크기를 따로 적용해야 한다.

### Batch=1 공식 architecture 예시 - 리뷰어 계산

MNv4-Conv-S Table 11의 stride-1 IB row를 택한다.

```text
input           [1, 14, 14, 96]
PW expansion    [1, 14, 14, 192]
3×3 DW          [1, 14, 14, 192]
PW projection   [1, 14, 14, 96]
residual add     [1, 14, 14, 96]
```

Convolution weight는

```math
96\times192+3^2\times192+192\times96
=38{,}592
```

이고 MAC은

```math
14^2\times38{,}592
=7{,}564{,}032
```

이다. Input/output tensor는 각각 18,816 elements, expansion은 37,632 elements다. FP16 logical size는 input/output 각각 `36.75 KiB`, expansion `73.5 KiB`다. Residual 때문에 input이 projection까지 live하면 branch의 tensor lifetime을 함께 봐야 한다.

이 값은 해당 한 block에 대한 리뷰어 계산이며 전체 MNv4-Conv-S의 논문 보고값 `3.8M parameters, 0.2G MACs`와 구분한다.

### UIB search ablation - 정확한 논문 값

| Search block space | Top-1 | GMACs | MParams | Pixel 8 EdgeTPU |
| --- | ---: | ---: | ---: | ---: |
| Full UIB | **83.3** | 6.2 | 33.0 | 2.68 ms |
| ConvNeXt-like only | 83.2 | 6.9 | 35.1 | 2.69 ms |
| IB only | 82.3 | **6.1** | **32.4** | **2.61 ms** |

UIB search는 IB-only보다 latency가 `0.07 ms` 늘지만 top-1이 `+1.0%p`다. ConvNeXt-only와는 latency가 거의 같고 parameter/MAC가 적으면서 `+0.1%p`다. 중요한 것은 ExtraDW 하나가 언제나 최고라는 주장이 아니라, stage별로 네 instantiation을 섞을 flexibility가 Pareto trade-off를 개선한다는 점이다.

## MNv4 architecture family

### Conv models

Appendix의 exact architecture table은 model size별로 input resolution부터 지정한다.

| Model | Input | Regular top-1 | Params | MACs |
| --- | ---: | ---: | ---: | ---: |
| MNv4-Conv-S | 224 | 73.8 | 3.8M | 0.2G |
| MNv4-Conv-M | 256 | 79.9 | 9.2M | 1.0G |
| MNv4-Conv-L | 384 | 82.9 | 31.0M | 5.9G |

단순 width multiplier로 한 architecture를 키우지 않고 size별 search space와 latency target으로 별도 NAS한다.

Conv-S는 `224 -> 112 -> 56 -> 28 -> 14 -> 7`로 downsample하고 최종 `7×7×128`을 `1×1×960`, `1×1×1280`, 1000-class로 보낸다. Conv-M은 `256 -> 128 -> 64 -> 32 -> 16 -> 8`, Conv-L은 `384 -> 192 -> 96 -> 48 -> 24 -> 12`다.

Appendix 관찰에 따르면 searchable stage 시작과 downsampling 부근에는 ExtraDW가 자주 선택된다. 두 depthwise가 receptive field를 늘려 resolution 감소를 보완하기 때문이다. Late stage에는 앞에서 spatial mixing이 충분히 이루어졌으므로 FFN과 ConvNeXt-like가 자주 선택되어 channel mixing에 capacity를 쓴다.

### Hybrid models

Hybrid-M/L은 late low-resolution stage에서 UIB와 Mobile MQA를 interleave한다. Hybrid-M Table 13에서는 `16×16×160`과 `8×8×256` stage에 MQA가 들어간다. Hybrid-L은 `24×24×192`와 `12×12×512` stage에 들어간다.

Attention을 early high-resolution stage에 넣지 않아 `N²` score와 softmax 비용을 제한한다. 그러나 hybrid model은 Pixel 4 Hexagon DSP에서 `Failed`로 표시되어 모든 platform에서 universal하지는 않다. Conv variants가 DSP 호환성을 담당한다.

## Mobile Multi-Query Attention

### MHSA와 MQA 차이

Multi-head self-attention은 head별 Q/K/V projection을 사용한다.

```text
Q: [B, heads, N, d_k]
K: [B, heads, M, d_k]
V: [B, heads, M, d_v]
```

MQA는 Q head는 유지하지만 K/V를 모든 head가 공유한다.

```text
Q: [B, N, heads, d_k]
K: [B, M, d_k]
V: [B, M, d_v]
```

Query head마다 다른 attention pattern을 만들 수 있지만 동일한 K/V memory bank를 읽는다. `batch=1`, low-resolution, high-channel인 mobile hybrid의 late stage에서 K/V projection parameter와 memory traffic을 줄이는 데 유리하다.

### Spatial reduction

`SR(X)`는 identity 또는 stride-2 `3×3` depthwise convolution이다. Query는 원래 resolution을 유지하고 K/V만 spatially 줄인다.

```math
\mathrm{attention}_j
=\operatorname{softmax}
\left(
\frac{(XW_j^Q)(SR(X)W^K)^T}{\sqrt{d_k}}
\right)
(SR(X)W^V)
```

```math
\mathrm{MobileMQA}(X)
=\operatorname{Concat}(mathrm{attention}_1,\dots,mathrm{attention}_h)W^O
```

Spatial reduction은 MQA와 별개의 절감 축이다. MQA는 K/V의 head dimension 복제를 없애고, SRA는 K/V token 수 `M`을 줄인다.

### Batch=1 tensor shape 예시 - 리뷰어 계산

Hybrid-M의 `16×16×160` stage를 바탕으로, 논문에 공개되지 않은 head config는 설명용으로 `h=4`, `d_k=d_v=40`이라 가정한다. 이 수치는 official config 보고값이 아니다.

Stride-2 K/V reduction을 사용하면

```text
X                 [1, 256, 160]
Q                 [1, 256, 4, 40]
SR(X)             [1, 64, 160]
shared K          [1, 64, 40]
shared V          [1, 64, 40]
attention logits  [1, 256, 4, 64]
output heads      [1, 256, 4, 40]
output            [1, 256, 160]
```

Attention score는 `65,536` elements로 FP16 `128 KiB`다. Downsampling이 없으면 `M=256`이므로 `262,144` elements, FP16 `512 KiB`가 된다.

MQA가 K/V를 공유해도 head별 query와 score는 남으므로 attention score memory 자체는 동일 `hNM`이다. 줄어드는 것은 K/V activation과 projection weight다. 이 예에서 K 하나는 MQA `64×40=2,560` elements지만 MHSA는 `64×4×40=10,240`으로 4배다.

### Parameter reasoning - 리뷰어 계산

`D=160`, `h×d=160`인 위 예에서 bias 제외 projection weight는

```math
\begin{aligned}
W_Q &: 160\times160=25{,}600,\\
W_K &: 160\times40=6{,}400,\\
W_V &: 160\times40=6{,}400,\\
W_O &: 160\times160=25{,}600.
\end{aligned}
```

합은 `64,000`이다. 같은 dimension의 MHSA가 Q/K/V/O 모두 `160×160`이면 `102,400`이므로 이 단순 예에서 `37.5%` 감소한다. 실제 Table 2의 full attention blocks는 다른 detail을 포함하므로 논문의 incremental parameter 감소 `25.5%`와 같을 필요는 없다.

## Mobile MQA ablation

### MHSA 대 MQA - 정확한 논문 값

MNv4-Conv-L base의 last stage에 attention block 3개를 추가한다.

| Model | Top-1 | MACs | Params | Pixel 7 EdgeTPU | Pixel 8 EdgeTPU | S23 GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 84.88 | 6.0G | 30.9M | 4.31 ms | 2.35 ms | 13.15 ms |
| +3 MHSA | **85.27** | 6.7G | 36.0M | 9.69 ms | 2.76 ms | 16.46 ms |
| +3 MQA | 85.24 | **6.5G** | **34.7M** | **5.16 ms** | **2.60 ms** | **15.10 ms** |

표의 `-84.2%/-39.0%/-41.1%` latency는 total model latency가 아니라 base를 뺀 **attention block overhead** 기준이다.

- Pixel 7: MHSA overhead `5.38 ms`, MQA `0.85 ms`, `84.2%` 감소
- Pixel 8: `0.41 -> 0.25 ms`, `39.0%` 감소
- S23 GPU: `3.31 -> 1.95 ms`, `41.1%` 감소

Accuracy 차이는 `-0.03%p`다. Incremental MAC는 `0.7 -> 0.5G`, parameter는 `5.1 -> 3.8M`이므로 각각 `28.6%`, `25.5%` 감소한다.

### K/V downsampling - 정확한 논문 값

MNv4-Hybrid-M, Samsung S23, penultimate `16×16` stage에서 stride-2 K/V downsampling을 비교한다.

| K/V downsample | Top-1 | MACs | CPU | GPU |
| --- | ---: | ---: | ---: | ---: |
| No | **80.77** | 1.285G | 15.8 ms | 7.4 ms |
| Yes | 80.71 | **1.245G** | **12.8 ms** | **5.9 ms** |

`-0.06%p`로 CPU `23%`, GPU `25%` efficiency gain을 얻는다. MAC 감소는 `3%`뿐인데 latency는 20% 이상 줄어, score/activation memory와 operational intensity 효과가 MAC에 포착되지 않음을 보여준다.

## Einsum layout 최적화

### 문제

Einsum은 간결하지만 runtime이 이를 transpose, reshape, batched matmul로 lower한다. Contracting/non-contracting index가 contiguous하지 않으면 tensor 전체를 읽고 쓰는 transpose가 생긴다. Transpose MAC는 0이지만 memory bandwidth와 latency를 소비한다.

원래 Q projection은 output 순서가 `b,h,n,k`다.

```python
Q = einsum("bnd,hdk->bhnk", X, P_q)
```

Mobile MQA는 weight layout과 output index를 바꿔 `b,n,h,k`를 직접 만든다.

```python
Q = einsum("bnd,dhk->bnhk", X, P_q)
```

Output projection도 `b,h,n,v`가 아니라 `b,n,h,v` layout을 유지한다.

```python
# slow: O와 P_o transpose 필요
Y = einsum("bhnv,hdv->bnd", O, P_o)

# fast: contracting/non-contracting 축이 contiguous
Y = einsum("bnhv,hvd->bnd", O, P_o)
```

Math는 같아도 physical layout이 달라 full tensor transpose를 제거한다.

### Exact result

| Model | Pixel 7 | Pixel 8 |
| --- | ---: | ---: |
| Base | 4.31 ms | 2.35 ms |
| +3 MQA, no Einsum optimization | 7.00 ms | 2.64 ms |
| +3 Mobile MQA, optimized | **5.16 ms** | **2.60 ms** |

Base를 뺀 attention overhead는 Pixel 7 `2.69 -> 0.85 ms`, 즉 `68.4%` 감소하고 Pixel 8은 `0.29 -> 0.25 ms`, `13.8%` 감소한다. 논문 본문의 "Pixel 7에서 약 3배"는 total model이 아니라 attention overhead 비율이다.

## NAS recipe

### Per-size search

EfficientNet식 고정 compound scaling 대신 S/M/L마다 별도 search와 resource constraint를 사용한다.

- Conv-S: 224 input, `285M MACs`와 Pixel 6 EdgeTPU `0.2 ms` target
- Conv-M: 256 input, Pixel 6 EdgeTPU `0.6 ms`
- Conv-L: 384 input, Pixel 6 `2.3 ms`와 Pixel 7 EdgeTPU `2.0 ms`

Search space를 device 간 cost-model correlation이 높은 standard component로 제한했기 때문에 EdgeTPU target search가 다른 device에서도 잘 transfer됐다는 것이 논문의 주장이다.

### Two-stage search

TuNAS parameter sharing이 작은 filter와 expansion을 편향되게 선호하는 문제를 완화한다.

1. Coarse search에서 macro/filter 선택을 좁힌다.
2. Fine search에서 expansion ratio 4를 고정하고 두 optional depthwise의 존재와 `3×3/5×5` kernel을 선택한다.

| Search | Val top-1 | Train top-1 | Pixel 6 EdgeTPU |
| --- | ---: | ---: | ---: |
| One-stage | 81.26 | 74.64 | 3.85 ms |
| Two-stage | **81.48** | **78.24** | **3.67 ms** |

Two-stage는 validation `+0.22%p`, training `+3.60%p`, latency `-4.68%`다.

### Offline distillation data for NAS

Supernet architecture가 계속 바뀌는 동안 augmentation/regularization hyperparameter가 reward noise를 만들 수 있다. 저자들은 offline JFT distillation dataset으로 TuNAS를 750 epochs 학습한다. Table 5에서 JFT-distill search는 ImageNet search보다 validation top-1이 `82.3 vs 82.4`로 `-0.1%p`지만 MAC `6.2 vs 7.2G`, parameter `34.4 vs 43.5M`, Pixel 4 GPU `51.0 vs 59.2 ms`, Pixel 6 CPU `67.3 vs 70.4 ms`로 효율적 architecture를 찾는다.

## ImageNet classification 결과

### Benchmark protocol

Appendix B의 실측 절차는 다음과 같다.

- 약 1000회 실행의 mean을 구한다.
- 이 과정을 5회 반복하고 means의 median을 보고한다.
- CPU는 fastest core affinity와 XNNPACK을 사용한다.
- Mobile CPU, Hexagon, EdgeTPU는 TFLite INT8이다.
- Mobile GPU는 FP16이다.
- iPhone 13 ANE는 CoreML MLProgram FP16과 FP16 MultiArray input이다.

같은 table의 device별 latency는 architecture portability를 보여주지만 precision이 모두 같지는 않다.

### 주요 regular-training 값

| Model | Top-1 | Params | MACs | Pixel 6 CPU | Pixel 8 EdgeTPU | iPhone 13 | Pixel 4 DSP | Pixel 7 GPU | S23 CPU | S23 GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MNv4-Conv-S | 73.8 | 3.8M | 0.2G | 2.4 | 0.7 | 0.6 | 2.4 | 8.4 | 1.8 | 2.0 |
| MNv4-Conv-M | 79.9 | 9.2M | 1.0G | 11.4 | 1.1 | 1.1 | 7.3 | 18.1 | 8.6 | 4.1 |
| MNv4-Hybrid-M | 80.7 | 10.5M | 1.2G | 14.3 | 1.5 | 미보고 | Failed | 17.9 | 10.8 | 5.9 |
| MNv4-Conv-L | 82.9 | 31.0M | 5.9G | 59.9 | 2.4 | 3.0 | 20.8 | 37.6 | 43.0 | 13.2 |
| MNv4-Hybrid-L | 83.4 | 35.9M | 7.2G | 87.6 | 3.8 | 미보고 | Failed | 61.3 | 61.8 | 18.1 |

Latency 단위는 모두 ms다. Missing과 unsupported(`Failed`)를 구분해야 한다. Hybrid가 Conv보다 정확하지만 DSP portability를 잃고 CPU latency도 늘어난다.

이전 MobileNet의 architecture 개선을 공정하게 보기 위해 저자들은 modern recipe로 다시 학습한 값도 사용한다. MobileNetV1 `74.0`, V2 `73.4`, V3 `75.5`로 원 논문 checkpoint보다 높다. 따라서 이 paper의 MobileNetV2 `73.4`를 2018 paper의 `72.0`과 같은 training recipe로 보아서는 안 된다.

## COCO object detection

RetinaNet, `384×384`, 256-d FPN P3-P7, 256-d prediction head 4 layers를 사용하고 FPN/head convolution은 depthwise-separable로 바꾼다. COCO 2017을 600 epochs, batch 2048, Adam, cosine schedule과 24-epoch warmup으로 학습한다.

| Backbone | COCO val AP | Detector MACs | Detector params | Pixel 6 CPU |
| --- | ---: | ---: | ---: | ---: |
| EfficientFormer-L1 | 29.5 | 6.54G | 12.77M | 84.3 ms |
| MobileNetV1 1.5× | 31.0 | 6.68G | 9.05M | 66.4 ms |
| MNv4-Conv-M | 32.6 | **5.06G** | 9.79M | **51.3 ms** |
| MobileNet Multi-AVG 1.5× | 32.7 | 5.42G | 9.51M | 58.1 ms |
| MobileNetV2 2.0× | 32.9 | 5.81G | 10.15M | 66.4 ms |
| MobileNetV3-L 2.0× | 33.2 | 4.99G | 17.92M | 59.9 ms |
| MNv4-Hybrid-M | **34.0** | 5.62G | 11.15M | 60.5 ms |

Conv-M은 MobileNetV2 2.0×보다 AP `-0.3`이지만 latency는 `22.7%` 짧다. Hybrid-M은 Conv-M보다 `+1.4 AP`, `+9.2 ms`다. Paper 본문은 이를 rounded `+1.6 AP`, `18%` slower로 서술한다. 표를 기준으로 계산하면 `34.0-32.6=1.4`, `60.5/51.3-1≈17.9%`다.

Small-object AP를 악화시켜 RandAugment에서 Shear와 Rotate를 제외했다는 training detail은 중요하지만, Table 7은 AP_S/AP_M/AP_L을 별도로 보고하지 않는다.

## Enhanced distillation

### Offline soft labels

Teacher는 EfficientNet-L2로 `480M` parameters, `290G` MACs, ImageNet-1K top-1 `87.5%`다. Candidate image를 먼저 augment하고 JPEG compress/decompress한 뒤 teacher의 1000-class softmax vector를 저장한다. Student training 중 teacher forward가 필요 없다.

Student는 horizontal flip 외 online augmentation을 끄고 final FC dropout만 regularization으로 사용한다. Teacher soft label과 student class probability 사이 cross-entropy, AdamW, cosine schedule과 warmup을 사용한다.

### Dataset mixing

- `D1`: Inception Crop + RandAugment `l2m9`을 적용한 ImageNet replica
- `D2`: Inception Crop + extreme Mixup의 ImageNet replica, Patient Teacher baseline
- `D3`: JFT-300M에서 ImageNet class별 최대 130K, 총 130M으로 resample한 subset, Inception Crop + RandAugment `l2m5`

Dynamic mixture가 fixed augmentation 하나보다 student에게 다양한 난이도와 image space를 제공한다.

### Exact ablation

| Dataset | Mixing | 400-epoch val/train | 2000-epoch val/train |
| --- | --- | ---: | ---: |
| D1 | - | 83.8 / 86.6 | - |
| D2 | - | 84.1 / 85.6 | - |
| D3 | - | 81.8 / 84.1 | - |
| D1 + D2 | 1:1 | 84.4 / 85.0 | - |
| D2 + D3 | 1:1 | 84.7 / 82.7 | - |
| D1 + D2 + D3 | 1:1:2 | **84.9 / 82.6** | **85.9 / 85.5** |

D3 단독은 D2보다 `-2.3%p`지만 D2+D3는 D2보다 `+0.6%p`다. JFT의 domain/noise를 단독으로 쓰는 것보다 ImageNet-derived data와 섞는 것이 중요하다.

### Model별 distillation gain

| Model | IN-1K only | Prior distill | Proposed distill |
| --- | ---: | ---: | ---: |
| Conv-S | 73.8 | - | 75.5 |
| Conv-M | 79.9 | 81.5 | 82.7 |
| Hybrid-M | 80.7 | 82.7 | 83.7 |
| Conv-L | 82.9 | 84.4 | 85.9 |
| Hybrid-L | 83.4 | 85.7 | **86.6** |

Appendix는 Conv-L 400-epoch distillation이 batch 16K, 128 TPU v5e에서 `2.3시간`이고 ordinary ImageNet training보다 `2.5배` 빠르다고 보고한다. Offline teacher cost와 dataset 생성/storage는 이 시간에 포함되지 않는 별도 비용이다.

Abstract/conclusion의 `87.0% at 3.8 ms`는 enhanced distillation과 JFT pretraining까지 결합한 Hybrid-Large claim이다. Regular Table 6의 `83.4% at 3.8 ms` 및 Table 9 distillation `86.6%`와 training condition을 구분해야 한다.

## 장점과 핵심 기여

- MobileNetV2 block을 두 optional depthwise로 일반화해 네 유명 block family를 하나의 search primitive로 통합했다.
- MQA와 asymmetric K/V downsampling을 mobile vision의 late-stage attention에 맞게 적용했다.
- Einsum index ordering이 transpose와 latency에 미치는 영향을 exact code와 device ablation으로 보여줬다.
- MAC뿐 아니라 operational intensity와 ridge point로 multi-hardware efficiency를 설명했다.
- CPU, GPU, DSP, ANE, EdgeTPU에서 같은 model family를 직접 benchmark했다.
- Per-size, two-stage NAS로 단순 scaling보다 각 resource region에 맞춘 architecture를 찾았다.
- Offline distillation dataset mixing으로 inference graph 변경 없이 정확도를 크게 높였다.
- Classification뿐 아니라 end-to-end RetinaNet detector latency를 측정했다.

## 한계와 비판적 관점

### 1. "Universal"은 완전한 호환성을 뜻하지 않는다

Hybrid-M/L은 Pixel 4 Hexagon에서 unsupported다. Conv variants가 폭넓게 동작하지만 attention variants는 platform coverage가 줄어든다.

### 2. Roofline은 software cost를 과소평가한다

실제 compiler fusion, cache, tensor layout, operator fallback을 모델링하지 않는다. 역설적으로 논문의 Einsum 사례가 zero-MAC transpose의 중요성을 보여준다.

### 3. Device별 precision이 다르다

CPU/DSP/EdgeTPU는 INT8, GPU/ANE는 FP16이다. Multi-device Pareto 비교에는 유용하지만 동일 precision의 architecture-only 비교는 아니다.

### 4. Tail latency, memory, energy 미보고

약 1000회 평균을 5번 반복한 median-of-means는 안정적이지만 p95/p99, peak memory, power, temperature, throttling은 없다.

### 5. NAS와 training budget이 크다

TuNAS 750 epochs, size별 search, JFT offline distillation, 2000-epoch student training이 필요하다. Deployment model은 작지만 search/data cost는 상당하다.

### 6. JFT access와 reproducibility

Enhanced 최고 결과는 JFT-300M subset을 사용한다. 비공개 data access가 없는 환경에서는 exact reproduction이 어렵다.

### 7. Mobile MQA는 attention score의 quadratic 항을 없애지 않는다

K/V head sharing은 projection과 K/V traffic을 줄이지만 logits는 여전히 `hNM`이다. Low-resolution late stage와 SRA가 함께 있어야 mobile-friendly하다.

### 8. Detection 분석 범위

RetinaNet 하나, 384 input, COCO overall AP 중심이다. NMS latency, AP_S/AP_M/AP_L, p95, sustained camera workload는 보고하지 않는다.

### 9. Pareto frontier는 benchmark implementation에 종속된다

Compiler/runtime version과 quantization recipe가 달라지면 순위가 바뀔 수 있다. "Mostly Pareto-optimal"은 논문이 측정한 model set과 platform/software 범위의 결과다.

## 자주 헷갈리는 지점

### UIB는 하나의 고정 block인가

아니다. 두 optional depthwise 선택으로 IB, ConvNeXt-like, FFN, ExtraDW 중 하나가 되는 super-block/search primitive다.

### ExtraDW는 depthwise를 두 번 연속 적용하는가

두 depthwise 사이에 pointwise expansion이 있다. Start DW는 input channel 공간, middle DW는 expanded channel 공간에서 동작한다.

### MQA는 query head도 하나인가

아니다. Query는 여러 head를 유지하고 K/V만 공유한다.

### MQA가 softmax를 제거하는가

아니다. Mobile MQA는 여전히 softmax attention이다. EfficientViT의 ReLU linear attention과 다르다.

### SRA는 Q/K/V를 모두 downsample하는가

아니다. Query resolution은 유지하고 K/V source `SR(X)`만 줄인다.

### MAC가 더 큰 model이 왜 더 빠를 수 있는가

High-RP hardware에서는 compute보다 memory bytes와 operator/dataflow가 병목일 수 있다. 큰 regular Conv2D가 peak compute를 잘 활용하면 작은 irregular graph보다 빠를 수 있다.

### 87.0%는 ImageNet-only architecture result인가

아니다. Enhanced distillation과 JFT pretraining이 결합된 최고 claim이다. Regular ImageNet-1K Hybrid-L은 `83.4%`다.

## 온디바이스 구현 관점

### Conv와 Hybrid 중 선택

- DSP 호환성과 가장 넓은 deployment coverage가 필요하면 MNv4-Conv를 우선한다.
- CPU/GPU/NPU에서 약간 높은 정확도를 위해 late attention을 감당할 수 있으면 Hybrid를 검토한다.
- Target compiler가 MQA layout과 K/V downsampling을 지원하는지 먼저 확인한다.

### Activation memory

UIB의 expanded tensor는 MobileNetV2와 마찬가지로 peak 후보다. ExtraDW는 spatial operation을 두 번 넣지만 start DW가 expansion 전 낮은 channel에서 실행되어 큰 kernel을 비교적 싸게 사용할 수 있다. Hybrid에서는 attention logits와 Q가 추가된다.

Paper는 parameter/MAC/latency를 풍부하게 제공하지만 peak activation은 보고하지 않는다. 다음 live set을 profiler로 측정해야 한다.

```text
UIB: residual input + expanded activation + conv workspace
MQA: Q + shared K/V + logits + softmax output + residual
Detector: P3-P7 FPN features + prediction heads + decoded boxes/NMS
```

### Quantization

Table 6의 CPU/DSP/EdgeTPU는 이미 INT8 benchmark다. 하지만 accuracy table은 regular training top-1이고 device별 quantized accuracy drop을 별도로 열거하지 않는다. Export checkpoint의 PTQ/QAT recipe와 per-channel scale을 확인해야 한다.

### Layout first

Mobile MQA의 가장 실용적인 교훈은 Einsum 식이 같아도 index order가 latency를 바꾼다는 것이다. Export 전후 graph에서 transpose count, tensor bytes, CPU fallback을 검사한다. High-level profiler의 MAC만 보면 이 병목은 보이지 않는다.

## 재현 계획

### 1단계: UIB unit test

1. UIB generic block에 `start_dw_kernel`, `middle_dw_kernel` option을 둔다.
2. 네 instantiation이 예상 graph와 weight 수를 갖는지 확인한다.
3. Stride 2가 어떤 depthwise에 적용되는지 official implementation과 맞춘다.
4. BN folding 후 export graph에 standard Conv/DW/Add만 남는지 확인한다.
5. Table 11의 Conv-S shape schedule을 end-to-end assert한다.

### 2단계: Mobile MQA

1. MHSA와 MQA output shape를 맞춘다.
2. Shared K/V parameter와 activation bytes를 hook으로 측정한다.
3. K/V downsampling on/off에서 logits가 `hN² -> hN(N/4)`로 줄어드는지 확인한다.
4. Slow/fast Einsum layout의 numerical output이 일치하는지 검증한다.
5. Export graph의 transpose 수와 bytes를 비교한다.

### 3단계: architecture checkpoint

1. Conv-S 또는 Conv-M official config를 선택한다.
2. Paper-reported parameter/MAC 근처인지 counting convention을 맞춘다.
3. ImageNet preprocessing과 regular-training checkpoint 정확도를 검증한다.
4. Distilled checkpoint는 data source와 recipe를 별도 label로 관리한다.

### 4단계: multi-device benchmark

| 항목 | 고정/기록 내용 |
| --- | --- |
| Input | model별 224/256/384 정확히 유지 |
| Batch | 1 |
| Precision | FP16, INT8를 별도 표로 분리 |
| CPU | affinity, thread, XNNPACK version |
| NPU/DSP | compiler, delegate, fallback |
| Timing | warm-up, p50, p95, cold start |
| Memory | allocator peak, external-memory traffic |
| Sustained | power, temperature, 10분 throttling |

NAS search를 재현하지 못하더라도 same checkpoint를 여러 device에서 profile해 "universal" claim을 자체 hardware 범위에서 검증할 수 있다.

### 5단계: detector

RetinaNet 384에서 backbone만이 아니라 FPN, head, decode, NMS를 포함한 p50/p95를 기록한다. AP/AP_S/AP_M/AP_L과 latency를 함께 비교한다. Camera pipeline에서는 preprocessing과 image copy도 포함한다.

## 구현 체크리스트

- [ ] UIB의 start/middle depthwise가 독립 optional인가?
- [ ] IB/CN-like/FFN/ExtraDW mapping이 정확한가?
- [ ] Stem에 FusedIB가 사용되는가?
- [ ] Model size별 input resolution이 224/256/384로 맞는가?
- [ ] Hybrid attention이 late low-resolution stage에만 있는가?
- [ ] MQA에서 Q는 multi-head, K/V는 shared인가?
- [ ] Spatial reduction이 K/V에만 적용되는가?
- [ ] Einsum output layout이 transpose를 유발하지 않는가?
- [ ] Paper latency의 precision과 backend를 기록했는가?
- [ ] Missing과 Failed를 구분했는가?
- [ ] Regular, prior-distilled, proposed-distilled, JFT-pretrained 정확도를 섞지 않았는가?
- [ ] MAC와 measured latency를 별도 column으로 유지하는가?
- [ ] Peak memory와 tail latency가 paper 미보고임을 명시했는가?

## 로드맵에서의 위치와 후속 연결

MobileNetV4는 효율적 backbone 단계의 종합점이다.

- MobileNetV2의 inverted residual과 linear bottleneck을 UIB search space로 일반화한다.
- DeiT가 보여준 training/distillation 중요성을 enhanced offline distillation로 확장한다.
- Swin처럼 attention을 late hierarchical stage에 배치하지만 fixed window 대신 MQA/SRA를 쓴다.
- EfficientViT처럼 실제 hardware latency를 중시하지만 softmax linearization 대신 K/V sharing과 spatial reduction을 택한다.

Detection 단계에서는 MNv4-Conv-M/Hybrid-M RetinaNet 결과가 anchor-free 또는 DETR 계열과 비교할 현실적 mobile CNN baseline이 된다. 압축/시스템 단계에서는 INT8 cross-device benchmark와 roofline analysis가 operator support, tensor bytes, memory bandwidth를 평가하는 직접적인 template다.

통합 프로젝트에서 공유 encoder 후보로 사용할 때는 universal Conv variant와 higher-accuracy Hybrid variant를 둘 다 profile하고, detector 상시 실행에는 Conv, event-driven VLM/segmenter에는 Hybrid feature를 쓰는 분리도 고려할 수 있다.

## 최종 평가

MobileNetV4의 가장 큰 공헌은 새 block 하나보다 **architecture, attention dataflow, NAS, distillation, hardware measurement를 하나의 mobile design discipline으로 묶은 것**이다. UIB는 단순한 optional DW 두 개로 넓은 block space를 표현하고, Mobile MQA는 K/V sharing뿐 아니라 실제 transpose 제거까지 다룬다. 다양한 device의 실측표는 MAC-only 사고의 한계를 매우 설득력 있게 보여준다.

한편 universal claim은 tested software/hardware와 mixed precision에 종속되고, hybrid는 DSP에서 실패한다. Peak memory, p95, 전력·온도도 없다. JFT와 대규모 distillation/search budget 때문에 최고 정확도의 완전한 재현성도 제한적이다. 따라서 후속 온디바이스 연구의 핵심은 paper 표를 복사하는 것이 아니라, **target runtime에서 operational intensity와 layout을 직접 측정하고 정확도·tail latency·memory·energy의 Pareto frontier를 다시 그리는 것**이다.

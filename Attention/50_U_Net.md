# 50. U-Net: Convolutional Networks for Biomedical Image Segmentation

## 논문 정보

- 제목: **U-Net: Convolutional Networks for Biomedical Image Segmentation**
- 저자: Olaf Ronneberger, Philipp Fischer, Thomas Brox
- 소속: University of Freiburg
- 공개: arXiv:1505.04597v1, 2015-05-18
- 발표: MICCAI 2015
- 논문: [arXiv](https://arxiv.org/abs/1505.04597) / [PDF](https://arxiv.org/pdf/1505.04597)
- 원본 파일: `50_U_Net.pdf`
- 원 구현/모델 안내: [U-Net project page](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- 핵심 키워드: encoder-decoder, skip concatenation, biomedical segmentation, valid convolution, overlap-tile, elastic deformation, weighted cross-entropy

## 한눈에 보는 요약

U-Net은 적은 annotation으로도 정밀한 pixel segmentation을 하기 위해 **context를 얻는 contracting path와 위치를 복원하는 symmetric expanding path를 skip concatenation으로 연결**한 구조다.

```text
input tile 572×572
 -> [3×3 valid conv ×2 + 2×2 max pool] ×4
 -> 1024-channel bottleneck
 -> [2×2 up-conv + crop-and-concat skip + 3×3 conv ×2] ×4
 -> 1×1 class projection
 -> output mask 388×388
```

논문의 세 가지 실용적 기여가 함께 작동한다.

1. Encoder feature를 decoder에 concatenate해 context와 localization을 동시에 사용
2. Valid convolution과 overlap-tile로 큰 microscopy image를 seam 없이 처리
3. Elastic deformation과 touching-cell border weight로 매우 적은 training image를 보완

EM segmentation challenge에서는 30장의 512×512 training image만 사용해 warping error 0.000353, Rand error 0.0382를 기록한다. Cell tracking challenge에서는 PhC-U373 IoU 0.9203, DIC-HeLa IoU 0.7756으로 당시 2위 결과를 크게 앞섰다.

현재의 작은 `same-padding U-Net`과 달리 원 논문 model은 모든 3×3 convolution이 unpadded다. 따라서 572×572 input에서 388×388 output만 나오며 skip feature를 crop해야 한다. 이 차이를 무시하면 원 구조의 tensor shape, tile halo, MAC 계산이 모두 달라진다.

## 문제 배경: 적은 image로 dense localization하기

Biomedical segmentation은 일반 classification과 조건이 다르다.

- Pixel 단위 annotation은 비싸서 training image가 수십 장뿐일 수 있다.
- 세포 경계처럼 1-몇 pixel의 구조가 중요하다.
- 큰 context가 있어야 cell과 background를 구분할 수 있다.
- Sliding-window CNN은 overlap patch를 반복 계산해 느리다.

큰 patch를 쓰면 context는 늘지만 pooling으로 위치가 거칠어지고, 작은 patch는 세밀하지만 전체 조직 구조를 보지 못한다. FCN은 whole-image dense prediction으로 중복을 줄였고, U-Net은 여기에 대칭 decoder와 high-resolution skip을 추가해 localization을 강화한다.

## 전체 architecture

### Contracting path

각 level에서 다음을 반복한다.

```text
3×3 valid convolution -> ReLU
3×3 valid convolution -> ReLU
2×2 max pool, stride 2
```

Downsampling할 때 channel은 `64 -> 128 -> 256 -> 512 -> 1024`로 두 배씩 증가한다. Valid convolution 하나마다 height와 width가 2 pixel씩 줄어든다.

### Expanding path

각 level에서 다음을 수행한다.

```text
2× upsampling + 2×2 up-convolution, channel 절반
 -> encoder feature를 center crop
 -> channel dimension으로 concatenate
 -> 3×3 valid convolution + ReLU
 -> 3×3 valid convolution + ReLU
```

마지막 1×1 convolution이 64-channel feature를 class 수로 투영한다. Figure 1의 binary segmentation example은 2 output channel이다. 전체 convolutional layer는 23개다.

## 원 논문의 exact spatial shape

Figure 1의 `572×572`, grayscale input, 2-class output을 그대로 추적하면 다음과 같다. Batch dimension은 1이다.

### Encoder

| 단계 | Operation | Output shape |
| --- | --- | ---: |
| Input | - | `1×1×572×572` |
| L1-1 | 3×3 conv | `1×64×570×570` |
| L1-2 | 3×3 conv | `1×64×568×568` |
| Pool1 | 2×2 / 2 | `1×64×284×284` |
| L2-1 | 3×3 conv | `1×128×282×282` |
| L2-2 | 3×3 conv | `1×128×280×280` |
| Pool2 | 2×2 / 2 | `1×128×140×140` |
| L3-1 | 3×3 conv | `1×256×138×138` |
| L3-2 | 3×3 conv | `1×256×136×136` |
| Pool3 | 2×2 / 2 | `1×256×68×68` |
| L4-1 | 3×3 conv | `1×512×66×66` |
| L4-2 | 3×3 conv | `1×512×64×64` |
| Pool4 | 2×2 / 2 | `1×512×32×32` |
| Bottom-1 | 3×3 conv | `1×1024×30×30` |
| Bottom-2 | 3×3 conv | `1×1024×28×28` |

### Decoder

| 단계 | Up/crop/concat | Conv outputs |
| --- | --- | ---: |
| Up1 | up `28 -> 56`, crop `64 -> 56`, concat `512+512` | `54 -> 52`, 512 ch |
| Up2 | up `52 -> 104`, crop `136 -> 104`, concat `256+256` | `102 -> 100`, 256 ch |
| Up3 | up `100 -> 200`, crop `280 -> 200`, concat `128+128` | `198 -> 196`, 128 ch |
| Up4 | up `196 -> 392`, crop `568 -> 392`, concat `64+64` | `390 -> 388`, 64 ch |
| Output | 1×1 conv | `1×2×388×388` |

Crop은 각각 encoder feature의 양쪽에서 동일한 pixel 수를 제거한다.

- 64에서 56: 한쪽 4 pixel
- 136에서 104: 한쪽 16 pixel
- 280에서 200: 한쪽 40 pixel
- 568에서 392: 한쪽 88 pixel

단순히 shape만 resize해서 concatenate하지 말고 receptive-field center 정렬까지 맞춰야 한다.

## Skip concatenation은 무엇을 전달하는가

Encoder의 shallow feature는 edge, texture, precise position을 보존한다. Bottleneck feature는 큰 receptive field와 semantic context를 가진다. Decoder는 두 정보를 channel 방향으로 이어 붙인다.

```math
Z_l=\mathrm{Concat}
\left(
\mathrm{Up}(D_{l+1}),
\mathrm{Crop}(E_l)
\right)
```

```math
D_l=\mathrm{Conv}_{3\times3}
\left(
\mathrm{Conv}_{3\times3}(Z_l)
\right)
```

Addition과 달리 concatenation은 encoder와 decoder feature를 별도 channel로 보존해 다음 convolution이 결합 방식을 학습한다. 대신 channel과 activation memory가 커진다. 논문은 concatenation과 addition을 직접 ablation하지 않는다.

## Receptive field, context, output crop

572 input에서 388 output이므로 전체 차이는 184 pixel, 중앙 정렬 시 output 한쪽에 필요한 input halo는 92 pixel이다.

```text
input tile:  92 halo + 388 valid output region + 92 halo
             = 572
```

Valid convolution은 full context가 있는 pixel만 출력한다. Image 경계에서는 없는 context를 mirror extrapolation으로 채운다. 이 정책은 zero padding으로 생기는 artificial dark border를 피하려는 선택이다.

## Overlap-tile strategy

큰 image 전체가 GPU memory에 들어가지 않을 때 572×572 input tile을 읽고 중앙 388×388만 출력한다. 다음 tile은 output region이 이어지도록 이동하되 input halo는 서로 겹친다.

```text
large image
 -> mirrored border extension
 -> overlapping 572×572 input tiles
 -> each tile yields central 388×388 prediction
 -> valid regions를 이어 붙임
```

Reviewer 계산으로 input/output tile 면적 비는 다음과 같다.

```math
\frac{572^2}{388^2}
=\frac{327{,}184}{150{,}544}
\approx2.17
```

즉 tile boundary context 때문에 output pixel 면적의 2.17배에 해당하는 input을 처리한다. Network 내부 compute가 정확히 2.17배라는 뜻은 아니지만 작은 tile일수록 halo redundancy가 커지는 이유를 보여준다. 더 큰 tile은 redundancy를 줄이는 대신 peak activation이 늘어난다.

## 원형 U-Net parameter 계산

논문은 parameter 수를 표로 보고하지 않는다. Figure 1의 grayscale input, bias가 있는 모든 conv, 2-class output을 기준으로 reviewer가 계산하면 다음과 같다.

일반 convolution parameter:

```math
P=k^2C_{in}C_{out}+C_{out}
```

2×2 up-convolution도 같은 weight-count 방식으로 계산한다. 23개 layer 합은 다음과 같다.

```math
P_{total}=31{,}030{,}658
```

순수 weight payload 하한은 FP32 약 118.4 MiB, FP16 약 59.2 MiB, INT8 약 29.6 MiB다. Framework metadata, quantization scale, alignment, activation은 제외했다.

Parameter가 가장 큰 layer 중 하나는 bottom의 `1024 -> 1024` 3×3 convolution이다.

```math
3\times3\times1024\times1024+1024
=9{,}438{,}208
```

이 layer 하나가 전체 parameter의 약 30.4%다. Deepest width를 줄이거나 depthwise-separable block으로 바꾸는 것이 model size에 큰 영향을 주는 이유다.

## MAC 계산

논문은 MACs/FLOPs를 보고하지 않는다. Figure 1의 모든 convolution output shape를 사용하고 bias, ReLU, pooling, crop, concat, softmax 비용을 제외한 reviewer 계산은 다음과 같다.

```math
\mathrm{MAC}_{conv}
=H_{out}W_{out}k^2C_{in}C_{out}
```

2×2 stride-2 transposed convolution은 output 위치가 겹치지 않는 이 구조에서 대응하는 multiply-accumulate를 포함해 합산했다.

```math
\mathrm{MAC}_{total}
\approx150.43\text{ G}
```

Multiply와 add를 각각 1 FLOP로 세면 약 300.86 GFLOPs에 대응하지만 profiler convention에 따라 다르다. 이 값은 **원 Figure 1의 572 -> 388 grayscale model에 대한 계산**이며 paper-reported number가 아니다.

상위 MAC layer는 decoder의 concat 뒤 convolution과 고해상도 early convolution이다. Parameter가 bottleneck에 집중되는 것과 달리 compute는 큰 `H×W`를 가진 여러 level에 넓게 분포한다. 따라서 weight pruning만으로 latency가 같은 비율로 줄지 않는다.

## Activation memory 계산

Decoder가 사용할 encoder skip을 full feature로 보관한다고 가정한 batch-1 payload는 다음과 같다.

| Skip | Elements | FP32 | FP16 |
| --- | ---: | ---: | ---: |
| `64×568×568` | 20,647,936 | 78.77 MiB | 39.39 MiB |
| `128×280×280` | 10,035,200 | 38.28 MiB | 19.14 MiB |
| `256×136×136` | 4,734,976 | 18.06 MiB | 9.03 MiB |
| `512×64×64` | 2,097,152 | 8.00 MiB | 4.00 MiB |
| 합계 | 37,515,264 | **143.11 MiB** | **71.56 MiB** |

이는 skip payload만이며 current activation, concat output, convolution workspace, allocator fragmentation은 빠져 있다. Inference에서는 필요한 중앙 crop만 일찍 저장하거나 recompute하여 memory를 줄일 수 있지만 원 graph와 scheduler가 자동으로 그렇게 한다고 가정하면 안 된다.

Final 2-class logits `2×388×388`은 FP32 약 1.15 MiB로 작다. U-Net에서는 class 수보다 wide skip과 decoder concatenation이 peak memory의 핵심이다.

Training은 backward를 위해 activation을 더 오래 저장하므로 batch 1을 사용했다. 논문은 "큰 tile을 선호하고 batch를 1로 줄였다"고 설명하지만 정확한 peak memory 수치는 보고하지 않는다.

## Pixelwise softmax와 weighted cross-entropy

Pixel `x`, class `k`의 logit을 `a_k(x)`라 하면 softmax는 다음과 같다.

```math
p_k(x)=
\frac{\exp(a_k(x))}
{\sum_{k'=1}^{K}\exp(a_{k'}(x))}
```

Ground-truth class를 `ell(x)`, pixel weight를 `w(x)`라 하면 일반적인 minimization cross-entropy는 다음과 같다.

```math
E=-\sum_{x\in\Omega}w(x)\log p_{\ell(x)}(x)
```

원 논문 Equation 1에는 leading minus가 보이지 않지만 본문은 cross-entropy로 deviation을 penalize한다고 설명한다. Maximum log-likelihood를 최대화하거나 loss에서 음수를 취해야 하므로 구현 시 sign을 확인해야 한다.

## Touching-cell border weight

Class frequency를 보정하는 `w_c(x)`에, 서로 가까운 두 cell boundary 사이를 강조하는 Gaussian term을 더한다.

```math
w(x)=w_c(x)+w_0\exp\left(
-\frac{(d_1(x)+d_2(x))^2}{2\sigma^2}
\right)
```

- `d1(x)`: 가장 가까운 cell border까지 거리
- `d2(x)`: 두 번째로 가까운 cell border까지 거리
- `w0=10`
- `sigma approximately 5 pixels`

두 cell 사이의 좁은 background에서는 `d1+d2`가 작아 weight가 커진다. 모델이 touching cells를 하나의 blob으로 합치지 않고 separation border를 학습하도록 한다.

이 weight map은 ground-truth instance geometry를 사용해 training 전에 계산한다. Inference에는 morphology나 distance transform이 필요 없다. 반면 semantic mask만 있고 instance identity가 없다면 같은 `d1,d2` map을 만들기 어렵다.

## Weight initialization

Convolution과 ReLU가 반복되는 network에서 feature variance를 유지하려고 incoming connection 수 `N`에 따라 Gaussian standard deviation을 정한다.

```math
W\sim\mathcal{N}\left(0,\frac{2}{N}\right)
```

즉 standard deviation은 다음과 같다.

```math
\mathrm{std}(W)=\sqrt{\frac{2}{N}}
```

3×3 convolution, input channel 64라면 `N=9×64=576`이다. 오늘날 He initialization으로 알려진 방식이다. 논문은 깊은 encoder-decoder의 서로 다른 path가 초기부터 excessive 또는 dead activation을 만들지 않도록 이 초기화를 강조한다.

## Data augmentation

수십 장뿐인 microscopy image에서 augmentation은 핵심 학습 신호다.

- Shift와 rotation
- Gray-value variation
- Random elastic deformation
- Contracting path 끝의 dropout

Elastic deformation은 coarse 3×3 grid의 displacement vector를 Gaussian standard deviation 10 pixel로 sample하고, pixel별 displacement를 bicubic interpolation으로 만든다.

```text
3×3 random displacement grid
 -> bicubic interpolation
 -> smooth dense displacement field
 -> image와 label에 같은 warp
```

Image에는 bicubic interpolation을 쓸 수 있지만 discrete label에는 class mixing을 막기 위해 nearest-neighbor interpolation을 사용하는 것이 일반적이다. 다만 label interpolation mode는 이 PDF 본문에 명확히 적혀 있지 않으므로 원 구현을 확인해야 한다.

## Training setup

- Framework: Caffe
- Optimizer: stochastic gradient descent
- Batch: 1 large tile
- Momentum: 0.99
- Loss: weighted pixelwise softmax cross-entropy
- Training time: 약 10시간
- Hardware: NVIDIA Titan GPU, 6 GB

Learning rate, weight decay, iteration 수 같은 세부 hyperparameter는 본문에 보고되지 않는다. Batch 1에서 높은 momentum을 써 이전 sample의 update를 오래 반영한다. BatchNorm은 원 architecture에 없으므로 small-batch BN 문제도 없다.

## 구현 pseudocode

```python
def double_conv_valid(x, out_channels):
    x = relu(conv2d(x, out_channels, kernel=3, padding=0))
    x = relu(conv2d(x, out_channels, kernel=3, padding=0))
    return x

def center_crop(x, target_hw):
    h, w = x.shape[-2:]
    th, tw = target_hw
    top = (h - th) // 2
    left = (w - tw) // 2
    return x[..., top:top + th, left:left + tw]

def up_block(x, skip, out_channels):
    x = conv_transpose2d(x, out_channels, kernel=2, stride=2)
    skip = center_crop(skip, x.shape[-2:])
    x = concatenate([skip, x], dim="channel")
    return double_conv_valid(x, out_channels)

def original_unet(x, num_classes=2):
    e1 = double_conv_valid(x, 64)
    e2 = double_conv_valid(max_pool2x2(e1), 128)
    e3 = double_conv_valid(max_pool2x2(e2), 256)
    e4 = double_conv_valid(max_pool2x2(e3), 512)
    b = double_conv_valid(max_pool2x2(e4), 1024)

    d4 = up_block(b, e4, 512)
    d3 = up_block(d4, e3, 256)
    d2 = up_block(d3, e2, 128)
    d1 = up_block(d2, e1, 64)
    return conv2d(d1, num_classes, kernel=1)
```

Weighted loss:

```python
prob = softmax(logits, dim="class")
ce = negative_log_likelihood(prob, label)
loss = (class_weight_plus_border_weight * ce).sum()
```

Modern framework에서 input 572가 output 388이 되는지 각 stage를 assertion으로 검증해야 한다.

## 실험 1: EM neuronal structure segmentation

Dataset은 Drosophila larva VNC의 serial-section transmission electron microscopy image다.

- Training: fully annotated 512×512 image 30장
- Test label: 비공개 challenge server
- Output: membrane probability
- Evaluation: 10개 threshold의 warping error, Rand error, pixel error

U-Net submission은 **입력을 7개 rotation으로 변환한 prediction을 평균**하고 별도 pre/post-processing 없이 평가했다.

| Method | Warping error | Rand error | Pixel error |
| --- | ---: | ---: | ---: |
| Human | 0.000005 | 0.0021 | 0.0010 |
| U-Net | **0.000353** | 0.0382 | 0.0611 |
| DIVE-SCI | 0.000355 | 0.0305 | 0.0584 |
| IDSIA sliding-window | 0.000420 | 0.0504 | 0.0613 |

U-Net은 ranking 기준인 warping error가 가장 낮지만 Rand error와 pixel error까지 모두 1위는 아니다. DIVE-SCI가 두 지표에서 더 낮다. 또한 7-rotation test-time ensemble을 썼으므로 single-pass latency와 challenge accuracy를 함께 인용하면 안 된다.

## 실험 2: ISBI Cell Tracking Challenge

### PhC-U373

- Phase-contrast microscopy
- Partially annotated training image 35장
- U-Net IoU 0.9203
- 2015 second-best 0.83

### DIC-HeLa

- Differential interference contrast microscopy
- Partially annotated training image 20장
- U-Net IoU 0.7756
- 2015 second-best 0.46

| Method | PhC-U373 IoU | DIC-HeLa IoU |
| --- | ---: | ---: |
| IMCB-SG 2014 | 0.2669 | 0.2935 |
| KTH-SE 2014 | 0.7953 | 0.4607 |
| 2015 second-best | 0.83 | 0.46 |
| U-Net | **0.9203** | **0.7756** |

논문은 동일 architecture가 서로 다른 microscopy modality에서 잘 작동함을 보여준다. 하지만 natural image multi-class segmentation이나 cross-site generalization을 평가하지 않는다.

## 속도와 memory 보고의 범위

논문은 512×512 image segmentation이 당시 recent GPU에서 1초 미만이라고 서술하지만 정확한 GPU model, precision, batch, single model/TTA 여부를 latency 표로 제공하지 않는다. Training은 Titan 6 GB에서 약 10시간이다.

따라서 다음 값은 원 논문에 없다.

- Exact single-pass latency
- p50/p95 latency
- MACs/FLOPs 표
- Parameter/serialized model size
- Peak activation/runtime workspace
- 전력과 온도

앞서 제시한 31.03M parameter, 150.43 GMAC, skip payload는 Figure 1을 바탕으로 한 reviewer 계산이다.

## 장점

- Context와 localization을 대칭 encoder-decoder로 명확하게 결합했다.
- Skip concatenation으로 fine feature를 decoder가 직접 재사용한다.
- 매우 적은 image에서 elastic augmentation의 중요성을 보여줬다.
- Touching instance의 경계를 loss weight로 직접 강조했다.
- Valid output과 mirror overlap-tile로 큰 image를 memory 제한 아래 처리했다.
- 복잡한 별도 post-processing 없이 강한 challenge 성능을 냈다.

## 한계와 비판적 관점

### 1. 핵심 요소별 ablation이 없다

Elastic deformation, border weight, skip concatenation, symmetric channel 수를 각각 제거한 표가 없다. 최종 gain을 어느 요소가 얼마나 만들었는지 분리할 수 없다.

### 2. 원 architecture의 compute가 크다

Reviewer 계산 31M parameter와 약 150G MAC은 오늘날 mobile 상시 segmenter에 무겁다. 고해상도 64-channel first stage가 큰 compute를 만든다.

### 3. Skip concatenation의 peak memory

Encoder feature를 오래 보존하고 decoder에서 channel을 두 배로 만든다. Training batch 1이 필요했던 이유와 연결되지만 논문은 peak를 직접 측정하지 않는다.

### 4. Valid convolution과 crop 복잡성

Output이 input보다 작고 tile halo 계산이 필요하다. Modern runtime의 fixed-shape compile과 camera frame edge 처리에는 same padding보다 관리가 어렵다.

### 5. Test-time rotation ensemble

EM headline 결과는 7개 rotated input 평균이다. Single-pass model accuracy와 전력 budget에서는 재평가가 필요하다.

### 6. Metric별 최우수는 아니다

Warping error 1위지만 DIVE-SCI가 Rand/pixel error에서 더 낮다. "모든 지표에서 압도"라고 요약하면 부정확하다.

### 7. Dataset 범위

수십 장의 microscopy dataset에서 강하지만 natural image, 많은 class, domain shift, scanner variation을 체계적으로 평가하지 않는다.

### 8. Loss equation sign과 구현 세부

Equation 1의 minus sign, elastic label interpolation, optimizer learning rate가 본문만으로 완전히 규정되지 않는다. 원 code 검증이 필요하다.

## 온디바이스 operator와 memory 관점

원 U-Net의 연산은 비교적 단순하다.

- 3×3 convolution + ReLU
- 2×2 max pooling
- 2×2 transposed convolution
- Crop + concatenate
- 1×1 convolution
- Softmax/argmax

Attention, LayerNorm, dynamic query는 없지만 다음이 병목이 될 수 있다.

### High-resolution convolution

첫 두 64-channel convolution은 spatial area가 매우 커서 MAC과 DRAM traffic이 크다. Parameter가 적은 early layer도 latency 비중이 클 수 있다.

### Skip retention과 concat

FP16 skip payload만 reviewer 계산 약 71.6 MiB다. NPU SRAM에 들어가지 않으면 DRAM write/read가 발생한다. Concat이 physical copy인지 view인지, 다음 convolution이 두 source를 직접 읽을 수 있는지 확인해야 한다.

### Transposed convolution

일부 NPU는 ConvTranspose를 지원하지 않거나 비효율적으로 구현한다. `Resize + Conv`로 바꿀 수 있지만 checkpoint 동등성을 잃으므로 fine-tuning이 필요하다.

### Tile size

작은 tile은 peak를 줄이지만 halo 중복과 kernel invocation을 늘린다. 큰 tile은 효율이 좋지만 memory peak가 높다. Target device에서 output tile 크기별 p95와 peak를 sweep해야 한다.

### Quantization

INT8 weight는 약 29.6 MiB 하한으로 줄일 수 있지만 boundary와 low-contrast microscopy pixel은 activation quantization에 민감할 수 있다. Overall IoU뿐 아니라 boundary F-score와 touching-cell split error를 확인해야 한다.

실용적 경량화 후보는 다음과 같다.

```text
base channel 64 -> 16/32
standard conv -> depthwise-separable conv
1024 bottleneck -> 256/512
concat -> projected concat 또는 addition
ConvTranspose -> bilinear resize + conv
full frame -> ROI/tile on demand
```

각 변경은 parameter뿐 아니라 skip activation, tile halo, operator support를 함께 측정해야 한다.

## 재현 체크리스트

- [ ] arXiv v1의 original valid-convolution architecture를 쓰는가?
- [ ] Input 572에서 output 388이 나오는가?
- [ ] Encoder shape 568/280/136/64와 decoder crop을 assertion했는가?
- [ ] Skip은 addition이 아니라 channel concatenation인가?
- [ ] Up-conv가 2×2, stride 2이며 channel을 절반으로 줄이는가?
- [ ] Final 1×1 convolution이 64에서 class 수로 mapping하는가?
- [ ] Batch 1, SGD, momentum 0.99를 기록했는가?
- [ ] Weight initialization std가 `sqrt(2/N)`인가?
- [ ] Border weight에서 두 nearest-cell distance를 올바르게 계산하는가?
- [ ] `w0=10`, `sigma about 5`를 확인했는가?
- [ ] Image와 label에 동일 elastic field를 적용하는가?
- [ ] Label warp interpolation이 class index를 섞지 않는가?
- [ ] Mirror padding과 92-pixel halo가 tile edge에서 일치하는가?
- [ ] EM 결과에 7-rotation TTA가 포함됐음을 표시했는가?
- [ ] Single-pass와 TTA latency를 별도로 측정했는가?
- [ ] Parameter/MAC 수치가 paper report가 아닌 계산임을 표시했는가?
- [ ] Peak skip/concat memory를 target runtime에서 측정했는가?
- [ ] IoU뿐 아니라 boundary와 instance separation metric을 기록했는가?

## 로드맵에서의 연결

U-Net은 FCN의 fully convolutional dense prediction을 더 강한 decoder로 확장한다.

```text
FCN: deep score + 얕은 score의 additive fusion
 -> U-Net: symmetric decoder + full feature concatenation
 -> DeepLabv3+: atrous context + lightweight boundary decoder
 -> Mask R-CNN: ROI별 instance mask
 -> SegFormer: hierarchical encoder + lightweight MLP fusion
 -> SAM/EdgeSAM: prompt-conditioned mask decoder
```

FCN과 U-Net을 같은 lightweight encoder로 구현해 비교하면 skip 방식의 trade-off가 선명해진다.

| 관점 | FCN-8s | U-Net |
| --- | --- | --- |
| Skip content | Class score map | Multi-channel feature |
| Fusion | Add | Concatenate |
| Decoder capacity | 작음 | 큼 |
| Boundary potential | 제한적 | 높음 |
| Activation memory | 상대적으로 작음 | 큼 |

온디바이스 로드맵에서는 상시 semantic segmenter로 U-Net 계열을 쓸 때 full-frame 실행 주기와 tile 크기를 조절하고, prompt segmenter는 사용자 요청 때만 실행하는 구성이 자연스럽다.

## 최종 평가

U-Net의 지속적인 영향력은 U자 모양 자체보다 **적은 데이터에서 context와 precise localization을 결합한 전체 recipe**에 있다. Symmetric decoder, skip concatenation, elastic deformation, border-weighted loss, overlap-tile이 biomedical constraint에 맞춰 하나의 system을 이룬다.

동시에 원형 model은 31M parameter, reviewer 계산 약 150G MAC, 큰 skip activation과 7-view TTA 때문에 현대 mobile deployment에 그대로 적합하지 않다. 재현의 핵심은 작은 modern U-Net을 이름만 가져오는 것이 아니라 원 논문의 valid-conv shape, crop alignment, border loss, augmentation을 이해한 뒤 **width, padding, upsampling, tile policy를 target hardware에 맞춰 다시 ablation하는 것**이다.

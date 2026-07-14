# 51. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

## 논문 정보

- 원본 파일: `51_DeepLabv3_Plus.pdf`
- 제목: **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation**
- 저자: Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam
- 발표: ECCV 2018
- 공식 링크: [arXiv:1802.02611](https://arxiv.org/abs/1802.02611)
- 태스크: semantic segmentation
- 핵심 키워드: DeepLabv3+, ASPP, atrous convolution, encoder-decoder, depthwise separable convolution, output stride, boundary refinement

## 한눈에 보는 요약

DeepLabv3+는 서로 다른 장점을 가진 두 계열을 결합한다. DeepLabv3의 Atrous Spatial Pyramid Pooling(ASPP)은 여러 유효 수용영역에서 문맥을 모으지만, 낮은 해상도의 logits를 한 번에 키우면 물체 경계가 흐려진다. U-Net형 encoder-decoder는 얕은 고해상도 feature를 이용해 경계를 복원하지만, 문맥을 다중 scale로 모으는 구조는 상대적으로 약하다. 이 논문은 **ASPP를 encoder로 유지하고, output stride 4의 단순한 decoder를 붙여 문맥과 경계 복원을 함께 얻는다.**

핵심 설계는 다음과 같다.

```text
image
  -> Xception/ResNet backbone with atrous convolution
  -> ASPP + image-level feature: H/16 x W/16 x 256
  -> bilinear upsample x4
  -> concatenate with H/4 x W/4 low-level feature reduced to 48 channels
  -> 3x3 conv, 256 channels x2
  -> class logits at output stride 4
  -> bilinear upsample x4 to the input resolution
```

논문이 직접 보여 준 가장 중요한 결과는 세 가지다.

1. ResNet-101, VOC val에서 decoder는 `77.21 -> 78.85 mIoU`로 개선하며 Multiply-Adds는 `81.02B -> 101.28B`가 된다.
2. modified Xception에서 ASPP와 decoder의 표준 convolution을 depthwise separable convolution으로 바꾸면 계산량을 약 `33%~41%` 줄이면서 비슷한 mIoU를 유지한다.
3. 추가 데이터와 고비용 test-time augmentation을 사용한 최종 모델은 PASCAL VOC 2012 test `89.0%`, Cityscapes test `82.1%` mIoU를 기록한다.

다만 마지막 최고 수치는 경량 단일-scale 배포 설정의 결과가 아니다. JFT/COCO 사전학습, 더 조밀한 output stride, multi-scale 및 flip 추론이 섞인 행을 분리해서 읽어야 한다. 논문은 모바일 장치 latency, peak memory, 전력, 온도를 보고하지 않는다.

## 연구 배경과 문제의식

### Semantic segmentation의 두 가지 요구

Semantic segmentation은 각 pixel에 class를 할당한다. 분류와 달리 다음 두 요구를 동시에 만족해야 한다.

- **넓은 문맥**: 작은 patch만 보면 `road`, `wall`, `sofa`처럼 모양이 비슷한 class를 구분하기 어렵다.
- **정확한 위치**: downsampling으로 얻은 강한 semantic feature만 사용하면 얇은 구조와 물체 경계가 사라진다.

공간 pyramid 계열은 서로 다른 dilation rate나 pooling grid로 문맥을 모은다. Encoder-decoder 계열은 깊은 feature를 단계적으로 확대하면서 얕은 feature를 합친다. DeepLabv3+의 질문은 단순하다.

> DeepLabv3의 다중-scale 문맥을 보존하면서, 복잡한 decoder 없이 경계만 효율적으로 복원할 수 있는가?

### DeepLabv3의 한계

DeepLabv3는 backbone의 마지막 downsampling을 atrous convolution으로 대체하고 ASPP로 여러 scale을 본다. 그러나 원래 방식은 낮은 해상도 logits를 bilinear interpolation으로 바로 원본 크기까지 올린다. 이 연산은 새로운 경계 정보를 만들지 못한다. 이미 사라진 얇은 물체와 세밀한 윤곽을 interpolation만으로 되찾을 수는 없다.

DeepLabv3+는 이 문제를 해결하기 위해 backbone의 output stride 4 지점에 있는 low-level feature 하나만 사용한다. 여러 skip을 계단식으로 연결하는 U-Net보다 decoder가 얕고, semantic feature가 low-level channel에 압도되지 않도록 `1x1` projection으로 low-level channel을 48까지 줄인다.

## 핵심 개념 1: Atrous convolution

### 정의

1차원 표기로 atrous convolution은 다음과 같다.

```math
y[i] = \sum_k x[i+r\cdot k]w[k]
```

`r`은 atrous rate 또는 dilation rate다. `r=1`이면 표준 convolution이다. 2차원 `3x3` kernel에서 rate가 `r`이면 kernel parameter와 tap 수는 9개로 유지되지만 유효 kernel 크기는 다음과 같이 커진다.

```math
k_{\mathrm{eff}} = k + (k-1)(r-1)
```

따라서 `3x3, r=2`의 유효 범위는 `5x5`, `r=6`은 `13x13`이다. Parameter와 이론적 MAC 수는 같은 channel의 표준 `3x3`과 동일하지만, 실제 장치에서는 간격이 벌어진 memory access와 kernel 지원 여부 때문에 latency가 같다고 볼 수 없다.

### Output stride

Output stride(OS)는 입력 공간 크기와 최종 encoder feature 크기의 비율이다.

```math
OS = \frac{H_{\mathrm{input}}}{H_{\mathrm{feature}}}
```

분류 backbone은 보통 `OS=32`다. 마지막 한 번의 stride를 제거하고 후속 convolution의 dilation을 2로 바꾸면 `OS=16`, 마지막 두 stride를 제거하고 dilation을 조정하면 `OS=8`의 조밀한 feature를 얻는다. 핵심은 receptive field를 크게 훼손하지 않고 공간 sample 수를 늘리는 것이다.

공간 원소 수는 대략 `1/OS^2`에 비례한다. 같은 channel 수라면 `OS=16`에서 `OS=8`로 바꿀 때 feature activation은 약 4배가 된다. 논문 표에서도 ResNet-101 baseline의 Multiply-Adds가 `81.02B`에서 eval OS 8의 `276.18B`로 급증한다.

## 핵심 개념 2: ASPP encoder

DeepLabv3 encoder의 마지막 feature에는 병렬 branch가 적용된다.

- `1x1` convolution
- 서로 다른 rate의 `3x3` atrous convolution 세 개
- image-level global pooling branch
- branch concatenate와 projection

논문의 도식은 OS 16에서 rate `6, 12, 18`을 보여 준다. 각 위치가 서로 다른 유효 수용영역을 관찰한 뒤 합쳐지므로 작은 물체의 국소 정보와 큰 물체의 문맥을 동시에 표현한다. 최종 DeepLabv3 feature는 256 channel이다.

ASPP는 여러 branch의 출력을 같은 해상도로 동시에 유지한다. 그러므로 최종 256-channel feature만 보고 peak memory를 추정하면 부족하다. 실제 peak에는 병렬 branch activation, concatenate buffer, BatchNorm 임시값, backend workspace가 포함된다.

## 핵심 개념 3: 단순한 decoder

Decoder의 동작은 다음과 같다.

1. `OS=16`, 256-channel encoder feature를 bilinear interpolation으로 4배 확대한다.
2. Backbone의 `OS=4` low-level feature를 `1x1` convolution으로 48 channel까지 줄인다.
3. 두 feature를 concatenate하여 304 channel을 만든다.
4. `3x3, 256` convolution 두 번으로 경계를 정제한다.
5. Class logits를 만든 뒤 4배 확대해 입력 크기로 복원한다.

Low-level feature가 256 또는 512 channel인 상태로 바로 concatenate되면 세밀한 texture가 256-channel semantic feature보다 수적으로 우세해지고 decoder 계산도 커진다. `48`은 단지 압축 숫자가 아니라 두 정보원의 균형을 맞추는 bottleneck이다.

논문은 Conv2와 Conv3를 모두 사용하는 더 깊은 decoder도 실험했지만 개선되지 않았다. 즉 DeepLabv3+의 기여는 "skip을 많이 붙인다"가 아니라 **필요한 고해상도 feature 한 개만 좁게 투입한다**는 데 있다.

## Atrous separable convolution

표준 `k x k` convolution의 parameter 수와 위치당 MAC은 다음과 같다.

```math
P_{\mathrm{standard}} = k^2 C_{in}C_{out}
```

Depthwise separable convolution은 channel별 spatial convolution과 `1x1` pointwise convolution으로 나눈다.

```math
P_{\mathrm{separable}} = k^2 C_{in}+C_{in}C_{out}
```

따라서 계산 비율은 대략 다음과 같다.

```math
\frac{P_{\mathrm{separable}}}{P_{\mathrm{standard}}}
=\frac{1}{C_{out}}+\frac{1}{k^2}
```

`k=3`, `C_out=256`이면 약 `0.115`, 즉 이상적인 곱셈 수는 표준 convolution의 약 11.5%다. 논문은 depthwise 단계에 dilation을 적용하는 **atrous separable convolution**을 ASPP와 decoder에 사용한다.

이 이론적 절감률이 end-to-end latency 절감률과 같지는 않다. Depthwise convolution은 arithmetic intensity가 낮고, dilation까지 결합하면 메모리 접근이 불규칙해질 수 있다. 모바일 NPU가 `depthwise + dilation` 조합을 단일 kernel로 지원하지 않으면 graph가 CPU로 fallback하거나 여러 연산으로 분해될 수 있다.

## Tensor shape: `batch=1`, 513x513 입력

논문의 VOC 학습 crop `513x513`, class 수 21, encoder OS 16을 기준으로 TensorFlow `SAME` padding을 가정한다.

| 지점 | Shape | 설명 |
| --- | --- | --- |
| 입력 | `1 x 513 x 513 x 3` | NHWC 표기 |
| Low-level Conv2 | `1 x 129 x 129 x 256` | OS 4 예시 |
| Low-level projection | `1 x 129 x 129 x 48` | `1x1` bottleneck |
| ASPP 출력 | `1 x 33 x 33 x 256` | OS 16 |
| ASPP upsample | `1 x 129 x 129 x 256` | 4배 확대 후 low-level과 정렬 |
| Concatenate | `1 x 129 x 129 x 304` | `256+48` |
| Decoder conv | `1 x 129 x 129 x 256` | `3x3` 두 번 |
| Logits | `1 x 129 x 129 x 21` | Class별 score |
| 출력 | `1 x 513 x 513 x 21` | Bilinear upsample |

513은 16으로 나누어떨어지지 않으므로 구현의 padding과 `align_corners` 규칙에 따라 중간 크기가 달라질 수 있다. Skip concatenate 직전에는 목표 low-level feature의 `H,W`를 명시적으로 사용해야 off-by-one 오류를 피할 수 있다.

### Activation memory 손계산

아래 값은 논문 보고값이 아니라 위 shape를 이용한 리뷰어 계산이다. Tensor 한 개의 storage만 계산하며 allocator, branch 동시 생존, gradient, workspace는 제외한다.

```math
M = B\cdot H\cdot W\cdot C\cdot \text{bytes per element}
```

| Tensor | 원소 수 | FP32 | FP16 |
| --- | ---: | ---: | ---: |
| ASPP 출력 `33x33x256` | 278,784 | 1.06 MiB | 0.53 MiB |
| Upsampled encoder `129x129x256` | 4,260,096 | 16.25 MiB | 8.13 MiB |
| Projected skip `129x129x48` | 798,768 | 3.05 MiB | 1.52 MiB |
| Concatenate `129x129x304` | 5,058,864 | 19.30 MiB | 9.65 MiB |
| Decoder output `129x129x256` | 4,260,096 | 16.25 MiB | 8.13 MiB |
| Full logits `513x513x21` | 5,527,449 | 21.09 MiB | 10.54 MiB |

Concat 입력과 출력, 다음 convolution 출력의 lifetime이 겹치면 decoder 구간만으로도 단일 tensor 최대값보다 훨씬 큰 peak가 발생한다. 학습에서는 activation 저장과 gradient 때문에 수 배 커진다. 추론에서도 full-resolution 21-channel logits가 오래 유지되면 오히려 마지막 tensor가 peak 후보가 된다. Argmax를 조기에 수행하거나 tiled output을 지원하면 이 비용을 줄일 수 있다.

`OS=8`의 ASPP 입력은 약 `65x65`가 되어 `33x33`보다 공간 원소가 약 3.88배다. Accuracy가 조금 오르더라도 backbone과 ASPP의 activation traffic이 크게 증가하므로 온디바이스 기본값으로는 OS 16이 더 현실적이다.

## 최소 구현 의사코드

아래 코드는 핵심 데이터 흐름을 보여 주는 축약형이다. 실제 재현에는 논문과 같은 modified aligned Xception, BatchNorm 설정, ASPP rate와 preprocessing이 필요하다.

```python
class DeepLabV3Plus(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone       # returns low_os4, high_os16
        self.aspp = ASPP(in_channels=2048, out_channels=256,
                         rates=(6, 12, 18))
        self.low_proj = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            SeparableConvBNReLU(304, 256, kernel_size=3),
            SeparableConvBNReLU(256, 256, kernel_size=3),
        )
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        input_hw = x.shape[-2:]
        low, high = self.backbone(x)
        high = self.aspp(high)
        high = F.interpolate(high, size=low.shape[-2:],
                             mode="bilinear", align_corners=False)
        low = self.low_proj(low)
        y = self.refine(torch.cat([high, low], dim=1))
        logits = self.classifier(y)
        return F.interpolate(logits, size=input_hw,
                             mode="bilinear", align_corners=False)
```

원 논문 TensorFlow 구현과 PyTorch 구현은 padding, BatchNorm epsilon/momentum, bilinear resize의 coordinate convention이 다를 수 있다. 공개 checkpoint와 pixel 단위로 비교하려면 이 세 항목부터 맞춰야 한다.

## 학습 설정

PASCAL VOC 2012는 20개 foreground class와 background를 포함한다.

- 원래 분할: train 1,464, val 1,449, test 1,456
- 추가 annotation을 포함한 `trainaug`: 10,582장
- Crop: `513x513`
- 초기 learning rate: `0.007`
- Schedule: poly policy
- Random scale augmentation
- OS 16에서는 BatchNorm parameter도 fine-tuning
- Decoder를 포함해 end-to-end 학습
- 평가 지표: 21개 class 평균 mIoU

Modified Xception의 ImageNet 사전학습은 Nesterov momentum `0.9`, initial LR `0.05`, 2 epoch마다 `0.94` decay, weight decay `4e-5`를 사용했다. 50 GPU에서 GPU당 batch 32의 비동기 학습이라는 설정이므로 소규모 재현 환경의 BatchNorm 통계와 optimization dynamics가 그대로 같을 것이라고 기대하면 안 된다.

## Decoder ablation

### Low-level channel 수

VOC val에서 low-level feature의 projection channel 수를 바꾼 결과다.

| Channel | 8 | 16 | 32 | 48 | 64 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| mIoU | 77.61 | 77.92 | 78.16 | **78.21** | 77.94 |

48까지 늘리면 좋아지지만 64에서는 떨어진다. 더 많은 detail channel이 항상 좋은 것이 아니며, high-level semantic feature와의 균형이라는 설명을 지지한다.

### Refinement 구조

| Low-level feature | Refinement | mIoU |
| --- | --- | ---: |
| Conv2 | `3x3, 256` 1회 | 78.21 |
| Conv2 | `3x3, 256` 2회 | **78.85** |
| Conv2 | `3x3, 256` 3회 | 78.02 |
| Conv2 | `3x3, 128` 1회 | 77.25 |
| Conv2 | `1x1, 256` 1회 | 78.07 |
| Conv2+Conv3 | 단계적 결합 | 78.61 |

두 번의 `3x3, 256`이 가장 좋다. Conv3 skip을 추가한 더 복잡한 decoder도 `78.61`로 열세다. 이 ablation은 decoder가 깊어서가 아니라 적절한 resolution과 channel bottleneck을 사용해서 효과가 난다는 점을 보여 준다.

## Accuracy와 계산량 trade-off

ResNet-101 기반 VOC val의 핵심 행을 분리하면 다음과 같다.

| Train OS | Eval OS | Decoder | MS/Flip | mIoU | Multiply-Adds |
| ---: | ---: | :---: | --- | ---: | ---: |
| 16 | 16 | - | - | 77.21 | 81.02B |
| 16 | 16 | O | - | **78.85** | 101.28B |
| 16 | 8 | - | - | 78.51 | 276.18B |
| 16 | 8 | O | - | **79.35** | 297.92B |
| 16 | 16 | O | MS | 80.09 | 898.69B |
| 16 | 16 | O | MS+Flip | 80.22 | 1797.23B |
| 32 | 32 | - | - | 75.43 | 52.43B |
| 32 | 32 | O | - | 77.37 | 74.20B |

Decoder는 OS 16에서 `+1.64 mIoU`에 약 `+20.26B` Multiply-Adds를 지불한다. Eval OS 8은 decoder보다 훨씬 큰 계산 증가를 만든다. Multi-scale과 flip은 mIoU를 더 올리지만 0.1~1%대 개선을 위해 계산이 한 자릿수 이상 증가한다. 실시간 배포 표로는 반드시 single-scale, no-flip 행을 사용해야 한다.

## Atrous separable convolution ablation

Modified Xception, train/eval OS 16의 single-scale 설정에서 다음 행이 중요하다.

| 설정 | mIoU | Multiply-Adds |
| --- | ---: | ---: |
| Decoder, 표준 convolution | 79.93 | 89.76B |
| Decoder, separable ASPP/decoder | 79.79 | **54.17B** |

리뷰어 계산으로 Multiply-Adds는 약 `39.7%` 줄고 mIoU 차이는 `-0.14`다. 논문 전체에서 가장 온디바이스 친화적인 결과다. 하지만 이는 실제 모바일 latency 측정이 아니라 연산 수 비교다. Depthwise atrous kernel의 backend 효율을 별도로 측정해야 한다.

## 최종 benchmark 결과

### PASCAL VOC 2012 test

| 모델 | mIoU |
| --- | ---: |
| DeepLabv3 | 85.7 |
| DeepLabv3-JFT | 86.9 |
| DeepLabv3+ Xception | 87.8 |
| DeepLabv3+ Xception-JFT | **89.0** |

### Cityscapes

- X-65 + ASPP: `77.33` val mIoU
- Decoder 추가: `78.79`
- Image-level branch 제거: `79.14`
- 더 깊은 X-71: `79.55`
- Coarse annotation까지 사용한 test: **82.1 mIoU**

Cityscapes에서 image-level feature 제거가 더 좋았다는 결과는 global pooling branch가 보편적으로 이득이라는 해석을 막는다. Dataset의 장면 구성과 crop 전략에 따라 global prior의 효과가 달라진다.

### 경계 성능

논문은 void label 주변을 dilation한 trimap에서 mIoU를 측정했다. 가장 좁은 band에서 decoder는 단순 bilinear upsampling보다 ResNet-101에서 `4.8%`, Xception에서 `5.4%` mIoU를 높였다. 전체 mIoU의 작은 개선보다 경계 근처 개선이 크므로 decoder의 목적과 증거가 잘 맞는다.

## 장점과 핵심 기여

1. **문맥과 경계를 분리해 설계했다.** ASPP는 semantic context, 얕은 skip decoder는 boundary recovery를 담당한다.
2. **Output stride를 명시적 계산 예산 손잡이로 만들었다.** 같은 학습 구조에서 OS 8/16/32를 비교할 수 있다.
3. **Decoder가 단순하다.** Skip 하나, 48-channel projection, `3x3` 두 번으로 충분하다는 ablation을 제시한다.
4. **Atrous separable convolution을 실용적으로 결합했다.** ASPP와 decoder의 계산량을 큰 정확도 손실 없이 줄였다.
5. **주장과 평가가 연결된다.** 전체 mIoU뿐 아니라 trimap 실험으로 경계 개선을 직접 측정했다.
6. **후속 경량 segmentation의 강한 baseline이 되었다.** Encoder 교체가 쉬워 MobileNet, EfficientNet 계열과 결합하기 좋다.

## 한계와 비판적 관점

### 1. 최고 정확도는 효율 설정과 다르다

VOC 89.0은 JFT 사전학습을 포함하고, benchmark 최고 설정에는 COCO, OS 8, multi-scale, flip 같은 비용 요소가 개입한다. 논문의 "faster"를 최종 SOTA 행과 곧바로 연결하면 안 된다.

### 2. 실제 latency와 memory가 없다

Multiply-Adds는 보고하지만 CPU/GPU/NPU latency, p50/p95, peak RSS, activation peak, 전력은 없다. Atrous depthwise convolution은 장치별 kernel 품질에 민감하므로 MAC만으로 실행 속도를 예측하기 특히 어렵다.

### 3. Decoder activation이 크다

Parameter는 가벼워도 OS 4에서 304-channel concat과 256-channel feature를 유지한다. 고해상도 입력에서는 decoder가 activation memory와 memory bandwidth 병목이 될 수 있다.

### 4. 학습 조건이 크고 복잡하다

ImageNet, COCO, JFT의 단계적 사전학습과 대규모 BatchNorm 학습은 작은 연구 환경에서 동일하게 재현하기 어렵다. 추가 데이터의 이득과 architecture의 이득을 분리해서 비교해야 한다.

### 5. Dilation의 grid effect를 직접 해결하지 않는다

큰 dilation rate는 sampling 위치가 성기게 떨어지는 gridding 문제를 만들 수 있다. ASPP의 여러 rate가 이를 완화하지만, content-adaptive sampling은 아니다.

### 6. 단일 skip은 얇은 구조에 한계가 있다

복잡한 multi-level decoder보다 효율적이지만, 매우 작은 물체나 가느다란 경계는 OS 4 feature 하나만으로 부족할 수 있다. 논문도 sofa/chair 혼동, 심한 occlusion, 드문 시점의 실패를 제시한다.

## 자주 헷갈리는 지점

### DeepLabv3와 DeepLabv3+의 차이는 backbone인가

핵심 차이는 decoder 추가다. 논문은 동시에 modified aligned Xception도 제안하므로 backbone 개선과 decoder 효과를 표에서 분리해 읽어야 한다.

### Atrous convolution은 upsampling인가

아니다. Feature map을 직접 키우는 연산이 아니라 stride를 제거한 상태에서 receptive field를 유지하도록 kernel sampling 간격을 넓힌다.

### Dilation을 키우면 MAC도 증가하는가

동일한 입력/output shape와 kernel tap 수라면 dilation 자체는 MAC를 늘리지 않는다. 그러나 stride 제거로 feature map을 조밀하게 만들면 전체 MAC와 activation은 크게 증가한다.

### Depthwise separable convolution과 atrous convolution은 경쟁 관계인가

아니다. 하나는 channel factorization, 다른 하나는 공간 sampling 간격에 관한 설계다. DeepLabv3+는 depthwise spatial convolution에 dilation을 함께 적용한다.

### Decoder가 deconvolution을 사용하는가

논문의 기본 decoder는 bilinear resize와 convolution을 사용한다. Learned transposed convolution이 필수는 아니다.

### 89.0 mIoU가 54.17B 설정의 결과인가

아니다. `54.17B`는 특정 VOC val single-scale, OS 16, separable 설정의 계산량이다. Test 최고 수치는 추가 사전학습과 더 비싼 평가 구성이 포함된 별도 조건이다.

## 온디바이스 관점

### 유리한 점

- Decoder 구조가 고정적이고 control flow가 없다.
- Bilinear resize, `1x1`, depthwise, pointwise convolution은 모바일 backend의 기본 operator다.
- Output stride와 input resolution로 accuracy-latency를 연속적으로 조절할 수 있다.
- Prompt나 autoregressive decoder가 없어 상시 실행 semantic segmenter에 적합하다.

### 위험한 operator와 graph pattern

- `depthwise convolution + dilation > 1`의 NPU 지원 여부
- ASPP 병렬 branch와 global pooling의 synchronization
- Resize 이후 large feature concatenate가 만드는 DRAM traffic
- NHWC/NCHW transpose가 branch마다 삽입되는지
- Dynamic input shape에서 bilinear resize가 accelerator 밖으로 나가는지
- BatchNorm folding 후 depthwise/pointwise fusion이 유지되는지

### 배포 우선순위

1. OS 16, single-scale, no-flip을 기준으로 고정한다.
2. 입력 해상도와 class 수를 실제 제품 조건으로 고정한다.
3. ASPP/decoder를 separable convolution으로 변환한 graph가 NPU에 전부 올라가는지 확인한다.
4. Concat 전후 tensor lifetime을 profiler로 확인한다.
5. FP16과 INT8에서 boundary mIoU 하락을 별도로 측정한다.
6. 평균 latency뿐 아니라 p50/p95, peak memory, 전력, 10분 지속 실행의 thermal throttling을 기록한다.

INT8에서는 depthwise channel별 range와 bilinear resize 주변 activation range가 민감할 수 있다. 전체 mIoU만 보면 얇은 경계 손실이 가려지므로 class별 IoU와 boundary F-score 또는 trimap mIoU를 함께 봐야 한다.

## 재현 계획

### 최소 재현

- Dataset: PASCAL VOC 2012 `trainaug/val`
- 입력: `513x513`
- Backbone: 먼저 ResNet-101로 논문 ablation을 맞춘 뒤 MobileNet 계열로 교체
- 기준: OS 16, single-scale, no-flip
- 비교 1: naive bilinear decoder 대 DeepLabv3+ decoder
- 비교 2: low-level projection channel `32/48/64`
- 비교 3: standard 대 atrous separable ASPP/decoder
- 지표: mIoU, boundary mIoU, MACs, parameters, peak activation, p50/p95 latency

### 반드시 고정할 구현 조건

- Resize의 `align_corners` 및 half-pixel convention
- `SAME` padding과 odd input 크기 처리
- Output stride별 stride/dilation 치환 위치
- ASPP rate
- BatchNorm 통계의 freeze 여부
- Ignore label과 loss resize 방향
- 원본 resolution로 복원한 뒤 평가하는지

### 작은 ablation 하나

가장 좋은 첫 실험은 low-level projection channel이다. `32, 48, 64`를 비교하면서 mIoU뿐 아니라 concat activation과 latency를 기록한다. 채널 48의 정확도 우위가 현대 backbone과 INT8에서도 유지되는지 확인할 수 있고, 논문의 설계 논리를 직접 검증한다.

## 로드맵에서의 위치

FCN은 fully convolutional dense prediction을 열었고, U-Net은 skip을 통한 localization 복원을 정립했다. DeepLabv3+는 이 두 흐름에 atrous multi-scale context와 효율적인 separable operator를 결합한다. 이후 SegFormer는 convolution decoder를 더 단순한 MLP head로 바꾸며, SAM 계열은 class 고정 semantic segmentation에서 promptable mask generation으로 문제를 확장한다.

온디바이스 프로젝트에서는 DeepLabv3+를 다음 기준선으로 쓰기 좋다.

- CNN 기반 상시 semantic segmenter
- Backbone과 decoder 비용을 분리해서 측정하기 쉬운 구조
- ROI crop이나 동적 해상도 정책의 효과를 실험하기 쉬운 구조
- INT8과 operator fusion의 장치 의존성을 드러내는 대표 모델

## 구현 체크리스트

- [ ] PDF의 OS 16/8 정의와 코드의 stride-dilation 치환이 일치하는가?
- [ ] ASPP 출력이 정확히 256 channel인가?
- [ ] Low-level feature를 48 channel로 줄였는가?
- [ ] Skip concatenate 직전 공간 크기를 tensor shape로 확인했는가?
- [ ] Decoder에 `3x3, 256` 두 번을 사용했는가?
- [ ] Standard와 separable convolution 결과를 같은 recipe로 비교했는가?
- [ ] MACs에 multi-scale/flip 횟수가 반영되었는가?
- [ ] Full-resolution logits를 포함해 peak activation을 측정했는가?
- [ ] NPU fallback과 layout transpose를 profiler에서 확인했는가?
- [ ] mIoU와 boundary 지표를 함께 보고했는가?
- [ ] `batch=1`, 고정 해상도에서 p50/p95 latency를 측정했는가?
- [ ] 10분 이상 지속 실행해 전력, 온도, throttling을 기록했는가?

## 최종 평가

DeepLabv3+의 강점은 화려한 decoder가 아니라 **어떤 정보는 낮은 해상도에서 모으고 어떤 정보는 높은 해상도에서 복원해야 하는지 명확히 분리한 것**이다. ASPP가 문맥을, 48-channel low-level skip과 두 번의 `3x3`이 경계를 담당한다. 논문의 ablation은 이 단순함이 우연이 아님을 설득력 있게 보여 준다.

온디바이스 관점에서는 parameter 수보다 activation과 operator support가 중요하다. OS 8과 multi-scale 평가는 정확도를 높이지만 계산량을 폭증시키며, OS 4 decoder의 concat은 메모리 peak 후보가 된다. 따라서 실제 배포 baseline은 `OS=16 + single-scale + separable ASPP/decoder`로 두고, 입력 해상도와 low-level channel을 장치별로 조절하는 것이 합리적이다.

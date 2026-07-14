# 32. CBAM: Convolutional Block Attention Module

## 논문 정보

- 제목: **CBAM: Convolutional Block Attention Module**
- 저자: Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon
- 발표: ECCV 2018
- 핵심 키워드: channel attention, spatial attention, average pooling, max pooling, sequential gating

## 한눈에 보는 요약

CBAM은 CNN feature를 channel과 spatial 두 축으로 순차적으로 재가중한다.

```math
\begin{aligned}
F'&=M_c(F)\otimes F,\\
F''&=M_s(F')\otimes F'.
\end{aligned}
```

- Channel attention `M_c ∈ R^{C×1×1}`은 “무엇이 중요한가”를 선택한다.
- Spatial attention `M_s ∈ R^{1×H×W}`은 “어디가 중요한가”를 선택한다.

두 branch 모두 average pooling과 max pooling을 함께 사용한다. Channel branch는 공간 축을 pooling한 두 descriptor를 shared MLP에 넣고, spatial branch는 channel 축을 pooling한 두 2D map을 concatenate해 7×7 convolution을 적용한다.

Full `C×H×W` attention map을 직접 예측하지 않고 channel gate와 spatial gate의 곱으로 factorize해 가볍다. ResNet-50 ImageNet top-1 error는 `24.56% → 22.66%`, COCO Faster R-CNN mAP는 `27.0 → 28.1`로 개선됐다.

<p align="center"><img src="https://github.com/user-attachments/assets/69122527-e541-4f50-9447-244cfa041c04" alt="CBAM channel and spatial attention modules" width="820"></p>
<p align="center"><sub>Figure 2 — channel attention과 spatial attention의 순차 구조</sub></p>

## SE에서 확장된 문제의식

SE block은 global average pooling으로 channel importance를 예측하지만 spatial 위치를 버린다. CBAM은 다음 두 관찰에서 확장한다.

1. Average descriptor는 전체 response를 담지만 salient feature의 강한 evidence를 희석할 수 있다.
2. Channel을 고른 뒤에도 feature map의 어느 위치를 볼지 정하는 spatial gate가 필요하다.

CBAM은 두 축을 동시에 거대한 tensor로 학습하기보다 순차 factorization한다.

## 전체 수식

입력 `F ∈ R^{C×H×W}`에 대해

```math
\begin{aligned}
F'&=M_c(F)\otimes F,\\
F''&=M_s(F')\otimes F'.
\end{aligned}
```

`⊗`는 broadcast element-wise multiplication이다. Channel gate는 H,W에 broadcast되고 spatial gate는 C에 broadcast된다.

최종 gate를 펼쳐 보면 대략

```math
A(c,h,w)=M_c(c)\cdot M_s(h,w\mid F')
```

형태의 low-rank factorization이다. Spatial gate가 channel-refined feature `F'`에 조건화되어 있어 완전히 독립인 단순 outer product보다는 유연하다.

## Channel attention module

Spatial 축에 average pooling과 max pooling을 각각 적용한다.

```math
\begin{aligned}
F_{\mathrm{avg}}^c&=\mathrm{AvgPool}_{H,W}(F)&&\in\mathbb{R}^{C\times1\times1},\\
F_{\mathrm{max}}^c&=\mathrm{MaxPool}_{H,W}(F)&&\in\mathbb{R}^{C\times1\times1}.
\end{aligned}
```

두 descriptor를 같은 MLP에 통과시켜 더하고 sigmoid를 적용한다.

```math
M_c(F)=\mathrm{sigmoid}\!\left(
\mathrm{MLP}(F_{\mathrm{avg}}^c)+\mathrm{MLP}(F_{\mathrm{max}}^c)
\right)
```

Shared MLP는 SE처럼 bottleneck을 갖는다.

```math
\begin{aligned}
\mathrm{MLP}(z)&=W_1\mathrm{ReLU}(W_0z),\\
W_0&:\ C\to C/r,\\
W_1&:\ C/r\to C.
\end{aligned}
```

두 pooling path가 weight를 공유하므로 parameter는 SE와 거의 같다. Average는 전체 activation 통계를, max는 가장 강한 salient response를 제공한다.

## Spatial attention module

Channel-refined `F'`에서 channel 축을 pooling한다.

```math
\begin{aligned}
F_{\mathrm{avg}}^s&=\mathrm{AvgPool}_{C}(F')&&\in\mathbb{R}^{1\times H\times W},\\
F_{\mathrm{max}}^s&=\mathrm{MaxPool}_{C}(F')&&\in\mathbb{R}^{1\times H\times W}.
\end{aligned}
```

두 map을 channel 방향으로 concatenate하고 convolution한다.

```math
M_s(F')=\mathrm{sigmoid}\!\left(
\mathrm{Conv}_{7\times7}\!\left([F_{\mathrm{avg}}^s;F_{\mathrm{max}}^s]\right)
\right)
```

Input channel은 2, output channel은 1이므로 7×7 convolution parameter는 매우 작다. Kernel 7은 3보다 넓은 local context로 salient region 경계를 판단한다.

## 왜 channel 먼저인가

논문은 세 arrangement를 비교한다.

- Channel → Spatial
- Spatial → Channel
- Channel과 Spatial 병렬

| 구성 | Top-1 error | Top-5 error |
| --- | ---: | ---: |
| Channel only(SE 형태) | 23.14 | 6.70 |
| Channel → Spatial | **22.66** | **6.31** |
| Spatial → Channel | 22.78 | 6.42 |
| Parallel | 22.95 | 6.59 |

Channel-first가 가장 좋다. 먼저 의미 있는 feature detector를 고른 뒤, 그 refined response에서 spatial saliency를 찾는 순서가 자연스럽다는 해석이다. Sequential은 두 gate가 서로 조건화되어 병렬보다 표현력이 높다.

## Pooling ablation

Channel attention에서 pooling 방법을 비교한다.

| 구성 | Top-1 error | Top-5 error |
| --- | ---: | ---: |
| ResNet-50 baseline | 24.56 | 7.50 |
| Average only(SE) | 23.14 | 6.70 |
| Max only | 23.20 | 6.83 |
| Average + Max | **22.80** | **6.52** |

Average와 max가 서로 보완적임을 보여준다. Max만으로도 SE와 비슷해 strongest response가 channel selection에 유용하다는 점도 드러난다.

## Spatial module ablation

Channel pooling과 kernel size를 비교한 결과다.

| Spatial descriptor | Kernel | Top-1 error | Top-5 error |
| --- | ---: | ---: | ---: |
| 1×1 conv projection | 3 | 22.96 | 6.64 |
| 1×1 conv projection | 7 | 22.90 | 6.47 |
| Avg+Max pool | 3 | 22.68 | 6.41 |
| Avg+Max pool | 7 | **22.66** | **6.31** |

Channel을 learned 1×1 projection으로 줄이는 것보다 parameter-free avg/max pooling이 좋고, 7×7이 3×3보다 약간 좋다.

## 계산량과 parameter

Channel MLP는 `2C²/r`, spatial conv는 `2×1×7×7=98` weight 정도다. ResNet-50 기준

```text
baseline : 25.56M params, 3.858 GFLOPs
CBAM     : 28.09M params, 3.864 GFLOPs
```

FLOPs 증가는 매우 작지만 parameter는 SE channel MLP 때문에 약 2.5M 늘어난다. 여러 block의 late-stage channel이 크기 때문이다.

## ImageNet 결과

| 모델 | Params | GFLOPs | Top-1 error | Top-5 error |
| --- | ---: | ---: | ---: | ---: |
| ResNet-50 | 25.56M | 3.858 | 24.56 | 7.50 |
| ResNet-50 + SE | 28.09M | 3.860 | 23.14 | 6.70 |
| ResNet-50 + CBAM | 28.09M | 3.864 | **22.66** | **6.31** |

CBAM은 같은 parameter 규모의 SE보다 top-1 error를 0.48%p 더 낮춘다. ResNet-101, WideResNet, MobileNet 등에서도 일관된 개선을 보고한다.

## COCO object detection

Faster R-CNN에 ImageNet-pretrained ResNet backbone을 사용한 결과 중 R50은 다음과 같다.

| Backbone | AP50 | AP75 | COCO AP |
| --- | ---: | ---: | ---: |
| ResNet-50 | 46.2 | 28.1 | 27.0 |
| ResNet-50 + CBAM | **48.2** | **29.2** | **28.1** |

Classification뿐 아니라 localization/detection에서 spatial gate가 유효함을 보여준다. VOC 2007 detection에서도 one-stage baseline을 개선했다.

## Grad-CAM 해석

논문 시각화에서 CBAM backbone은 baseline이나 SE보다 target object 영역에 더 집중하고 배경 activation이 줄어드는 경향을 보인다. Channel attention이 relevant detector를 고르고 spatial attention이 object region을 강조한다는 설계 직관과 맞는다.

다만 Grad-CAM과 gate map은 model output의 원인 전체를 설명하지 않는다. Attention visualization을 qualitative sanity check로만 보는 것이 안전하다.

## 장점과 기여

- Channel과 spatial attention을 가벼운 sequential module로 통합했다.
- Average와 max pooling의 보완성을 두 branch 모두에서 활용했다.
- Full 3D attention map 대신 두 factor gate로 parameter/compute를 줄였다.
- 다양한 CNN에 plug-in 가능하고 classification·detection에서 일관된 이득을 보였다.
- Ablation으로 pooling, kernel, order 선택을 체계적으로 검증했다.

## 한계와 비판적 관점

### 1. Factorized gate의 표현 한계

Channel마다 서로 다른 spatial map을 만들지 않는다. 모든 channel이 같은 `M_s(h,w)`를 공유하고, 모든 위치가 같은 channel gate를 공유한다.

### 2. Feature mixing이 아니다

Non-local/Transformer처럼 위치 간 value를 가져오지 않고 기존 activation을 곱으로 조절한다. Long-range dependency를 직접 생성하지 않는다.

### 3. Spatial gate도 local convolution

7×7 receptive field로 saliency를 판단하므로 image 전체 global spatial relation은 여러 CNN layer에 의존한다.

### 4. Parameter overhead

FLOPs는 작지만 channel MLP가 `C²`에 비례한다. 매우 넓은 network나 tiny device에서는 부담이 될 수 있다.

### 5. 최신 training recipe와 재평가

2018 ResNet recipe 기준 improvement다. Strong augmentation, normalization, modern backbone에서 절대 이득은 달라질 수 있다.

## SE와 비교

```text
SE:
  GAP -> MLP -> channel gate

CBAM:
  (GAP + GMP) -> shared MLP -> channel gate
  then
  channel avg/max maps -> 7x7 conv -> spatial gate
```

CBAM은 SE의 channel descriptor를 풍부하게 하고 공간 축을 추가한다.

## 구현 체크리스트

- Channel avg/max pooling axis가 H,W인가?
- 두 descriptor가 같은 MLP weight를 공유하는가?
- Spatial attention은 channel-refined `F'`를 입력으로 받는가?
- Spatial pooling axis가 C인가?
- 7×7 convolution padding이 H,W를 유지하는가?
- Gate를 residual branch 안/밖 어느 위치에 넣었는지 baseline과 맞는가?

## 온디바이스 관점

CBAM은 pairwise attention이 없어 memory와 compute가 작고, 7×7 conv도 input channel 2라 가볍다. 하지만 GAP/GMP, small MLP, sigmoid, broadcast가 여러 operator로 쪼개지면 kernel launch와 memory pass가 latency를 늘릴 수 있다.

NPU에서는 channel branch를 1×1 conv로, spatial branch를 supported depthwise/standard conv로 fuse할 수 있는지 확인해야 한다. Tiny model에서는 모든 block보다 late stage 또는 bottleneck block에 선택적으로 넣는 편이 효율적일 수 있다.

## 최종 평가

CBAM은 channel “what”과 spatial “where”를 순차적으로 분해해 CNN feature를 정제한다. Avg/max descriptor, shared MLP, 7×7 spatial convolution이라는 단순한 구성으로 SE보다 나은 ImageNet·COCO 결과를 얻었다. 위치 간 새로운 정보를 전달하는 self-attention은 아니지만, 작고 범용적인 gating module로서 CNN attention 설계의 대표 기준점이다.

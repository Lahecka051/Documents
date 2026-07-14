# 31. Squeeze-and-Excitation Networks

## 논문 정보

- 제목: **Squeeze-and-Excitation Networks**
- 저자: Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu
- 발표: CVPR 2018 / TPAMI
- 핵심 키워드: channel attention, squeeze, excitation, global average pooling, feature recalibration

## 한눈에 보는 요약

Squeeze-and-Excitation(SE) block은 CNN feature map의 channel마다 input-dependent gate를 예측해 중요한 channel은 키우고 덜 중요한 channel은 줄인다.

```math
\begin{aligned}
\mathrm{Squeeze}:&\quad C\times H\times W\to C
&&\text{(global average pooling)},\\
\mathrm{Excitation}:&\quad C\to C/r\to C
&&\text{(two FC layers, ReLU, sigmoid)},\\
\mathrm{Scale}:&\quad U_c\leftarrow s_cU_c.
\end{aligned}
```

Convolution이 local receptive field 안에서 spatial·channel 정보를 섞지만, channel 간 dependency를 명시적으로 모델링하지 않는다는 문제의식에서 출발한다. SE는 image 전체 spatial context를 한 vector로 압축하고 그 vector로 모든 channel의 gate를 공동 예측한다.

`r=16`에서 계산 overhead는 매우 작고 다양한 ResNet/ResNeXt/MobileNet/ShuffleNet에 일관된 개선을 보였다. SE-ResNet-50은 ImageNet top-5 error를 `7.48% → 6.62%`로 낮췄고, SENet ensemble은 ILSVRC 2017에서 top-5 error `2.251%`로 1위를 기록했다.

<p align="center"><img src="https://github.com/user-attachments/assets/ce481d90-6608-4181-9b11-b7f131304b33" alt="Squeeze and Excitation block" width="820"></p>
<p align="center"><sub>Figure 1 원본·확대 패널 — squeeze·excitation·channel recalibration 흐름</sub></p>

## 문제의식: Channel dependency

Convolution output `U ∈ R^{C×H×W}`에서 각 output channel은 여러 input channel을 local kernel로 조합한다. 그러나 filter response가 어떤 image에서는 중요하고 다른 image에서는 불필요할 수 있어도 convolution weight 자체는 입력에 따라 바뀌지 않는다.

SE block은 global image context를 보고 channel response를 동적으로 recalibrate한다. Spatial location마다 다른 weight를 만드는 것이 아니라 channel 하나에 scalar 하나를 적용한다.

## Squeeze: Global information embedding

Feature map channel `u_c`를 global average pooling한다.

```math
z_c=F_{sq}(u_c)=\frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W}u_c(i,j)
```

결과 `z ∈ R^C`는 각 channel의 전체 image response를 요약한다. Local convolution만으로는 멀리 떨어진 spatial 위치를 여러 layer에 걸쳐 모아야 하지만 GAP은 한 번에 global receptive field를 제공한다.

Average를 쓰므로 spatial arrangement는 사라진다. “어디”가 중요한지는 표현하지 않고 “어떤 channel이 전체적으로 활성화되었는가”를 담는다.

## Excitation: Adaptive channel gate

Descriptor `z`를 bottleneck MLP에 넣는다.

```math
\begin{aligned}
s&=\mathrm{sigmoid}\!\left(W_2\mathrm{ReLU}(W_1z)\right),\\
W_1&\in\mathbb{R}^{C/r\times C},\\
W_2&\in\mathbb{R}^{C\times C/r},\\
s&\in(0,1)^C.
\end{aligned}
```

첫 FC는 channel을 `C/r`로 줄이고 두 번째가 다시 `C`로 확장한다. Reduction ratio `r`은 parameter와 capacity를 조절한다.

Sigmoid는 channel마다 독립 gate를 허용한다. Softmax를 쓰면 channel끼리 합 1을 경쟁해야 하지만, SE는 여러 channel이 동시에 중요할 수 있다고 본다.

## Scale

```math
\tilde x_c=s_c\cdot u_c
```

Scalar `s_c`를 `H×W` 전체에 broadcast한다. 같은 channel의 모든 spatial position이 동일 비율로 조정된다.

Residual network에서는 보통 residual branch의 convolution 뒤, skip connection과 더하기 전에 SE를 넣는다.

```math
\begin{aligned}
\mathrm{residual}&=\mathrm{ConvBlock}(x),\\
\mathrm{residual}&=\mathrm{SE}(\mathrm{residual}),\\
y&=x+\mathrm{residual}.
\end{aligned}
```

따라서 identity path는 유지되고 residual feature만 input-dependent하게 gate된다.

## Parameter와 FLOPs

Excitation MLP parameter는 bias를 제외하면

```math
C\left(\frac{C}{r}\right)+\left(\frac{C}{r}\right)C=\frac{2C^2}{r}
```

이다. Spatial 크기와 무관하고 convolution FLOPs에 비해 작다. 모든 block에 넣으면 channel이 큰 late stage에서 parameter가 늘어난다.

논문에서 ResNet-50은 약 `3.86 GFLOPs`, SE-ResNet-50은 `3.87 GFLOPs`다. Training batch 256의 forward는 190ms에서 209ms, single-image inference는 164ms에서 167ms로 보고된다. FLOPs overhead는 작지만 작은 FC와 GAP의 실제 latency는 hardware에 따라 다르다.

## Reduction ratio `r`

작은 `r`은 bottleneck이 넓어 표현력과 parameter가 크고, 큰 `r`은 가볍지만 channel interaction capacity가 줄어든다. 논문은 `r=16`을 기본으로 사용한다.

ImageNet ablation에서 넓은 범위의 `r`이 baseline을 개선했고 performance가 비교적 robust했다. Accuracy를 거의 유지하면서 late stage의 SE parameter를 줄이는 변형도 가능했다. 따라서 16은 이론적 최적값이 아니라 실용적 절충이다.

## 왜 GAP인가

Global max pooling과 비교했을 때 average pooling이 더 좋은 ImageNet error를 보였다. Max는 가장 강한 위치 하나만 남겨 sparse evidence에 민감하고, average는 object의 전체 response distribution을 반영한다.

후속 CBAM은 average와 max를 함께 사용해 상호보완적 descriptor를 만들지만, SE의 핵심은 하나의 안정적인 global descriptor로 channel dependency를 학습하는 것이다.

## Excitation nonlinearity

Sigmoid 대신 단순 linear, ReLU, tanh 등을 비교한다. Gate가 channel 간 nonlinear relation을 모델링하고 여러 channel을 독립적으로 활성화해야 하므로 `ReLU bottleneck + sigmoid output`이 가장 안정적이다.

너무 단순한 excitation은 baseline 아래로 떨어질 수 있어, 개선이 parameter 추가만이 아니라 gating function의 선택에 의존함을 보인다.

## ImageNet 결과

대표 결과는 다음과 같다.

| 모델 | Top-1 error | Top-5 error | GFLOPs |
| --- | ---: | ---: | ---: |
| ResNet-50 | 24.80 | 7.48 | 3.86 |
| SE-ResNet-50 | **23.29** | **6.62** | 3.87 |

SE-ResNet-50은 계산량을 거의 늘리지 않고 top-1 error를 1.51%p, top-5를 0.86%p 개선했다. Top-5 6.62는 더 깊은 ResNet-101의 6.52에 근접한다.

다른 backbone에서도 일관된다.

- SE-ResNet-101 top-5 error 6.07
- SE-ResNeXt-50 top-5 error 5.49
- SE-ResNeXt 계열에서 baseline 대비 지속적 개선
- MobileNet top-1 error `28.4 → 25.3`, top-5 `9.4 → 7.7`

MobileNet에서는 569→572 MFLOPs, parameter 4.2M→4.7M으로 비교적 작은 비용으로 큰 accuracy 개선을 보였다.

## ILSVRC 2017

SENet architecture ensemble은 test top-5 error `2.251%`로 1위를 기록했다. 2016 winning entry의 2.991%보다 relative 약 25% 감소다.

이 숫자는 단일 SE block의 독립 효과가 아니라 architecture, training, ensemble 전체 결과지만 SE design의 scale-up 가능성을 보여준다.

## 다른 dataset과 task

Places365에서 ResNet-50 top-1/top-5 accuracy `57.9/38.0` 대비 SE-ResNet-50은 `61.0/40.4`를 기록했다. CIFAR-10/100에서도 여러 backbone의 error를 줄였고, detection backbone으로 사용할 때도 baseline보다 개선됐다.

즉 SE가 ImageNet class taxonomy에만 맞춘 trick이 아니라 general feature recalibration임을 보여준다.

## Stage별 역할 해석

논문은 excitation activation을 layer 깊이와 class별로 시각화한다.

- 초기 stage의 gate는 class 간 비슷해 low-level edge/color feature를 일반적으로 recalibrate한다.
- 후반 stage는 class별로 다른 activation pattern을 보여 semantic specialization이 강하다.
- 같은 class image에서는 더 유사한 channel selection이 나타난다.

이는 SE가 단순 normalization scale이 아니라 input-dependent conditional computation을 수행한다는 해석을 뒷받침한다. 다만 gate 시각화 역시 causal explanation은 아니다.

## 장점과 기여

- Channel dependency를 명시적으로 모델링하는 간단한 plug-in block을 제시했다.
- GAP으로 global context를 매우 싸게 가져왔다.
- Bottleneck MLP와 sigmoid로 input-dependent channel gate를 만들었다.
- 다양한 CNN family와 dataset에서 일관된 improvement를 보였다.
- FLOPs overhead가 거의 없고 기존 residual block에 쉽게 통합된다.

## 한계와 비판적 관점

### 1. Spatial 위치를 잃는다

GAP 후 channel scalar 하나만 남기므로 object가 어디에 있는지, 여러 region이 어떤 관계인지 표현하지 못한다.

### 2. Channel gate만 있고 새로운 spatial interaction은 없다

Non-local attention처럼 먼 위치 feature를 섞지 않는다. 기존 convolution feature를 재가중할 뿐이다.

### 3. Late-stage parameter 증가

MLP parameter가 `C²`에 비례한다. Channel이 매우 큰 model에서는 `r`을 써도 parameter overhead가 무시하기 어려울 수 있다.

### 4. Sigmoid 범위

Gate가 0~1이라 원 feature를 직접 증폭한다기보다 상대적 억제/유지에 가깝다. Residual과 다음 layer가 보상하지만 표현을 제한할 수 있다.

### 5. FLOPs와 latency 차이

GAP와 작은 FC는 mobile accelerator에서 memory synchronization을 유발할 수 있다. 0.01 GFLOPs 증가가 항상 negligible latency는 아니다.

## CBAM과 차이

| 방법 | Channel gate | Spatial gate | Global pairwise mixing |
| --- | --- | --- | --- |
| SE | GAP + MLP | 없음 | 없음 |
| CBAM | Avg/Max + shared MLP | Avg/Max + 7×7 conv | 없음 |
| Non-local | projection에 포함 | position pair attention | 있음 |

SE는 가장 가볍고 channel “what”에 집중한다.

## 구현 체크리스트

- GAP가 batch/channel을 제외한 H,W에만 적용되는가?
- `C/r`이 0이 되지 않도록 minimum channel을 두는가?
- Residual branch의 어느 지점에 SE를 넣는가?
- Sigmoid gate shape가 `[B,C,1,1]`로 올바르게 broadcast되는가?
- Quantization에서 sigmoid와 small FC를 device가 지원하는가?
- Parameter 수뿐 아니라 single-batch latency를 측정했는가?

## 온디바이스 관점

SE는 global pairwise attention이 없어 activation memory가 `O(CHW)`이고 overhead가 작아 mobile CNN에 적합하다. 실제 MobileNet 결과도 강하다. GAP와 FC를 fuse하거나 1×1 convolution으로 표현하면 NPU 배포가 쉽다.

다만 very small model에서는 FC parameter와 sigmoid가 상대적으로 커질 수 있다. Hard-sigmoid, ECA류 1D channel convolution, stage별 선택적 삽입을 비교할 수 있다.

## 최종 평가

SE Networks는 attention을 복잡한 token pair 관계가 아니라 **global context가 channel response를 조건부로 조절하는 문제**로 풀었다. 구조가 단순하고 계산량이 작으며 거의 모든 CNN backbone에 붙일 수 있다는 점이 강점이다. Spatial 정보와 위치 간 상호작용을 직접 모델링하지 않는 한계는 있지만, channel attention의 표준을 만든 가장 영향력 있는 vision block 중 하나다.

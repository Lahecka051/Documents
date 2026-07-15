# 36. MobileNetV2: Inverted Residuals and Linear Bottlenecks

## 논문 정보

- 제목: **MobileNetV2: Inverted Residuals and Linear Bottlenecks**
- 저자: Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
- 소속: Google Inc.
- 발표: CVPR 2018
- arXiv: [https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)
- PDF 기준 버전: arXiv v4, 2019-03-21
- 원본 파일: `36_MobileNetV2.pdf`
- 핵심 키워드: depthwise separable convolution, inverted residual, linear bottleneck, ReLU6, activation memory, SSDLite

## 한눈에 보는 요약

MobileNetV2는 모바일 CNN block을 다음 세 단계로 재구성한다.

```text
얇은 입력 bottleneck
 -> 1×1 pointwise expansion + ReLU6
 -> 3×3 depthwise convolution + ReLU6
 -> 1×1 linear projection
 -> 조건이 맞으면 얇은 bottleneck끼리 residual add
```

일반적인 ResNet bottleneck이 `넓음 -> 좁음 -> 넓음`이고 넓은 표현 사이를 shortcut으로 연결한다면, MobileNetV2의 inverted residual은 `좁음 -> 넓음 -> 좁음`이며 **좁은 표현 사이를 연결**한다. 넓은 중간 tensor는 비선형 변환의 표현력을 제공하고, 얇은 입출력 tensor는 block 사이에서 전달되는 정보와 메모리 비용을 담당한다.

두 번째 핵심은 마지막 projection에 ReLU를 넣지 않는 **linear bottleneck**이다. 저자들은 낮은 차원에서 ReLU가 좌표를 0으로 접으면 정보를 복원할 수 없다고 보고, 비선형성은 충분히 넓힌 중간 공간에만 둔다. 이 주장은 엄밀한 보편 정리라기보다 저차원 manifold 직관과 실험으로 뒷받침되는 설계 가설이다.

논문이 보고한 기본 `MobileNetV2 1.0, 224`는 ImageNet top-1 `72.0%`, `3.4M` parameters, `300M` multiply-adds, Google Pixel 1의 단일 big CPU core에서 TF-Lite `75 ms`이다. MobileNetV1의 `70.6%`, `4.2M`, `575M`, `113 ms`보다 정확도와 비용을 함께 개선했다. 다만 이 latency는 2018년 Pixel 1 CPU와 당시 TF-Lite에 한정되며, peak memory와 현대 NPU latency는 논문이 직접 보고하지 않는다.

## 연구 배경과 문제의식

### 모바일 모델에서는 FLOPs만 줄여서는 충분하지 않다

모바일 추론은 다음 제약을 동시에 받는다.

- 연산량과 parameter 수
- 중간 activation이 차지하는 peak memory
- cache에 들어가지 못해 발생하는 DRAM traffic
- framework와 accelerator가 실제로 지원하는 operator
- 작은 kernel 호출, tensor layout 변환, fusion 여부

MobileNetV1은 표준 convolution을 depthwise convolution과 pointwise convolution으로 분해해 계산량을 크게 줄였다. 그러나 더 높은 정확도를 얻으면서도 중간 표현의 메모리를 작게 유지하고, 특수한 group shuffle 같은 연산 없이 범용 framework에서 실행할 block이 필요했다.

MobileNetV2의 문제 설정은 단순히 "더 적은 MAC으로 ImageNet 정확도를 높이자"가 아니다. 논문은 분류뿐 아니라 COCO detection과 PASCAL VOC segmentation까지 같은 backbone을 적용하고, parameter, multiply-add, 실제 Pixel latency, 중간 tensor materialization을 함께 본다.

### 표준 convolution과 depthwise separable convolution

입력과 출력 shape를 각각

```math
X\in\mathbb{R}^{h\times w\times d_i},
\qquad
Y\in\mathbb{R}^{h\times w\times d_j}
```

라 하고 kernel 크기를 `k×k`라 하자. stride 1, same padding인 표준 convolution의 weight와 MAC은 다음과 같다.

```math
\#W_{\mathrm{conv}}=k^2d_id_j,
\qquad
\mathrm{MAC}_{\mathrm{conv}}=hwd_id_jk^2
```

Depthwise separable convolution은 channel마다 하나의 spatial filter를 적용한 뒤 `1×1` pointwise convolution으로 channel을 섞는다.

```math
\mathrm{MAC}_{\mathrm{DSConv}}
=hwd_i k^2+hwd_id_j
=hwd_i(k^2+d_j)
```

표준 convolution 대비 비율은

```math
\frac{\mathrm{MAC}_{\mathrm{DSConv}}}
{\mathrm{MAC}_{\mathrm{conv}}}
=\frac{1}{d_j}+\frac{1}{k^2}
```

이다. `k=3`이고 `d_j`가 충분히 크면 약 `1/9`에 가까워진다. 논문은 이를 8-9배 적은 계산이라고 설명한다. 다만 depthwise convolution은 arithmetic intensity가 낮아, MAC 감소가 같은 비율의 latency 감소로 이어지지는 않는다.

## Linear bottleneck의 핵심 직관

### ReLU가 정보를 잃는 방식

ReLU는 각 좌표에 대해

```math
\mathrm{ReLU}(x)=\max(0,x)
```

를 적용한다. 음수 값을 모두 같은 0으로 보내므로 일반적으로 역함수가 없다. 저자들의 직관은 실제 image activation이 전체 고차원 공간을 채우지 않고 더 낮은 차원의 manifold 근처에 놓인다는 가정에서 출발한다.

- 표현 공간이 충분히 넓으면 어떤 channel이 0으로 접혀도 다른 channel이 정보를 보존할 가능성이 있다.
- bottleneck처럼 channel 수가 작은 공간에서는 ReLU가 서로 다른 입력을 같은 출력으로 collapse할 위험이 커진다.
- 따라서 bottleneck projection 뒤에는 비선형성을 제거하고, ReLU는 expansion된 고차원 공간에서 사용한다.

이 논문의 부록은 ReLU 뒤에도 역변환 가능한 embedding 조건을 논의하지만, 자연 image의 activation manifold가 실제로 항상 그 조건을 만족한다고 증명하지는 않는다. "ReLU는 저차원에서 무조건 나쁘다"가 아니라, **좁은 정보 통로의 끝에 불필요한 ReLU를 두지 않는 것이 이 architecture와 학습 설정에서 더 좋았다**가 정확한 결론이다.

### Linear bottleneck은 block 전체가 선형이라는 뜻이 아니다

마지막 `1×1` projection만 activation function 없이 선형이다. 앞의 expansion과 depthwise 뒤에는 ReLU6가 있으므로 block 전체 함수는 비선형이다.

```math
F(X)=P\left(\mathrm{ReLU6}\left(
D\left(\mathrm{ReLU6}(E(X))\right)
\right)\right)
```

- `E`: `1×1` expansion
- `D`: `3×3` depthwise convolution
- `P`: activation 없는 `1×1` projection

Residual 조건을 만족하면 최종 출력은

```math
Y=X+F(X)
```

이다.

## Inverted residual block

입력이 `B×H×W×C_in`, expansion ratio가 `t`, 출력 channel이 `C_out`, stride가 `s`라 하자. 논문의 TensorFlow 표기와 달리 아래에서는 읽기 쉽게 NHWC를 쓴다.

| 단계 | 연산 | 출력 shape | activation |
| --- | --- | --- | --- |
| 입력 | - | `B×H×W×C_in` | - |
| Expansion | `1×1`, `C_in -> tC_in` | `B×H×W×tC_in` | ReLU6 |
| Spatial filtering | `3×3` depthwise, stride `s` | `B×H/s×W/s×tC_in` | ReLU6 |
| Projection | `1×1`, `tC_in -> C_out` | `B×H/s×W/s×C_out` | **없음** |
| Shortcut | identity add | 위와 같음 | `s=1`이고 `C_in=C_out`일 때만 |

`t=1`인 첫 bottleneck은 expansion `1×1`을 생략하는 구현이 일반적이다. 논문의 기본 모델은 첫 stage를 제외하면 `t=6`을 사용한다. ReLU6는

```math
\mathrm{ReLU6}(x)=\min(\max(0,x),6)
```

이며 bounded range가 저정밀 계산에 더 안정적이라는 이유로 선택됐다.

### 왜 residual이 "inverted"인가

고전적인 bottleneck residual은 넓은 tensor를 shortcut에 보관하고, residual branch에서 channel을 줄였다가 다시 넓힌다. MobileNetV2는 얇은 tensor를 shortcut에 보관하고 residual branch 안에서 channel을 확장한다.

```text
고전 residual:        wide -- narrow -- wide
                     |_______________|

MobileNetV2:          thin -- wide -- thin
                     |______________|
```

따라서 shortcut으로 오래 살아 있어야 하는 tensor가 얇다. 이것이 parameter 수보다 **activation lifetime** 관점에서 중요한 설계다.

## Batch=1 tensor shape와 수치 예시

### 예시: `28×28×32 -> 14×14×64`, `t=6`, `stride=2`

입력을 다음과 같이 두자.

```math
X\in\mathbb{R}^{1\times28\times28\times32}
```

1. `1×1` expansion은 channel을 `32 -> 192`로 늘린다.

```math
E(X)\in\mathbb{R}^{1\times28\times28\times192}
```

2. stride 2 depthwise convolution은 spatial size를 절반으로 줄인다.

```math
D(E(X))\in\mathbb{R}^{1\times14\times14\times192}
```

3. linear projection은 channel을 `192 -> 64`로 줄인다.

```math
Y\in\mathbb{R}^{1\times14\times14\times64}
```

Stride와 channel이 모두 달라 identity shortcut은 없다.

### 이 block의 parameter 계산 - 리뷰어 계산값

BatchNorm affine parameter와 bias를 제외한 convolution weight 수는

```math
\begin{aligned}
\#W_{expand}&=32\times192=6{,}144,\\
\#W_{dw}&=3\times3\times192=1{,}728,\\
\#W_{project}&=192\times64=12{,}288,\\
\#W_{total}&=20{,}160.
\end{aligned}
```

이 `20,160`은 위에서 임의로 택한 단일 block에 대한 리뷰어 계산값이며 논문의 model 전체 parameter 보고값이 아니다.

### 이 block의 MAC 계산 - 리뷰어 계산값

Stride 2에서는 expansion과 나머지 두 연산의 spatial 크기가 다르므로 각각 계산해야 한다.

```math
\begin{aligned}
\mathrm{MAC}_{expand}
&=28^2\times32\times192
=4{,}816{,}896,\\
\mathrm{MAC}_{dw}
&=14^2\times192\times3^2
=338{,}688,\\
\mathrm{MAC}_{project}
&=14^2\times192\times64
=2{,}408{,}448,\\
\mathrm{MAC}_{total}
&=7{,}564{,}032.
\end{aligned}
```

따라서 이 예에서는 pointwise convolution 두 개가 MAC의 약 `95.5%`를 차지한다. Depthwise를 더 최적화하는 것만으로 전체 latency가 크게 줄지 않을 수 있으며, `1×1` GEMM/conv의 fusion과 memory layout이 중요하다.

### Activation memory - 리뷰어 계산값

각 tensor의 원소 수와 FP16 저장량은 다음과 같다. 아래 값은 tensor 하나만 저장할 때의 크기이며 runtime peak가 아니다.

| Tensor | 원소 수 | FP16 bytes | 대략적 크기 |
| --- | ---: | ---: | ---: |
| 입력 `1×28×28×32` | 25,088 | 50,176 | 49.0 KiB |
| Expansion `1×28×28×192` | 150,528 | 301,056 | 294.0 KiB |
| DW 출력 `1×14×14×192` | 37,632 | 75,264 | 73.5 KiB |
| Projection `1×14×14×64` | 12,544 | 25,088 | 24.5 KiB |

가장 큰 단일 tensor는 expansion 출력이다. 그러나 실제 peak memory는 allocator, in-place ReLU, convolution workspace, layout conversion, residual tensor lifetime에 따라 달라진다. 이 표의 합을 곧바로 peak memory로 부르면 안 된다.

## 전체 MobileNetV2-1.0 구조

논문의 Table 2에서 `t`는 expansion ratio, `c`는 output channel, `n`은 반복 수, `s`는 첫 block의 stride다. 같은 stage의 나머지 block은 stride 1이다.

| 입력 | 연산 | `t` | `c` | `n` | `s` |
| --- | --- | ---: | ---: | ---: | ---: |
| `224²×3` | `3×3 conv` | - | 32 | 1 | 2 |
| `112²×32` | bottleneck | 1 | 16 | 1 | 1 |
| `112²×16` | bottleneck | 6 | 24 | 2 | 2 |
| `56²×24` | bottleneck | 6 | 32 | 3 | 2 |
| `28²×32` | bottleneck | 6 | 64 | 4 | 2 |
| `14²×64` | bottleneck | 6 | 96 | 3 | 1 |
| `14²×96` | bottleneck | 6 | 160 | 3 | 2 |
| `7²×160` | bottleneck | 6 | 320 | 1 | 1 |
| `7²×320` | `1×1 conv` | - | 1280 | 1 | 1 |
| `7²×1280` | global average pool | - | - | 1 | - |
| `1×1×1280` | classifier `1×1` | - | classes | 1 | - |

초기 convolution 뒤에 17개 bottleneck block이 있고 마지막 `1×1`, pooling, classifier가 이어진다. 논문 본문은 이를 초기 convolution 뒤의 19 residual bottleneck layers라고 표현하지만, Table 2의 반복 수를 합하면 bottleneck block은 `1+2+3+4+3+3+1=17`이다. 구현을 검증할 때는 공식 architecture table과 코드의 block count를 기준으로 삼는 편이 안전하다.

Width multiplier `α`와 입력 resolution은 accuracy-latency trade-off knob다. 기본 모델은 `α=1.0`, `224×224`이며, 논문은 resolution `96-224`, width multiplier `0.35-1.4`를 탐색한다. 가장 작은 점부터 큰 점까지 계산량은 약 `7M-585M` MAdds, parameter는 `1.7M-6.9M` 범위다. `α<1`에서도 마지막 1280-channel convolution에는 multiplier를 적용하지 않아 작은 모델의 정확도를 보완한다.

## Activation memory와 논문의 channel-splitting 제안

### 논문이 정의한 graph peak memory

논문은 실행 graph `G`의 schedule 중 필요한 live tensor와 operator 내부 저장공간의 최대를 최소화하는 문제로 memory를 정의한다. 거의 직렬인 graph에서는 다음과 같이 단순화한다.

```math
M(G)=\max_{op\in G}
\left[
\sum_{A\in op_{in}}|A|
+\sum_{B\in op_{out}}|B|
+|op|
\right]
```

여기서 `|op|`는 kernel workspace 같은 operator 내부 저장공간이다. 이는 parameter file size가 아니라 **추론 중 live activation과 workspace의 peak**를 보는 식이다.

### 넓은 tensor를 완전히 materialize하지 않는 방법

Expansion `A`, channel-wise 비선형/Depthwise 연산 `N`, projection `B`를

```math
F(X)=(B\circ N\circ A)(X)
```

라고 하자. Expanded channel을 `t`개 조각으로 나누면

```math
F(X)=\sum_{i=1}^{t}(B_i\circ N\circ A_i)(X)
```

처럼 각 조각의 projection 결과를 누적할 수 있다. `N`이 channel-wise이므로 아직 계산하지 않은 expanded channel이 현재 조각에 필요하지 않다. 이론상 한 channel씩 처리하면 넓은 expansion tensor 전체를 보관하지 않고도 결과를 만들 수 있다.

논문의 Table 3은 FP16 activation을 가정할 때 각 해상도에서 materialize해야 하는 최대 channel/memory를 비교한다. 최대치는 MobileNetV1 `1600 KB`, MobileNetV2 `400 KB`, ShuffleNet `600 KB`다. 예를 들어 MobileNetV2의 `112×112×16` bottleneck은

```math
112\times112\times16\times2
=401{,}408\ \text{bytes}
\approx 400\ \text{decimal KB}
```

로 표의 값과 대응한다.

중요한 단서가 있다. 논문은 channel을 너무 잘게 나누면 하나의 큰 matrix multiplication이 여러 작은 연산으로 바뀌어 cache miss와 launch overhead가 늘어난다고 직접 지적한다. 당시 구현에서는 `t=2-5` 정도의 split이 실용적이었다. 따라서 "inverted residual이면 framework가 자동으로 400 KB peak를 보장한다"고 해석하면 안 된다. 실제 deployment에서는 compiler가 expansion-DW-projection을 tile/fuse하는지 확인해야 한다.

## 핵심 block 구현 예시

아래 코드는 논문의 계산 순서를 드러내기 위한 PyTorch형 pseudocode다. BatchNorm folding, quantization scale, channel rounding은 생략했다.

```python
class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, stride, expand_ratio):
        super().__init__()
        hidden = int(round(cin * expand_ratio))
        self.use_skip = (stride == 1 and cin == cout)

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(cin, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ]

        layers += [
            nn.Conv2d(
                hidden, hidden, 3,
                stride=stride, padding=1,
                groups=hidden, bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            # projection 뒤에는 ReLU6를 두지 않는다.
        ]
        self.branch = nn.Sequential(*layers)

    def forward(self, x):
        y = self.branch(x)
        return x + y if self.use_skip else y
```

검증할 핵심은 세 가지다.

1. Depthwise의 `groups`가 expanded channel 수와 같은가?
2. Projection 뒤 activation이 정말 없는가?
3. Skip은 `stride=1`이고 input/output channel이 같을 때만 적용되는가?

## 실험 결과

### ImageNet classification - 논문 보고값

아래 latency는 모두 Google Pixel 1의 단일 large CPU core, TF-Lite에서 측정됐다. ShuffleNet은 당시 group convolution과 channel shuffle을 효율적으로 지원하지 않아 latency가 보고되지 않았다.

| Network | Top-1 (%) | Params | MAdds | CPU latency |
| --- | ---: | ---: | ---: | ---: |
| MobileNetV1 | 70.6 | 4.2M | 575M | 113 ms |
| ShuffleNet 1.5× | 71.5 | 3.4M | 292M | 미보고 |
| ShuffleNet 2× | 73.7 | 5.4M | 524M | 미보고 |
| NASNet-A | 74.0 | 5.3M | 564M | 183 ms |
| MobileNetV2 1.0 | 72.0 | 3.4M | 300M | **75 ms** |
| MobileNetV2 1.4 | **74.7** | 6.9M | 585M | 143 ms |

MobileNetV2 1.0은 MobileNetV1보다 `275M` 적은 MAdds와 `38 ms` 짧은 latency로 top-1이 `1.4%p` 높다. 그러나 NASNet-A와 MobileNetV2 1.4처럼 MAdds가 비슷해도 latency가 `183 ms`와 `143 ms`로 다르다. Operator topology와 구현이 MAC만큼 중요하다는 근거다.

### SSDLite와 COCO detection - 논문 보고값

SSDLite는 SSD prediction head의 표준 convolution을 depthwise separable convolution으로 바꾼다. MobileNetV2와 80-class prediction을 구성했을 때 head 비교는 다음과 같다.

| Head | Params | MAdds |
| --- | ---: | ---: |
| SSD | 14.8M | 1.25B |
| SSDLite | **2.1M** | **0.35B** |

`320×320` 입력, COCO `trainval35k -> test-dev` 결과는 다음과 같다.

| Detector | mAP | Params | MAdds | Pixel 1 CPU |
| --- | ---: | ---: | ---: | ---: |
| SSD300 | 23.2 | 36.1M | 35.2B | 미보고 |
| SSD512 | 26.8 | 36.1M | 99.5B | 미보고 |
| YOLOv2 | 21.6 | 50.7M | 17.5B | 미보고 |
| MobileNetV1 + SSDLite | **22.2** | 5.1M | 1.3B | 270 ms |
| MobileNetV2 + SSDLite | 22.1 | **4.3M** | **0.8B** | **200 ms** |

본문에는 MobileNetV2 SSDLite가 세 모델 중 가장 정확하다는 문장이 있으나 Table 6에서는 MobileNetV1 SSDLite `22.2`가 MobileNetV2 SSDLite `22.1`보다 `0.1` 높다. 표 기준으로는 정확도가 사실상 비슷하고 MobileNetV2가 비용과 latency에서 우수하다고 정리하는 것이 정확하다.

### Mobile DeepLabv3와 PASCAL VOC - 논문 보고값

| Backbone/head | OS | ASPP | Multi-scale+flip | mIoU | Params | MAdds |
| --- | ---: | :---: | :---: | ---: | ---: | ---: |
| MNet V1 | 16 | ✓ | - | 75.29 | 11.15M | 14.25B |
| MNet V1 | 8 | ✓ | ✓ | 78.56 | 11.15M | 941.9B |
| MNet V2* | 16 | ✓ | - | 75.70 | 4.52M | 5.8B |
| MNet V2* | 8 | ✓ | ✓ | 78.42 | 4.52M | 387B |
| MNet V2* | 16 | - | - | **75.32** | **2.11M** | **2.75B** |
| MNet V2* | 8 | - | ✓ | 77.33 | 2.11M | 152.6B |
| ResNet-101 | 16 | ✓ | - | 80.49 | 58.16M | 81.0B |
| ResNet-101 | 8 | ✓ | ✓ | 82.70 | 58.16M | 4870.6B |

`MNet V2*`는 1280-channel 마지막 feature가 아니라 320-channel second-last feature에 DeepLabv3 head를 붙인다. 이 선택이 head compute를 크게 줄인다.

## Ablation을 어떻게 읽어야 하는가

### 1. Linear bottleneck과 shortcut 위치

Figure 6은 projection 뒤 ReLU6를 둔 모델보다 linear bottleneck이 학습 후반의 ImageNet top-1에서 더 높고, expanded representation 사이의 shortcut 또는 residual 없음보다 bottleneck 사이 shortcut이 더 높음을 보인다. 하지만 figure만 제공되고 최종 수치 표는 없다. 따라서 그래프를 눈대중으로 소수점 수치화해서 인용하지 않는다.

이 결과가 보여주는 것은 동일한 계산 graph에서 activation과 shortcut 위치가 정확도에 중요하다는 점이다. 다만 linear bottleneck 모델이 함수 집합 면에서 더 강해서 이기는 것은 아니다. 저자도 ReLU가 선형 영역에 머물도록 만들 수 있으므로 비선형 모델의 표현 집합이 더 크다고 지적한다. 개선은 optimization과 정보 보존의 inductive bias로 해석해야 한다.

### 2. 정확한 값이 있는 head ablation

Table 7의 `MNet V2*, OS=16` 두 행은 ASPP 제거 효과를 동일 backbone과 output stride에서 비교할 수 있다.

| 설정 | mIoU | Params | MAdds |
| --- | ---: | ---: | ---: |
| ASPP 포함 | 75.70 | 4.52M | 5.8B |
| ASPP 제거 | 75.32 | 2.11M | 2.75B |
| 변화 | `-0.38%p` | `-2.41M` | `-3.05B` |

ASPP를 제거하면 mIoU는 `0.38%p`만 낮아지는 반면 MAdds는 약 `52.6%` 감소한다. 이는 on-device segmentation에서 backbone뿐 아니라 dense head를 함께 단순화해야 한다는 정량적 ablation이다.

반대로 `MNet V2* + ASPP`에서 `OS=16, single-scale`의 `75.70 mIoU, 5.8B`를 `OS=8, multi-scale+flip`의 `78.42 mIoU, 387B`로 바꾸면 `+2.72%p`를 얻지만 계산은 약 `66.7배`가 된다. 정확도만 보고 test-time augmentation을 채택하면 모바일 목표와 어긋난다.

## 장점과 핵심 기여

- 범용 `1×1`, depthwise `3×3`, BatchNorm, ReLU6, add만으로 강한 모바일 block을 만들었다.
- 좁은 shortcut과 넓은 일시적 변환을 분리해 capacity와 expressiveness를 독립적으로 생각할 설계 언어를 제공했다.
- Linear bottleneck으로 저차원 표현의 불필요한 정보 손실을 줄였다.
- Parameter와 MAdds뿐 아니라 Pixel 1 실제 latency와 activation materialization까지 분석했다.
- 같은 backbone 원리를 SSDLite detection과 Mobile DeepLabv3 segmentation으로 확장했다.
- 이후 EfficientNet, MobileNetV3/V4, modern mobile hybrid block의 출발점이 된 단순하고 재사용 가능한 primitive를 제시했다.

## 한계와 비판적 관점

### 1. Manifold 설명은 설계 직관이지 완전한 증명이 아니다

실제 학습 activation이 가정한 저차원 manifold에 놓이는지, 각 layer에서 ReLU가 어느 정도 정보를 잃는지 직접 측정하지 않는다. Figure 1의 synthetic spiral은 가능성을 설명하지만 자연 image network의 인과적 증거는 아니다.

### 2. 특수 memory schedule은 자동으로 실현되지 않는다

Channel-wise split은 이론적으로 expansion tensor materialization을 줄이지만, 일반 runtime은 전체 tensor를 먼저 만들 수 있다. Split을 과도하게 하면 작은 GEMM과 cache miss 때문에 오히려 느려진다는 점도 논문이 인정한다.

### 3. Depthwise convolution은 hardware에 따라 느릴 수 있다

MAC은 매우 작지만 memory-bound이고 kernel launch 비중이 커질 수 있다. 특히 NPU가 depthwise 또는 per-channel quantization을 비효율적으로 처리하면 표의 CPU 우위가 재현되지 않는다.

### 4. 현대적인 peak memory와 energy가 없다

Table 3은 FP16 tensor materialization의 분석값이고 profiler로 측정한 end-to-end peak RSS가 아니다. 전력, 에너지, 온도, throttling, p95 latency도 보고하지 않는다.

### 5. Detection 정확도 주장의 작은 불일치

본문은 MobileNetV2 SSDLite가 가장 정확하다고 표현하지만 Table 6에서는 MobileNetV1 SSDLite가 `0.1 mAP` 높다. 차이가 작더라도 표와 문장을 구분해 읽어야 한다.

### 6. Small-object와 dense boundary 분석이 제한적이다

COCO 전체 mAP와 VOC mIoU는 제공하지만 AP_S/AP_M/AP_L, category별 성능, boundary quality는 없다. 초기 stride와 작은 channel이 세부 정보에 미치는 영향을 평가하기 어렵다.

## 자주 헷갈리는 지점

### Inverted residual과 depthwise separable convolution은 같은가

아니다. Depthwise separable convolution은 spatial filtering과 channel mixing을 분해하는 연산이다. Inverted residual은 이를 `expand -> depthwise -> linear project` 순서로 배치하고 얇은 입출력 사이에 shortcut을 두는 block topology다.

### Linear bottleneck이면 activation이 전혀 없는가

아니다. 마지막 projection 뒤에만 activation이 없다. Expansion과 depthwise 뒤에는 ReLU6가 있다.

### Expansion ratio 6은 출력 channel의 6배인가

논문 정의는 **입력 bottleneck channel `C_in`의 6배**다. `C_out`이 아니다.

### Stride 2 block에도 residual add가 있는가

없다. Spatial shape가 다르므로 identity add를 사용할 수 없다. 논문의 기본 구현은 projection shortcut을 추가하지 않는다.

### MAdd 300M은 FLOPs 300M과 같은가

논문은 multiply-add 한 쌍을 하나의 MAdd로 센다. 어떤 profiler는 multiply와 add를 각각 한 FLOP으로 세어 약 `600 MFLOPs`로 표시한다. 비교 전 counting convention을 맞춰야 한다.

### Parameter가 작으면 activation memory도 항상 작은가

아니다. Parameter는 weight 저장량이고 activation은 `H×W×C`, tensor lifetime, workspace에 좌우된다. Expansion tensor는 parameter가 거의 늘지 않아도 순간적으로 매우 클 수 있다.

## 온디바이스 구현 관점

### Operator 지원과 fusion

가장 먼저 target runtime에서 다음 graph가 하나 또는 소수 kernel로 fusion되는지 확인해야 한다.

```text
Conv1x1 + folded BN + ReLU6
 -> DWConv3x3 + folded BN + ReLU6
 -> Conv1x1 + folded BN
 -> Add
```

BatchNorm은 inference에서 convolution weight와 bias로 fold할 수 있다. ReLU6는 INT8 calibration range를 제한하는 데 유리하지만, runtime이 clamp를 별도 kernel로 실행하면 overhead가 생길 수 있다.

### Memory bandwidth

High-resolution stage의 expansion tensor가 SRAM/cache를 벗어나 DRAM에 기록됐다가 다시 읽히면 depthwise의 낮은 MAC 이점이 희석된다. Compiler trace에서 tensor allocation뿐 아니라 external-memory bytes를 확인해야 한다. MobileNetV2의 진짜 강점은 parameter 수가 아니라 thin shortcut과 tile 가능한 channel-wise middle에 있다.

### Quantization

논문은 ReLU6가 low-precision에 robust하다고 설명하지만 전체 INT8 정확도 표는 제공하지 않는다. 실제 PTQ에서는 다음을 별도로 확인해야 한다.

- Depthwise의 per-channel weight scale 지원
- Residual add 양쪽 activation scale 정렬
- Linear projection 출력의 outlier와 calibration
- 마지막 classifier와 첫 stem을 FP16/INT8 중 무엇으로 둘지

### Latency 측정 원칙

논문의 `75 ms`를 현재 device의 기준값으로 사용하지 말고 같은 조건에서 다시 측정한다.

- `batch=1`, fixed `224×224`
- warm-up 후 최소 수백 회
- p50과 p95
- big core affinity와 thread 수 고정
- CPU/NPU delegate와 precision 명시
- cold-start compile time과 steady-state latency 분리
- peak memory, 전력, 온도, throttling 함께 기록

## 재현 계획

### 최소 재현 1: block 단위

1. `C_in=C_out=32`, `H=W=28`, `t=6`, `stride=1` block을 구현한다.
2. Projection 뒤 ReLU가 없는지 graph를 출력한다.
3. Residual 포함/제거, projection ReLU6 포함/제거의 2×2 ablation을 수행한다.
4. Hook으로 모든 tensor shape, dtype, bytes를 기록한다.
5. profiler의 MAC과 손 계산값이 같은 counting convention에서 일치하는지 확인한다.

### 최소 재현 2: ImageNet backbone

1. 공식 MobileNetV2-1.0 architecture와 channel schedule을 맞춘다.
2. `224×224`, width multiplier 1.0에서 논문 보고값 `3.4M`, `300M MAdds` 근처인지 확인한다.
3. 공식 pretrained checkpoint의 ImageNet preprocessing과 top-1 evaluation protocol을 고정한다.
4. MobileNetV1과 같은 runtime, precision, thread 설정에서 비교한다.
5. activation peak를 eager runtime과 compiled runtime에서 각각 측정한다.

### 최소 재현 3: on-device ablation

| 실험 축 | 설정 |
| --- | --- |
| Precision | FP32, FP16, INT8 |
| Width | 0.35, 0.5, 0.75, 1.0 |
| Resolution | 160, 192, 224 |
| Backend | CPU, GPU delegate, NPU |
| Metric | top-1, p50/p95, peak memory, energy/image |

각 실험은 parameter와 MAC만 다시 적는 대신 실제 operator별 latency와 DRAM traffic을 남긴다. 특히 `1×1`, depthwise, add, layout conversion의 비율을 분리해야 한다.

## 구현 체크리스트

- [ ] Expansion channel이 `round(C_in × t)`인가?
- [ ] 첫 `t=1` block에서 불필요한 expansion을 생략했는가?
- [ ] Depthwise `groups=hidden_dim`인가?
- [ ] Projection 뒤 ReLU/ReLU6가 없는가?
- [ ] Shortcut 조건이 `stride=1 and C_in=C_out`인가?
- [ ] Width multiplier 뒤 channel rounding 규칙이 checkpoint와 같은가?
- [ ] `α<1`에서도 마지막 1280 channel 처리 규칙을 맞췄는가?
- [ ] BatchNorm을 inference graph에 fold했는가?
- [ ] MAdd와 FLOP counting convention을 기록했는가?
- [ ] 분석 activation 크기와 profiler peak memory를 구분했는가?
- [ ] TF-Lite/Core ML/NNAPI delegate가 depthwise와 residual add를 fusion하는가?
- [ ] 평균뿐 아니라 p95 latency와 thermal steady state를 측정했는가?

## 로드맵에서의 위치와 후속 연결

MobileNetV2는 효율적 백본 단계에서 가장 먼저 읽어야 할 기준점이다. 이후 논문을 볼 때 다음 질문의 기준을 제공한다.

- DeiT: 강한 convolutional inductive bias 없이도 data recipe로 효율을 얻을 수 있는가?
- Swin Transformer: high-resolution에서 계층적 feature map과 local attention이 CNN stage를 어떻게 대체하는가?
- EfficientViT: attention을 선형화했을 때 실제 hardware latency와 activation memory가 좋아지는가?
- MobileNetV4: inverted bottleneck을 UIB로 일반화하고 attention을 섞어 여러 accelerator에서 동시에 Pareto-optimal할 수 있는가?

Detection과 segmentation 단계에서도 MobileNetV2의 교훈은 그대로 남는다. Backbone만 경량화하고 SSD/DeepLab head를 그대로 두면 head가 전체 비용을 지배한다. SSDLite와 ASPP 제거 실험은 **end-to-end pipeline의 모든 stage를 같은 효율 기준으로 설계해야 한다**는 초기 사례다.

## 최종 평가

MobileNetV2의 가장 오래 남은 공헌은 단순히 `300M MAdds`짜리 모델을 만든 것이 아니다. 이 논문은 모바일 block을 **얇은 장기 상태, 넓은 일시적 계산, 선형 정보 통로**로 분해하고, parameter/FLOPs와 activation lifetime을 함께 설계해야 한다는 관점을 정립했다.

Linear bottleneck의 manifold 설명은 경험적 직관에 가깝고, channel-splitting memory 이점은 compiler가 실제로 지원해야 한다. 그럼에도 표준 operator만으로 ImageNet, detection, segmentation에서 일관된 효율을 보였고, Pixel 실측까지 제시했다는 점에서 강한 기준선이다. 온디바이스 연구에서는 이 architecture를 복제하는 데서 끝내지 말고, **expanded tensor가 실제로 materialize되는지와 MAC 감소가 p95 latency·DRAM traffic·전력 감소로 이어지는지**를 profiler로 확인해야 한다.

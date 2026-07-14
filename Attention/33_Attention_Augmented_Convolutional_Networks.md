# 33. Attention Augmented Convolutional Networks

## 논문 정보

- 제목: **Attention Augmented Convolutional Networks**
- 저자: Irwan Bello, Barret Zoph, Ashish Vaswani, Jonathon Shlens, Quoc V. Le
- 발표: ICCV 2019
- 핵심 키워드: attention augmentation, 2D relative self-attention, convolution, global context, vision attention

## 한눈에 보는 요약

이 논문은 convolution을 channel/spatial gate로 재가중하는 대신, convolution branch와 self-attention branch가 **서로 다른 output feature map을 직접 생성**하고 channel 방향으로 concatenate하는 Attention-Augmented Convolution(AAConv)을 제안한다.

```math
\operatorname{AAConv}(X)=\operatorname{Concat}\!\left[
\operatorname{Conv}(X),\operatorname{MHA}_{\mathrm{2D\ relative}}(X)
\right]
```

Convolution은 local pattern과 translation equivariance라는 강한 inductive bias를 제공하고, self-attention은 image 전체의 content-dependent long-range interaction을 제공한다. 두 연산을 경쟁시키기보다 output channel budget을 나눠 병렬 결합한다.

Vision에 맞게 attention score에 relative height와 relative width embedding을 별도로 추가한다. ResNet-50에서 ImageNet top-1은 `76.4 → 77.7`, RetinaNet R50의 COCO AP는 `36.8 → 38.2`로 개선됐으며 parameter 수는 거의 유지됐다.

![Attention-Augmented Convolution의 convolution·relative attention 병렬 구조](https://github.com/user-attachments/assets/f034dee9-c16f-4bd3-b5fc-b6ad6edbc4e8)

## 기존 attention module과 차이

SE와 CBAM은 convolution이 만든 feature `F`에 gate를 곱한다.

```math
F_{\mathrm{out}}=\operatorname{gate}(F)\otimes F
```

AAConv의 attention branch는 value를 global weighted sum해 **새 feature channel**을 만든다.

```math
\begin{aligned}
F_{\mathrm{attn}}&=\operatorname{softmax}\!\left(QK^{\top}+\mathrm{relative\ bias}\right)V,\\
F_{\mathrm{out}}&=\operatorname{concat}(F_{\mathrm{conv}},F_{\mathrm{attn}}).
\end{aligned}
```

따라서 단순 feature selection보다 표현력이 크고, 위치 간 정보를 실제로 이동시킨다.

## Multi-head self-attention on 2D feature

Input `X ∈ R^{H×W×C_in}`를 `N=HW` sequence로 펼친다.

```math
\begin{aligned}
Q&=XW_q&&\in\mathbb{R}^{N\times d_k},\\
K&=XW_k&&\in\mathbb{R}^{N\times d_k},\\
V&=XW_v&&\in\mathbb{R}^{N\times d_v}.
\end{aligned}
```

`N_h` head로 나누어 head별 score와 output을 계산한다.

```math
\begin{aligned}
O_h&=\operatorname{softmax}\!\left(\frac{Q_hK_h^{\top}}{\sqrt{d_k^h}}\right)V_h,\\
\operatorname{MHA}(X)&=W_o\operatorname{Concat}_h(O_h).
\end{aligned}
```

Score는 `[N,N]`이라 global attention cost는 `O((HW)²)`. 논문은 주로 28×28 이하 stage에 적용해 비용을 관리한다.

## 2D relative position embedding

Content dot product만 사용하면 permutation에 대해 equivariant하고 2D 위치 관계를 알기 어렵다. Absolute position은 image resolution 변화와 translation에 불리할 수 있다. 논문은 query position `i=(i_y,i_x)`, key `j=(j_y,j_x)`의 상대 차이를 height와 width로 분리한다.

```math
\operatorname{logit}(i,j)=q_i^{\top}k_j
+q_i^{\top}r^H_{j_y-i_y}
+q_i^{\top}r^W_{j_x-i_x}
```

- `r^H`: relative vertical displacement embedding
- `r^W`: relative horizontal displacement embedding

전체 2D offset마다 embedding을 두는 대신 두 1D table을 합쳐 parameter를 줄인다. Spatial relation이 translation에 따라 보존되며 resolution별 relative range를 interpolation할 수 있다.

## Memory-efficient relative logits

Naive하게 모든 `(i,j)` pair의 relative vector를 만들면 `[H,W,H,W,d]` tensor가 필요하다. 논문은 1D relative-to-absolute indexing trick을 height와 width에 각각 적용해 explicit pair tensor 없이 logits를 만든다.

```text
relative_logits_width(Q, r_W)
relative_logits_height(Q, r_H)
```

각 연산은 einsum과 reshape/padding으로 모든 query-key 상대 offset을 효율적으로 배치한다. 그래도 attention weight 자체 `[HW,HW]` memory는 남는다.

## Attention-Augmented Convolution

원 convolution output channel을 `F_out`이라 하자. 그중 `d_v` channel을 attention에 배정하고 convolution은 `F_out-d_v` channel을 만든다.

```math
\begin{aligned}
\mathrm{conv}_{\mathrm{out}}&=\operatorname{Conv}_{k\times k}(X,\mathrm{channels}=F_{\mathrm{out}}-d_v),\\
\mathrm{attn}_{\mathrm{out}}&=\operatorname{MHA}_{\mathrm{relative}}(X,\mathrm{value\ channels}=d_v),\\
\operatorname{AAConv}(X)&=\operatorname{Concat}(\mathrm{conv}_{\mathrm{out}},\mathrm{attn}_{\mathrm{out}}).
\end{aligned}
```

Attention이 단순 추가 branch가 아니라 convolution channel 일부를 대체하므로 parameter budget을 비슷하게 유지할 수 있다.

## Hyperparameter `κ`와 `υ`

논문은 channel ratio를 다음처럼 정의한다.

```math
\kappa=\frac{d_k}{F_{\mathrm{out}}},
\qquad
\upsilon=\frac{d_v}{F_{\mathrm{out}}}
```

- `κ`: query/key dimension 비율
- `υ`: attention output channel 비율

대부분 ResNet 실험에서 `κ=2υ=0.2`, 즉 `d_k=0.2F_out`, `d_v=0.1F_out`에 가까운 설정을 사용한다. Attention ratio가 커지면 global branch capacity와 quadratic compute/memory가 늘고 convolution local bias는 줄어든다.

## Downsampling

Stride convolution을 대체할 때 attention branch는 query output resolution을 맞춰야 한다. 논문은 attention output을 average pooling하거나 필요한 resolution로 downsample한 뒤 convolution output과 concatenate한다.

Relative embedding도 feature resolution에 맞춰 bilinear interpolation한다. Branch의 spatial shape가 정확히 같아야 한다.

## Parameter 비교

Standard `k×k` convolution parameter는

```math
k^2C_{\mathrm{in}}F_{\mathrm{out}}
```

이다. AAConv는 convolution output channel을 줄이는 대신 Q/K/V/output projection을 추가한다. 적절한 `d_k,d_v`에서는 3×3 convolution channel을 attention으로 대체하면서 전체 parameter가 오히려 약간 감소할 수 있다.

FLOPs는 attention의 `(HW)²` 항 때문에 stage에 따라 늘지만 late stage에서는 H,W가 작아 manageable하다. Parameter 수와 compute가 반드시 함께 움직이지 않는 구조다.

## CIFAR-100 결과

| 모델 | Params | GFLOPs | Top-1 | Top-5 |
| --- | ---: | ---: | ---: | ---: |
| Wide-ResNet-28-10 | 36.3M | 10.4 | 80.3 | 95.0 |
| GE-Wide-ResNet | 36.3M | 10.4 | 79.8 | 95.0 |
| SE-Wide-ResNet | 36.5M | 10.4 | 81.0 | **95.3** |
| AA-Wide-ResNet | 36.2M | 10.9 | **81.6** | 95.2 |

같은 parameter 규모에서 global feature generation이 channel reweighting보다 top-1에서 강했다.

## ImageNet 결과

| 모델 | GFLOPs | Params | Top-1 | Top-5 |
| --- | ---: | ---: | ---: | ---: |
| ResNet-34 | 7.4 | 21.8M | 73.6 | 91.5 |
| AA-ResNet-34 | 7.1 | 20.7M | **74.7** | **92.0** |
| ResNet-50 | 8.2 | 25.6M | 76.4 | 93.1 |
| SE-ResNet-50 | 8.2 | 28.1M | 77.5 | 93.7 |
| AA-ResNet-50 | 8.3 | 25.8M | **77.7** | **93.8** |
| ResNet-101 | 15.6 | 44.5M | 77.9 | 94.0 |
| AA-ResNet-101 | 16.1 | 45.4M | **78.7** | **94.4** |

AA-ResNet-50은 baseline보다 top-1 1.3%p 높고 더 깊은 ResNet-101에 근접한다. Parameter는 SE-ResNet-50보다 작다.

## COCO RetinaNet 결과

| Backbone | GFLOPs | Params | AP | AP50 | AP75 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet-50 | 182 | 33.4M | 36.8 | 54.5 | 39.5 |
| SE-ResNet-50 | 183 | 35.9M | 36.5 | 54.0 | 39.1 |
| AA-ResNet-50 | 182 | 33.1M | **38.2** | **56.5** | **40.7** |
| ResNet-101 | 243 | 52.4M | 38.5 | 56.4 | 41.2 |
| AA-ResNet-101 | 245 | 51.7M | **39.2** | **57.8** | **41.9** |

R50에서 AP가 1.4%p 개선됐다. 이 실험에서는 ImageNet pretrained backbone 없이 RetinaNet을 scratch에서 학습해 attention augmentation의 detection 학습 효과를 보였다.

## Attention ratio와 full attention

Attention output 비율을 높여 convolution channel을 줄여도 상당히 robust했다. `κ=υ=1`에 가까운 fully attentional ResNet-50 변형도 경쟁적인 ImageNet accuracy를 보였다.

그러나 best result는 convolution과 attention 혼합에서 나왔다. Local inductive bias와 global content interaction이 상호보완적이라는 논문의 중심 결론이다.

## Position encoding ablation

| 모델 | Position | Top-1 | Top-5 |
| --- | --- | ---: | ---: |
| AA-ResNet-50 | 없음 | 77.5 | 93.7 |
| AA-ResNet-50 | 2D sine | 77.5 | 93.7 |
| AA-ResNet-50 | CoordConv | 77.5 | 93.8 |
| AA-ResNet-50 | Relative | **77.7** | **93.8** |

혼합 모델에서는 차이가 작지만 attention 비율이 커질수록 relative position의 중요성이 커진다. Fully attentional 변형은 relative encoding으로 top-1이 2.8%p 개선됐다. Convolution branch가 줄면 공간 inductive bias를 position encoding이 대신해야 한다.

## 장점과 기여

- Convolution과 attention output channel을 병렬 concatenate하는 명확한 hybrid primitive를 제시했다.
- Height/width 분리 2D relative self-attention을 효율적으로 구현했다.
- Parameter budget을 유지하면서 global feature channel을 추가했다.
- ResNet, MnasNet, CIFAR, ImageNet, COCO에서 일관된 improvement를 보였다.
- Gate형 attention과 feature-generating self-attention의 차이를 분명히 했다.

## 한계와 비판적 관점

### 1. Quadratic spatial memory

Attention map이 `[HW,HW]`라 early high-resolution stage에 적용하기 어렵다. 논문도 마지막 3 stage에 주로 사용한다.

### 2. Branch balance hyperparameter

`κ,υ`를 architecture와 resolution에 맞춰 조정해야 한다. Attention channel이 너무 적으면 효과가 작고, 너무 많으면 local bias와 efficiency를 잃는다.

### 3. Relative implementation 복잡도

2D relative-to-absolute indexing과 resize가 off-by-one 오류에 민감하다. Variable resolution deployment 검증이 필요하다.

### 4. Parameter와 latency 불일치

Parameter가 비슷해도 global attention은 convolution보다 memory access가 크고 hardware kernel 지원이 약할 수 있다.

### 5. Modern architecture에서의 위치

ViT/Swin/ConvNeXt 이후 pure/hierarchical Transformer와 stronger CNN이 발전했다. AAConv의 절대 성능보다 local-global hybrid 원칙이 더 지속적인 공헌이다.

## 구현 체크리스트

- Flatten한 `(h,w)`와 relative height/width index가 일치하는가?
- `d_k,d_v`가 head 수로 나누어지는가?
- Convolution output channel이 정확히 `F_out-d_v`인가?
- Stride block에서 두 branch의 H,W가 같은가?
- Relative embedding resize가 offset 중심을 보존하는가?
- Early stage에서 `[HW,HW]` peak memory를 확인했는가?

## 온디바이스 관점

AAConv는 local convolution과 global attention을 한 block에서 병렬 실행해 operator fusion이 어렵고 high-resolution memory가 크다. Mobile backbone 실험은 가능성을 보였지만 실제 NPU latency는 attention kernel 지원에 크게 좌우된다.

온디바이스에서는 attention을 late low-resolution stage에만 넣고 `υ`를 작게 유지하거나 window attention으로 제한하는 구성이 현실적이다. Relative bias는 resolution generalization에 유리하지만 interpolation cost를 compile-time에 처리하는 편이 좋다.

## 최종 평가

Attention-Augmented Convolution은 convolution을 버리거나 단순 gate로 보정하는 대신, **local convolution feature와 global self-attention feature를 동일한 output 안에서 병렬적으로 공존**시킨다. 2D relative position과 channel-budget 대체를 통해 parameter를 유지하면서 ImageNet·COCO를 개선했다. Quadratic spatial cost는 한계지만, 이후 vision model의 local-global hybrid 설계를 선명하게 예고한 연구다.

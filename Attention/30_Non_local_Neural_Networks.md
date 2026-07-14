# 30. Non-local Neural Networks

## 논문 정보

- 제목: **Non-local Neural Networks**
- 저자: Xiaolong Wang, Ross Girshick, Abhinav Gupta, Kaiming He
- 발표: CVPR 2018
- 핵심 키워드: non-local operation, global dependency, space-time attention, video recognition, residual block

## 한눈에 보는 요약

Non-local Neural Networks는 convolution과 recurrence가 local neighborhood를 반복해 장거리 정보를 전달하는 한계를 해결하기 위해, 한 위치의 출력을 **모든 위치 feature의 weighted sum**으로 계산하는 generic non-local operation을 제안한다.

```math
y_i=\frac{1}{C(x)}\sum_jf(x_i,x_j)g(x_j)
```

`f`는 위치 `i,j`의 affinity, `g`는 value transformation이다. Embedded Gaussian을 쓰면 식은 사실상 dot-product self-attention과 같다. 이 non-local response를 projection하고 원 입력에 residual로 더해 기존 CNN/I3D에 삽입한다.

논문은 video의 space-time 전체, image의 spatial 전체에서 global interaction을 한 block으로 계산한다. Kinetics에서 5-block Non-local I3D가 ResNet-101 기준 `74.4 → 76.0` top-1로 개선되었고, COCO detection·segmentation·keypoint에도 일관된 이득을 보였다.

<p align="center"><img src="https://github.com/user-attachments/assets/0fcd45ba-eb31-4db1-8862-1a539983180f" alt="Non-local block" width="760"></p>
<p align="center"><sub>원 논문 Figure 2 — affinity·global aggregation·residual로 이어지는 non-local block</sub></p>

## 문제의식

Convolution의 receptive field는 layer를 쌓으면 커지지만 먼 두 위치 정보가 만나려면 여러 hop을 거쳐야 한다. Video에서 서로 떨어진 frame의 같은 object, image에서 멀리 떨어진 body part 관계는 local filter만으로 직접 모델링하기 어렵다.

Recurrent model도 time step을 순차적으로 지나므로 dependency path가 길다. Non-local operation은 거리와 무관하게 모든 위치 pair를 한 layer에서 연결한다.

## Generic non-local operation

입력 `x`의 position index는 image의 `(h,w)` 또는 video의 `(t,h,w)`를 펼친 값이다.

```math
y_i=\frac{1}{C(x)}\sum_{\forall j}f(x_i,x_j)g(x_j)
```

- `i`: output을 만들 query 위치
- `j`: 모든 source 위치
- `f`: pairwise affinity scalar
- `g`: source feature의 value transform
- `C(x)`: normalization factor

Convolution과 달리 `j` 집합이 local kernel에 제한되지 않는다. Output size는 input과 같고, 위치 수를 `N`이라 하면 pairwise map은 `[N,N]`이다.

## Pairwise function의 네 가지 형태

### Gaussian

```math
\begin{aligned}
f(x_i,x_j)&=\exp\!\left(x_i^{\top}x_j\right),\\
C(x)&=\sum_jf(x_i,x_j).
\end{aligned}
```

Feature projection 없이 원 feature 내적을 softmax-like normalization한다.

### Embedded Gaussian

```math
\begin{aligned}
\theta(x_i)&=W_{\theta}x_i,\\
\phi(x_j)&=W_{\phi}x_j,\\
f(x_i,x_j)&=\exp\!\left(\theta(x_i)^{\top}\phi(x_j)\right),\\
C(x)&=\sum_jf(x_i,x_j).
\end{aligned}
```

Row normalization을 적용하면

```math
y=\operatorname{softmax}\!\left(\theta(x)\phi(x)^{\top}\right)g(x)
```

가 되어 Transformer의 scaled dot-product 이전 형태와 거의 같다. 논문은 이 관계를 명시하며 self-attention을 더 일반적인 non-local family의 special case로 본다.

### Dot product

```math
\begin{aligned}
f(x_i,x_j)&=\theta(x_i)^{\top}\phi(x_j),\\
C(x)&=N.
\end{aligned}
```

Exponential/softmax 없이 위치 수로 scaling한다. Weight가 음수일 수 있다.

### Concatenation

```math
f(x_i,x_j)=\operatorname{ReLU}\!\left(w_f^{\top}[\theta(x_i),\phi(x_j)]\right)
```

Query/key feature를 concatenate한 learned compatibility다. 계산은 더 복잡하지만 generic pair function의 범위를 보여준다.

## Non-local block

Non-local output을 channel projection하고 residual로 더한다.

```math
z_i=W_zy_i+x_i
```

`W_z` 뒤에 BatchNorm을 두고 BN scale을 0으로 초기화한다. 처음 삽입했을 때 `z=x`인 identity block이 되므로 pretrained CNN을 깨뜨리지 않고 fine-tune할 수 있다.

```text
input x
 ├─ theta -> Q
 ├─ phi   -> K
 └─ g     -> V
QK^T -> normalization -> weighted V -> W_z -> + x
```

이 zero-init residual 전략은 이후 attention block을 pretrained backbone에 넣는 일반적인 기법으로 이어진다.

## Efficient modification

Channel cost를 줄이기 위해 `theta,phi,g`의 output channel을 원 channel의 절반으로 설정한다. 또한 `phi(x)`와 `g(x)` path에 spatial/temporal subsampling을 적용할 수 있다.

```math
\begin{aligned}
\#Q\text{ positions}&=N,\\
\#K/V\text{ positions after pooling}&=N/2\text{ or }N/4,\\
\text{attention map shape}&=[N\times N/s].
\end{aligned}
```

모든 query가 subsampled global positions를 보므로 non-local 성질은 유지하지만 fine detail은 줄 수 있다. Pairwise cost는 여전히 position 수의 곱이다.

## Space, time, spacetime

Video tensor에서 index `j`의 범위를 바꿔 세 구성을 만들 수 있다.

- Space-only: 각 frame 안의 모든 spatial 위치
- Time-only: 같은 spatial location의 여러 frame
- Spacetime: 모든 frame의 모든 spatial 위치

Spacetime은 움직이는 object가 frame마다 위치를 바꿔도 content affinity로 직접 연결할 수 있다. 3D convolution의 고정 local kernel과 다른 점이다.

## 계산 복잡도와 memory

Position 수 `N=THW`, projected channel `d`라 하면

```math
\begin{aligned}
\text{affinity compute}&:\ O(N^2d),\\
\text{attention memory}&:\ O(N^2),\\
\text{value aggregation}&:\ O(N^2d_v).
\end{aligned}
```

Subsampling은 key/value `N`을 `N/s`로 줄여 `O(N²/s)`로 낮춘다. 그러나 고해상도 feature의 quadratic 병목은 남으므로 논문은 주로 res3/res4처럼 downsample된 stage에 block을 배치한다.

## Kinetics ablation: Pairwise function

ResNet-50 C2D baseline과 non-local block 하나의 결과다.

| 구성 | Top-1 | Top-5 |
| --- | ---: | ---: |
| C2D baseline | 71.8 | 89.7 |
| Gaussian | 72.5 | 90.2 |
| Embedded Gaussian | 72.7 | 90.5 |
| Dot product | **72.9** | 90.3 |
| Concatenation | 72.8 | **90.5** |

형태 차이는 작고 모두 baseline보다 낫다. 특정 similarity 식보다 global aggregation 자체가 주된 이득이라는 저자 해석을 뒷받침한다.

## Block 수와 stage

| Backbone | 구성 | Top-1 | Top-5 |
| --- | --- | ---: | ---: |
| R50 | baseline | 71.8 | 89.7 |
| R50 | 1 block | 72.7 | 90.5 |
| R50 | 5 blocks | 73.8 | 91.0 |
| R50 | 10 blocks | **74.3** | **91.2** |
| R101 | baseline | 73.1 | 91.0 |
| R101 | 5 blocks | **75.1** | **91.7** |

한 block을 res2, res3, res4에 넣으면 top-1이 72.7~72.9로 비슷했고, res5는 72.3으로 이득이 작았다. 너무 늦은 저해상도 stage보다 spatial structure가 남은 중간 stage가 유리하다.

## Space-time ablation

5-block R50에서

| 범위 | Top-1 | Top-5 |
| --- | ---: | ---: |
| Baseline | 71.8 | 89.7 |
| Space-only | 72.9 | 90.8 |
| Time-only | 73.1 | 90.5 |
| Spacetime | **73.8** | **91.0** |

공간 또는 시간 한 축만으로도 개선되지만 전체 spacetime 연결이 가장 좋았다.

## 3D convolution과 비교·결합

R101 기준 5-block NL-C2D는 baseline 대비 parameter/FLOPs 약 `1.2×/1.2×`로 top-1 `75.1`을 기록했다. I3D 3×3×3은 `1.5×/1.8×` 비용으로 74.1이었다.

또한 I3D에 non-local block을 추가하면 R50 `73.3 → 74.9`, R101 `74.4 → 76.0`으로 개선됐다. 3D convolution의 local motion modeling과 non-local global relation이 상호보완적임을 보여준다.

긴 128-frame clip에서는 NL-I3D R101이 `77.7/93.3` top-1/top-5를 기록했다.

## Charades와 COCO

Charades에서도 5-block NL-I3D가 strong I3D baseline보다 classification mAP를 개선했다. COCO에서는 Mask R-CNN backbone에 non-local block 하나를 넣어 R50/R101/ResNeXt의 box와 mask AP를 일관되게 높였다. Keypoint head에 4개 block을 넣었을 때도 long-range body-part relation 덕분에 AP가 개선됐다.

COCO의 single block은 baseline 계산의 5% 미만을 추가했다. 다만 고해상도 feature에 직접 적용하면 비용이 급증하므로 배치 stage가 중요하다.

## 장점과 기여

- Local operation을 반복하지 않고 모든 위치 dependency를 한 block에서 모델링했다.
- Pairwise function을 generic family로 정리하고 self-attention과 연결했다.
- Zero-init residual로 pretrained CNN에 안전하게 삽입했다.
- Video space-time, image detection, segmentation, pose에서 범용성을 보였다.
- Transformer 이전/동시기에 vision global attention의 대표 구조가 되었다.

## 한계와 비판적 관점

### 1. Quadratic position cost

Feature map이 크면 `[THW,THW]` affinity가 memory를 지배한다. Subsampling과 중간 stage 배치가 필요하다.

### 2. Positional information이 약하다

Affinity는 content similarity 중심이고 explicit relative position bias가 없다. 멀리 있는 비슷한 pattern을 잘 연결하지만 공간 관계 자체를 학습하기 어렵다.

### 3. 여러 instantiation 차이가 작다

Pairwise function의 이론적 차이보다 experimental 차이가 작다. Genericity는 장점이지만 어떤 form이 언제 필요한지 명확한 원칙은 부족하다.

### 4. Attention map의 해석

시각화에서 같은 object 영역에 weight가 모여도 causal explanation은 아니다. Feature routing pattern으로 봐야 한다.

### 5. 현대 kernel과 resolution

당시 custom implementation 기준 비용이며, 현대 FlashAttention이나 window attention과 같은 조건에서 재평가가 필요하다.

## 관련 연구와 연결

- **Transformer self-attention**: Embedded Gaussian non-local operation과 거의 같은 수식이다.
- **Attention-Augmented Convolution**: Convolution feature와 relative self-attention feature를 병렬 concatenate한다.
- **ViT**: CNN 내부 일부 block이 아니라 patch sequence 전체를 Transformer로 처리한다.
- **Efficient attention**: Non-local `[N,N]` memory를 줄이기 위한 linear/low-rank 연구가 뒤따른다.

## 구현 체크리스트

- Flatten 순서가 `(T,H,W)` positional mapping과 일치하는가?
- Softmax가 key position `j` 축에 적용되는가?
- `theta,phi,g` channel projection과 transpose shape가 맞는가?
- Subsampling이 K와 V에 같은 위치 mapping으로 적용되는가?
- `W_z`/BN zero-init으로 초기 block이 identity인가?
- High-resolution stage에서 affinity peak memory를 측정했는가?

## 온디바이스·비전 관점

Non-local block은 global relation을 제공하지만 high-resolution mobile vision에는 `[N,N]` memory가 부담이다. 낮은 resolution stage에 1개 block만 넣거나 key/value pooling, window+global token, linear attention으로 대체하는 것이 현실적이다.

고정 feature size에서는 dense GEMM으로 구현 가능해 irregular sparsity보다 NPU 친화적일 수 있다. 그러나 DRAM traffic과 attention map materialization을 피하려면 fused tiled kernel이 필요하다.

## 최종 평가

Non-local Neural Networks는 vision의 장거리 관계를 **모든 위치의 content-weighted aggregation**이라는 간결한 식으로 정리했다. Embedded Gaussian form은 self-attention과 직접 연결되며, residual block으로 CNN/I3D에 쉽게 삽입해 video와 COCO에서 효과를 보였다. Quadratic spatial cost와 약한 positional bias는 한계지만, 현대 vision attention의 출발점 중 하나다.

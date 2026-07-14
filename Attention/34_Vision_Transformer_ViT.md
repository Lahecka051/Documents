# 34. An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale

## 논문 정보

- 제목: **An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale**
- 저자: Alexey Dosovitskiy et al.
- 발표: ICLR 2021
- 핵심 키워드: Vision Transformer, patch embedding, class token, large-scale pretraining, transfer learning

## 한눈에 보는 요약

Vision Transformer(ViT)는 image를 fixed-size patch로 나누고 각 patch를 word token처럼 다뤄, 거의 수정하지 않은 표준 Transformer encoder로 classification한다.

```text
image H×W×C
 -> N = HW/P² patches of P×P×C
 -> linear patch embeddings
 -> prepend [class] token + position embeddings
 -> Transformer encoder
 -> class-token representation -> classifier
```

ViT의 핵심 주장은 vision-specific convolution 없이도 충분히 큰 data로 pretrain하면 Transformer가 강력한 image representation을 학습한다는 것이다. ImageNet 규모만으로 학습할 때는 비슷한 크기의 ResNet보다 몇 %p 낮지만, ImageNet-21k나 JFT-300M으로 pretrain하면 CNN baseline을 넘고 transfer efficiency가 좋아진다.

JFT-300M pretrained ViT-H/14는 ImageNet `88.55%`, ImageNet-ReaL `90.72%`, CIFAR-100 `94.55%`, VTAB 평균 `77.63%`를 기록했다. 논문의 진짜 공헌은 단일 수치보다 **data scale이 inductive bias 부족을 상쇄한다**는 경험 법칙이다.

<p align="center"><img src="https://github.com/user-attachments/assets/8a462cac-4240-4352-8572-7b9270cc80a3" alt="Vision Transformer model overview" width="820"></p>
<p align="center"><sub>Figure 1 — patch embedding·class token·Transformer encoder 구조</sub></p>

## Image를 token sequence로 만들기

Input image `x ∈ R^{H×W×C}`를 `P×P` patch로 나눈다. Patch 수는

```math
N=\frac{HW}{P^2}
```

이다. 각 patch를 flatten하면

```math
x_p\in\mathbb{R}^{N\times(P^2C)}
```

이고 trainable projection `E ∈ R^{(P²C)×D}`로 hidden dimension `D`에 보낸다.

```math
x_pE\in\mathbb{R}^{N\times D}
```

Patchify + linear projection은 kernel size와 stride가 모두 P인 convolution으로 동일하게 구현할 수 있다. 차이는 이후 모든 patch가 global self-attention으로 상호작용한다는 점이다.

## Class token과 position embedding

BERT의 `[CLS]`처럼 learned vector `x_class`를 sequence 앞에 붙인다.

```math
z_0=\left[x_{\mathrm{class}};x_p^1E;\ldots;x_p^NE\right]+E_{\mathrm{pos}}
```

`E_pos ∈ R^{(N+1)×D}`는 learned 1D absolute position embedding이다. 2D-aware embedding도 실험했지만 주 setting에서 큰 이득이 없어 단순 1D를 사용했다.

최종 encoder layer의 class-token state `z_L^0`만 classification head에 넣는다. Patch token은 image evidence를 담고 class token이 self-attention을 통해 전체 정보를 모은다.

## Transformer encoder

Pre-LayerNorm block을 사용한다.

```math
\begin{aligned}
z'_l&=\mathrm{MSA}\!\left(\mathrm{LN}(z_{l-1})\right)+z_{l-1},\\
z_l&=\mathrm{MLP}\!\left(\mathrm{LN}(z'_l)\right)+z'_l.
\end{aligned}
```

MLP는 두 linear layer와 GELU로 구성된다. Encoder 내부는 NLP Transformer와 동일하며 2D convolution, pooling hierarchy가 없다.

Self-attention score는 patch 수에 대해 `[N,N]`이므로 complexity는

```math
O(N^2D)=O\!\left(\left(\frac{HW}{P^2}\right)^2D\right)
```

이다. Patch size를 절반으로 줄이면 token 수가 4배, attention FLOPs는 약 16배가 된다.

## Model variants

| 모델 | Layers | Hidden D | MLP | Heads | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

`ViT-L/16`은 Large model과 16×16 patch, `ViT-H/14`는 Huge와 14×14 patch를 뜻한다. Smaller patch는 더 세밀하지만 compute가 훨씬 크다.

## Inductive bias

CNN은 architecture에 다음 bias를 내장한다.

- Locality: small convolution kernel
- Translation equivariance: 같은 filter를 모든 위치에 공유
- Hierarchy: stage별 resolution 감소와 receptive field 증가

ViT에는 이 bias가 훨씬 적다.

- Patch extraction 단계에만 local/2D structure가 명시된다.
- Self-attention은 모든 patch pair를 동등하게 볼 수 있다.
- Position embedding을 제외하면 spatial relation을 data에서 학습해야 한다.

이 때문에 small/medium dataset에서는 sample efficiency가 낮지만, data가 충분하면 더 약한 bias가 오히려 flexible scaling을 가능하게 한다.

## Higher-resolution fine-tuning

Pretraining보다 높은 resolution로 fine-tune하면 patch size를 유지하므로 token 수가 늘어난다. Learned position embedding 길이가 맞지 않으므로 원 2D grid로 reshape해 bilinear interpolation하고 새 grid에 맞춘다. Class token position은 별도로 유지한다.

```text
pretrain grid: 14×14
fine-tune grid: 24×24
E_pos patch part -> 2D interpolate
```

이 절차는 절대 position embedding을 variable resolution에 사용하는 실용적 방법이지만, training 범위를 크게 벗어난 aspect ratio나 resolution에서는 distortion이 생길 수 있다.

## Hybrid model

Raw patch 대신 ResNet feature map을 token sequence로 사용할 수 있다. CNN stage output의 각 spatial location을 1×1 patch처럼 projection해 Transformer에 넣는다.

Hybrid는 convolution inductive bias와 Transformer global interaction을 결합한다. 작은 compute/data regime에서는 유리할 수 있지만 large-scale에서 pure ViT가 더 단순하고 잘 scale한다는 것이 논문의 경향이다.

## Pretraining dataset

- ImageNet-1k: 1.3M image
- ImageNet-21k: 약 14M image, 21K class
- JFT-300M: 약 300M image, noisy multi-label

같은 model을 data scale별로 pretrain해 transfer를 비교한다. 이 controlled scaling 실험이 논문의 중심이다.

## Data 규모에 따른 결과

ImageNet만으로 pretrain하면 ViT-L이 ViT-B보다 오히려 나쁘고 comparable ResNet에 뒤진다. ImageNet-21k에서는 격차가 줄고, JFT-300M에서 Large/Huge model의 capacity가 본격적으로 드러난다.

```text
small data  : convolutional bias가 sample efficiency에 유리
large data  : ViT의 약한 bias와 scale이 더 높은 ceiling 제공
```

따라서 “ViT가 CNN보다 항상 낫다”가 아니라 pretraining data와 regularization이 architecture 성능을 결정한다.

## State-of-the-art transfer 결과

| 모델 | Pretrain | ImageNet | ImageNet-ReaL | CIFAR-100 | VTAB | TPUv3 core-days |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ViT-H/14 | JFT | **88.55** | **90.72** | **94.55** | **77.63** | 2.5K |
| ViT-L/16 | JFT | 87.76 | 90.54 | 93.90 | 76.28 | 0.68K |
| ViT-L/16 | ImageNet-21k | 85.30 | 88.62 | 93.25 | 72.72 | **0.23K** |
| BiT-L R152×4 | JFT | 87.54 | 90.54 | 93.51 | 76.29 | 9.9K |
| Noisy Student | JFT unlabeled+IN | 88.4/88.5 | 90.55 | - | - | 12.3K |

ViT-L/16은 같은 JFT를 사용한 BiT-L보다 모든 listed task에서 높고 pretraining core-days는 약 14분의 1이다. ViT-H/14는 ImageNet 당시 최고 수준과 경쟁하면서 VTAB transfer도 강했다.

## Training compute 관점

ViT는 동일 parameter 수에서 CNN보다 항상 싸다는 뜻은 아니다. Patch 수와 model 크기에 따라 FLOPs가 크다. 논문의 효율 주장은 대규모 pretraining에서 **목표 transfer accuracy에 도달하는 compute**가 BiT ResNet보다 낮다는 empirical scaling curve에 근거한다.

Batch 4096, Adam, high weight decay 0.1, warmup/linear decay 등 large-scale recipe가 사용된다. Fine-tuning은 SGD momentum과 높은 resolution을 사용한다. Architecture 비교와 training recipe를 분리해 읽어야 한다.

## Learned representation 분석

### Patch embedding filters

첫 projection weight를 시각화하면 edge/color blob 같은 low-level basis가 나타나 CNN 초기 filter와 유사하다.

### Position embedding

Learned position embedding cosine similarity를 보면 같은 row/column 또는 가까운 2D 위치가 유사해진다. 1D table로 시작해도 data에서 2D image topology를 회복한다.

### Attention distance

초기 layer 일부 head는 local neighbor를, 다른 head는 image 전체를 본다. 깊은 layer에서는 attention distance가 전반적으로 커진다. CNN의 고정 local-to-global hierarchy와 달리 ViT는 layer 1부터 global head를 가질 수 있다.

## 장점과 기여

- Image를 patch token으로 바꾸는 최소 수정만으로 표준 Transformer를 vision에 적용했다.
- CNN 없이도 large-scale pretraining에서 SOTA급 transfer를 달성했다.
- Dataset scale과 inductive bias의 관계를 체계적으로 분석했다.
- Simple architecture가 accelerator에서 효율적으로 scale함을 보였다.
- 이후 vision backbone 연구의 중심을 convolution에서 Transformer로 이동시켰다.

## 한계와 비판적 관점

### 1. Data hunger

ImageNet 규모에서는 strong CNN보다 낮다. JFT-300M 같은 비공개 대규모 data 의존은 재현성과 접근성의 한계다.

### 2. Quadratic token attention

고해상도 dense prediction에서는 patch 수가 커져 full attention이 비싸다. 원 ViT는 hierarchical/local attention이 없다.

### 3. Patch boundary와 fine detail

큰 patch는 내부 spatial detail을 한 vector로 압축한다. Small object와 pixel-level task에는 불리할 수 있다.

### 4. Absolute position interpolation

Resolution/aspect ratio 변화에 interpolation이 필요하고 training grid를 크게 벗어난 generalization이 보장되지 않는다.

### 5. Classification 중심

원 논문은 주로 image-level transfer다. Detection/segmentation에는 feature pyramid와 multi-scale adaptation이 추가로 필요하다.

## 구현 체크리스트

- `N=HW/P²`이고 H,W가 patch size로 정확히 나누어지는가?
- Patch flatten channel order와 projection weight가 일치하는가?
- Class token이 position embedding과 함께 sequence 첫 위치에 들어가는가?
- Pre-LN residual 순서가 논문과 맞는가?
- Resolution 변경 시 class token을 제외한 patch position만 2D interpolation하는가?
- Patch size별 attention memory와 FLOPs를 실제 profiler로 확인했는가?

## 온디바이스 관점

ViT는 convolution hierarchy가 없어 patch token 수가 곧 memory/compute를 결정한다. `P=16/32`는 token을 줄여 mobile deployment에 유리하지만 small object detail을 잃는다. Dense GEMM과 LayerNorm은 NPU에 맞을 수 있으나 softmax와 dynamic resolution 지원이 중요하다.

온디바이스에서는 DeiT-style distillation, smaller patch/token pruning, window attention, hybrid stem, quantization을 함께 고려해야 한다. 모델 parameter뿐 아니라 attention activation peak를 측정해야 한다.

## 최종 평가

ViT는 image classification에 필요한 vision-specific architecture를 극단적으로 줄이고, patch token과 표준 Transformer만으로 대규모 data의 힘을 보여줬다. Small data에서는 CNN bias가 유리하지만 data와 compute가 커질수록 ViT가 더 효율적으로 scale하고 transfer한다. 이후 vision Transformer 전체 계보를 만든 기준점이며, 동시에 data 의존성과 quadratic high-resolution cost라는 명확한 한계를 남겼다.

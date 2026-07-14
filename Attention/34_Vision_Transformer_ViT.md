# 34. An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale

## 논문 정보

- 제목: **An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale**
- 저자: Alexey Dosovitskiy et al.
- 발표: ICLR 2021
- 핵심 키워드: Vision Transformer, patch embedding, class token, inductive bias, large-scale pretraining, transfer learning

## 한눈에 보는 요약

Vision Transformer(ViT)는 이미지를 고정 크기 patch로 나누고 각 patch를 word token처럼 다뤄, 거의 수정하지 않은 표준 Transformer encoder로 분류한다.

```text
image H×W×C
 -> N = HW/P² patches of P×P×C
 -> flatten + shared linear projection
 -> prepend [class] token + add position embeddings
 -> L Transformer encoder blocks
 -> final LayerNorm of class token
 -> classifier
```

핵심 질문은 "CNN의 모든 layer에 강하게 내장된 locality와 translation equivariance 같은 convolutional prior 없이도 image representation을 잘 학습할 수 있는가?"이다. ImageNet 규모에서는 큰 ViT가 비슷한 compute의 ResNet보다 불리하지만, ImageNet-21k와 JFT-300M으로 사전학습하면 격차가 역전된다. 논문은 이를 **대규모 데이터가 부족한 image-specific inductive bias를 학습으로 보완할 수 있다는 경험적 증거**로 해석한다.

JFT-300M 사전학습 ViT-H/14는 ImageNet `88.55%`, ImageNet-ReaL `90.72%`, CIFAR-100 `94.55%`, VTAB 19개 과제 평균 `77.63%`를 기록했다. 단, 이 결과가 "약한 inductive bias가 언제나 더 좋다"는 인과적 증명은 아니다. 데이터 크기, 정규화, optimizer, 학습 기간과 fine-tuning recipe가 함께 작용한 결과로 읽어야 한다.

<p align="center"><img src="https://github.com/user-attachments/assets/8a462cac-4240-4352-8572-7b9270cc80a3" alt="Vision Transformer model overview" width="820"></p>
<p align="center"><sub>Figure 1 - patch embedding, class token, Transformer encoder 구조</sub></p>

## 기존 CNN과 무엇이 다른가

CNN은 image용 구조를 강하게 내장한다. 작은 kernel은 가까운 위치부터 결합하고, 같은 kernel을 전체 image에 공유하며, stage별 downsampling으로 local-to-global hierarchy를 만든다. ViT는 이 대부분을 고정 규칙으로 주지 않고 patch 사이의 관계를 self-attention으로 학습한다.

| 관점 | CNN | 원형 ViT |
| --- | --- | --- |
| 기본 단위 | pixel neighborhood | non-overlapping patch token |
| 한 layer의 정보 교환 | 작은 kernel 안의 local interaction | 모든 token 사이의 global interaction |
| 공간 prior | locality, 2D neighborhood, translation equivariance | 고정 patch grid와 position embedding 정도 |
| 가중치 | 상대 offset별 kernel이 입력과 무관하게 고정 | query-key 내용에 따라 image마다 동적으로 변화 |
| 해상도 변화 | convolution은 크기에 독립적 | token 수와 position embedding 길이가 변함 |

따라서 ViT의 장점은 image-specific rule이 적어 표현이 유연하고 accelerator-friendly한 dense matrix 연산으로 scale하기 쉽다는 점이다. 반대로 작은 데이터에서는 CNN이 이미 알고 있는 공간 규칙까지 데이터에서 배워야 하므로 sample efficiency가 낮을 수 있다.

## Image를 token sequence로 만들기

입력 image를 다음과 같이 두자.

```math
x\in\mathbb{R}^{H\times W\times C}
```

이를 `P×P` 크기의 겹치지 않는 patch로 나누면 patch 수는

```math
N=\frac{HW}{P^2}
```

이고, flatten한 patch sequence는

```math
x_p\in\mathbb{R}^{N\times(P^2C)}
```

이다. 모든 patch에 같은 trainable projection을 적용한다.

```math
E\in\mathbb{R}^{(P^2C)\times D},
\qquad
x_pE\in\mathbb{R}^{N\times D}
```

여기서 `D`는 Transformer 전체에서 유지되는 hidden dimension이다. Batch dimension을 생략한 구체적인 shape는 다음과 같다.

| 설정 | Patch grid | Patch token `N` | CLS 포함 길이 `S=N+1` | Encoder input |
| --- | ---: | ---: | ---: | ---: |
| 224×224, P=16, ViT-B | 14×14 | 196 | 197 | 197×768 |
| 224×224, P=32, ViT-B | 7×7 | 49 | 50 | 50×768 |
| 384×384, P=16, ViT-B | 24×24 | 576 | 577 | 577×768 |

Patchify와 linear projection은 `kernel_size=P`, `stride=P`, `out_channels=D`인 convolution 하나로 동일하게 구현할 수 있다. Projection weight가 모든 patch에 공유되므로 ViT에도 완전히 bias가 없는 것은 아니다. 다만 patch 안의 pixel은 하나의 `D`차원 token으로 선형 투영되고, 이후에는 모든 patch가 global attention graph에 들어간다.

고정된 non-overlapping grid는 또 다른 trade-off를 만든다. 물체가 `P`의 배수만큼 이동하면 patch 배열도 함께 이동하지만, 1 pixel처럼 patch 경계를 가로지르는 이동은 patch 내용 자체를 바꾼다. 따라서 patchification은 pixel-level translation equivariance를 보장하지 않는다.

## Class token과 position embedding

BERT의 `[CLS]`처럼 학습 가능한 vector `x_class`를 sequence 앞에 붙이고 position embedding을 더한다.

```math
z_0=\left[x_{\mathrm{class}};x_p^1E;\ldots;x_p^NE\right]+E_{\mathrm{pos}}
```

```math
E_{\mathrm{pos}}\in\mathbb{R}^{(N+1)\times D}
```

기본 설정은 learned 1D absolute position embedding이다. Patch를 raster order로 나열하므로 table 자체는 1D이지만, 학습 후에는 가까운 row와 column의 position vector가 비슷해지는 2D 구조가 나타난다.

최종 image representation은 마지막 class-token state에 LayerNorm을 적용한 값이다.

```math
y=\mathrm{LN}\!\left(z_L^0\right)
```

Class token은 고정된 average pooling이 아니다. 다른 patch token과 양방향으로 self-attention에 참여하면서 분류에 필요한 정보를 학습한다. 사전학습 때는 hidden layer 하나가 있는 MLP head를 붙이고, downstream fine-tuning 때는 이를 제거한 뒤 class 수 `K`에 맞는 zero-initialized linear head `D×K`로 교체한다.

Appendix의 class token 대 global average pooling 실험에서는 learning rate를 각각 맞추면 두 방식이 비슷하게 동작했다. 따라서 ViT의 성공을 class token 자체의 우월성으로 해석하면 안 된다.

## Transformer encoder

ViT는 Pre-LayerNorm encoder block을 사용한다.

```math
\begin{aligned}
z'_l&=\mathrm{MSA}\!\left(\mathrm{LN}(z_{l-1})\right)+z_{l-1},\\
z_l&=\mathrm{MLP}\!\left(\mathrm{LN}(z'_l)\right)+z'_l,
\qquad l=1,\ldots,L.
\end{aligned}
```

MLP는 `D -> D_MLP -> D`의 두 linear layer와 GELU로 구성된다. MLP는 각 token에 독립적으로 같은 weight를 적용하고, token 사이의 정보 이동은 MSA가 담당한다. Encoder 내부에는 convolution, spatial pooling, feature pyramid가 없다.

## Multi-head self-attention과 계산량

CLS를 포함한 sequence 길이를 `S=N+1`, head dimension을 `D_h`라 하면 한 head는 다음을 계산한다.

```math
Q=ZW_q,\qquad K=ZW_k,\qquad V=ZW_v
```

```math
A=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{D_h}}\right)
\in\mathbb{R}^{S\times S},
\qquad
\mathrm{SA}(Z)=AV
```

여러 head의 출력을 concatenate한 뒤 다시 `D`차원으로 projection한다.

```math
\mathrm{MSA}(Z)
=\mathrm{Concat}\!\left[\mathrm{SA}_1(Z),\ldots,\mathrm{SA}_h(Z)\right]W_o
```

한 encoder block의 주요 계산량은 대략

```math
O(SD^2)+O(SD\,D_{\mathrm{MLP}})+O(S^2D)
```

이다. 첫 항은 Q/K/V와 output projection, 둘째 항은 token-wise MLP, 셋째 항은 attention score와 value aggregation의 quadratic 부분이다. 보통 `D_MLP`가 `D`의 상수배이므로 앞의 두 항을 합쳐 `O(SD²)`로 쓰기도 한다. 표준적으로 attention map을 materialize하는 구현의 memory는 head 수를 포함해 `O(hS²)`이다.

Patch size를 절반으로 줄이면 patch 수가 약 4배가 되고 attention의 `S²` 항은 약 16배가 된다. 그러나 projection과 MLP는 `S`에 선형이므로 **전체 model FLOPs가 반드시 16배가 되는 것은 아니다**.

## Model variants

| 모델 | Layers `L` | Hidden `D` | MLP size | Heads | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

`ViT-L/16`은 Large model과 16×16 patch, `ViT-H/14`는 Huge model과 14×14 patch를 뜻한다. 작은 patch는 parameter 수를 거의 늘리지 않고 token 수와 계산량을 크게 늘려 더 세밀한 입력을 제공한다.

Appendix의 scaling ablation에서는 depth 증가가 가장 큰 개선을 보였지만 16 layer 이후부터 diminishing return이 나타났다. Width만 키우는 효과는 상대적으로 작았고, patch를 줄이는 방식은 parameter 증가 없이 성능을 안정적으로 높였다. 저자들은 한 축만 극단적으로 키우기보다 depth, width, patch size를 함께 scale하는 구성이 robust하다고 결론냈다.

## Inductive bias를 정확히 이해하기

Inductive bias는 모델이 유한한 학습 데이터에서 보지 못한 sample로 일반화할 때, 어떤 함수나 규칙을 더 쉽게 선택하도록 만드는 구조적 사전 가정이다. 여기서 bias는 오류나 linear layer의 `+b` 항이 아니다.

### CNN의 image-specific bias

Convolution의 한 위치 출력을 단순화하면 다음과 같다.

```math
y_p=\sum_{\delta\in\mathcal K}W_\delta x_{p+\delta}
```

- **Locality**: 작은 kernel `K` 안의 가까운 pixel부터 결합한다.
- **2D neighborhood structure**: 위, 아래, 좌, 우 같은 상대 offset에 서로 다른 weight를 둔다.
- **Weight sharing**: 같은 `W_δ`를 모든 공간 위치에 사용한다.
- **Translation equivariance**: 경계, padding, stride 효과를 무시하면 입력이 이동할 때 feature map도 같이 이동한다.

```math
\mathrm{Conv}(T_\Delta x)=T_\Delta\mathrm{Conv}(x)
```

Stage별 downsampling과 receptive-field 증가는 convolution 자체의 필수 성질이라기보다 일반적인 CNN architecture가 추가로 부여하는 hierarchy bias이다. 이런 강한 prior가 자연 image와 잘 맞으면 edge와 texture를 적은 데이터로 효율적으로 배울 수 있지만, 입력 내용에 따라 멀리 떨어진 위치를 한 layer에서 직접 연결하는 유연성은 제한된다.

### ViT의 bias는 적지만 0은 아니다

Position embedding이 없는 self-attention은 token 순서를 임의로 바꿔도 출력이 같은 순서를 따라 바뀐다.

```math
\mathrm{SA}(\Pi Z)=\Pi\,\mathrm{SA}(Z)
```

즉 permutation equivariant하다. 모든 patch pair가 한 layer에서 직접 연결될 수 있지만 실제 attention weight는 query-key 내용에 따라 달라진다. 이 구조는 global, content-dependent relation을 선호한다는 자체적인 bias를 가지지만, "옆 patch", "위쪽", "가까운 거리"라는 2D geometry는 알지 못한다.

ViT에도 다음 가정은 남아 있다.

- Image를 고정된 2D patch grid로 자른다.
- 같은 patch projection을 모든 위치에 공유한다.
- MLP는 각 token에 같은 함수를 적용하므로 token 단위로 local하고 weight-shared이다.
- Attention은 모든 token이 pairwise relation으로 상호작용한다고 가정한다.
- Learned position embedding으로 각 절대 위치의 identity를 제공한다.

하지만 absolute position embedding은 locality를 강제하지 않고, 상대 거리나 방향을 구조적으로 보장하지 않는다. 입력을 옮겼을 때 다른 absolute vector가 더해지므로 전체 ViT는 정확한 translation equivariance도 갖지 않는다. 논문이 직접 2D 구조를 수동으로 사용하는 지점은 주로 초기 patch extraction과 고해상도 fine-tuning 때의 2D position interpolation이다.

### 데이터 규모와 bias의 관계

강한 bias는 가능한 해의 범위를 좁혀 작은 데이터에서 sample efficiency를 높인다. 반대로 충분히 큰 데이터가 있다면 모델이 필요한 locality와 2D relation을 직접 학습할 수 있고, 덜 제한된 global attention이 유리할 수 있다.

이 논문의 결과는 이 해석과 일치하지만 보편적인 정리는 아니다. 정확한 결론은 다음에 가깝다.

> 이 논문의 model, regularization, compute 범위에서는 작은 데이터에서 ResNet의 convolutional bias가 유리했고, 데이터가 커질수록 ViT의 상대 성능과 큰 model의 활용도가 높아졌다.

## Position embedding ablation

ViT-B/16의 ImageNet 5-shot linear evaluation 결과는 다음과 같다.

| Position encoding | 5-shot accuracy (%) |
| --- | ---: |
| 없음 | 61.382 |
| Learned 1D, input에 1회 추가 | **64.206** |
| Learned 2D | 64.001 |
| Relative position | 64.032 |

위치 정보가 전혀 없을 때보다 약 2.6-2.8%p 좋아지므로 patch의 공간 위치와 순서 정보는 중요하다. 반면 1D, 2D, relative 방식 사이의 차이는 매우 작다. 저자들은 14×14처럼 pixel grid보다 훨씬 작은 patch grid에서는 여러 방식으로 공간 관계를 학습하기가 비슷하게 쉽다고 추측한다.

따라서 "1D position만으로 충분하다"는 결과는 모든 해상도와 dense task에 대한 보편적 결론이 아니라, 이 논문의 patch-level classification 설정에서 얻은 ablation 결과다.

## Higher-resolution fine-tuning

ViT는 모든 layer의 hidden dimension이 해상도와 무관하므로 더 긴 sequence를 처리할 수 있다. 하지만 patch size를 유지한 채 해상도를 올리면 token 수와 position embedding 길이가 달라진다.

```text
pretrain 224×224, P=16: 14×14 = 196 patches
fine-tune 384×384, P=16: 24×24 = 576 patches
```

Pretrained position table에서 class-token 부분은 분리해 유지하고, patch 부분을 원래 2D grid로 reshape한 뒤 새 grid 크기에 맞춰 2D interpolation한다. 이 과정은 sequence 길이를 맞추는 실용적인 해법이지만 새로운 resolution이나 aspect ratio에서 기하학적 equivariance를 보장하는 것은 아니다.

224에서 384로 바꾸면 CLS 포함 sequence 길이는 `197 -> 577`이 되고, 한 head의 attention matrix 원소 수는 약 8.6배가 된다. Higher-resolution fine-tuning의 정확도 이득과 memory 증가를 함께 봐야 한다.

일반 fine-tuning 실험은 주로 384 resolution을 사용했다. State-of-the-art 비교용 ImageNet 결과에서는 ViT-L/16을 512, ViT-H/14를 518 resolution으로 fine-tune하고 Polyak averaging도 사용했다.

## Hybrid model

Raw image patch 대신 CNN feature map을 Transformer input으로 사용할 수 있다. Feature map의 `1×1` spatial location을 한 token처럼 flatten하고 `D`차원으로 projection하는 것이 대표적인 special case다.

```text
image
 -> ResNet feature extractor
 -> feature map H'×W'×C'
 -> H'W' tokens + linear projection
 -> class/position embeddings
 -> Transformer encoder
```

논문의 hybrid는 modified ResNet50의 stage 4 output을 쓰거나, stage 4의 layer를 stage 3으로 옮겨 4배 긴 sequence를 만들었다. Hybrid 이름의 `/16`, `/32`는 이 경우 patch size가 아니라 ResNet의 total downsampling ratio를 뜻한다.

Controlled scaling 결과에서 hybrid는 작은 model과 compute budget에서 pure ViT보다 약간 좋았지만, 규모가 커지면 차이가 사라졌다. 이는 CNN front-end의 local prior가 저비용 영역에서 도움이 된다는 결과이지, 모든 작은 데이터 조건에서 hybrid가 항상 우월하다는 증명은 아니다.

## 실험 설계와 평가 protocol

### Pretraining dataset

- ImageNet-1k: 1.3M images, 1K classes
- ImageNet-21k: 약 14M images, 21K classes
- JFT-300M: 약 303M images, 18K noisy multi-label classes

Pretraining dataset은 downstream test set과 중복되는 image를 제거했다. Transfer 대상은 ImageNet/ReaL, CIFAR-10/100, Oxford-IIIT Pets, Oxford Flowers-102와 VTAB이다.

VTAB은 task당 1,000개의 training sample을 사용하는 19개 transfer task 모음이다.

- Natural: 일반 자연 image
- Specialized: medical, satellite 등 전문 domain
- Structured: localization과 geometric reasoning이 중요한 task

논문은 full fine-tuning accuracy와 frozen representation 위의 linear few-shot accuracy를 구분한다. Few-shot metric은 제한된 label로 regularized least-squares classifier를 맞추므로, full network를 재학습하는 fine-tuning과 동일한 지표가 아니다.

## Training과 fine-tuning recipe

모든 architecture를 가능한 한 같은 조건에서 비교하기 위해 ResNet도 Adam으로 사전학습했다.

| 단계 | 주요 설정 |
| --- | --- |
| Pretraining | Adam `β1=0.9`, `β2=0.999`, batch 4096, 10K-step warmup |
| JFT | weight decay 0.1, dropout 0.0, 7 또는 14 epochs |
| ImageNet-21k | weight decay 0.03, dropout 0.1, 30 또는 90 epochs |
| ImageNet-1k | weight decay 0.3, dropout 0.1, 300 epochs, gradient clipping 1 |
| Fine-tuning | SGD momentum 0.9, batch 512, cosine decay, no weight decay, gradient clipping 1 |

모든 사전학습 resolution은 224였다. 강한 regularization은 ImageNet에서 scratch training할 때 특히 중요했다. 따라서 architecture만 보고 결과를 해석하면 안 되며, optimizer와 regularization을 포함한 recipe도 비교의 일부다.

## 데이터 규모에 따른 결과

논문은 두 가지 방식으로 data requirement를 확인한다.

### Dataset을 ImageNet에서 JFT까지 확대

ImageNet 사전학습에서는 ViT-L이 ViT-B보다 오히려 낮았다. ImageNet-21k에서는 두 크기의 성능이 비슷해지고, JFT-300M에서야 Large/Huge model의 추가 capacity가 분명한 이득을 냈다.

```text
작은 데이터 + 큰 ViT : model capacity를 활용하기 전에 overfitting
큰 데이터 + 큰 ViT   : 추가 capacity와 global interaction이 transfer 성능으로 연결
```

이 실험에서는 작은 dataset별로 weight decay, dropout, label smoothing을 조정했으므로, data 크기뿐 아니라 regularization 효과도 포함된다.

### 같은 hyperparameter로 JFT subset 비교

보다 통제된 실험에서는 JFT의 9M, 30M, 90M, 300M subset에 같은 hyperparameter를 적용했다. ViT-B/32는 9M에서 비슷한 compute의 ResNet50보다 크게 낮았지만 90M 이상에서는 앞섰다. ViT-L/16과 ResNet152x2도 유사한 교차를 보였다.

이 결과가 논문의 "large-scale training trumps inductive bias" 주장을 가장 직접적으로 뒷받침한다. 다만 교차점인 90M은 이 실험 설정에 해당하며 다른 architecture나 augmentation에서는 달라질 수 있다.

## State-of-the-art transfer 결과

아래 값은 Table 2의 fine-tuning accuracy이며 세 번의 run 평균이다.

| 모델 | Pretrain | ImageNet | ReaL | C10 | C100 | Pets | Flowers | VTAB | TPUv3 core-days |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-H/14 | JFT | **88.55** | **90.72** | **99.50** | **94.55** | **97.56** | 99.68 | **77.63** | 2.5K |
| ViT-L/16 | JFT | 87.76 | 90.54 | 99.42 | 93.90 | 97.32 | **99.74** | 76.28 | 0.68K |
| ViT-L/16 | ImageNet-21k | 85.30 | 88.62 | 99.15 | 93.25 | 94.67 | 99.61 | 72.72 | **0.23K** |
| BiT-L R152×4 | JFT | 87.54 | 90.54 | 99.37 | 93.51 | 96.62 | 99.63 | 76.29 | 9.9K |
| Noisy Student | ImageNet + unlabeled JFT | 88.4/88.5 | 90.55 | - | - | - | - | - | 12.3K |

ViT-L/16은 BiT-L보다 ImageNet, CIFAR, Pets, Flowers에서 높고 ImageNet-ReaL은 동률이며 VTAB은 `0.01%p` 낮다. 따라서 "모든 listed task에서 높다"가 아니라 **대부분의 task에서 비슷하거나 우수하다**가 정확하다. Pretraining core-days는 BiT-L의 약 `6.9%`만 사용했다(`0.68K / 9.9K ≈ 1/14.6`).

TPUv3 core-days는 사용 core 수와 일수를 곱한 hardware-dependent proxy다. FLOPs나 실제 비용, 다른 accelerator에서의 wall-clock time과 완전히 같지는 않다.

## Controlled compute scaling

JFT-300M에서 여러 크기의 ResNet, ViT, hybrid를 같은 축의 pretraining compute로 비교했을 때 다음 경향이 나타났다.

- ViT는 5개 transfer dataset 평균에서 같은 성능에 도달하는 데 ResNet보다 약 `2-4×` 적은 compute를 사용했다.
- Hybrid는 작은 compute budget에서 pure ViT보다 약간 높았다.
- Model이 커지면 hybrid와 pure ViT의 차이가 사라졌다.
- 실험한 범위에서 ViT scaling curve는 뚜렷하게 포화되지 않았다.

Table 2의 최고 결과와 controlled scaling table은 epoch와 evaluation recipe가 다르므로 숫자를 직접 섞어 비교하면 안 된다. 특히 Table 2의 ImageNet 열은 ViT-L/16과 ViT-H/14에 각각 512/518 resolution과 Polyak averaging을 사용했다. 또한 compute efficiency는 architecture뿐 아니라 dense-kernel 구현과 TPU utilization에도 영향을 받는다.

## Learned representation 분석

### Patch embedding filters

첫 projection weight의 principal component를 image로 복원하면 edge, oriented line, color blob 같은 low-level basis가 나타난다. Convolution으로 강제하지 않아도 large-scale data에서 CNN 초기 filter와 비슷한 local representation이 학습된 것이다.

### Position embedding

Learned position embedding의 cosine similarity를 보면 가까운 patch, 같은 row, 같은 column의 vector가 비슷해진다. 초기 1D table에는 2D 정보가 없지만 학습 과정에서 raster sequence와 image statistics로부터 2D topology가 나타난다.

### Attention distance

논문은 각 query와 key 사이의 pixel distance를 attention weight로 가중 평균해 attention distance를 정의한다.

```math
d_i=\sum_j A_{ij}\lVert p_i-p_j\rVert
```

초기 layer에서도 일부 head는 image 전체를 보고, 다른 head는 매우 local한 범위를 본다. Head별 차이가 크며 depth가 깊어질수록 평균 distance가 대체로 증가한다. ResNet front-end가 있는 hybrid에서는 초기 local head가 덜 두드러졌는데, CNN이 이미 local feature extraction을 담당했기 때문으로 해석할 수 있다.

논문의 Attention Rollout visualization은 head 평균과 layer별 attention matrix의 재귀적 결합을 통해 output token에서 input patch로 이어지는 attention을 보여준다. Object와 관련된 영역에 집중하는 사례가 나타나지만, 이를 곧바로 완전한 causal explanation으로 해석해서는 안 된다.

## Preliminary self-supervision

저자들은 BERT와 유사한 masked patch prediction도 예비적으로 실험했다.

- Patch embedding의 50%를 corruption한다.
- 그중 80%는 `[mask]`, 10%는 random patch, 10%는 그대로 둔다.
- Target은 corrupted patch의 3-bit mean color, 즉 512-way classification이다.
- ViT-B/16을 JFT에서 1M steps, 약 14 epochs 학습한다.

ImageNet accuracy는 `79.9%`로 scratch training보다 약 `2%p` 높았지만 supervised pretraining보다 약 `4%p` 낮았다. 논문 당시의 단순 objective가 가능성을 보인 수준이며, 이후 masked image modeling 연구가 해결할 여지를 남겼다.

## 장점과 기여

- Image를 patch token으로 바꾸는 최소 수정만으로 표준 Transformer를 vision에 적용했다.
- CNN 없이도 large-scale supervised pretraining에서 강한 transfer 성능을 달성했다.
- Model size와 dataset size의 상호작용을 통제 실험으로 보여줬다.
- Architecture별 accuracy뿐 아니라 pretraining compute와 transfer efficiency를 비교했다.
- Position embedding, attention distance, patch filter를 분석해 ViT가 2D structure를 학습하는 과정을 관찰했다.
- 이후 vision backbone 연구의 중심을 convolution-only model에서 Transformer와 hybrid 설계로 확장했다.

## 한계와 비판적 관점

### 1. Large-scale data 의존

ImageNet 규모에서는 strong CNN보다 sample efficiency가 낮다. JFT-300M은 비공개 대규모 dataset이므로 최고 결과의 재현성과 접근성에도 한계가 있다. 이후의 training recipe가 이 격차를 줄였다는 점을 고려하면, 원 논문의 data hunger를 architecture 하나의 고정 속성으로만 보면 안 된다.

### 2. Quadratic token attention

고해상도에서 token 수가 증가하면 attention map이 `S²`으로 커진다. 원형 ViT에는 window, sparse attention, hierarchical downsampling이 없어 dense prediction과 고해상도 입력의 memory 부담이 크다.

### 3. Patch boundary와 fine detail

큰 non-overlapping patch는 내부 spatial detail을 한 vector로 투영하고 sub-patch translation에 민감할 수 있다. Small object나 pixel-level boundary에 불리할 가능성이 있다. 이는 원 논문의 classification 결과로 직접 입증한 결론이라기보다 이후 dense prediction 관점에서 드러난 구조적 한계다.

### 4. Absolute position과 resolution 변화

새 grid에서는 position interpolation이 필요하다. 이는 shape mismatch를 해결하지만 training grid 밖의 resolution, aspect ratio, translation에 대한 정확한 geometric generalization을 보장하지 않는다.

### 5. Classification 중심 평가

원 논문은 주로 image-level transfer를 평가했다. Detection과 segmentation에는 multi-scale feature와 local detail을 얻기 위한 별도 설계가 흔히 필요하다.

### 6. Architecture와 recipe의 분리 어려움

최고 성능 비교에는 다른 epoch, fine-tuning resolution, Polyak averaging이 포함된다. 논문의 controlled compute study가 이를 보완하지만, "Transformer가 본질적으로 몇 배 효율적이다"라는 보편적 결론으로 확대하면 안 된다.

## 후속 연구와의 연결

- **DeiT**: 강한 augmentation, regularization, distillation로 ImageNet-only training의 data efficiency를 개선했다.
- **Swin/PVT 계열**: window attention과 hierarchy를 도입해 고해상도 dense task의 비용을 줄였다.
- **MAE 계열**: masked image modeling을 효과적인 self-supervised pretraining으로 발전시켰다.
- **Hybrid/local-global backbone**: convolution 또는 local attention과 global attention을 stage별로 조합했다.

따라서 ViT의 지속적인 공헌은 모든 convolution을 없앤 최종 구조라기보다, image를 token sequence로 보고 Transformer scaling law를 적용할 수 있다는 설계 공간을 연 데 있다.

## 구현 체크리스트

- `N=HW/P²`이고 `H,W`가 patch size로 정확히 나누어지는가?
- CLS를 포함한 sequence 길이가 `N+1`인지 확인했는가?
- Patch projection을 convolution으로 구현할 때 `kernel_size=stride=P`가 맞는가?
- Class token과 patch token 모두에 대응하는 position embedding이 더해지는가?
- Encoder가 Pre-LN 순서이며 최종 class token에도 LayerNorm을 적용하는가?
- Head 수가 바뀔 때 `D_h=D/h`와 projection shape가 일치하는가?
- Fine-tuning 때 pretrained MLP head를 제거하고 새 linear head를 붙이는가?
- Resolution 변경 시 class token을 제외한 patch position만 2D interpolation하는가?
- Attention memory를 `N`이 아니라 CLS를 포함한 `S=N+1`로 계산하는가?
- Patch size와 resolution별 total FLOPs, peak activation, latency를 실제 profiler로 확인했는가?
- Pixel shift와 aspect-ratio 변화에 대한 robustness를 별도로 검증했는가?

## 온디바이스 관점

ViT는 parameter 수보다 token 수가 activation memory와 latency를 크게 좌우한다. 큰 patch는 `S²` attention 비용을 줄이지만 small object와 fine detail을 잃을 수 있다. 작은 patch는 parameter를 거의 늘리지 않으면서 compute와 memory를 급격히 키운다.

Dense GEMM은 NPU에 잘 맞을 수 있지만 LayerNorm, softmax, transpose, dynamic shape가 병목이 될 수 있다. 실제 deployment에서는 다음을 함께 봐야 한다.

- Fixed input resolution과 compile-time position interpolation
- Hybrid convolution stem 또는 hierarchical token reduction
- Window/sparse attention과 token pruning
- Weight뿐 아니라 activation과 softmax의 quantization sensitivity
- Average latency 외에 peak memory와 thermal throttling

따라서 mobile ViT 선택은 parameter 수만 비교해서는 안 되며, target resolution에서의 sequence length와 attention kernel 지원을 기준으로 해야 한다.

## 최종 평가

ViT는 image classification에 필요하다고 여겨졌던 vision-specific architecture를 극단적으로 줄이고, patch token과 표준 Transformer만으로 large-scale representation learning이 가능함을 보여줬다. 가장 중요한 교훈은 "Transformer가 CNN보다 항상 우월하다"가 아니라 **inductive bias, dataset scale, model capacity, training compute가 함께 최적점을 결정한다**는 것이다.

작은 데이터에서는 convolutional prior가 학습을 돕고, 큰 데이터에서는 ViT가 필요한 공간 규칙을 직접 학습하면서 global interaction과 scale을 활용할 수 있었다. 동시에 quadratic high-resolution cost, absolute position interpolation, 대규모 데이터 의존이라는 한계를 남겼다. 이 장점과 한계가 이후 DeiT, hierarchical vision Transformer, masked image modeling 연구의 출발점이 됐다.

# 53. SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers

## 논문 정보

- 원본 파일: `53_SegFormer.pdf`
- 제목: **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers**
- 저자: Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo
- 발표: NeurIPS 2021
- 공개: arXiv:2105.15203, 제공된 PDF는 v3(2021)
- 링크: [https://arxiv.org/abs/2105.15203](https://arxiv.org/abs/2105.15203)
- 핵심 키워드: semantic segmentation, Mix Transformer, hierarchical encoder, efficient self-attention, overlapped patch embedding, Mix-FFN, All-MLP decoder

## 한눈에 보는 요약

SegFormer는 semantic segmentation용 Transformer를 복잡한 decoder 없이 효율적으로 구성한 모델이다. 핵심은 두 부분이다.

1. `1/4, 1/8, 1/16, 1/32` 해상도의 feature를 만드는 hierarchical Mix Transformer(MiT) encoder
2. 네 feature를 같은 channel로 projection하고 `1/4` 해상도에서 concat하는 가벼운 All-MLP decoder

```text
image
 -> overlap patch embedding, stride 4
 -> MiT stage 1: H/4  x W/4
 -> MiT stage 2: H/8  x W/8
 -> MiT stage 3: H/16 x W/16
 -> MiT stage 4: H/32 x W/32

F1,F2,F3,F4
 -> per-level linear projection
 -> all upsample to H/4 x W/4
 -> concatenate
 -> linear fusion
 -> per-pixel class logits
 -> upsample to input resolution
```

MiT는 absolute positional embedding을 쓰지 않는다. 대신 FFN 안의 `3×3 depthwise convolution`과 zero padding이 위치에 민감한 local interaction을 제공한다. 따라서 학습과 다른 test resolution에서도 positional table interpolation이 필요 없다.

High-resolution self-attention 비용은 key/value token을 spatial reduction해 줄인다. Stage 1부터 4까지 구현의 spatial reduction ratio는 `[8,4,2,1]`이며, key token 수는 각각 `[64,16,4,1]`배로 줄어든다. Query는 원 해상도를 유지하므로 dense output의 spatial detail은 보존한다.

ADE20K single-scale에서 SegFormer-B0는 `3.8M params`, `8.4G FLOPs`, `37.4 mIoU`, B5는 `84.7M params`, `183.3G FLOPs`, `51.0 mIoU`를 기록했다. B5 multi-scale은 `51.8 mIoU`, Cityscapes validation은 `84.0 mIoU`다. B0는 Cityscapes short side 512에서 `47.6 FPS`, `71.9 mIoU`였지만, 논문은 GPU 종류별 peak memory나 모바일 p50/p95 latency를 제공하지 않는다.

## Semantic segmentation의 요구 사항

Image classification은 image 전체 label 하나를 예측하지만 semantic segmentation은 모든 pixel에 class를 할당한다. 좋은 encoder는 동시에 두 종류의 정보를 보존해야 한다.

- Boundary와 작은 객체를 위한 high-resolution local detail
- 넓은 scene context와 class 관계를 위한 global receptive field

ViT는 image를 큰 non-overlapping patch로 나눠 single low-resolution token map을 만든다. Classification에는 적합하지만 segmentation에서는 다음 문제가 있다.

1. `16×16` patch로 작은 경계 detail이 일찍 사라진다.
2. Encoder가 single scale만 출력해 decoder가 해상도를 복원해야 한다.
3. Full self-attention이 high-resolution에서 quadratic하게 증가한다.
4. Learned positional embedding은 test resolution 변화 시 interpolation이 필요하다.

CNN segmentation model은 FPN, ASPP, dilated convolution, context module 등으로 이를 보완했지만 decoder가 무거워질 수 있다. SegFormer는 encoder가 local-to-global multi-scale feature를 직접 만들면 decoder는 단순한 MLP fusion으로 충분하다는 가설을 검증한다.

## 전체 구조

입력 image를 `x in R^{B×3×H×W}`라 하자. Encoder는 네 feature를 출력한다.

```math
F_i\in\mathbb{R}^{B\times C_i\times H/2^{i+1}\times W/2^{i+1}},
\qquad i\in\{1,2,3,4\}.
```

MiT-B0의 channel은 `[32,64,160,256]`, B1-B5는 `[64,128,320,512]`다. Stage가 깊어질수록 spatial resolution은 절반, channel은 증가한다.

모든 stage는 다음 순서다.

```text
overlapped patch embedding / merging
 -> LayerNorm
 -> repeated Transformer blocks:
      efficient self-attention
      residual
      Mix-FFN with depthwise 3x3 conv
      residual
```

## Overlapped patch embedding

### 첫 stage

ViT의 non-overlapping `16×16` patch 대신 kernel 7, stride 4, padding 3의 convolution과 동등한 projection을 사용한다.

```text
input B x 3 x H x W
 -> Conv2d(kernel=7, stride=4, padding=3, out=C1)
 -> B x C1 x H/4 x W/4
```

Kernel이 stride보다 크므로 이웃 token의 receptive field가 겹친다. Patch 경계 양쪽의 local continuity를 보존한다.

### Stage 2-4

각 stage transition은 kernel 3, stride 2, padding 1이다.

```text
Fi -> Conv2d(3x3, stride 2, padding 1) -> F(i+1)
```

Non-overlapping `2×2` merge보다 주변 context를 함께 본다. 이는 learnable downsampling이면서 CNN의 local inductive bias를 Transformer에 넣는 방식이다.

### Arbitrary resolution

Convolution projection은 input 크기에 독립적이며 fixed token table이 없다. 다만 arbitrary resolution이라는 말이 모든 크기에서 accuracy가 같다는 뜻은 아니다. Padding, aspect ratio, training crop 분포와 decoder upsampling이 여전히 영향을 준다.

## Efficient self-attention

Standard attention은:

```math
\operatorname{Attention}(Q,K,V)
=\operatorname{Softmax}\left(\frac{QK^\top}{\sqrt{d_{head}}}\right)V.
```

Query와 key가 모두 `N=HW` token이면 attention score가 `N×N`이고 complexity는 `O(N^2)`다.

SegFormer는 query length는 유지하고 key/value의 spatial resolution만 줄인다. 구현 관점에서 spatial reduction ratio를 `r`라 하면:

```text
K,V feature: H x W
 -> spatial reduction r x r
 -> H/r x W/r
 -> key length N/r^2
```

따라서 attention matrix shape는:

```math
[B,h,N,N/r^2]
```

이고 score complexity는 약 `O(N^2/r^2)`다.

### 논문 표기 주의

본문 식은 sequence length reduction factor `R`을 사용해 `N/R`로 쓰고 stage별 `[64,16,4,1]`을 언급한다. Appendix Table 6과 공개 구현의 `sr_ratio`는 spatial ratio `[8,4,2,1]`이다.

```text
spatial sr_ratio:       8, 4, 2, 1
token reduction factor: 64,16,4,1
```

둘은 모순된 architecture가 아니라 ratio 정의가 다른 것이다. 구현에서 `sr_ratio=64`를 넣으면 key를 지나치게 줄이므로 주의해야 한다.

### Stage별 key 수

512×512 input에서 token 수는 다음과 같다.

| Stage | Query map | `N_q` | spatial ratio | `N_k` |
| --- | ---: | ---: | ---: | ---: |
| 1 | 128×128 | 16,384 | 8 | 256 |
| 2 | 64×64 | 4,096 | 4 | 256 |
| 3 | 32×32 | 1,024 | 2 | 256 |
| 4 | 16×16 | 256 | 1 | 256 |

흥미롭게도 이 예시에서는 모든 stage의 key length가 256이다. Early stage는 많은 query가 축약된 global summary를 보고, final stage는 full attention을 수행한다.

## Mix-FFN과 위치 정보

Standard Transformer FFN은 각 token을 독립적으로 처리한다. SegFormer는 두 linear layer 사이에 `3×3 depthwise convolution`을 넣는다.

```math
x_{out}=\operatorname{MLP}_2
\left(
\operatorname{GELU}
\left(
\operatorname{DWConv}_{3\times3}
(\operatorname{MLP}_1(x_{in}))
\right)
\right)+x_{in}.
```

Depthwise conv는 channel별로 인접 token을 섞어 local continuity를 준다. Zero padding 때문에 border와 interior가 다르게 처리되어 absolute location에 대한 단서를 제공한다. 저자들은 이를 이용해 explicit positional encoding을 제거한다.

### 무엇을 주장할 수 있고 없는가

- 주장 가능: fixed positional embedding 없이도 segmentation accuracy가 높고 resolution 변화에 강했다.
- 과도한 주장: depthwise conv가 수학적으로 완전한 absolute coordinate를 유일하게 encoding한다.

Padding에서 유출되는 위치 신호는 translation equivariance를 약화할 수 있으며, 극단적으로 큰 resolution이나 다른 padding 정책의 extrapolation을 보장하지 않는다.

## MiT model family

| Model | channels | block depths | heads | decoder C |
| --- | --- | --- | --- | ---: |
| B0 | `[32,64,160,256]` | `[2,2,2,2]` | `[1,2,5,8]` | 256 |
| B1 | `[64,128,320,512]` | `[2,2,2,2]` | `[1,2,5,8]` | 256 |
| B2 | 동일 | `[3,3,6,3]` | 동일 | 768 |
| B3 | 동일 | `[3,3,18,3]` | 동일 | 768 |
| B4 | 동일 | `[3,8,27,3]` | 동일 | 768 |
| B5 | 동일 | `[3,6,40,3]` | 동일 | 768 |

Stage 3에 대부분의 block을 배치한다. Stage 3은 충분한 spatial detail과 높은 semantic capacity의 균형점이라 깊이를 늘려도 final full-resolution attention만큼 비싸지 않다.

Expansion ratio는 B0 stage 1-2가 8, 이후 대부분 4다. B1-B4 stage 1-2도 8, B5는 모두 4다. 정확한 variant를 재현할 때 channel/depth만 맞추고 FFN ratio를 놓치면 parameter와 FLOPs가 달라진다.

## All-MLP decoder

네 encoder feature를 같은 channel `C`로 projection한다.

```math
\hat F_i=\operatorname{Linear}_{C_i\rightarrow C}(F_i).
```

모두 `H/4×W/4`로 bilinear upsample하고 channel axis로 concat한다.

```math
\begin{aligned}
\tilde F_i&=\operatorname{Upsample}_{H/4,W/4}(\hat F_i),\\
F&=\operatorname{Linear}_{4C\rightarrow C}
(\operatorname{Concat}(\tilde F_1,\ldots,\tilde F_4)),\\
M&=\operatorname{Linear}_{C\rightarrow N_{cls}}(F).
\end{aligned}
```

여기서 MLP는 spatial token마다 같은 linear layer를 적용하므로 implementation에서는 `1×1 convolution`과 동등하다. Decoder 자체에는 attention, ASPP, multi-layer 3×3 conv가 없다.

### Local과 global의 결합

저자들의 effective receptive field 분석에서 early MiT stage는 local pattern, stage 4는 non-local context를 보였다. Decoder가 모든 stage를 합치면 boundary detail과 global context를 동시에 쓸 수 있다.

MiT-B2에서 stage 4만 decoder에 쓰면 `43.1 mIoU`, stage 1-4를 모두 쓰면 `45.4 mIoU`였다. Global feature 하나만으로 충분하지 않다는 직접 ablation이다.

## Batch=1 tensor shape와 activation 예시

### 가정

`B=1`, `512×512` image, SegFormer-B0, ADE20K 150 classes, decoder `C=256`, FP16을 가정한다. 아래는 **리뷰어 계산**이며 논문은 peak-memory table을 제공하지 않는다.

### Encoder output

| Tensor | shape | elements | FP16 payload |
| --- | ---: | ---: | ---: |
| F1 | `1×32×128×128` | 524,288 | 1.00 MiB |
| F2 | `1×64×64×64` | 262,144 | 0.50 MiB |
| F3 | `1×160×32×32` | 163,840 | 0.31 MiB |
| F4 | `1×256×16×16` | 65,536 | 0.13 MiB |
| 합계 | - | 1,015,808 | 1.94 MiB |

Final stage output은 작지만 early attention의 score tensor가 더 클 수 있다.

### Attention score

Head 수까지 포함한 layer당 score element는:

| Stage | score shape | elements | FP16 |
| --- | ---: | ---: | ---: |
| 1 | `1×1×16384×256` | 4,194,304 | 8.00 MiB |
| 2 | `1×2×4096×256` | 2,097,152 | 4.00 MiB |
| 3 | `1×5×1024×256` | 1,310,720 | 2.50 MiB |
| 4 | `1×8×256×256` | 524,288 | 1.00 MiB |

Stage 1은 channel이 작아도 query 수가 많아 attention memory가 가장 크다. Softmax kernel이 score를 완전히 materialize하는지, fused attention을 쓰는지에 따라 실제 peak가 달라진다.

### Decoder activation

각 projected feature를 128×128×256으로 upsample하면 하나당 4,194,304 element, FP16 8 MiB다. 네 개를 모두 보존하면 32 MiB이고 concat tensor `1×1024×128×128`도 32 MiB다.

Fusion 전후 buffer가 동시에 live하면 decoder만 60 MiB 이상이 될 수 있다. ADE20K quarter-resolution logits는:

```math
128\cdot128\cdot150=2{,}457{,}600
```

FP16 약 4.69 MiB다. 이를 full 512×512×150으로 먼저 upsample하면 약 75 MiB이므로, argmax 전에 full class logits를 materialize하는 방식은 모바일 peak memory에 불리하다.

## Decoder parameter와 MAC 계산

### B0, ADE20K reviewer calculation

Bias를 제외한 channel projection:

```math
(32+64+160+256)\cdot256=131{,}072.
```

Fusion과 classifier:

```math
1024\cdot256+256\cdot150=300{,}544.
```

합계 `431,616 weights`, 논문의 B0 decoder `0.4M`과 일치한다.

512×512에서 ideal MAC은 대략:

- 네 native-resolution projection: 0.26G
- quarter-resolution `1024→256` fusion: 4.29G
- `256→150` classifier: 0.63G
- 합계: 약 5.18G MACs

Upsampling cost는 제외했다. Decoder parameter는 작지만 high-resolution fusion MAC과 activation은 무시할 수 없다. 논문의 B0 total `8.4G FLOPs`와 count convention이 다를 수 있으므로 reviewer MAC을 paper FLOPs와 직접 더하면 안 된다.

## 학습 objective

SegFormer는 특별한 auxiliary loss, OHEM, class-balanced loss 없이 per-pixel semantic classification loss로 학습한다. 일반적으로 upsampled logit과 label 사이의 cross-entropy다.

```math
L_{seg}=-\frac{1}{|\Omega|}\sum_{u\in\Omega}
\log p_{u,y_u}.
```

Ignore label은 denominator와 loss에서 제외해야 한다. 논문의 단순 objective는 architecture 비교를 깨끗하게 만들지만, class imbalance가 큰 custom dataset에서는 rare-class loss가 필요할 수 있다.

## 구현 pseudocode

```python
class EfficientAttention(nn.Module):
    def forward(self, x, H, W, sr_ratio):
        q = self.q(x)                                  # [B, N, C]
        if sr_ratio > 1:
            feat = x.transpose(1, 2).reshape(B, C, H, W)
            reduced = self.sr_conv(feat)               # kernel=stride=sr_ratio
            reduced = reduced.flatten(2).transpose(1, 2)
            reduced = self.norm(reduced)
        else:
            reduced = x
        k, v = split(self.kv(reduced))
        attn = softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        return self.proj(attn @ v)


class MixFFN(nn.Module):
    def forward(self, x, H, W):
        z = self.fc1(x)
        z = z.transpose(1, 2).reshape(B, hidden, H, W)
        z = self.depthwise_conv3x3(z)
        z = gelu(z).flatten(2).transpose(1, 2)
        return self.fc2(z)


def segformer(image):
    features = mit_encoder(image)                       # F1...F4
    projected = []
    for feat, linear in zip(features, level_projections):
        z = linear(flatten_spatial(feat))
        z = restore_spatial(z)
        projected.append(bilinear_resize(z, size=features[0].spatial))
    fused = fusion_1x1(concat(projected, dim=channel))
    logits_quarter = classifier_1x1(fused)
    return bilinear_resize(logits_quarter, image.spatial)
```

Deployment에서는 full-resolution logits를 만들지 않고 quarter-resolution argmax 또는 tiled upsample-argmax를 지원하면 peak memory를 크게 줄일 수 있다.

## 실험 설정

- Encoder ImageNet-1K pretraining, decoder random initialization
- Framework: MMSegmentation
- 8 Tesla V100
- AdamW, initial LR `6×10^-5`, poly schedule
- ADE20K/Cityscapes 160k iterations, COCO-Stuff 80k
- Ablation 40k iterations
- Batch 16 for ADE20K/COCO-Stuff, 8 for Cityscapes
- Random resize 0.5-2.0, horizontal flip, random crop
- Crop: ADE20K 512, B5 640; Cityscapes 1024; COCO-Stuff 512
- Cityscapes는 1024×1024 sliding-window inference
- Metric: mean Intersection over Union

FPS와 FLOPs는 input/crop scale에 따라 크게 변하므로 table의 ADE20K와 Cityscapes 수치를 섞어 비교하면 안 된다.

## Model scaling 결과

### ADE20K

| Model | encoder M | decoder M | FLOPs | mIoU SS | mIoU MS |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 3.4 | 0.4 | 8.4G | 37.4 | 38.0 |
| B1 | 13.1 | 0.6 | 15.9G | 42.2 | 43.1 |
| B2 | 24.2 | 3.3 | 62.4G | 46.5 | 47.5 |
| B3 | 44.0 | 3.3 | 79.0G | 49.4 | 50.0 |
| B4 | 60.8 | 3.3 | 95.7G | 50.3 | 51.1 |
| B5 | 81.4 | 3.3 | 183.3G | 51.0 | 51.8 |

Model이 커질수록 mIoU는 꾸준히 오르지만 B4→B5 single-scale gain은 0.7, FLOPs는 거의 두 배다. 온디바이스에서는 B0-B2가 더 현실적인 operating point다.

### Cityscapes와 COCO-Stuff

- B0 Cityscapes validation: `76.2 SS / 78.1 MS`
- B5 Cityscapes validation: `82.4 SS / 84.0 MS`
- B5 Cityscapes test, ImageNet-1K only: `82.2`
- B5 + Mapillary pretraining Cityscapes test: `83.1`
- B5 COCO-Stuff: `46.7 mIoU`, `84.7M params`

Dataset split, extra pretraining과 single/multi-scale 여부를 함께 표기해야 한다.

## 핵심 ablation

### Decoder channel C

MiT-B2 ADE20K에서 C 256/512/768/1024/2048의 mIoU는 `44.9/45.0/45.4/45.2/45.6`이었다. C=768 이후 거의 포화하지만 FLOPs는 `62.4→304.4G`까지 증가한다. B2-B5에 768, real-time B0-B1에 256을 선택한 근거다.

### Positional embedding 대 Mix-FFN

Cityscapes:

| Encoder type | inference resolution | mIoU |
| --- | ---: | ---: |
| PE | 768×768 | 77.3 |
| PE | 1024×2048 | 74.0 |
| Mix-FFN | 768×768 | 80.5 |
| Mix-FFN | 1024×2048 | 79.8 |

Resolution 변화 시 PE model은 3.3 point, Mix-FFN은 0.7 point 차이다. 다만 두 model의 absolute accuracy도 달라 위치 방식만의 순수 extrapolation 효과를 완전히 분리한 것은 아니다.

### CNN encoder에 같은 decoder

All-MLP decoder와 ResNet50/101/ResNeXt101의 ADE20K mIoU는 `34.7/38.7/39.8`, MiT-B2 stage 1-4는 `45.4`였다. Decoder의 단순함은 Transformer encoder가 이미 넓은 context를 제공한다는 조건에 의존한다.

## 속도와 정확도

Paper-reported single-scale table:

- ADE20K B0: `3.8M params`, `8.4G FLOPs`, `50.5 FPS`, `37.4 mIoU`
- Cityscapes B0 short side 1024: `15.2 FPS`, `76.2 mIoU`
- Cityscapes B0 short side 512: `47.6 FPS`, `71.9 mIoU`
- ADE20K B4: `15.4 FPS`, `51.1 mIoU`로 표에는 multi-scale 결과가 함께 표시된 문맥에 주의
- ADE20K B5: `9.8 FPS`, `51.8 mIoU`

논문은 latency의 GPU model/측정 protocol 일부를 본문에서 충분히 세분화하지 않으며 p50/p95, peak memory, power, thermal throttling은 보고하지 않는다.

## Robustness 결과

Cityscapes-C에서 SegFormer-B5는 clean `82.4`였고, Gaussian noise `57.8`, impulse `63.4`, shot `52.3`, brightness `81.0`, contrast `77.7`, saturation `80.1`, snow `68.4`, fog `78.5`, frost `49.9` 등 기존 DeepLabv3+ 계열보다 높은 corruption mIoU를 보였다.

그러나 robustness 결과는 model capacity, pretraining과 clean accuracy도 함께 다르다. Mix-FFN 하나가 모든 robustness를 만든다고 인과적으로 단정할 수 없다.

## 작은 객체와 경계 분석

SegFormer가 single-scale ViT보다 dense detail에 유리한 이유는:

1. 첫 patch stride가 4로 작다.
2. Overlap projection이 patch boundary continuity를 보존한다.
3. Stage 1/2 feature를 decoder가 직접 사용한다.
4. Depthwise 3×3 conv가 local interaction을 계속 공급한다.
5. Deep stage attention이 scene-level context를 제공한다.

하지만 output은 quarter resolution에서 만들어져 bilinear upsample된다. Thin structure, 1-2 pixel boundary와 작은 객체는 여전히 smoothing될 수 있다. 논문은 ADE20K/Cityscapes class-size별 IoU나 boundary F-score를 주지 않는다.

## 장점과 기여

1. Hierarchical Transformer와 lightweight decoder의 균형을 명확히 제시했다.
2. High-resolution attention을 sequence reduction으로 실용화했다.
3. Absolute positional embedding 없이 resolution-flexible encoder를 만들었다.
4. Overlap patch와 depthwise conv로 local inductive bias를 넣었다.
5. B0-B5 scaling family로 accuracy-latency operating point를 제공했다.
6. 단순 architecture로 ADE20K, Cityscapes, COCO-Stuff에서 강한 결과를 냈다.

## 한계와 비판적 관점

### 1. Decoder parameter와 activation은 다르다

Decoder가 0.4M parameter여도 four-way upsample와 concat은 큰 activation을 만든다. "Lightweight"를 model size만으로 해석하면 모바일 peak를 놓친다.

### 2. Efficient attention도 early-stage score가 크다

512 crop B0 stage 1 score는 reviewer calculation FP16 8 MiB/layer다. Resolution이 두 배면 query 수와 reduced key 수가 함께 증가해 attention score가 16배 가까이 커질 수 있다.

### 3. Full-resolution output 비용

Class 수가 많을 때 H×W×C logits가 크다. Semantic mask만 필요하면 upsample 전 argmax/fused resize 전략이 필요하다.

### 4. Position-free라는 표현의 한계

Explicit table은 없지만 convolution padding과 hierarchy가 강한 position/locality bias를 준다. 위치 정보가 전혀 없는 모델은 아니다.

### 5. Edge memory 미검증

논문 결론도 100k memory 수준 chip에서 동작할지는 불명확하다고 인정한다. 실제 모바일 latency와 peak memory는 보고하지 않는다.

## 자주 헷갈리는 지점

### SegFormer와 MiT는 같은가

MiT는 encoder, SegFormer는 MiT와 All-MLP decoder를 합친 segmentation model이다.

### All-MLP decoder는 spatial mixing을 전혀 안 하는가

Decoder linear는 위치별 channel mixing만 하지만 encoder feature 자체가 attention과 depthwise conv로 spatial context를 포함한다. Upsampling과 multi-scale alignment도 spatial operation이다.

### Positional encoding이 완전히 없는가

Learned/sinusoidal positional vector는 없다. Overlap conv, zero padding, hierarchy가 implicit positional bias를 제공한다.

### Reduction ratio가 `[64,16,4,1]`인가 `[8,4,2,1]`인가

Sequence-length 감소 factor는 전자, spatial stride parameter는 후자다. 공개 구현에는 `[8,4,2,1]`을 넣는다.

### Decoder output이 full resolution인가

기본 logit은 `H/4×W/4×N_cls`이고 평가를 위해 input resolution으로 upsample한다.

## 온디바이스 관점

### 유리한 연산

- Overlap patch와 Mix-FFN은 Conv2d/depthwise Conv2d로 구현 가능
- Decoder linear는 1×1 conv
- RoI/NMS/dynamic query가 없는 static dense graph
- B0 3.8M model은 storage가 작음

### 위험한 연산

- `QK^T`, softmax, reshape/transpose
- Sequence reduction 전후 layout conversion
- Multi-level bilinear resize와 1024-channel concat
- Full-resolution many-class logits
- LayerNorm 지원 부족 시 CPU fallback

CNN FLOPs와 Transformer FLOPs가 같아도 transpose와 softmax memory traffic 때문에 latency가 다를 수 있다.

### 최적화 실험

- 512/384/256 input에서 mIoU, peak, p50/p95 비교
- Decoder C 256보다 더 작은 128/64 ablation
- F4→F1 upsample를 sequential fusion으로 바꿔 concat peak 절감
- FP16 attention, INT8 conv/linear mixed precision
- Full-resolution logits 대신 quarter argmax + label resize
- Stage 1 attention kernel의 fused softmax 여부 확인

Semantic segmentation은 detector처럼 event 때만 실행하는 것이 아니라 frame마다 돌 수 있으므로 thermal throttling과 sustained FPS를 반드시 측정해야 한다.

## 재현 체크리스트

- [ ] PDF v3/NeurIPS 2021 metadata와 arXiv link를 기록한다.
- [ ] Overlap patch kernel/stride/padding `(7,4,3)`, 이후 `(3,2,1)`을 맞춘다.
- [ ] Variant별 channel, depth, head, expansion ratio를 확인한다.
- [ ] Spatial sr_ratio `[8,4,2,1]`과 token reduction factor를 혼동하지 않는다.
- [ ] K/V reduction conv와 LayerNorm 위치를 확인한다.
- [ ] Mix-FFN의 depthwise 3×3와 zero padding을 구현한다.
- [ ] Absolute positional embedding을 추가하지 않는다.
- [ ] B0/B1 decoder C=256, B2-B5 C=768을 맞춘다.
- [ ] 네 feature를 quarter resolution에서 concat한다.
- [ ] Dataset별 crop, iterations, batch와 sliding-window test를 구분한다.
- [ ] Single/multi-scale mIoU를 혼동하지 않는다.
- [ ] FLOPs/FPS의 input resolution을 함께 기록한다.
- [ ] Encoder/decoder parameter와 activation memory를 분리한다.
- [ ] Target에서 batch=1 peak, p50/p95, sustained FPS, power/temperature를 측정한다.

## 로드맵에서의 연결

- **FPN**과 같은 multi-scale principle을 Transformer hierarchy와 MLP fusion으로 구현한다.
- **Swin Transformer**와 달리 window partition 없이 reduced global keys를 사용한다.
- **Mask2Former**는 semantic segmentation을 mask classification/set prediction으로 바꾸고 multi-scale pixel decoder와 masked attention을 사용한다.
- **SAM/EdgeSAM**은 prompt 기반 요청형 segmentation이며 상시 semantic segmenter인 SegFormer와 실행 조건이 다르다.

통합 시스템에서는 SegFormer-B0/B1을 camera frame마다 실행하는 semantic map 후보로 두고, SAM 계열은 box/point prompt가 있을 때만 실행하는 구성이 합리적이다. Shared encoder를 연구할 때는 SegFormer의 quarter/high-level feature와 SAM mask decoder가 요구하는 embedding resolution/channel을 맞추는 비용을 측정해야 한다.

## 최종 평가

SegFormer의 핵심은 Transformer를 segmentation에 사용했다는 사실보다, encoder와 decoder 사이의 계산 책임을 잘 배분한 데 있다. Encoder가 high-resolution local detail과 low-resolution global context를 모두 만들기 때문에 decoder는 단순한 channel projection과 fusion으로 강한 mask를 낸다.

논문의 parameter/FLOPs/accuracy 결과는 B0가 효율적인 semantic segmentation baseline임을 보여 준다. 그러나 온디바이스에서 가장 큰 위험은 parameter가 아니라 attention score, four-way upsample/concat, full-resolution logits의 activation이다. `512×512` B0 예시에서도 decoder concat이 FP16 32 MiB이므로, 실제 배포 연구는 peak live tensor와 operator fusion을 중심으로 진행해야 한다.

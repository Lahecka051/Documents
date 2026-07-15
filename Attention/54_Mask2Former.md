# 54. Masked-attention Mask Transformer for Universal Image Segmentation

## 논문 정보

- 원본 파일: `54_Mask2Former.pdf`
- 제목: **Masked-attention Mask Transformer for Universal Image Segmentation**
- 저자: Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, Rohit Girdhar
- 발표: CVPR 2022
- 공개: arXiv:2112.01527, 제공된 PDF는 v3(2022)
- 링크: [https://arxiv.org/abs/2112.01527](https://arxiv.org/abs/2112.01527)
- 핵심 키워드: Mask2Former, universal segmentation, mask classification, masked attention, set prediction, bipartite matching, point-sampled mask loss, panoptic segmentation

## 한눈에 보는 요약

Mask2Former는 semantic, instance, panoptic segmentation을 동일한 architecture와 mask-classification objective로 처리하는 universal segmentation model이다. 모든 pixel을 독립 classifying하는 대신 고정 수 query가 binary mask 하나와 category 하나를 예측한다.

```text
image
 -> backbone
 -> multi-scale pixel decoder
      1/32, 1/16, 1/8 decoder features
      1/4 per-pixel embedding
 -> 100 learnable mask queries
 -> 9-layer Transformer decoder
      masked cross-attention
      query self-attention
      FFN
 -> query class + binary mask
 -> task-specific output formatting
```

핵심 연산은 **masked attention**이다. Standard cross-attention이 image 전체 위치를 보는 것과 달리 이전 decoder layer가 예측한 foreground mask 내부만 다음 query update에 허용한다.

```math
X_l=\mathrm{softmax}(M_{l-1}+Q_lK_l^\top)V_l+X_{l-1}.
```

Mask 밖 위치에는 `-infinity`, 안에는 0을 더한다. Query가 처음부터 segment 주변 feature에 집중하므로 convergence가 빨라지고 instance/panoptic 성능이 크게 오른다. 문맥은 query 간 self-attention을 통해 교환한다.

Small object를 위해 pixel decoder의 `1/32,1/16,1/8` feature를 decoder layer에 round-robin으로 넣는다. 모든 layer가 모든 scale을 concat하는 naive 방식보다 compute가 작다. 기본은 3 scale cycle을 세 번 반복한 9 decoder layers, 100 queries다.

Training mask loss는 full-resolution mask 전체가 아니라 `K=12,544=112×112` sampled points에서 계산한다. Hungarian matching cost에도 point sample을 사용한다. 논문 보고 기준 image당 training memory가 `18GB→6GB`, 3배 감소했다.

COCO에서 Swin-L model은 panoptic `57.8 PQ`, instance `50.1 AP`; ADE20K semantic은 FaPN pixel decoder와 `57.7 mIoU`를 기록했다. R50 model은 44M parameters, 226G FLOPs이며 COCO instance에서 batch=1 V100 `9.7 FPS`다. Masked attention은 sparse semantic constraint지만 일반 dense attention에 additive mask를 넣는 구현이면 FLOPs가 줄지 않으며, 논문 ablation에서도 masked version FLOPs가 오히려 조금 높다.

## Segmentation task의 세 종류

### Semantic segmentation

모든 pixel에 category를 할당한다. 같은 class의 서로 다른 instance를 구분하지 않는다.

### Instance segmentation

Countable object, 즉 thing마다 독립 mask와 class를 낸다. 같은 category의 두 사람을 서로 다른 mask로 구분한다.

### Panoptic segmentation

Thing instance와 amorphous stuff 영역을 하나의 non-overlapping scene partition으로 합친다. 모든 pixel이 정확히 하나의 segment에 속해야 한다.

전통적으로 세 task는 서로 다른 head와 post-processing을 사용했다. Mask2Former는 segment의 의미만 바꾸고 기본 output을 동일하게 둔다.

```text
N binary masks + N class predictions
```

Universal이라는 말은 동일 checkpoint 하나가 모든 dataset/task를 동시에 해결한다는 뜻이 아니다. 논문에서는 architecture와 loss는 같지만 task와 dataset별로 따로 학습한다.

## Per-pixel classification과 mask classification

### Per-pixel classification

Semantic FCN은 각 pixel마다 `C` class probability를 낸다.

```math
P\in\mathbb{R}^{H\times W\times C}.
```

Instance identity를 표현하려면 추가 grouping이나 detector가 필요하다.

### Mask classification

Mask2Former는 query `i`마다:

- class distribution `p_i`
- binary mask probability `m_i(x,y)`

를 예측한다. Query 수가 100이면 output set은 최대 100 segment slot이다. DETR처럼 GT segment set과 one-to-one matching한다.

Mask probability는 query에서 만든 mask embedding과 per-pixel embedding의 dot product로 계산한다.

```math
\hat m_i(u)=\sigma(e_i^\top f_{pixel}(u)).
```

여기서 `e_i in R^C`는 query의 mask embedding, `f_pixel(u) in R^C`는 1/4-resolution pixel embedding이다. 동일 구조로 thing과 stuff를 모두 표현한다.

## Meta architecture

### Backbone

ResNet 또는 Swin Transformer가 multi-level low-resolution feature를 추출한다.

### Pixel decoder

Backbone feature를 upsample/fuse해 두 종류의 output을 만든다.

1. Transformer decoder cross-attention용 multi-scale feature: input의 `1/32,1/16,1/8`
2. Binary mask dot-product용 high-resolution per-pixel embedding: `1/4`

Default pixel decoder는 Deformable DETR의 multi-scale deformable attention 6 layers를 사용한다. C3-C5 수준의 feature 사이를 attention으로 교환하고, final 1/8 feature에 lateral upsampling을 붙여 1/4 pixel embedding을 만든다.

FPN, Semantic FPN, FaPN, BiFPN도 교체 가능하지만 세 task 전체에서 MSDeformAttn이 가장 일관됐다.

### Transformer decoder

기본 query 수 `N=100`, hidden channel `C=256`, 9 layers다. 각 layer는 순서를 바꾼 다음 block을 쓴다.

```text
masked cross-attention first
 -> add & norm
 -> query self-attention
 -> add & norm
 -> FFN
 -> add & norm
```

각 intermediate layer와 decoder 진입 전 learnable query feature에도 auxiliary class/mask supervision을 건다.

## Masked attention

### Standard cross-attention

Layer `l`의 query feature를 `X_l in R^{N×C}`, 현재 scale image feature를 `K_l,V_l in R^{H_lW_l×C}`라 하면:

```math
X_l=\mathrm{softmax}(Q_lK_l^\top)V_l+X_{l-1}.
```

초기 query가 image 전체를 보므로 object region에 집중하는 pattern을 학습하는 데 오래 걸릴 수 있다.

### Attention mask 생성

이전 layer의 mask prediction을 현재 feature resolution으로 resize하고 0.5에서 binarize한다.

```math
\mathcal{M}_{l-1}(i,u)=
\begin{cases}
0,&\hat m_{l-1,i}(u)\ge0.5,\\
-\infty,&\text{otherwise}.
\end{cases}
```

Decoder 전 initial learnable query `X_0`도 mask proposal `M_0`을 예측해 첫 layer attention을 제한한다.

### Masked update

```math
X_l=\mathrm{softmax}
\left(\mathcal{M}_{l-1}+Q_lK_l^\top\right)V_l+X_{l-1}.
```

Softmax probability는 predicted foreground 밖에서 0이다. Query는 자기 mask 내부의 evidence를 모으고, query self-attention으로 다른 segment와 global context를 교환한다.

### Hard threshold의 의미

Attention mask 생성의 threshold operation은 이전 mask를 discrete support로 사용한다. Mask loss gradient는 mask prediction으로 가지만, threshold를 통한 다음 attention support의 gradient는 일반적으로 전달되지 않는다. 잘못된 초기 mask가 true object를 제외하면 다음 layer가 회복하기 어려울 수 있으므로 learnable proposal과 auxiliary losses가 중요하다.

### Sparse support와 sparse compute는 다르다

Equation은 mask 밖 weight를 0으로 만들지만 일반 implementation은 여전히 full `N×HW` score를 계산하고 additive mask를 적용한다. 따라서 attention focus는 sparse해도 arithmetic/memory가 자동으로 foreground pixel 수에 비례하지 않는다.

실제로 ablation에서 full Mask2Former는 226G FLOPs, masked attention을 제거한 model은 213G였다. Masked attention의 주된 이점은 compute 감소보다 optimization과 representation이다.

## Multi-scale round-robin strategy

Pixel decoder의 세 feature를 낮은 해상도부터 사용한다.

```text
layer 1: 1/32
layer 2: 1/16
layer 3: 1/8
layer 4: 1/32
layer 5: 1/16
layer 6: 1/8
layer 7: 1/32
layer 8: 1/16
layer 9: 1/8
```

각 feature에는 sinusoidal positional embedding과 learnable scale-level embedding을 더한다.

모든 layer에 3-scale concat을 넣는 naive multi-scale은 247G FLOPs, efficient round-robin은 226G였다. AP/PQ는 거의 같고 semantic mIoU는 efficient 방식이 더 높았다.

High-resolution layer는 작은 객체와 boundary를 돕지만 `1/8` token 수가 `1/32`보다 16배다. Round-robin은 이를 9번 모두 쓰지 않고 3번만 사용한다.

## Optimization 변경

### Cross-attention first

Standard decoder는 self-attention부터 한다. 그러나 initial query가 image-independent하면 첫 self-attention에는 image signal이 없다. Mask2Former는 먼저 image feature를 읽고 그다음 query끼리 관계를 교환한다.

### Learnable query feature

DETR식 zero feature + learnable positional query 대신 content feature `X_0`도 learnable하게 한다. Decoder 전부터 mask/class auxiliary loss를 주므로 initial query가 mask proposal 역할을 한다.

AR@100은 decoder 전 `50.3`, layer 3 `56.8`, layer 6 `57.4`, layer 9 `57.7`로 refinement된다.

### Dropout 제거

Decoder residual과 attention dropout을 제거했다. 이 setting에서는 dropout이 성능을 낮췄으며 제거해도 compute가 늘지 않는다. 다른 dataset 규모나 regularization 조건에 일반화되는 절대 규칙은 아니다.

## Set prediction과 loss

### Bipartite matching

Prediction query set과 GT segment set 사이 최소-cost one-to-one matching을 구한다. Matching cost에는 class와 sampled mask BCE/Dice가 들어간다.

Unmatched query는 no-object class target을 갖는다. 같은 GT에 query 여러 개가 positive가 되지 않으므로 detector식 anchor assignment가 필요 없다.

### Mask loss

```math
L_{mask}=\lambda_{ce}L_{BCE}+\lambda_{dice}L_{dice},
\qquad\lambda_{ce}=5,\ \lambda_{dice}=5.
```

전체 matched prediction loss는:

```math
L=L_{mask}+\lambda_{cls}L_{cls}.
```

Matched GT의 class weight는 `lambda_cls=2.0`, unmatched no-object는 `0.1`이다.

Dice loss는 foreground overlap을 직접 정규화해 mask size imbalance에 강하고, BCE는 point별 probability calibration을 보완한다. Exact smoothing epsilon과 reduction은 code version을 기록해야 한다.

### Point-sampled loss

Full mask 대신 `K=12,544` point를 사용한다.

- Matching: 모든 prediction/GT에 동일한 uniformly sampled point set
- Final matched loss: pair마다 importance sampling한 서로 다른 point set

Matching에서 동일 point를 써야 cost matrix의 query-GT pair가 같은 evidence 위에서 비교된다. Final loss는 uncertain point를 더 뽑아 boundary 학습 효율을 높인다.

## Task별 output 변환

### Semantic

Query class probability와 mask probability를 합쳐 pixel별 class score를 만든다.

```math
s_c(u)=\sum_i p_i(c)\,m_i(u).
```

Pixel마다 가장 높은 class를 선택한다. 같은 class query는 합쳐질 수 있다.

### Instance

Thing query mask 각각을 instance로 유지한다. Final confidence는 class confidence와 foreground mask pixel의 평균 mask probability를 곱한다. NMS 없이 query ranking을 사용한다.

### Panoptic

Class/score threshold 후 mask score 경쟁으로 non-overlapping segment map을 만든다. Stuff category의 여러 query는 같은 category로 merge할 수 있다. Architecture는 같지만 output formatting rule은 task별로 다르다.

## Batch=1 tensor shape 계산

### 800×1280 image, R50 model 가정

Paper inference의 short side 800 조건을 단순화해 divisible shape를 사용한다. Hidden `C=256`, query `N=100`이다. 아래는 **리뷰어 계산**이다.

| Tensor | shape | elements | FP16 payload |
| --- | ---: | ---: | ---: |
| 1/32 feature | `1×256×25×40` | 256,000 | 0.49 MiB |
| 1/16 feature | `1×256×50×80` | 1,024,000 | 1.95 MiB |
| 1/8 feature | `1×256×100×160` | 4,096,000 | 7.81 MiB |
| 1/4 pixel embedding | `1×256×200×320` | 16,384,000 | 31.25 MiB |
| queries | `1×100×256` | 25,600 | 0.049 MiB |

Per-pixel embedding 하나가 multi-scale transformer feature 전체보다 훨씬 크다.

### Cross-attention score

8 heads를 가정하면 layer별 dense score payload는:

| Scale | shape | FP16 |
| --- | ---: | ---: |
| 1/32 | `1×8×100×1000` | 1.53 MiB |
| 1/16 | `1×8×100×4000` | 6.10 MiB |
| 1/8 | `1×8×100×16000` | 24.41 MiB |

1/8 attention layer가 peak에 민감하다. Masked support가 작더라도 dense score를 materialize하면 크기는 그대로다.

### Binary mask output

100 query와 1/4-resolution 64,000 pixel의 dot-product output은:

```math
100\cdot64{,}000=6{,}400{,}000
```

FP16 약 `12.21 MiB`다. 각 auxiliary layer mask를 training에 보존하면 여러 배가 된다. Full 800×1280 mask를 100개 materialize하면 FP16 약 195 MiB이므로 quarter-resolution 유지가 중요하다.

## Reviewer MAC 분석

### Query-pixel mask dot product

```math
100\cdot64{,}000\cdot256\approx1.64\ \text{G MACs}
```

Intermediate decoder layer마다 mask proposal을 만들면 이 비용이 반복된다.

### Cross-attention score와 value aggregation

한 3-scale cycle에서 QK와 attention-value 두 matmul만 근사하면:

```math
2\cdot100\cdot(1000+4000+16000)\cdot256
\approx1.08\ \text{G MACs}.
```

Cycle 3회면 약 3.23G MACs이며 projection, self-attention, FFN, pixel decoder는 제외한다. Paper total R50 226G FLOPs와 count 범위가 다르다. 위 값은 mechanism 이해용 reviewer calculation이다.

## Training memory

Paper-reported COCO R50 image당 memory:

| Matching cost | Final loss | AP | PQ | mIoU | memory |
| --- | --- | ---: | ---: | ---: | ---: |
| full mask | full mask | 41.0 | 50.3 | 45.9 | 18G |
| point | point | 43.7 | 51.9 | 47.2 | 6G |

Final loss만 point로 바꿔도 6G였고 accuracy가 유지됐다. Matching도 point로 바꾸면 accuracy까지 개선됐다. 이 memory는 training 환경 측정이며 inference peak memory가 아니다.

## 구현 pseudocode

```python
def predict_masks(query, pixel_embedding):
    mask_embed = mask_mlp(query)                    # [B,N,C]
    return einsum("bnc,bchw->bnhw", mask_embed, pixel_embedding)


def masked_cross_attention(query, image_feat, prev_mask, pos, level_embed):
    # Resize previous query masks to current feature scale.
    support = interpolate(sigmoid(prev_mask), image_feat.spatial) >= 0.5
    additive_mask = where(support, 0.0, -INF)        # [B,N,H*W]
    key = flatten(image_feat) + pos + level_embed
    return multihead_attention(query=query, key=key, value=key,
                               attention_mask=additive_mask)


def mask2former(image):
    backbone_features = backbone(image)
    multi_scale, pixel_embedding = pixel_decoder(backbone_features)
    query = learnable_query_features.expand(batch)
    predictions = []

    prev_mask = predict_masks(query, pixel_embedding)  # M0 proposal
    predictions.append(class_and_mask(query, prev_mask))

    for layer_idx in range(9):
        level = layer_idx % 3                         # 1/32,1/16,1/8
        query = masked_cross_attention(query, multi_scale[level], prev_mask,
                                       pos[level], level_embedding[level])
        query = query_self_attention(query)
        query = ffn(query)
        prev_mask = predict_masks(query, pixel_embedding)
        predictions.append(class_and_mask(query, prev_mask))
    return predictions


def training_loss(predictions, targets):
    shared_points = uniform_points(K=12544)
    cost = class_cost(predictions, targets)
    cost += point_bce_dice_cost(predictions.masks, targets.masks, shared_points)
    matching = hungarian(cost)
    loss = 0
    for pred, gt in matching:
        points = importance_sample(pred.mask, K=12544)
        loss += 2.0 * class_ce(pred.cls, gt.cls)
        loss += 5.0 * mask_bce(pred.mask, gt.mask, points)
        loss += 5.0 * dice_loss(pred.mask, gt.mask, points)
    loss += no_object_ce(weight=0.1)
    return loss + auxiliary_losses(predictions[:-1])
```

## Training 및 evaluation protocol

- Framework: Detectron2
- Optimizer: AdamW
- Initial LR `1e-4`, weight decay `0.05`
- Backbone LR multiplier `0.1`
- LR drop at 0.9, 0.95 fraction of steps
- COCO R50/R101 기본 50 epochs, batch 16
- LSJ scale 0.1-2.0 후 1024×1024 crop
- Inference short side 800, long side <=1333
- FLOPs: 100 validation images 평균
- FPS: V100, batch=1, 전체 val 평균, post-processing 포함
- Semantic setting은 MaskFormer protocol과 별도 crop/schedule 사용

Swin-L best model은 200 queries, 100 epochs, ImageNet-22K pretrained backbone이다. R50 100-query 50-epoch 결과와 동일 recipe로 보아서는 안 된다.

## 주요 결과

### COCO panoptic

| Model | queries | epochs | PQ | PQ thing | PQ stuff | AP thing | mIoU pan | params | FLOPs | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R50 | 100 | 50 | 51.9 | 57.7 | 43.0 | 41.7 | 61.7 | 44M | 226G | 8.6 |
| R101 | 100 | 50 | 52.6 | 58.5 | 43.7 | 42.6 | 62.4 | 63M | 293G | 7.2 |
| Swin-L | 200 | 100 | 57.8 | 64.2 | 48.1 | 48.6 | 67.4 | 216M | 868G | 4.0 |

MaskFormer R50의 46.5 PQ보다 +5.4이면서 300→50 epochs로 줄었다.

### COCO instance

| Model | AP | AP small | AP medium | AP large | Boundary AP | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R50 | 43.7 | 23.4 | 47.2 | 64.8 | 30.6 | 9.7 |
| R101 | 44.2 | 23.8 | 47.7 | 66.7 | 31.1 | 7.8 |
| Swin-L | 50.1 | 29.9 | 53.9 | 72.1 | 36.2 | 4.0 |

R50 MaskFormer의 34.0 AP, 16.4 AP_S보다 크게 높다. 다만 Swin-L AP_S 29.9는 specialized HTC++ 31.0보다 낮고, 저자도 small-object 한계를 인정한다.

### ADE20K semantic

- R50: `47.2 SS / 49.2 MS`
- Swin-T: `47.7 / 49.6`
- Swin-L default: `56.1 / 57.3`
- Swin-L + FaPN pixel decoder: `56.4 / 57.7`

Best semantic 수치는 default MSDeformAttn이 아니라 FaPN pixel decoder임을 함께 기록해야 한다.

## 핵심 ablation

### Masked attention

Full R50은 instance AP `43.7`, panoptic PQ `51.9`, semantic mIoU `47.2`. Masked attention 제거 시 `37.8/47.1/45.5`로 각각 -5.9/-4.8/-1.7이다. Instance-level grouping에 특히 중요하다.

Foreground/background attention weight 분석에서 standard cross-attention은 평균 foreground 0.20, background 0.80이었다. Masked attention은 foreground 0.59, background 0.41로 segment 내부에 더 집중했다. Binarized mask와 feature resize 때문에 foreground 밖 weight가 이론상 0이어야 하는 layer도 있지만, 논문 분석의 foreground 정의/예측 mask 오차에 따라 GT background weight가 남을 수 있다.

### High-resolution feature

High-resolution strategy 제거 시 AP/PQ/mIoU가 `41.5/50.2/46.1`. 1/8 single scale은 `44.0/51.8/47.4`로 정확하지만 239G, efficient multi-scale은 `43.7/51.9/47.2`에 226G다.

### Pixel decoder

FPN, Semantic FPN, FaPN, BiFPN, MSDeformAttn의 AP는 각각 `41.5,42.1,42.4,43.5,43.7`. 한 task만 보면 다른 decoder가 유리할 수 있지만 세 task 평균 일관성 때문에 MSDeformAttn을 선택했다.

### Optimization changes

Learnable query 제거 -0.8 AP, cross-attention-first 제거 -0.5 AP, dropout 제거를 되돌리면 -0.7 AP였다. 세 변경은 FLOPs 증가 없이 총 -1.4 AP 차이를 만들었다.

## Small object와 boundary

Small object 개선 요소:

- 1/8 feature를 decoder가 주기적으로 사용
- 1/4 pixel embedding으로 mask boundary 생성
- Point importance sampling이 uncertain boundary에 집중
- Masked attention이 작은 foreground를 background context에 희석시키지 않음

R50은 MaskFormer보다 AP_S +7.0이지만 large-object gain +10.6이 더 크다. Query budget, coarse attention support, quarter-resolution mask 때문에 작은 object가 여전히 어렵다. Dilated backbone과 small-object loss가 후속 과제로 제시된다.

Boundary AP에서 R50 30.6, Swin-L 36.2는 mask quality 이점을 보여 준다. 그러나 1/4 resolution dot-product mask이므로 매우 얇은 structure에는 해상도 한계가 있다.

## 장점과 기여

1. 세 segmentation task를 같은 mask-classification architecture로 강하게 해결했다.
2. Previous mask를 attention support로 재사용하는 masked attention을 제시했다.
3. Multi-scale round-robin으로 small-object feature와 compute를 균형화했다.
4. Learnable proposals, attention order, dropout 제거로 수렴을 개선했다.
5. Point-sampled matching/loss로 training memory를 3배 줄였다.
6. Specialized model을 넘어서는 panoptic/instance/semantic 결과를 보였다.

## 한계와 비판적 관점

### 1. Universal checkpoint는 아니다

Architecture가 universal할 뿐 task/dataset별 별도 학습이 필요하다. Panoptic-trained model을 다른 task로 변환하면 task-specific model보다 조금 낮다.

### 2. Masked attention이 compute-sparse하지 않다

Dense attention score를 먼저 계산하면 foreground support가 작아도 latency와 memory는 줄지 않는다. True block-sparse kernel이 필요하다.

### 3. Pixel decoder가 무겁다

6-layer multi-scale deformable attention과 high-resolution embedding은 custom operator와 큰 activation을 요구한다.

### 4. Query 수 upper bound

100 query보다 segment가 많은 crowd scene은 recall upper bound가 생긴다. Swin-L은 200으로 늘리지만 self-attention과 mask output 비용이 증가한다.

### 5. Small-object 성능

개선됐지만 best specialized model보다 낮은 AP_S이고 저자도 한계로 명시한다.

### 6. Training memory와 inference memory는 다르다

6GB는 point loss를 쓴 training image당 측정이다. Batch=1 inference peak, p50/p95, energy는 보고하지 않는다.

## 자주 헷갈리는 지점

### Mask2Former는 SAM처럼 prompt를 받는가

아니다. Learnable query가 dataset category segment를 자동 예측한다. User point/box prompt encoder가 없다.

### Masked attention과 mask loss의 mask는 같은가

같은 query mask prediction에서 나오지만 역할이 다르다. 하나는 다음 attention support, 다른 하나는 GT supervision 대상이다.

### 0.5 threshold 밖은 계산하지 않는가

수학적으로 weight는 0이지만 dense implementation은 score를 계산한 뒤 mask할 수 있다. Operator를 확인해야 한다.

### NMS가 필요한가

Hungarian set prediction 때문에 detector식 NMS는 기본적으로 없다. Task output conversion에서 mask overlap resolution과 score filtering은 존재한다.

### Point loss는 112×112 grid인가

K가 12,544라 같은 개수지만 실제로는 sampled points다. Matching은 uniform, final loss는 importance sampling을 사용한다.

## 온디바이스 관점

### 병목 operator

- MSDeformAttn의 bilinear gather/custom kernel
- 1/8 dense multi-head attention
- 1/4 pixel embedding
- query-pixel einsum
- 9회 auxiliary mask generation
- resize/threshold/additive mask

NPU가 deformable attention이나 dynamic attention mask를 지원하지 않으면 CPU/GPU fallback이 발생할 수 있다.

### 요청형 실행

Mask2Former는 image마다 scene 전체 segment를 예측하므로 상시 semantic segmentation에는 SegFormer보다 무겁다. SAM처럼 image embedding을 cache하고 prompt decoder만 반복하는 구조도 아니다.

### 최적화 방향

- R50보다 MobileNet/efficient hierarchical backbone
- Pixel decoder를 FPN/BiFPN으로 교체해 custom op 제거
- Queries 100→50 ablation
- Decoder 9→3/6 layers early exit
- 1/8 layer 횟수 감소
- Mask dot-product INT8, attention FP16 mixed precision
- Full-resolution mask는 selected top queries만 upsample

Paper R50도 9.7 FPS V100이므로 원형은 모바일 real-time에 적합하지 않다. Accuracy 손실과 operator support를 함께 평가해야 한다.

## 재현 체크리스트

- [ ] CVPR 2022/arXiv v3 metadata를 기록한다.
- [ ] Backbone, pixel decoder, Transformer decoder를 분리한다.
- [ ] Default pixel decoder 6 MSDeformAttn layers와 1/4 embedding을 확인한다.
- [ ] Multi-scale feature `1/32,1/16,1/8` round-robin 순서를 맞춘다.
- [ ] Decoder 9 layers, default queries 100을 확인한다.
- [ ] Cross-attention first, learnable query feature, dropout 0을 적용한다.
- [ ] Previous mask resize, sigmoid, threshold 0.5, additive -infinity를 구현한다.
- [ ] Every layer와 X0에 auxiliary class/mask loss를 건다.
- [ ] Matching shared uniform points와 final importance points를 구분한다.
- [ ] K=12,544, BCE/Dice weights 5/5, class matched/no-object 2/0.1을 맞춘다.
- [ ] Task별 post-processing과 score definition을 구분한다.
- [ ] R50 50 epoch와 Swin-L 100 epoch/200 query를 혼동하지 않는다.
- [ ] SS/MS result, ImageNet-1K/22K pretraining을 기록한다.
- [ ] Training memory와 inference peak를 분리한다.
- [ ] Target batch=1 peak, p50/p95, custom-op fallback, power/temperature를 측정한다.

## 로드맵에서의 연결

- **DETR/Deformable DETR**의 set matching과 multi-scale attention을 segmentation으로 확장한다.
- **SegFormer**는 per-pixel semantic specialist, Mask2Former는 query-mask universal architecture다.
- **SAM**은 category-free promptable mask를 예측하며 supervised category set을 가진 Mask2Former와 목적이 다르다.
- **Video segmentation**에서는 query/mask memory를 시간축으로 확장하는 아이디어가 SAM 2와 연결된다.

통합 프로젝트에서 Mask2Former는 panoptic scene decomposition teacher로 가치가 크다. 모바일 student에는 full query decoder를 그대로 배포하기보다 SegFormer/FCOS head로 distill하거나, selected ROI에만 query mask decoder를 실행하는 방향이 현실적이다.

## 최종 평가

Mask2Former의 중요한 기여는 세 task의 output을 query-mask set으로 통일하고, mask prediction을 다음 attention의 inductive bias로 다시 사용한 것이다. Masked attention은 query가 자기 segment에 집중하도록 해 300 epoch MaskFormer를 50 epoch strong model로 바꿨다. Point sampling은 universal mask model의 training memory 장벽도 크게 낮췄다.

반면 원형은 44M R50 model도 226G FLOPs, V100 9.7 FPS이며 1/4 embedding과 dense masked-attention matrix가 크다. 온디바이스에서는 "mask가 sparse하다"는 직관보다 실제 tensor materialization과 custom operator support를 봐야 한다. Mask2Former는 강력한 accuracy/teacher baseline이지만, 상시 모바일 segmenter로 쓰려면 pixel decoder, query 수, layer 수를 상당히 줄여야 한다.

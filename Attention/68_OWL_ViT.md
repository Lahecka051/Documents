# 68. Simple Open-Vocabulary Object Detection with Vision Transformers

## 논문 정보

- 원본 파일: `68_OWL_ViT.pdf`
- 제목: **Simple Open-Vocabulary Object Detection with Vision Transformers**
- 모델명: OWL-ViT, Vision Transformer for Open-World Localization
- 저자: Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf, Neil Houlsby
- 소속: Google Research
- 발표: ECCV 2022
- 링크: [arXiv:2205.06230](https://arxiv.org/abs/2205.06230)
- 코드: [Scenic OWL-ViT](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)
- 핵심 키워드: open-vocabulary detection, zero-shot detection, image-conditioned detection, ViT token prediction, Hungarian matching, text query

## 한 문장 요약

OWL-ViT는 contrastive image-text pretrained ViT의 pooling을 제거하고 모든 patch output token에 box head와 text-conditioned classification head를 직접 붙여, decoder와 region proposal 없이 open-vocabulary 및 one-shot detection을 수행한다.

## 문제 설정

closed-vocabulary detector는 학습 때 정해진 `C`개 class에 대해서만 고정 classifier weight를 학습한다. 새로운 category를 추가하려면 annotation과 재학습이 필요하다. OWL-ViT는 classifier weight를 text encoder가 만든 query embedding으로 바꾼다.

```text
closed vocabulary:
image feature -> learned W_class -> fixed C logits

open vocabulary:
text strings -> text encoder -> query embeddings
image token -> object embedding
object embedding x query embedding -> per-query logits
```

inference 시 `"giraffe"`, `"red screwdriver"`, `"damaged bearing"`처럼 query set을 image마다 바꿀 수 있다. text로 설명하기 어려운 object는 crop image의 embedding을 query로 사용해 one-shot detection도 할 수 있다.

논문은 다음 두 단계를 사용한다.

1. 수십억 image-text pair로 image encoder와 text encoder를 contrastive pretrain
2. 공개 detection data에서 두 encoder와 작은 detection head를 end-to-end fine-tune

RPN crop distillation, pseudo-label stage, multimodal fusion encoder를 추가하지 않는 단순함이 핵심이다.

## 전체 구조

```text
image: B x 3 x H x W
  -> Vision Transformer
  -> patch output tokens: B x N x D
  -> optional CLIP class-token modulation
  -> classification projection -> B x N x E
  -> box MLP -> B x N x 4

text query strings: C independent sequences
  -> Text Transformer
  -> query embeddings: C x E

similarity
  -> logits: B x N x C
  -> sigmoid query probabilities
  -> boxes + labels + scores
```

DETR처럼 set prediction과 bipartite matching을 사용하지만 transformer decoder와 learned object query가 없다. image patch token 자체가 object slot이다. 따라서 가능한 prediction 수는 image token 수 `N`이다.

## Image token이 detection slot이 되는 과정

patch token output을 `z_i in R^D`라 하자. 각 token에서 box와 object embedding을 독립 head로 만든다.

```math
o_i=W_o z_i+b_o
```

```math
\hat b_i=\mathrm{MLP}_{box}(z_i)
```

text query `q_c`는 category name이나 description을 text transformer에 넣어 얻는다.

```math
q_c=f_T(t_c)
```

normalize와 learned scale을 생략해 쓰면 classification logit은 dot product다.

```math
\ell_{ic}=o_i^\top q_c
```

```math
p_{ic}=\sigma(\ell_{ic})
```

softmax가 아니라 independent sigmoid를 사용하므로 한 object에 `toy`, `elephant`, `toy elephant`처럼 여러 label이 동시에 참일 수 있다. federated detection dataset의 non-disjoint label space와 잘 맞는다.

query는 서로 독립된 text sequence로 encoding한다. image와 text 사이 early fusion이 없으므로 query embedding을 image와 무관하게 미리 cache할 수 있고 한 image에 1,203개 LVIS query를 넣을 수 있다. query를 바꿀 때 image encoder를 다시 실행할 필요도 없다.

## Class token과 class assignment를 구분하기

OWL-ViT에서 "class token"은 두 의미로 혼동되기 쉽다.

### CLIP의 global class token

public CLIP vision encoder는 patch token 외에 global class token을 출력한다. detection에서는 class token 하나로 box를 예측하지 않는다. 논문은 다음 선택을 비교한다.

- global class token을 버림
- class token을 각 feature-map token과 element-wise multiply한 뒤 LayerNorm

후자가 대부분 architecture에서 더 좋아 사용한다.

```math
\tilde z_i=\mathrm{LN}(z_i\odot z_{cls})
```

global semantic context가 모든 local token을 gate하는 셈이다. 이 class token은 ground-truth object와 match되는 detector slot이 아니다.

### Ground-truth object와 image token의 assignment

training에서는 Hungarian bipartite matching이 `N`개 prediction과 `M`개 ground-truth object 사이의 one-to-one assignment를 찾는다.

```math
\hat\pi=
\arg\min_{\pi\in\Pi}
\sum_{j=1}^{M}
\mathcal{C}(y_j,\hat y_{\pi(j)})
```

matching cost는 classification, box L1, generalized IoU 정보를 사용한다. match된 token은 해당 object box와 여러 valid label을 학습하고, 나머지 token은 background 또는 negative query에 낮은 score를 내도록 학습한다. 이 assignment가 duplicate box를 억제하는 주된 메커니즘이다.

## Box location bias

각 output token은 원래 2D patch grid 위치가 있다. self-attention 이후 strict locality는 사라지지만 box head의 초기 prediction center를 그 patch 위치에 bias한다.

```math
\hat b_i=
\mathrm{decode}(
\Delta x_i,\Delta y_i,\hat w_i,\hat h_i;
x_i^{grid},y_i^{grid})
```

anchor를 고정한 detector라기보다 box center의 initial prior다. bipartite matching의 permutation symmetry를 깨고 학습을 빠르게 한다. R26+B/32 ablation에서 location bias를 제거하면 LVIS AP가 1.2, rare AP가 1.1 point 낮아진다. pure ViT-B/32에서는 하락이 약 2.8과 2.9 point로 더 크다. convolutional hybrid의 spatial bias가 일부 역할을 대신한다는 해석이 가능하다.

## Tensor shape와 후보 수

### 대표 architecture

| Backbone | Detection input | Patch/stride | Grid | Prediction token `N` |
| --- | ---: | ---: | ---: | ---: |
| ViT-B/32 | 768 x 768 | 32 | 24 x 24 | 576 |
| ViT-B/16 | 768 x 768 | 16 | 48 x 48 | 2,304 |
| ViT-L/16 | 768 x 768 | 16 | 48 x 48 | 2,304 |
| ViT-H/14 | 840 x 840 | 14 | 60 x 60 | 3,600 |
| CLIP ViT-L/14 | 840 x 840 | 14 | 60 x 60 | 3,600 |

논문은 최소 576 token이 당시 LVIS image의 최대 instance 294개보다 많아 candidate 수 병목이 아니라고 설명한다. 그러나 token 수는 vision attention, query similarity, postprocess 비용을 동시에 키운다.

### LVIS 1,203 query 예시

CLIP L/14@840은 다음 logit matrix를 만든다.

```math
3600\times1203=4,330,800\ \text{logits/image}
```

FP16 raw payload는 약 8.26MiB다. box tensor `3600 x 4`는 FP16 약 28KiB에 불과하다. open-vocabulary에서는 box regression보다 token-query similarity와 score filtering이 post-head memory의 큰 부분이다.

7개 prompt template를 한 번에 모두 materialize하면 query 축이 7배가 될 수 있다. 구현은 prompt별 probability를 streaming average하거나 query embedding을 ensemble한 형태로 최적화해 peak를 줄여야 한다. 논문은 mobile memory를 보고하지 않는다.

### Vision self-attention

3,600 image token의 dense attention score는 head당 `12.96M` element다. FP16 matrix 한 장은 약 24.7MiB이며 layer와 head를 곱하면 naive materialization은 매우 크다. FlashAttention류 kernel이 memory를 줄일 수 있지만 840 해상도의 large ViT가 on-device에 무거운 근본 원인은 남는다.

## Contrastive pretraining

image token은 pretraining 단계에서 multi-head attention pooling으로 한 image embedding으로 모으고, text representation은 EOS token에서 얻는다. image와 matching caption을 batch 내에서 가깝게 만드는 symmetric contrastive loss를 사용한다.

```math
\mathcal{L}_{ITC}
=\frac{1}{2}
\left(
\mathrm{CE}(I T^\top/\tau,\mathrm{diag})
+\mathrm{CE}(T I^\top/\tau,\mathrm{diag})
\right)
```

LiT 계열 실험은 3.6B unique image-text pair에서 scratch pretraining하고 configuration에 따라 8B, 12B, 16B, 24B examples를 반복해 본다. public CLIP checkpoint를 사용하는 model도 평가한다.

detection head는 전체 parameter의 최대 1.1%라 거의 모든 parameter가 image-level pretraining의 이득을 받는다. 반면 image classification 성능이 높다고 detection transfer가 자동으로 좋은 것은 아니다. zero-shot ImageNet과 LVIS rare AP의 correlation은 0.73이고 pretraining task performance와 ImageNet의 correlation은 0.98이다.

## Detection loss

match된 prediction에 box L1, generalized IoU, classification loss를 동일 weight로 더한다.

```math
\mathcal{L}
=\mathcal{L}_{focal}
+\mathcal{L}_{L1}
+\mathcal{L}_{gIoU}
```

focal sigmoid loss hyperparameter는 `alpha=0.3`, `gamma=2.0`이다. softmax가 아닌 이유는 category가 non-disjoint하고 하나의 instance가 여러 label을 가질 수 있기 때문이다.

federated dataset은 모든 category를 모든 image에서 exhaustively annotation하지 않는다. training query set은 다음으로 구성한다.

- image에 present라고 annotation된 positive category
- absent라고 확인된 real negative category
- dataset frequency에 비례해 sampling한 pseudo-negative

positive는 sampling에서 제외하고 최소 50 negative가 되도록 추가한다. random negative를 제거하면 R26+B/32의 LVIS rare AP가 2.8 point 낮아진다.

## Dataset 결합과 annotation 정리

object-level training은 약 2M image 규모의 공개 data를 조합한다.

| Dataset | Train images | Box instances | Categories/특성 |
| --- | ---: | ---: | --- |
| OpenImages V4 | 1.7M | 14.6M | 601, federated |
| Visual Genome | 84.5K | 약 2M | free-text dense objects |
| Objects365 | 608.5K | 9.2M | 365 |
| LVIS | 100K | 평가 중심 | 1,203, long tail/federated |
| COCO | 118K | 약 0.9M | 80 |

대부분 ablation은 OI:VG를 70:30으로 sampling하고 큰 model 표는 O365:VG를 80:20으로 사용한다. dataset 크기에 비해 VG를 크게 oversample하는 이유는 넓은 semantic label space가 example당 더 많은 open-vocabulary supervision을 주기 때문이다.

COCO와 LVIS validation/test image가 VG, OI, O365에 겹칠 수 있어 strict deduplication한다. VG에서는 6.7K image와 156K instance가 제거된다. zero-shot LVIS rare 평가는 rare category name과 matching되는 모든 box annotation도 detection training에서 제거한다. image-text pretraining 자체에서 category concept을 본 것은 허용한다.

overlap하는 box가 synonym 또는 hierarchical multi-label인 경우 IoU 0.7-0.9에서 하나의 instance로 merge하고 label을 합친다. 그렇지 않으면 같은 object에 `mug`와 `jug` box를 따로 예측하도록 강제해 open-vocabulary generalization이 나빠질 수 있다.

## Prompting

category name만 넣으면 image-text pretraining의 caption distribution과 차이가 난다. training에서는 CLIP의 80개 template 중 random prompt를 선택한다.

```text
"a photo of a {category}"
"an image of a {category}"
...
```

inference에서는 CLIP이 선정한 7개 best prompt의 probability를 평균한다. prompt ensemble을 제거하면 R26+B/32에서 LVIS AP 2.8, rare AP 5.5, COCO AP 5.9 point가 낮아진다. 효과가 크므로 evaluation prompt를 model architecture의 일부처럼 고정해야 한다.

## NMS는 사용하는가

논문의 핵심 pipeline은 DETR식 one-to-one set prediction이므로 **NMS를 요구하지 않는다**. Hungarian matching이 한 ground-truth를 한 image token에 배정하고 나머지는 background로 학습한다. paper는 decoder뿐 아니라 proposal generation과 non-maximum suppression이 없는 단순 detector를 지향한다.

일반 inference 흐름은 다음과 같다.

```text
sigmoid scores
 -> query별 또는 전체 threshold
 -> token-query pair flatten
 -> top-k selection
 -> box coordinate conversion and clipping
 -> output
```

실제 deployment에서 duplicate가 문제라면 optional NMS를 넣을 수 있지만 이는 논문 결과의 기본 구성과 다르다.

### Open-vocabulary에서 NMS가 특히 까다로운 이유

- class-agnostic NMS: `dog`와 `animal`, `mug`와 `cup`처럼 같은 box에 유효한 여러 query 중 하나를 지워 버린다.
- class-aware NMS: synonym query마다 duplicate box를 남길 수 있다.
- 작은 겹친 물체: 높은 IoU suppression이 실제 두 instance를 하나로 만들 수 있다.
- 1,203 query: query별 NMS loop가 CPU와 p95 latency를 키울 수 있다.

따라서 optional postprocess를 평가한다면 `no NMS`, exact-string class-aware NMS, synonym-group NMS, class-agnostic NMS를 별도 실험해야 한다. paper AP와 비교할 때는 no-NMS 결과를 기준으로 유지한다.

## Training recipe

default text encoder는 12 layers, hidden 512, MLP 2048, 8 heads다. detection fine-tuning batch는 256이고 최대 140K step이다. image encoder learning rate보다 text encoder LR을 100배 작게 하는 것이 중요하다.

R26+B/32 baseline의 대표 설정은 다음과 같다.

- pretraining: 8B seen examples, batch 16K, LR `3e-4`, weight decay `1e-5`
- detection: 70K steps, batch 256
- image LR: `2e-4`
- text LR: `2e-6`
- input: 768
- OI:VG = 0.7:0.3
- mosaic single/2x2/3x3 = 0.5:0.33:0.17
- random negatives: 사용

text encoder를 image와 같은 LR로 fine-tune하면 rare AP가 8.5 point 떨어지고, 완전히 freeze해도 2.3 point 떨어진다. 넓은 language semantics를 잊지 않되 detection label에 약하게 적응하는 중간점이 필요하다.

## Detection training pseudocode

```python
def owl_vit_train_step(image, annotations, query_names):
    # independent text encoding
    text_query = normalize(text_encoder(prompt(query_names)))  # [C, E]

    token, cls_token = vision_encoder(image)                    # [N, D], [D]
    token = layer_norm(token * cls_token[None, :])

    object_emb = normalize(class_projection(token))             # [N, E]
    pred_box = box_mlp(token, grid_center_bias=True)             # [N, 4]
    logits = object_emb @ text_query.T                           # [N, C]

    match = hungarian_match(
        logits, pred_box, annotations,
        costs=["focal", "l1", "giou"],
    )
    return (
        focal_sigmoid_loss(logits, match.multi_labels)
        + l1_box_loss(pred_box, match.boxes)
        + generalized_iou_loss(pred_box, match.boxes)
    )
```

## Open-vocabulary 결과

LVIS v1.0 val에서 rare label annotation을 training에서 제거한 결과다. 값은 세 fine-tuning run 평균이다.

| Backbone | Pretrain | Detection data | Resolution | LVIS AP | LVIS rare AP |
| --- | --- | --- | ---: | ---: | ---: |
| ViT-B/32 | LiT | O365, VG | 768 | 23.3 | 19.7 |
| R26+B/32 | LiT | O365, VG | 768 | 25.7 | 21.6 |
| ViT-B/16 | LiT | O365, VG | 768 | 26.7 | 23.6 |
| ViT-L/16 | LiT | O365, VG | 768 | 30.9 | 28.8 |
| ViT-H/14 | LiT | O365, VG | 840 | 33.6 | 30.6 |
| ViT-B/32 | CLIP | O365, VG | 768 | 22.1 | 18.9 |
| ViT-B/16 | CLIP | O365, VG | 768 | 27.2 | 20.6 |
| ViT-L/14 | CLIP | O365, VG | 840 | 34.6 | 31.2 |

best CLIP L/14은 전체 34.6 AP, unseen rare 31.2 AP다. COCO에서는 43.5 AP와 64.7 AP50을 기록하고 O365를 빼고 학습한 model의 O365 AP는 15.8이라고 본문이 요약한다. 다만 COCO와 O365 category가 training label과 겹치므로 이 결과는 논문 정의상 zero-shot이 아니라 open-vocabulary transfer다.

## Image-conditioned one-shot 결과

query crop에 대해 model inference를 먼저 수행하고 query box와 IoU 0.65보다 큰 prediction 중 주변 embedding과 가장 다른 foreground embedding을 고른다. 후보가 없는 약 10%에서는 `"an image of an object"` text embedding으로 fallback한다.

COCO unseen split 평균 AP50은 다음과 같다.

| Method | 1-shot unseen AP50 | 10-shot unseen AP50 |
| --- | ---: | ---: |
| SiamMask | 16.8 | 미보고 |
| CoAE | 22.0 | 미보고 |
| AIT | 24.3 | 미보고 |
| OWL-ViT | 41.8 | 46.8 |

seen category는 1-shot 49.1, query 10개 embedding을 평균하면 55.1이다. image query와 target image를 early fusion하지 않아 query embedding을 cache하고 많은 prototype을 동시에 비교할 수 있다.

## 핵심 ablation

R26+B/32 baseline은 LVIS 21.0, rare 18.9, COCO 30.9, OI 54.1이다.

| 제거/변경 | LVIS delta | Rare delta | COCO delta | 해석 |
| --- | ---: | ---: | ---: | --- |
| VG만 사용 | -14.5 | -14.0 | -23.6 | dataset 다양성과 규모 모두 필요 |
| OI만 사용 | -6.9 | -5.7 | -4.2 | VG의 넓은 label space 중요 |
| image/text 같은 LR | -3.0 | -8.5 | -0.5 | language forgetting이 zero-shot에 치명적 |
| prompt ensemble 없음 | -2.8 | -5.5 | -5.9 | prompt distribution shift 큼 |
| prompt 전혀 없음 | -1.2 | -1.3 | -0.6 | train과 test prompting 모두 영향 |
| random negative 없음 | -1.0 | -2.8 | -0.4 | open label discrimination 약화 |
| mosaic 없음 | -2.3 | -1.5 | -1.7 | scale diversity 중요 |
| overlap instance merge 없음 | -0.8 | -1.3 | -0.6 | multi-label box 정리가 유효 |
| location bias 없음 | -1.2 | -1.1 | -1.3 | token-grid prior가 학습 안정화 |

mosaic을 제거하고 2배 또는 3배 더 오래 학습해도 성능이 회복되지 않고 rare AP는 더 낮아진다. augmentation의 이득은 단순 training step 증가와 다르다.

## Scaling 결과 해석

- image-level zero-shot 성능은 detection transfer에 필요하지만 충분하지 않다.
- 작은 compute에서는 hybrid가 pure ViT보다 효율적이다.
- model이 커질수록 pure ViT가 hybrid를 추월한다.
- 같은 overall LVIS AP에서 pure ViT는 rare AP가 더 높은 경향이 있다.
- 작은 model을 너무 오래 pretrain하면 image classification은 계속 좋아져도 detection AP는 peak 후 정체될 수 있다.
- model size와 pretraining duration, detection fine-tuning을 함께 scale해야 한다.

이 결과는 semantic generalization과 localization inductive bias가 같은 방향으로 움직이지 않을 수 있음을 보여 준다.

## On-device 관점

논문은 parameter 수, latency, peak memory, power를 보고하지 않는다. 최고 model은 840 x 840 CLIP ViT-L/14와 3,600 image token을 사용하므로 mobile 실시간 detector로 보기 어렵다.

배포 비용은 다음처럼 나눌 수 있다.

```math
T_{total}
=T_{text\ query}
+T_{vision}
+T_{similarity}
+T_{filter/topk}
+T_{optional\ NMS}
```

query set이 고정되면 `T_text_query`는 startup에 한 번만 내고 embedding을 cache한다. frame마다 vision encoder가 지배할 가능성이 높다. query 수가 수천이면 similarity와 top-k도 무시할 수 없다.

### Reviewer activation 예시

ViT-L/14 hidden dimension을 실제 checkpoint의 `D`라 할 때 output activation은 `3600 x D`, FP16 payload는 `7200D` bytes다. attention의 theoretical compute는 layer당 대략 `O(N^2D+ND^2)`다. 840에서 420으로 해상도를 절반으로 낮추면 token은 4분의 1, attention score 항은 16분의 1이 된다. 정확도 저하는 별도 fine-tuning으로 측정해야 한다.

### Operator 위험

- large fixed-resolution ViT attention
- positional embedding interpolation
- 3,600 x C similarity GEMM
- sigmoid와 large top-k
- dynamic query count
- optional NMS의 CPU loop

NPU는 dynamic `C`를 싫어할 수 있으므로 query count bucket 또는 maximum query padding이 필요하다. text encoder는 별도 graph로 두고 embedding cache를 persistent storage에 저장할 수 있다.

## 강점

1. image-text pretrained encoder를 detection으로 옮기는 구조가 매우 단순하다.
2. text와 image query를 같은 classification interface에서 처리한다.
3. decoder, RPN, early fusion, 기본 NMS 없이 end-to-end set prediction을 수행한다.
4. dataset deduplication과 unseen rare label 제거를 명시해 zero-shot 평가를 엄격히 한다.
5. data, learning rate, prompting, negative, mosaic, location bias를 세밀하게 ablation한다.
6. one-shot unseen AP50을 24.3에서 41.8로 크게 개선한다.

## 한계와 비판적 해석

### 1. Mobile latency와 memory evidence가 없다

large input과 global ViT는 온디바이스에 매우 무겁다. FLOPs 비교 일부 외에 실제 hardware 수치가 없어 실시간성을 판단할 수 없다.

### 2. Token 수가 object 수보다 지나치게 많다

3,600 slot이 최대 수백 object를 위해 사용된다. decoder를 없앤 단순함과 compute 효율은 같은 말이 아니다.

### 3. Query 수가 커지면 output가 커진다

image-text fusion은 없지만 `N x C` score가 필요하다. LVIS 1,203 query와 7 prompt에서 memory 및 postprocess가 커진다.

### 4. Zero-shot은 localized annotation 기준이다

rare category의 detection box는 보지 않았지만 image-text pretraining에서 concept과 image를 보았을 수 있다. 완전한 data-free unseen concept은 아니다.

### 5. Prompt sensitivity가 크다

prompt ensemble 제거 시 rare AP가 5.5 point 떨어진다. 사용자 query wording이 달라지는 real-world interface에서는 calibration과 prompt robustness가 필요하다.

### 6. NMS-free라고 duplicate가 완전히 사라지는 것은 아니다

Hungarian training은 duplicate를 억제하지만 distribution shift와 synonym query에서는 겹친 prediction이 생길 수 있다. deployment postprocess 정책을 별도 검증해야 한다.

## 재현 체크리스트

- [ ] 정확한 LiT 또는 CLIP image/text checkpoint 확인
- [ ] detection input 768 또는 840과 position interpolation 확인
- [ ] CLIP class token multiply 후 LayerNorm 구현
- [ ] patch token마다 box와 class embedding head 부착
- [ ] focal alpha 0.3, gamma 2.0 확인
- [ ] L1, gIoU, classification loss equal weighting
- [ ] Hungarian one-to-one assignment과 multi-label target 확인
- [ ] image LR 대비 text LR 100분의 1 적용
- [ ] 최소 50 pseudo-negative sampling과 frequency weighting 확인
- [ ] overlap multi-label instance merge 구현
- [ ] crop 후 original box area 60% rule 확인
- [ ] train 80 prompt, test 7 prompt ensemble 고정
- [ ] LVIS/COCO image dedup과 rare annotation 제거 확인
- [ ] no-NMS paper baseline과 optional NMS 결과 분리
- [ ] p50/p95, peak memory, power를 target device에서 추가 측정

## 온디바이스 연구 로드맵

1. ViT-B/32@768의 576-slot baseline부터 export한다.
2. text query embedding을 offline cache하고 `C={20,100,1203}`을 비교한다.
3. input `384,512,768`에서 token, AP, vision latency를 측정한다.
4. FP16, image INT8, text INT8, head FP16 조합을 비교한다.
5. no NMS와 여러 NMS 정책의 duplicate rate, AP, p95를 비교한다.
6. detector를 매 frame 실행한다면 query set과 vision feature cache 전략을 추가한다.

필수 결과 표에는 AP와 rare AP뿐 아니라 vision latency, query encoding latency, similarity/top-k, optional NMS, peak memory, energy, temperature를 넣어야 한다.

## 최종 평가

OWL-ViT의 핵심은 open-vocabulary detection을 복잡한 multimodal fusion 문제가 아니라 `image token과 query embedding의 matching`으로 단순화한 것이다. 모든 patch token이 한 box slot이 되고 Hungarian matching이 ground-truth를 token에 배정한다. CLIP의 global class token은 detector slot이 아니라 local feature를 modulation하는 context로 사용한다. 기본 inference는 NMS-free다.

이 단순함은 확장성과 one-shot flexibility를 주지만 mobile 효율을 자동으로 주지는 않는다. CLIP L/14@840은 3,600 image token과 LVIS 기준 4.33M query logits를 만든다. 논문에는 latency와 peak memory가 없다. 따라서 OWL-ViT는 강한 open-vocabulary teacher/reference로 사용하고, 온디바이스 student에서는 smaller hierarchical encoder, query cache, token reduction과 postprocess를 함께 설계하는 것이 현실적이다.

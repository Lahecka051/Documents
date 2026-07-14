# 44. FCOS: Fully Convolutional One-Stage Object Detection

## 논문 정보

- 원본 파일: `44_FCOS.pdf`
- 제목: **FCOS: Fully Convolutional One-Stage Object Detection**
- 저자: Zhi Tian, Chunhua Shen, Hao Chen, Tong He
- 발표: ICCV 2019
- 공개: arXiv:1904.01355, 제공된 PDF는 v5(2019)
- 링크: [https://arxiv.org/abs/1904.01355](https://arxiv.org/abs/1904.01355)
- 핵심 키워드: anchor-free detection, one-stage detector, per-pixel prediction, FPN, center-ness, IoU loss, focal loss, NMS

## 한눈에 보는 요약

FCOS는 object detection을 anchor box matching 문제가 아니라 semantic segmentation과 유사한 **per-pixel dense prediction**으로 다시 정의한다. FPN의 각 위치가 어떤 객체 내부에 있으면 그 위치에서 category와 box의 네 변까지 거리 `(l,t,r,b)`를 직접 예측한다.

```text
image
 -> backbone + FPN P3-P7
 -> every spatial location:
      C class scores
      4 distances (left, top, right, bottom)
      1 center-ness score
 -> decode boxes
 -> class score x center-ness
 -> NMS
```

Anchor 기반 RetinaNet은 위치마다 9개의 미리 정한 box를 놓고 각 anchor를 GT와 IoU로 matching한다. FCOS는 anchor scale·ratio, positive/negative IoU threshold와 anchor-GT IoU matrix가 필요 없다. COCO 80-class 기준 location당 output은 RetinaNet의 `9×(80+4)`가 아니라 `80+4+1`이다.

단순히 객체 내부의 모든 위치를 positive로 두면 중심에서 먼 위치가 부정확한 box를 높은 class score로 낼 수 있다. FCOS는 위치가 객체 중심에 가까운 정도를 나타내는 **center-ness**를 별도 branch로 예측하고 inference score에 곱한다. 이 branch는 COCO minival AP를 `33.5→37.1`로 크게 높였다.

FPN은 두 가지 문제를 해결한다. 각 level이 담당하는 regression distance 범위를 제한해 scale을 분리하고, 겹친 GT가 한 위치를 동시에 positive로 만드는 ambiguity를 줄인다. FPN을 쓰면 best possible recall은 `95.55→98.40%`, 서로 다른 category box 사이의 ambiguous positive 비율은 `17.84→3.75%`로 감소했다.

최종 single-model, single-scale 결과는 ResNet-101-FPN `41.5 AP`, ResNeXt-64x4d-101-FPN 기본 `43.2 AP`, 여러 개선을 포함하면 `44.7 AP`다. 다만 논문은 실제 FPS/latency, parameter, peak memory를 보고하지 않는다.

## 왜 anchor를 제거하려 했는가

### Anchor는 reference이자 training sample

Faster R-CNN과 RetinaNet에서 anchor는 미리 정한 scale/ratio의 reference box다. 각 anchor는 GT와 IoU로 positive, ignore, negative가 결정되고 box offset을 회귀한다.

Anchor 방식에는 다음 비용이 있다.

1. Scale, aspect ratio, anchor 수를 dataset마다 설계해야 한다.
2. Positive/negative IoU threshold가 AP에 민감하다.
3. High recall을 위해 매우 많은 anchor를 dense하게 놓는다.
4. Training마다 anchor-GT IoU와 matching을 계산한다.
5. 대부분의 anchor가 negative라 class imbalance가 심해진다.

논문은 짧은 변 800 image의 FPN에 180k개가 넘는 anchor가 생길 수 있다고 지적한다. FCOS는 feature location 자체를 sample로 사용해 이 축을 없앤다.

### 기존 anchor-free의 어려움

YOLOv1은 객체 중심 근처의 제한된 위치만 box를 예측해 recall이 낮았다. DenseBox 계열은 box 내부의 모든 위치에서 네 변 거리를 회귀했지만, 서로 겹친 일반 객체에서 한 위치가 어느 GT를 예측해야 하는지 ambiguous했다. CornerNet은 corner pair grouping과 embedding 같은 복잡한 post-processing이 필요했다.

FCOS의 해법은 다음 조합이다.

- 모든 inside-box location을 활용해 recall 확보
- FPN level range로 scale과 overlap ambiguity 분리
- center-ness로 중심에서 먼 low-quality prediction 억제

## Location을 input 좌표로 매핑하기

Stride `s`인 feature map `F_i in R^{H×W×C}`의 index `(x,y)`는 input image의 다음 좌표에 대응한다.

```math
(x_s,y_s)=\left(\left\lfloor\frac{s}{2}\right\rfloor+xs,
\left\lfloor\frac{s}{2}\right\rfloor+ys\right).
```

즉 feature cell receptive field의 중심 근처를 sample point로 쓴다. Modern implementation에서는 흔히 `(x+0.5)s,(y+0.5)s`로 표현하며 짝수 stride에서는 같은 의미다.

GT box를 왼쪽 위 `(x_0,y_0)`, 오른쪽 아래 `(x_1,y_1)`이라 하자. Location `(x_s,y_s)`가 box 내부면 네 target은:

```math
\begin{aligned}
l^*&=x_s-x_0, & t^*&=y_s-y_0,\\
r^*&=x_1-x_s, & b^*&=y_1-y_s.
\end{aligned}
```

모두 양수다. Prediction `(l,t,r,b)`를 box로 decode하면:

```math
(\hat x_0,\hat y_0,\hat x_1,\hat y_1)
=(x_s-l,\ y_s-t,\ x_s+r,\ y_s+b).
```

Anchor offset과 달리 width/height log ratio를 예측하지 않는다. 현재 point에서 경계까지 실제 거리라는 해석 가능한 target이다.

## Positive assignment

### 단일 level의 기본 규칙

- Location이 어떤 GT box 내부: positive, 그 GT class가 target
- 어느 GT에도 속하지 않음: background
- 여러 GT 내부: 면적이 가장 작은 GT를 선택

작은 GT를 선택하는 이유는 큰 box와 작은 box가 겹칠 때 작은 객체의 supervision이 큰 객체에 묻히는 것을 줄이기 위해서다.

### FPN level range

FCOS는 `P3-P7`을 사용하며 stride는 `{8,16,32,64,128}`이다. 각 location의 target에서:

```math
d_{max}=\max(l^*,t^*,r^*,b^*)
```

를 계산하고 level별 범위에 들어올 때만 positive로 둔다.

| Level | stride | regression range |
| --- | ---: | ---: |
| P3 | 8 | `[0,64]` |
| P4 | 16 | `[64,128]` |
| P5 | 32 | `[128,256]` |
| P6 | 64 | `[256,512]` |
| P7 | 128 | `[512,∞)` |

논문 표기에서는 `m2=0,m3=64,...,m6=512,m7=∞`이고, 범위 밖 location을 negative로 만든다. 경계값 포함 convention은 implementation과 일치시켜야 한다.

이 assignment는 anchor IoU는 없지만 hand-designed scale range는 남는다. 따라서 "hyperparameter-free detector"라기보다 **anchor-shape와 IoU matching hyperparameter를 제거한 detector**라고 말하는 것이 정확하다.

## Network architecture

### Backbone과 FPN

P3-P5는 backbone C3-C5에 FPN top-down/lateral connection을 적용해 만든다. P6, P7은 각각 앞 level에 stride-2 convolution을 적용한다. 모든 level은 256 channel이다.

논문 기본 FCOS는 RetinaNet과 매우 비슷한 tower를 사용해 anchor 유무를 공정하게 비교한다.

### Classification tower

```text
P_l [B,256,H_l,W_l]
 -> 4 x (3x3 conv, 256 channels, ReLU; GN in reported strong setup)
 -> 3x3 conv, C class logits
```

COCO에서는 `C=80` independent sigmoid score를 낸다. Classification loss는 RetinaNet의 focal loss다.

### Regression tower

별도의 4개 `3×3,256-channel` conv를 거쳐 4-channel `(l,t,r,b)`를 낸다. Target이 양수이므로 raw output `x`를:

```math
\exp(s_i x)
```

로 바꾼다. `s_i`는 FPN level별 trainable scalar다. Head convolution은 level 간 공유하지만, level마다 담당 distance scale이 다르므로 scalar가 output scale을 보정한다.

후속 개선 row의 "Normalization"은 target을 level stride로 나누는 방식이며 원 논문의 기본 `exp(s_i x)`와 구분해야 한다.

### Center-ness branch

원 논문 그림의 기본 구성에서는 classification tower 옆에 single conv layer로 center-ness logit 하나를 낸다. 제출 후 실험은 regression tower에 붙이는 것이 조금 더 좋음을 보였고 최종 개선 표에는 이 변경이 포함된다.

Head weight는 P3-P7에서 공유된다. FPN level마다 별도 tower를 복제하지 않는다.

## Loss function

Center-ness를 잠시 제외한 기본 detection loss는:

```math
L=\frac{1}{N_{pos}}\sum_{x,y}L_{cls}(p_{x,y},c^*_{x,y})
+\frac{\lambda}{N_{pos}}\sum_{x,y}\mathbb{1}[c^*_{x,y}>0]
L_{reg}(t_{x,y},t^*_{x,y}),
```

여기서:

- `L_cls`: focal loss
- `L_reg`: UnitBox의 IoU loss
- `lambda=1`
- `N_pos`: 모든 FPN level의 positive location 수
- regression은 positive location만 계산

IoU loss는 좌표별 독립 Smooth L1과 달리 네 경계가 만드는 box 전체의 overlap을 최적화한다. UnitBox 계열의 대표 형태는 `-log(IoU)`이며, 제공된 FCOS 본문은 이를 참조해 사용한다고 명시한다. 재현에서는 사용 code version의 IoU/GIoU 구현과 epsilon을 확인해야 한다. 논문의 후속 개선 row에서는 GIoU로 교체해 AP를 더 높인다.

Center-ness branch에는 positive location에서 binary cross-entropy를 추가한다.

## Center-ness

### Target 정의

```math
c^*=\sqrt{
\frac{\min(l^*,r^*)}{\max(l^*,r^*)}
\cdot
\frac{\min(t^*,b^*)}{\max(t^*,b^*)}
}.
```

Box 중심에서는 좌우 거리와 상하 거리가 비슷해 `c*=1`에 가깝다. 한 변에 가까워지면 한 ratio가 0으로 가므로 center-ness도 0으로 감소한다. Square root는 감소가 너무 빠르지 않게 한다.

### 수치 예시

한 location의 target이:

```text
(l*, t*, r*, b*) = (20, 10, 60, 30)
```

이면:

```math
c^*=\sqrt{\frac{20}{60}\cdot\frac{10}{30}}
=\sqrt{\frac{1}{9}}=\frac{1}{3}.
```

이 위치가 class probability 0.9를 내더라도 inference ranking score는 논문 설명대로 `0.9×1/3=0.3` 수준으로 낮아진다. 중심에서 먼 위치의 부정확한 box가 NMS 전에 위로 올라오는 것을 막는다.

### 왜 predicted box에서 계산하면 안 되는가

Regression output `(l,t,r,b)`에서도 같은 식을 계산할 수 있어 보인다. 그러나 Table 4에서 regression vector로 계산한 center-ness는 `33.5 AP`로 개선이 없었고, 별도 learned branch는 `37.1 AP`였다. Regression 값의 좌우 대칭성이 실제 localization quality와 항상 일치하지 않으며, 학습된 confidence가 더 유용했다.

### Center sampling과의 차이

기본 FCOS는 box 내부의 모든 valid-range location을 positive로 쓰고 center-ness는 inference score를 낮춘다. 후속 "center sampling"은 아예 GT 중심 주변만 positive로 제한한다. 둘은 같은 것이 아니며 결합하면 성능이 더 좋아졌다.

## Best Possible Recall과 ambiguity

### Best Possible Recall(BPR)

BPR은 GT 하나가 최소 한 training sample에라도 할당될 수 있는지 보는 upper-bound 진단 지표다.

| Method | BPR |
| --- | ---: |
| RetinaNet, official low-quality match >=0.4 | 90.92% |
| RetinaNet, 모든 low-quality match | 99.23% |
| FCOS, P4 only | 95.55% |
| FCOS, FPN | 98.40% |

FCOS는 anchor가 없어도 FPN 위치가 거의 모든 GT 내부에 있어 높은 coverage를 얻는다. 그러나 BPR 98.4%가 실제 recall 98.4%라는 뜻은 아니다. Model이 그 upper bound를 달성한다는 보장이 없다.

### Ambiguous locations

FPN 없이 P4만 쓰면 positive 중 여러 GT에 속하는 비율이 `23.16%`, 서로 다른 category만 세면 `17.84%`다. FPN range로 scale을 나누면 각각 `7.14%`, `3.75%`로 줄어든다.

같은 category의 겹침은 어느 box를 택해도 class target이 같고, 겹치지 않은 다른 location이 각 instance를 예측할 수 있어 덜 치명적이다. 서로 다른 category의 overlap은 A class를 예측하면서 B box를 회귀하는 불일치가 생길 수 있어 더 중요하다.

Inference detection 중 ambiguous location에서 나온 비율은 전체 `2.3%`, 서로 다른 category만 보면 `1.5%`였다. Ambiguity가 0은 아니지만 FPN과 최소-area rule로 실제 영향이 제한된다는 근거다.

## Batch=1 tensor shape 계산

### 800×1024 입력

논문 Figure 2와 동일한 `800×1024`, `B=1`, 80 classes를 사용한다.

| Level | feature | locations | class output | reg output | center output |
| --- | ---: | ---: | ---: | ---: | ---: |
| P3 | `1×256×100×128` | 12,800 | `1×80×100×128` | `1×4×100×128` | `1×1×100×128` |
| P4 | `1×256×50×64` | 3,200 | `1×80×50×64` | `1×4×50×64` | `1×1×50×64` |
| P5 | `1×256×25×32` | 800 | `1×80×25×32` | `1×4×25×32` | `1×1×25×32` |
| P6 | `1×256×13×16` | 208 | `1×80×13×16` | `1×4×13×16` | `1×1×13×16` |
| P7 | `1×256×7×8` | 56 | `1×80×7×8` | `1×4×7×8` | `1×1×7×8` |

총 location은 `17,064`다.

### Output payload

```math
\begin{aligned}
\text{class} &:17{,}064\cdot80=1{,}365{,}120,\\
\text{box} &:17{,}064\cdot4=68{,}256,\\
\text{center} &:17{,}064.
\end{aligned}
```

FP16 payload는 각각 약 `2.60 MiB`, `0.13 MiB`, `0.033 MiB`다. 같은 grid의 RetinaNet은 anchor 9개 때문에 class/box output이 약 9배다. FCOS의 "9× fewer output variables" 주장이 이 tensor에서 직접 보인다.

### Box decode 예시

P3 stride 8의 feature index `(x=10,y=20)`은 input에서 대략 `(84,164)`다. Prediction이 `(l,t,r,b)=(20,30,40,50)`이면:

```text
box = (84-20, 164-30, 84+40, 164+50)
    = (64, 134, 124, 214)
```

별도 anchor width/height나 exp decode가 필요하지 않는다. 단, network raw regression에는 positive 보장을 위해 `exp(s_i x)`가 이미 적용되어 있다.

## Parameter, MAC, activation memory

### Head parameter: 리뷰어 계산

두 tower의 4개 `3×3,256→256` conv:

```math
2\cdot4\cdot3\cdot3\cdot256\cdot256=4{,}718{,}592.
```

COCO output conv:

```math
3\cdot3\cdot256\cdot(80+4+1)=195{,}840.
```

합계는 bias/GN을 제외해 `4,914,432 weights`, FP16 약 `9.37 MiB`, FP32 약 `18.75 MiB`다. Level별 scalar 5개는 무시 가능한 크기다. Backbone과 FPN은 제외했다.

### Head MAC: 리뷰어 계산

17,064 locations에서 ideal MAC은:

- 8개 256-channel tower conv: 약 `80.5G MACs`
- class/reg/center output conv: 약 `3.34G MACs`
- 총 약 `83.9G MACs`

RetinaNet보다 output conv가 작아 계산을 줄이지만, 대부분은 두 4-layer tower에 남아 있다. Anchor-free가 곧 head-free 또는 ultra-light라는 뜻은 아니다.

### Activation memory

Pyramid 전체의 256-channel intermediate 한 세트는:

```math
17{,}064\cdot256=4{,}368{,}384\ \text{elements}
```

FP16 약 `8.33 MiB`다. 두 tower의 4개 layer output을 training용으로 모두 보존하면 단순 payload가 약 `66.7 MiB`다. FCOS는 final output memory를 9배 줄이지만 tower intermediate는 RetinaNet과 거의 같다.

논문은 parameter, MAC, peak memory, p50/p95 latency를 직접 보고하지 않는다. 위 값은 reviewer calculation이며 실제 peak에는 FPN/backbone, GN, gradient, allocator workspace가 더해진다.

## 구현 pseudocode

```python
def assign_fcos_targets(points_by_level, gt_boxes, gt_labels):
    targets = []
    ranges = [(0,64), (64,128), (128,256), (256,512), (512, INF)]
    for points, (low, high) in zip(points_by_level, ranges):
        # distances: [num_points, num_gt, 4]
        l = points.x[:, None] - gt_boxes.x0[None, :]
        t = points.y[:, None] - gt_boxes.y0[None, :]
        r = gt_boxes.x1[None, :] - points.x[:, None]
        b = gt_boxes.y1[None, :] - points.y[:, None]
        distances = stack(l, t, r, b)

        inside = distances.min(-1) > 0
        in_range = (distances.max(-1) >= low) & (distances.max(-1) <= high)
        valid = inside & in_range

        area = gt_boxes.area[None, :].masked_fill(~valid, INF)
        matched = area.argmin(dim=1)       # smallest-area valid GT
        positive = area.min(dim=1) < INF
        targets.append(make_targets(matched, positive, distances, gt_labels))
    return targets


def fcos_head(feature, level):
    cls_feat = classification_tower(feature)
    reg_feat = regression_tower(feature)
    cls_logits = class_predictor(cls_feat)             # [B,C,H,W]
    distances = exp(level_scale[level] * reg_predictor(reg_feat))
    center_logits = center_predictor(cls_feat)          # original paper layout
    return cls_logits, distances, center_logits


def fcos_loss(outputs, targets):
    pos = targets.labels > 0
    cls = focal_loss(outputs.cls, targets.one_hot).sum()
    reg = iou_loss(decode(outputs.dist[pos]), targets.box[pos]).sum()
    ctr_target = sqrt(
        min(l, r) / max(l, r) * min(t, b) / max(t, b)
    )
    ctr = bce_with_logits(outputs.center[pos], ctr_target).sum()
    return (cls + reg + ctr) / max(1, pos.sum())
```

Distributed training에서는 `N_pos`를 GPU별이 아니라 all-reduce한 평균/합계 convention에 맞춰야 single-GPU와 gradient scale이 일치한다.

## Training과 inference

### 기본 training

- COCO trainval35k 학습, minival ablation, test-dev 최종
- ResNet-50 기본 backbone, ImageNet pretrained
- SGD 90k iterations, total batch 16
- LR 0.01, 60k와 80k에서 10배 감소
- momentum 0.9, weight decay 0.0001
- 기본 input: short side 800, long side <=1333
- 새 layer는 RetinaNet과 같은 prior initialization
- strong setup에서 head final predictor를 제외하고 GroupNorm

최종 state-of-the-art 비교는 short side를 640-800으로 random scale하고 180k iterations를 사용한다. 기본 90k minival 결과와 recipe가 다르다.

### Inference

논문은 RetinaNet과 같은 post-processing hyperparameter를 사용한다.

- class confidence threshold 0.05
- level별 top-scoring candidate 제한
- box decode
- level merge
- NMS, 기본 비교 0.5; FCOS tuned experiment는 0.6

Center-ness를 class score에 곱해 ranking/NMS score로 사용한다. FCOS는 proposal-free지만 NMS-free는 아니다.

## Ablation 결과

### Center-ness

| 설정 | AP | AP50 | AP75 | AP small | AP medium | AP large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 없음 | 33.5 | 52.6 | 35.2 | 20.8 | 38.5 | 42.6 |
| predicted regression에서 계산 | 33.5 | 52.4 | 35.1 | 20.8 | 37.8 | 42.8 |
| learned branch | 37.1 | 55.9 | 39.8 | 21.3 | 41.0 | 47.8 |

AP `+3.6`의 큰 개선이다. 특히 medium/large와 AP75가 좋아져 low-quality high-score box 억제라는 설명과 일치한다.

### 순수 FCOS 구성

동일 조건의 appendix ablation:

| 구성 | AP |
| --- | ---: |
| RetinaNet, 1 anchor/location | 32.5 |
| RetinaNet, 9 anchors/location | 35.7 |
| pure FCOS on C5-derived pyramid | 35.7 |
| P5로 P6/P7 생성 | 35.8 |
| + GroupNorm | 36.3 |
| + level scalar | 36.4 |
| + IoU loss | 36.6 |

Pure FCOS가 9-anchor RetinaNet과 동등하면서 output은 약 9배 적다. 최종 성능 전체를 anchor 제거 하나의 효과로 돌리면 안 되고 GN, scalar, IoU loss와 schedule을 구분해야 한다.

### 개선 stack

ResNet-50 minival에서 base strong FCOS `37.1 AP` 이후:

- center-ness를 regression branch로 이동: `37.4`
- center sampling: `38.1`
- GIoU: `38.3`
- regression normalization: `38.6`

최종 `44.7 AP` 모델은 이 개선들을 포함한다. 원 제출판의 가장 순수한 FCOS와 수정된 strong recipe를 혼동하지 않아야 한다.

## 주요 COCO 결과

Single-model, single-scale test-dev:

| Method | Backbone | AP | AP50 | AP75 | AP small | AP medium | AP large |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RetinaNet | ResNet-101-FPN | 39.1 | 59.1 | 42.3 | 21.8 | 42.7 | 50.2 |
| FCOS | ResNet-101-FPN | 41.5 | 60.7 | 45.0 | 24.4 | 44.8 | 51.6 |
| FCOS | ResNeXt-32x8d-101-FPN | 42.7 | 62.2 | 46.1 | 26.0 | 45.6 | 52.6 |
| FCOS | ResNeXt-64x4d-101-FPN | 43.2 | 62.8 | 46.6 | 26.5 | 46.2 | 53.3 |
| FCOS + improvements | ResNeXt-64x4d-101-FPN | 44.7 | 64.1 | 48.4 | 27.6 | 47.5 | 55.6 |

같은 ResNet-101-FPN에서 RetinaNet보다 `+2.4 AP`, small AP는 `+2.6`이다. 논문은 FPS/latency table을 제공하지 않으므로 accuracy가 높고 output이 적다는 사실만으로 실제 장치에서 더 빠르다고 단정할 수 없다.

## FCOS를 RPN으로 사용한 실험

Class-agnostic FCOS head를 Faster R-CNN proposal generator로 바꾸면:

| Proposal method | samples | AR100 | AR1k |
| --- | ---: | ---: | ---: |
| RPN + FPN + GN | 약 200k | 44.7 | 56.9 |
| FCOS, center-ness 없음 | 약 66k | 48.0 | 59.3 |
| FCOS + center-ness | 약 66k | 52.8 | 60.3 |

Anchor 수가 적은 정도가 아니라 class-agnostic proposal recall도 개선되었다. Location-based regression이 two-stage proposal에도 적용 가능한 generic formulation이라는 증거다.

## 작은 객체 분석

FCOS는 작은 객체에서 anchor mismatch를 피하고 box 내부의 여러 location을 positive로 사용할 수 있다. FPN P3 stride 8과 `[0,64]` range가 작은 객체를 담당한다.

장점:

- 정해진 anchor shape와 작은 GT의 IoU가 낮아 positive가 사라지는 문제 감소
- GT 내부의 가능한 여러 location이 regressor supervision에 참여
- FPN으로 높은 spatial resolution 유지

한계:

- 객체가 stride보다 너무 작아 feature point를 하나도 포함하지 않으면 BPR에서 누락
- P3보다 이른 backbone downsampling에서 detail이 사라질 수 있음
- center-ness는 작은 box에서 positive 영역을 더 강하게 집중시킬 수 있음
- Crowd scene의 same-scale overlap ambiguity는 완전히 사라지지 않음

논문 final result의 `27.6 AP_S`는 강하지만 640-800 scale jitter와 강한 backbone이 포함된다. 작은 객체 이득을 anchor-free 하나로만 해석하면 안 된다.

## NMS와 score quality

FCOS class score는 category confidence이지 localization IoU의 직접 예측값이 아니다. Box 경계 근처의 location도 같은 class target 1을 받으므로 class score만 높고 IoU가 낮은 box가 생길 수 있다. Center-ness가 이를 보정한다.

```text
classification: "이 위치는 고양이 내부인가?"
center-ness:    "이 위치가 좋은 고양이 box를 만들기 쉬운가?"
regression:     "네 경계까지 거리는 얼마인가?"
```

최종 ranking score가 localization quality와 더 잘 상관되면 NMS가 좋은 box를 먼저 보존한다. 그러나 one-to-one prediction constraint가 아니므로 중복 제거 자체는 여전히 NMS가 담당한다.

## 장점과 핵심 기여

1. Generic object detection을 간단한 per-location FCN 문제로 정식화했다.
2. Anchor scale/ratio와 anchor-GT IoU matching을 제거했다.
3. Output variable과 matching memory를 크게 줄였다.
4. FPN range assignment로 recall과 overlap ambiguity를 해결했다.
5. Center-ness로 classification-localization quality mismatch를 효과적으로 완화했다.
6. One-stage detector뿐 아니라 RPN 대체로도 높은 recall을 보였다.

## 한계와 비판적 관점

### 1. 완전히 hyperparameter-free는 아니다

FPN regression range, center sampling radius(개선판), NMS threshold, score threshold가 남는다. Anchor-specific parameter가 없다는 표현이 더 정확하다.

### 2. Assignment가 GT box 내부 규칙에 의존한다

같은 scale·다른 class 객체가 많이 겹치면 최소-area heuristic이 불완전하다. Crowd/occlusion에서 instance identity를 명시적으로 modeling하지 않는다.

### 3. Center-ness는 별도 quality proxy다

GT box geometry에서 만든 target이며 실제 predicted IoU와 완전히 같지 않다. 후속 quality-aware classification은 class와 IoU quality를 더 직접 결합한다.

### 4. NMS가 남는다

Proposal과 anchor는 없지만 dense duplicate prediction, top-k, class-wise NMS는 필요하다.

### 5. Tower compute는 여전히 크다

Final output은 작아졌지만 8개의 256-channel 3×3 conv가 head MAC와 training activation 대부분을 차지한다.

### 6. 시스템 지표 부족

논문은 parameter, FLOPs, FPS, peak memory, p50/p95, 전력·온도를 제공하지 않는다. "Faster training/testing" 주장은 anchor matching 제거의 방향성은 타당하지만 device latency 증거는 없다.

## 자주 헷갈리는 지점

### Anchor-free는 proposal-free와 같은가

FCOS one-stage는 둘 다 맞다. 하지만 FCOS formulation을 RPN으로 쓰면 그 출력은 second-stage proposal이 된다.

### 모든 box 내부 pixel이 positive인가

Feature-map location 중 GT 내부이면서 해당 FPN level regression range를 만족하는 위치다. 실제 input의 모든 pixel을 직접 sample하는 것은 아니다.

### Center-ness가 training positive를 제거하는가

원 기본 방식에서는 아니다. 모든 valid location을 positive로 유지하고 center-ness BCE를 학습한 뒤 inference score를 낮춘다. Center sampling은 별도 후속 개선이다.

### Center-ness는 object center heatmap인가

정확한 center point만 1인 Gaussian heatmap이 아니다. `(l,r,t,b)` ratio로 box 내부 전체에 연속 target을 만든다.

### FCOS는 softmax를 쓰는가

RetinaNet처럼 class별 independent sigmoid와 focal loss를 쓴다. Background는 모든 class가 0이다.

### IoU threshold가 전혀 없는가

Training assignment에는 anchor-GT IoU threshold가 없다. Evaluation AP와 NMS에는 IoU threshold가 여전히 존재한다.

## 온디바이스 관점

### 유리한 점

- Anchor generation과 anchor-GT matching 제거
- Final output tensor가 RetinaNet 대비 약 9배 작음
- RoIAlign 없는 fully convolutional static graph
- 대부분의 operator가 Conv/GN/ReLU/Sigmoid/Exp라 accelerator mapping 가능

### 주의할 점

- `exp`와 GroupNorm 지원이 NPU마다 다르다.
- GN이 CPU fallback이면 작은 head conv 절감보다 synchronization 비용이 클 수 있다.
- Level별 scale과 decode, top-k, NMS가 후처리에 남는다.
- 4+4 tower의 dense 3×3 conv가 절대 compute를 지배한다.
- P3 high-resolution activation이 peak memory에 크게 기여한다.

### 실험 설계

동일 MobileNet-FPN에서 RetinaNet과 FCOS를 비교할 때:

- input resolution과 FPN channel 동일
- tower depth와 normalization 동일
- 동일 confidence/top-k/NMS implementation
- AP/AP_S뿐 아니라 output tensor, peak memory, p50/p95 측정
- anchor matching은 training time/memory에서 별도 측정

FCOS가 output memory는 확실히 줄이지만, inference latency가 얼마나 줄지는 final conv가 전체 시간에서 차지하는 비율과 post-processing 구현에 달려 있다.

## 재현 체크리스트

- [ ] 제공 PDF v5의 기본과 post-submission improvements를 구분한다.
- [ ] P3-P7 stride `{8,16,32,64,128}`와 256 channel을 확인한다.
- [ ] Location 좌표가 `(x+0.5)s,(y+0.5)s`와 일치하는지 검증한다.
- [ ] `(l,t,r,b)`의 좌표 convention과 inside test를 맞춘다.
- [ ] Regression range `[0,64,128,256,512,∞]`를 구현한다.
- [ ] Multiple GT에서 minimum-area assignment를 확인한다.
- [ ] Class는 C sigmoid + focal loss, background는 all-zero target이다.
- [ ] Regression은 positive만 IoU loss를 적용한다.
- [ ] Level별 trainable scalar와 exp 적용 위치를 확인한다.
- [ ] Center-ness target의 두 min/max ratio와 square root를 맞춘다.
- [ ] Original cls-branch center와 improved reg-branch center를 구분한다.
- [ ] GN 사용 여부와 gradient clipping protocol을 기록한다.
- [ ] Base 90k와 final 180k scale-jitter schedule을 구분한다.
- [ ] NMS 0.5/0.6, threshold, top-k를 명시한다.
- [ ] AP/AP50/AP75/AP_S/M/L과 BPR/ambiguity를 함께 검증한다.
- [ ] Parameter/MAC/activation은 reviewer 계산과 profiler 측정을 구분한다.
- [ ] Target device에서 batch=1 peak memory, p50/p95, fallback, 전력·온도를 측정한다.

## 로드맵에서의 연결

- **Faster R-CNN/RPN**: anchor와 proposal cascade의 기준점
- **FPN**: scale별 semantic feature와 FCOS range assignment의 기반
- **RetinaNet**: 같은 FPN/tower/focal loss를 사용해 anchor 제거 효과를 비교할 수 있는 가장 가까운 baseline
- **ATSS**: anchor-based와 anchor-free의 차이보다 adaptive sample assignment가 중요할 수 있음을 보임
- **YOLOX/RTMDet**: 실시간 anchor-free dense detector의 engineering 방향
- **DETR 계열**: dense NMS 방식 대신 Hungarian one-to-one set prediction으로 중복 자체를 다룸

최종 프로젝트의 camera-every-frame detector 후보로 FCOS류가 적합한 이유는 fully convolutional static graph와 작은 output이다. 그러나 모바일에서는 8-layer tower를 줄이고, supported normalization을 쓰며, NMS를 device 안에서 끝내는 설계가 함께 필요하다.

## 최종 평가

FCOS의 가장 큰 기여는 anchor를 제거한 사실보다, detection을 segmentation과 같은 location-wise prediction 언어로 단순화하면서도 generic object overlap과 scale 문제를 실제로 해결했다는 데 있다. FPN range는 assignment ambiguity를 줄이고, center-ness는 class confidence와 localization quality 사이의 간극을 메운다.

Anchor-free는 자동으로 더 빠르거나 더 정확하다는 보장이 아니다. 이 논문에서도 강한 결과는 FPN, focal loss, GN, center-ness, 긴 schedule과 후속 개선의 결합이다. 다만 `800×1024`에서 final output이 RetinaNet보다 약 9배 작고 anchor matching이 사라지는 것은 명확한 시스템 이점이다. 온디바이스 연구에서는 이 이점을 tower compute·P3 activation·NMS p95와 함께 측정할 때 FCOS의 실제 가치를 판단할 수 있다.

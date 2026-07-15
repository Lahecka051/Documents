# 43. Focal Loss for Dense Object Detection

## 논문 정보

- 원본 파일: `43_RetinaNet_Focal_Loss.pdf`
- 제목: **Focal Loss for Dense Object Detection**
- 저자: Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár
- 발표: ICCV 2017
- 공개: arXiv:1708.02002
- 링크: [https://arxiv.org/abs/1708.02002](https://arxiv.org/abs/1708.02002)
- 핵심 키워드: focal loss, RetinaNet, one-stage detector, class imbalance, hard examples, dense anchors, FPN, sigmoid classification

## 한눈에 보는 요약

이 논문은 one-stage detector가 two-stage detector보다 정확도가 낮았던 주된 원인을 architecture보다 **극단적인 foreground-background class imbalance**에서 찾는다. Dense detector는 한 image에서 `10^4-10^5`개의 후보 위치를 평가하지만 실제 객체와 대응하는 positive는 매우 적다. 쉬운 background 각각의 cross-entropy는 작더라도 수가 너무 많아 합계 loss와 gradient를 지배한다.

Focal Loss는 cross-entropy에 `(1-p_t)^gamma`를 곱해 잘 분류된 sample의 기여를 연속적으로 낮춘다.

```math
\mathrm{FL}(p_t)=-\alpha_t(1-p_t)^\gamma\log p_t.
```

- `p_t`가 작아 틀린 sample: modulating factor가 1에 가까워 학습 신호를 유지
- `p_t`가 1에 가까운 쉬운 sample: factor가 0으로 빠르게 감소
- `gamma=0`: alpha-balanced cross-entropy로 복귀
- 논문의 기본값: `gamma=2`, `alpha=0.25`

이를 검증하기 위해 저자들은 ResNet-FPN 위에 classification과 box regression subnet을 붙인 단순 one-stage detector **RetinaNet**을 설계한다. RetinaNet은 FPN의 `P3-P7` 모든 위치에 9 anchors를 놓고, 각 anchor에 대해 80개 independent sigmoid class score와 4개 box offset을 예측한다. Proposal stage는 없고 focal loss를 약 100k anchor 전체에 적용한다.

COCO test-dev에서 ResNet-101-FPN RetinaNet은 `39.1 AP`, ResNeXt-101-FPN은 `40.8 AP`를 기록했다. 당시 Faster R-CNN 계열과 one-stage detector를 모두 넘어섰다. 논문의 핵심 결론은 "one-stage가 본질적으로 부정확하다"가 아니라, **dense imbalance에 맞는 objective를 쓰면 단순한 one-stage도 높은 정확도를 낼 수 있다**는 것이다.

## Detection의 class imbalance

### Dense one-stage detector

One-stage detector는 feature pyramid의 모든 위치와 anchor를 직접 classifying한다.

```text
image
 -> backbone + FPN
 -> every level, every location, every anchor
 -> class logits + box offsets
 -> decode + threshold + NMS
```

대부분의 anchor는 어떤 객체와도 겹치지 않는 쉬운 background다. 예를 들어 positive:negative가 `1:1000`이면 negative 하나의 loss가 positive보다 훨씬 작아도 천 개를 합친 값은 클 수 있다.

이 문제는 두 가지를 일으킨다.

1. 유용하지 않은 쉬운 negative 계산에 training capacity를 소비한다.
2. negative gradient가 rare positive를 압도해 초기 학습이 불안정하거나 degenerate model이 된다.

### Two-stage detector가 imbalance를 완화하는 방법

Faster R-CNN은 loss 함수가 특별해서가 아니라 cascade와 sampling으로 분포를 바꾼다.

1. RPN이 사실상 무한한 위치를 1-2천 proposal로 줄인다.
2. Second stage는 positive:negative를 보통 1:3처럼 biased sampling한다.

즉 two-stage의 proposal은 쉬운 background를 대량 제거하는 hard filtering이고, RoI sampling은 implicit class balancing이다. RetinaNet은 이 역할을 loss의 continuous reweighting으로 수행한다.

### Focal Loss는 robust loss와 반대 방향이다

Huber 같은 robust loss는 큰 오차의 outlier 영향을 줄인다. Focal Loss는 반대로 잘 맞힌 inlier, 특히 easy negative의 영향을 줄인다. 어려운 sample을 버리는 것이 아니라 쉬운 sample의 weight를 낮춰 상대적으로 hard sample에 집중한다.

## Cross-entropy에서 Focal Loss까지

### Binary cross-entropy

Label `y in {+1,-1}`, positive probability `p`에 대해:

```math
\mathrm{CE}(p,y)=
\begin{cases}
-\log p,&y=+1,\\
-\log(1-p),&y=-1.
\end{cases}
```

Ground-truth class에 대한 확률을 하나로 쓰면:

```math
p_t=\begin{cases}
p,&y=+1,\\
1-p,&y=-1,
\end{cases}
\qquad
\mathrm{CE}(p_t)=-\log p_t.
```

이 표기는 positive/negative를 동일한 식으로 다룬다. `p_t`가 높을수록 정답을 확신한다.

### Alpha-balanced CE

Class frequency imbalance만 보정하려면 positive에 `alpha`, negative에 `1-alpha`를 둔다.

```math
\mathrm{CE}_\alpha(p_t)=-\alpha_t\log p_t.
```

하지만 alpha는 class만 구분한다. 쉬운 negative와 어려운 negative를 같은 weight로 취급한다. 논문 Table 1a에서 alpha-balanced CE의 최고 AP는 `31.1`이었다.

### Focal modulating factor

```math
\mathrm{FL}(p_t)=-\alpha_t(1-p_t)^\gamma\log p_t.
```

`gamma`는 focusing parameter다.

- `gamma=0`: factor가 1, CE와 같음
- `gamma`가 커질수록 easy sample이 더 빨리 감소
- hard example `p_t≈0`: factor가 거의 1
- easy example `p_t≈1`: factor가 거의 0

Alpha와 gamma는 역할이 다르다.

- `alpha_t`: positive와 negative class의 전체 중요도 균형
- `(1-p_t)^gamma`: 같은 class 안에서도 difficulty에 따른 균형

둘은 상호작용하므로 독립적으로 고르면 안 된다. 논문에서는 gamma가 커질수록 easy negative가 이미 억제되므로 최적 alpha도 낮아지는 경향이 있었다.

## 수치로 이해하는 focusing

Alpha를 잠시 빼고 `gamma=2`라 하자.

| `p_t` | CE `-log(p_t)` | focal factor `(1-p_t)^2` | FL |
| ---: | ---: | ---: | ---: |
| 0.10 | 2.3026 | 0.81 | 1.8651 |
| 0.50 | 0.6931 | 0.25 | 0.1733 |
| 0.90 | 0.1054 | 0.01 | 0.00105 |
| 0.99 | 0.01005 | 0.0001 | 0.0000010 |

`p_t=0.9`인 easy sample은 CE보다 100배 작고, `p_t≈0.968`이면 약 1000배 작아진다. 반면 `p_t=0.1`인 심하게 틀린 sample은 약 19%만 줄어 여전히 큰 gradient를 낸다.

Negative anchor에서 model이 background를 `0.99`로 맞혔다면 ground-truth 확률 `p_t=0.99`다. 수십만 개가 있어도 각각의 loss가 거의 0이 된다. 이 덕분에 RetinaNet은 RPN처럼 256 anchors를 sampling하지 않고 모든 anchor를 사용할 수 있다.

## Gradient 관점

Logit을 `x`, label을 `y in {+1,-1}`, `p_t=sigma(yx)`라 하면 alpha를 제외한 논문 Appendix의 derivative는:

```math
\frac{\partial\mathrm{FL}}{\partial x}
=y(1-p_t)^\gamma
\left(\gamma p_t\log p_t+p_t-1\right).
```

CE의 derivative는:

```math
\frac{\partial\mathrm{CE}}{\partial x}=y(p_t-1).
```

Focal Loss는 단지 loss graph를 보기 좋게 바꾸는 것이 아니라 easy sample의 gradient magnitude를 실제로 줄인다. 논문의 converged-model 분석에서 positive loss 분포는 gamma 변화에 비교적 덜 민감했지만, negative loss는 `gamma=2`일 때 극소수 hard negatives에 집중되었다.

실제 구현은 `sigmoid -> log`를 별도 연산으로 나누지 않고 logits에서 numerically stable하게 계산해야 한다. FP16에서 `p≈0` 또는 `p≈1`을 직접 log하면 underflow/overflow가 생길 수 있다.

## 초기 prior가 필요한 이유

일반적인 zero-bias sigmoid head는 초기 `p=0.5`를 낸다. Dense image의 수만 negative가 모두 foreground 0.5로 시작하면 첫 iteration의 loss가 매우 크고 불안정하다.

논문은 foreground prior `pi=0.01`이 되도록 final classification bias를 초기화한다.

```math
b=-\log\frac{1-\pi}{\pi}.
```

`pi=0.01`이면 `b≈-4.595`이고 `sigmoid(b)=0.01`이다. 이는 loss를 바꾸는 것이 아니라 model의 초기 상태를 background-heavy data 분포에 맞추는 것이다.

논문의 standard CE 실험은 이 초기화 없이 발산했고, prior initialization만 적용한 ResNet-50 RetinaNet CE가 `30.2 AP`를 얻었다. 따라서 focal loss 효과와 안정적인 prior initialization 효과를 구분해야 한다.

## RetinaNet 전체 구조

```text
image
 -> ResNet C3/C4/C5
 -> FPN P3/P4/P5 + strided P6/P7
 -> at every level:
      classification subnet -> H x W x (A*K)
      box subnet            -> H x W x (A*4)
 -> score threshold / top-k
 -> anchor decode
 -> merge levels
 -> class-wise NMS
```

### FPN P3-P7

모든 level은 256 channel이며 input 대비 stride는 `{8,16,32,64,128}`다.

- P3-P5: C3-C5에 top-down/lateral FPN 적용
- P6: C5에 `3×3 stride-2` conv
- P7: P6에 ReLU 후 `3×3 stride-2` conv

원 FPN과 달리 P2를 쓰지 않아 high-resolution compute를 줄이고, P7을 추가해 large object를 보완한다. P6도 단순 pooling이 아니라 learned strided conv다.

### Anchors

Level base size는 P3부터 P7까지 `32,64,128,256,512` pixels다. 각 level에서:

- ratios: `{1:2,1:1,2:1}`
- within-octave scales: `{2^0,2^(1/3),2^(2/3)}`
- 총 `A=9 anchors/location`

전체가 input 기준 약 `32-813` pixel scale을 덮는다.

### Anchor assignment

- 어떤 GT와 IoU `>=0.5`: 해당 GT/class positive
- 모든 GT와 IoU `[0,0.4)`: background
- IoU `[0.4,0.5)`: ignore
- anchor 하나는 최대 GT 하나에 할당

Positive anchor는 길이 `K` one-hot target을 가지지만 classification은 softmax 하나가 아니라 `K`개의 independent sigmoid binary classifier다. Background는 모든 class target이 0이다.

## Classification subnet

각 FPN level의 `[B,256,H_l,W_l]`에 다음을 적용한다.

```text
3x3 conv 256 -> 256 + ReLU
3x3 conv 256 -> 256 + ReLU
3x3 conv 256 -> 256 + ReLU
3x3 conv 256 -> 256 + ReLU
3x3 conv 256 -> A*K
sigmoid
```

COCO에서 `A=9`, `K=80`이므로 최종 channel은 `720`이다. Subnet parameter는 P3-P7 사이에 공유된다.

## Box regression subnet

Classification과 같은 4-layer 구조지만 parameter는 별개다. Final output은 `4A=36` channel이며 class-agnostic box offset을 예측한다. Box loss는 positive anchor에만 standard Smooth L1을 적용한다.

Classification과 regression subnet이 구조가 같다는 것이 weight도 공유한다는 뜻은 아니다. 서로 다른 task의 feature를 분리한다.

## Training objective와 normalization

Image별 classification focal loss는 ignore를 제외한 약 100k anchors 전체에 대해 합산하고 **positive로 할당된 anchor 수**로 나눈다. Total loss는:

```math
L=L_{focal}+L_{smooth\ L1}.
```

Box regression은 positive만 사용한다. Negative가 압도적으로 많지만 focal factor가 easy negative를 거의 0으로 만들기 때문에 total anchor 수로 normalize하지 않는다.

Positive가 없는 image에서 denominator를 어떻게 처리하는지는 구현상 `max(1,N_pos)` 같은 안전장치가 필요하다. 논문 문장만 복사해 0으로 나누면 안 된다.

## Batch=1 tensor shape 계산

### 800×1024 예시

`B=1`, COCO 80 classes, 9 anchors/location을 가정한다. 아래는 논문 architecture에 따른 **리뷰어 계산**이며 실제 padding에 따라 한 칸 차이가 날 수 있다.

| Level | feature shape | locations | anchors |
| --- | ---: | ---: | ---: |
| P3 | `1×256×100×128` | 12,800 | 115,200 |
| P4 | `1×256×50×64` | 3,200 | 28,800 |
| P5 | `1×256×25×32` | 800 | 7,200 |
| P6 | `1×256×13×16` | 208 | 1,872 |
| P7 | `1×256×7×8` | 56 | 504 |
| 합계 | - | 17,064 | 153,576 |

논문의 "약 100k anchors"는 image aspect ratio와 input scale에 따른 규모 표현이다. 이 예시에서는 약 154k다.

### Dense output memory

Classification logit 수:

```math
17{,}064\cdot9\cdot80=12{,}286{,}080.
```

- FP16 payload: 약 `23.43 MiB`
- FP32 payload: 약 `46.87 MiB`

Box output은:

```math
17{,}064\cdot9\cdot4=614{,}304
```

- FP16 약 `1.17 MiB`
- FP32 약 `2.34 MiB`

80-class dense classification tensor가 regression보다 약 20배 크다. Class 수가 커지는 open-vocabulary 방향에서는 이 `location×anchor×class` 축이 가장 먼저 memory 병목이 된다.

### Head intermediate activation

한 `256-channel` intermediate map을 pyramid 전체에서 합치면:

```math
17{,}064\cdot256=4{,}368{,}384\ \text{elements}
```

FP16 약 `8.33 MiB`다. Classification과 box branch의 4개 conv output을 training backward용으로 모두 보존한다고 단순 가정하면 8개가 약 `66.7 MiB`다. 이는 FPN/backbone, logits, gradient, optimizer state를 제외한 값이다. Inference engine은 buffer를 재사용할 수 있으므로 실제 peak는 profiler로 측정해야 한다.

## Parameter와 MAC 계산

### Head parameter: 리뷰어 계산

Classification subnet, bias 제외:

```math
4(3\cdot3\cdot256\cdot256)
+3\cdot3\cdot256\cdot720
=4{,}018{,}176.
```

Box subnet:

```math
4(3\cdot3\cdot256\cdot256)
+3\cdot3\cdot256\cdot36
=2{,}442{,}240.
```

합계 `6,460,416 weights`, FP16 약 `12.32 MiB`, FP32 약 `24.64 MiB`다. Level마다 반복 적용하지만 weight는 공유하므로 parameter 수는 다섯 배가 되지 않는다.

### Head MAC: 리뷰어 계산

`800×1024` 예시의 17,064 locations에서 ideal MAC은:

- classification subnet: 약 `68.6G MACs`
- box subnet: 약 `41.7G MACs`
- 합계: 약 `110.3G MACs`

Backbone과 FPN을 제외한 head만의 값이다. Classification final conv가 `256→720`이라 약 `28.3G MACs`를 차지한다. 따라서 RetinaNet은 one-stage라 pipeline은 단순하지만 head가 가볍다는 뜻은 아니다.

논문은 모델별 total FLOPs를 표로 주지 않고 M40 runtime을 보고한다. 위 값은 reviewer calculation이며 FLOP를 `2×MAC`으로 세는 도구에서는 숫자가 두 배로 표시된다.

## Inference와 NMS

논문의 inference 절차는 다음과 같다.

1. 모든 FPN level에서 sigmoid class score 계산
2. confidence `<0.05` 제거
3. 각 level에서 score 상위 최대 1,000 prediction만 box decode
4. 모든 level을 합침
5. NMS IoU threshold `0.5`

이 top-k는 dense output의 후처리 비용을 제한한다. Anchor마다 80 class score가 있으므로 "1,000 boxes"가 아니라 class-anchor prediction ranking이라는 구현 세부를 확인해야 한다.

RetinaNet은 proposal-free이지만 NMS-free는 아니다. Focal Loss는 중복 prediction을 억제하는 set constraint가 아니라 training imbalance를 해결한다. 같은 객체 주변의 여러 anchor가 높은 score를 낼 수 있어 class-wise NMS가 필요하다.

## 구현 pseudocode

```python
def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    # stable BCE-with-logits; do not compute log(sigmoid(x)) directly
    ce = binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    modulating = (1 - p_t).pow(gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return alpha_t * modulating * ce


def retinanet(image, targets=None):
    p3, p4, p5, p6, p7 = resnet_fpn(image)
    cls_all, box_all, anchor_all = [], [], []

    for feat, level in zip((p3, p4, p5, p6, p7), range(3, 8)):
        cls = shared_class_subnet(feat)       # [B, A*K, H, W]
        box = shared_box_subnet(feat)         # [B, A*4, H, W]
        anchors = make_anchors(level, 3_scales, 3_ratios)
        cls_all.append(flatten_anchor_class(cls))
        box_all.append(flatten_anchor_box(box))
        anchor_all.append(anchors)

    if targets is not None:
        labels, matched_gt = match_iou(anchor_all, targets,
                                       positive=0.5, negative=0.4)
        valid = labels != IGNORE
        positive = labels > BACKGROUND
        cls_loss = sigmoid_focal_loss(cls_all[valid], one_hot(labels[valid]))
        cls_loss = cls_loss.sum() / max(1, positive.sum())
        reg_loss = smooth_l1(box_all[positive], encode(matched_gt[positive]))
        reg_loss = reg_loss.sum() / max(1, positive.sum())
        return cls_loss + reg_loss

    candidates = []
    for cls, box, anchors in per_level_outputs():
        keep = sigmoid(cls) > 0.05
        keep = topk_by_score(keep, k=1000)
        candidates += decode(anchors[keep], box[keep])
    return classwise_nms(candidates, iou_threshold=0.5)
```

Classification final bias를 `-4.595` 부근으로 초기화하고, ignore anchor가 all-zero background로 잘못 들어가지 않도록 mask해야 한다.

## 실험 설정

- COCO trainval35k 학습, minival ablation, test-dev 최종 평가
- ImageNet pretrained ResNet-50/101-FPN
- 8 GPU synchronized SGD, 총 batch 16
- 기본 90k iterations
- LR 0.01, 60k와 80k에서 10배 감소
- momentum 0.9, weight decay 0.0001
- 기본 augmentation은 horizontal flip
- 기본 ablation image scale: 짧은 변 600
- loss: focal classification + Smooth L1 box regression

최종 800-scale 모델은 scale jitter와 1.5배 긴 training을 사용해 Table 1e의 모델보다 1.3 AP 높다. 단순히 backbone 이름과 input 800만 맞춰서는 최종 수치가 재현되지 않는다.

## Focal Loss ablation

### Gamma와 alpha

ResNet-50-FPN, 600 scale:

| gamma | alpha | AP | AP50 | AP75 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.75 | 31.1 | 49.4 | 33.0 |
| 0.5 | 0.50 | 32.9 | 51.7 | 35.2 |
| 1.0 | 0.25 | 33.7 | 52.0 | 36.2 |
| 2.0 | 0.25 | 34.0 | 52.5 | 36.5 |
| 5.0 | 0.25 | 32.2 | 49.6 | 34.8 |

`gamma=2`가 최고였지만 0.5-2 범위도 강하다. 너무 큰 gamma는 충분한 sample까지 억제해 성능이 떨어질 수 있다.

### CE 대비 개선

최적 alpha-balanced CE는 `31.1 AP`, 같은 network의 focal loss는 `34.0 AP`, 즉 `+2.9 AP`다. Architecture를 바꾼 비교가 아니라 objective만 바꾼 controlled result라는 점이 중요하다.

### OHEM 비교

ResNet-101-FPN에서 best OHEM은 `32.8 AP`, focal loss는 `36.0 AP`였다. OHEM은 hard sample을 discrete selection하고 easy sample을 완전히 버리며 batch size와 NMS threshold가 필요하다. Focal Loss는 모든 sample에 연속 weight를 주고 별도 mining pipeline이 없다.

### Anchor density

| scales/location | ratios | AP |
| ---: | ---: | ---: |
| 1 | 1 | 30.3 |
| 1 | 3 | 32.4 |
| 2 | 3 | 34.2 |
| 3 | 3 | 34.0 |
| 4 | 3 | 33.8 |

한 square anchor만으로도 30.3 AP이지만 6-9 anchors/location에서 약 4 AP 개선되고 이후 포화한다. Dense anchor 수를 계속 늘린다고 accuracy가 오르지 않는다.

## 주요 결과와 속도

### COCO test-dev

| Method | Backbone | AP | AP50 | AP75 | AP small | AP medium | AP large |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Faster R-CNN + FPN | ResNet-101-FPN | 36.2 | 59.1 | 39.0 | 18.2 | 39.0 | 48.2 |
| DSSD513 | ResNet-101-DSSD | 33.2 | 53.3 | 35.2 | 13.0 | 35.4 | 51.1 |
| RetinaNet | ResNet-101-FPN | 39.1 | 59.1 | 42.3 | 21.8 | 42.7 | 50.2 |
| RetinaNet | ResNeXt-101-FPN | 40.8 | 61.1 | 44.1 | 24.1 | 44.2 | 51.2 |

RetinaNet-101은 당시 가장 가까운 one-stage DSSD보다 `+5.9 AP`, 강한 two-stage보다도 높았다.

### Accuracy-speed trade-off

M40 GPU에서 논문 Table 1e:

| Backbone | short side | AP | AP small | time |
| --- | ---: | ---: | ---: | ---: |
| R50 | 400 | 30.5 | 11.2 | 64 ms |
| R50 | 600 | 34.3 | 16.2 | 98 ms |
| R50 | 800 | 35.7 | 18.9 | 153 ms |
| R101 | 400 | 31.9 | 11.6 | 81 ms |
| R101 | 600 | 36.0 | 17.4 | 122 ms |
| R101 | 800 | 37.8 | 20.2 | 198 ms |

ResNet-101-600 RetinaNet은 Faster R-CNN FPN의 172 ms보다 빠른 122 ms에서 같은 36.0급 AP를 보였다. 다만 이 표는 평균 latency이고 p50/p95, peak memory, power를 보고하지 않는다.

## 작은 객체 분석

RetinaNet은 FPN P3의 stride 8과 32-pixel base anchor를 사용해 small object를 처리한다. P2를 제외해 FPN 원형보다 finest resolution은 낮지만, 800-scale final model은 `21.8 AP_S`, ResNeXt model은 `24.1 AP_S`를 기록했다.

작은 객체에서 focal loss가 특히 유리한 이유는 다음과 같다.

- 작은 GT는 positive anchor 수가 적다.
- 주변의 background/near-miss anchor는 매우 많다.
- CE에서는 이 negative 합계가 rare positive를 압도한다.
- Focal Loss는 이미 background로 확신한 위치를 낮추고 boundary의 hard negative와 rare positive에 학습을 집중한다.

그러나 작은 객체 성능이 focal loss만의 결과는 아니다. FPN, input scale 800, anchor scale, scale jitter가 함께 작용한다. Table 1e에서 R101의 short side를 400→800으로 키우면 AP_S가 `11.6→20.2`로 크게 오르지만 latency도 `81→198 ms`로 증가한다.

## 장점과 핵심 기여

1. Dense one-stage detector의 accuracy gap을 class imbalance 관점에서 명확히 진단했다.
2. CE와 호환되는 한 줄의 modulating factor로 easy example gradient를 낮췄다.
3. Sampling/OHEM 없이 모든 anchor를 학습에 사용할 수 있게 했다.
4. Focal Loss의 gamma/alpha, loss 분포, OHEM 비교를 폭넓게 ablation했다.
5. RetinaNet으로 one-stage speed와 two-stage 이상 accuracy를 동시에 입증했다.
6. 이후 detection, segmentation, medical imaging 등 불균형 dense task의 표준 loss가 되었다.

## 한계와 비판적 관점

### 1. Focal Loss가 모든 imbalance를 해결하지는 않는다

Class frequency, object scale, annotation noise, localization quality imbalance는 별도 문제다. Alpha와 gamma도 dataset에 따라 tuning이 필요할 수 있다.

### 2. Anchor design이 남는다

9 anchors/location, IoU 0.4/0.5 threshold, scale/ratio가 필요하다. Focal Loss는 anchor matching의 수작업 규칙을 없애지 않는다.

### 3. Dense class output이 크다

80 classes에서 classification logit만 수십 MiB다. Category가 수천 개로 늘면 `A×K` final conv와 activation이 선형 증가한다.

### 4. NMS가 필요하다

Loss는 sample importance를 조절할 뿐 중복 box에 one-to-one constraint를 주지 않는다. Threshold, top-k와 class-wise NMS가 latency와 tail을 만든다.

### 5. 최종 결과의 recipe 의존

39.1 AP는 800 input, scale jitter, 1.5× schedule을 포함한다. 기본 ablation 34.0과 직접 비교할 때 training protocol 차이를 빼먹으면 안 된다.

### 6. 논문은 모바일 지표를 제공하지 않는다

M40 평균 runtime만 있고 peak memory, p95, energy, thermal throttling, operator fallback 데이터가 없다.

## 자주 헷갈리는 지점

### Focal Loss는 hard negative mining인가

효과는 hard example에 집중한다는 점에서 비슷하지만 sample을 discrete하게 선택하지 않는다. 모든 valid anchor의 loss를 계산한 뒤 easy sample weight를 연속적으로 낮춘다.

### Alpha와 gamma는 같은 역할인가

아니다. Alpha는 positive/negative class balance, gamma는 prediction difficulty를 조절한다.

### `p_t`는 항상 foreground probability인가

아니다. Ground-truth class의 확률이다. Negative sample에서는 `p_t=1-p`다.

### Background class logit이 따로 있는가

RetinaNet은 anchor당 K independent sigmoid를 사용한다. Background는 K target이 모두 0이며 별도 softmax background channel이 없다.

### Focal Loss를 box regression에도 쓰는가

아니다. 논문은 classification에 focal loss, box regression에 Smooth L1을 쓴다.

### RetinaNet은 anchor-free인가

아니다. FPN 모든 위치에 9 anchors를 놓는 dense anchor-based detector다.

### One-stage라 NMS가 없는가

아니다. Confidence threshold, per-level top-k, box decode, NMS가 필요하다.

## 온디바이스 관점

### 장점

- RoIAlign/동적 proposal batching이 없어 graph가 비교적 규칙적이다.
- Backbone, FPN, head가 대부분 dense 3×3 convolution이라 NPU 친화적이다.
- Head weight가 level 간 공유되어 model storage는 제한된다.

### 병목

- 8개의 256-channel 3×3 head conv가 pyramid 전체에서 반복된다.
- 80-class `A×K` logit의 activation과 memory write가 크다.
- Sigmoid/threshold/top-k/decode/NMS가 accelerator 밖으로 나갈 수 있다.
- Input resolution 상승이 AP_S를 크게 올리지만 activation·MAC·latency를 면적 비례로 늘린다.

### 최적화 우선순위

1. Backbone과 FPN channel을 줄인 경량 variant를 먼저 측정
2. Classification/box tower depth를 4→2 등으로 ablation
3. Anchor 수를 9→3 또는 anchor-free FCOS와 비교
4. Per-level top-k를 낮춰 NMS candidate 제한
5. INT8에서 final sigmoid logit와 box delta calibration 분리 확인
6. Decode+top-k+NMS의 device-side fused implementation 사용

FP16/INT8 model size만 보면 head weight 6.46M은 작아 보이지만, 실제 peak는 dense activation과 FPN P3에서 결정될 가능성이 높다. p95는 candidate 수와 CPU fallback에 민감하다.

## 재현 체크리스트

- [ ] ICCV 2017/arXiv 제공 PDF의 최종 table을 기준으로 한다.
- [ ] P3-P7 stride와 P6/P7 생성 방식을 맞춘다.
- [ ] 256 channel, head depth 4, 3×3 conv를 확인한다.
- [ ] Level 간 head 공유, cls/reg branch 간 비공유를 확인한다.
- [ ] Anchors: ratios 3 × within-octave scales 3 = 9/location.
- [ ] Positive IoU >=0.5, negative <0.4, 사이 ignore를 구현한다.
- [ ] K sigmoid와 all-zero background target을 사용한다.
- [ ] Focal Loss `gamma=2`, `alpha=0.25` 및 stable logits 구현을 확인한다.
- [ ] Classification final bias를 foreground prior 0.01로 초기화한다.
- [ ] Loss를 positive anchor 수로 normalize한다.
- [ ] Regression은 positive에만 Smooth L1을 적용한다.
- [ ] Inference threshold 0.05, level당 top 1000, NMS 0.5를 기록한다.
- [ ] Basic 600-scale ablation과 final 800-scale longer schedule을 구분한다.
- [ ] AP/AP50/AP75/AP_S/M/L를 모두 비교한다.
- [ ] Parameter/MAC은 backbone, FPN, head를 분리한다.
- [ ] Peak activation, p50/p95, NMS fallback, 전력/온도를 target에서 측정한다.

## 로드맵에서의 연결

- **Faster R-CNN**은 sampling과 proposal cascade로 imbalance를 줄였다.
- **FPN**은 RetinaNet의 P3-P7 multi-scale backbone이 된다.
- **FCOS**는 focal classification을 유지하면서 anchor와 IoU assignment를 제거하고 location별 box distance를 회귀한다.
- **ATSS/quality focal 계열**은 sample assignment와 classification score가 localization quality를 더 잘 반영하도록 확장한다.
- **Deformable DETR**도 classification에 focal loss를 도입하지만 box 중복은 Hungarian matching으로 다룬다.

온디바이스 비교에서는 RetinaNet과 FCOS를 동일 ResNet/MobileNet-FPN, 동일 input, 동일 NMS 구현으로 맞춰야 한다. 그래야 focal loss의 학습 이점과 anchor output channel의 추론 비용을 분리할 수 있다.

## 최종 평가

이 논문의 가장 강한 점은 복잡한 detector module을 제안한 것이 아니라, one-stage detector의 실패 원인을 loss distribution에서 찾아 controlled experiment로 입증한 데 있다. Focal Loss는 수많은 easy negative를 hard discard하지 않고 gradient가 거의 0이 되도록 만들어 dense training을 가능하게 했다.

RetinaNet의 역사적 결과는 focal loss의 효과를 분명히 보여 주지만, 현재 온디바이스 관점에서는 dense head의 계산과 activation을 따로 봐야 한다. `800×1024` 예시에서 classification logits만 FP16 약 23.4 MiB이고 head가 리뷰어 계산 약 110G MAC이므로, one-stage라는 이름이 자동으로 가볍다는 뜻은 아니다. 이 논문에서 가져가야 할 핵심은 "모든 sample을 같은 weight로 학습하지 말라"는 원리와, accuracy·activation memory·NMS tail을 함께 측정하는 실험 설계다.

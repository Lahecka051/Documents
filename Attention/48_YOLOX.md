# 48. YOLOX: Exceeding YOLO Series in 2021

## 논문 정보

- 제목: **YOLOX: Exceeding YOLO Series in 2021**
- 저자: Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, Jian Sun
- 소속: Megvii Technology
- 공개: arXiv:2107.08430v2, 2021-08-06
- 논문: [arXiv](https://arxiv.org/abs/2107.08430) / [PDF](https://arxiv.org/pdf/2107.08430)
- 원본 파일: `48_YOLOX.pdf`
- 공식 구현: [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- 핵심 키워드: anchor-free YOLO, decoupled head, SimOTA, dynamic-k, Mosaic, MixUp, real-time detection

## 한눈에 보는 요약

YOLOX는 YOLOv3를 통제된 baseline으로 삼아 2021년 당시 detector 연구의 핵심 개선을 한 단계씩 이식한 기술 보고서다.

```text
YOLOv3-SPP baseline
 -> decoupled classification/regression head
 -> Mosaic + MixUp
 -> anchor-free, one prediction per location
 -> 3×3 center positives
 -> SimOTA dynamic label assignment
 = YOLOX-DarkNet53
```

이 순서로 COCO val AP가 `38.5 -> 47.3`으로 올라간다. 이후 CSPNet backbone과 PAN neck을 사용해 S/M/L/X family로 scale하며, 모바일용 depthwise YOLOX-Nano도 제시한다.

핵심 기여는 다음 세 가지다.

1. Classification과 box regression의 optimization conflict를 줄이는 decoupled head
2. Domain-specific anchor template를 제거하고 위치당 prediction을 3개에서 1개로 줄인 anchor-free head
3. Prediction-ground-truth loss를 global하게 비교하고 object마다 positive 수를 바꾸는 SimOTA

YOLOX-L은 `640×640`, FP16, batch 1, Tesla V100에서 AP 50.0, latency 14.5 ms를 보고한다. 다만 이 latency는 post-processing을 제외한다. YOLOX의 기본 최종 model도 NMS-free가 아니며, optional end-to-end variant는 AP와 속도가 모두 낮아 최종 family에서 제외됐다.

## 왜 YOLO를 다시 설계했는가

YOLOv4와 YOLOv5는 강한 anchor-based pipeline이지만 다음 heuristic이 따라온다.

- Dataset별 anchor clustering
- Location당 여러 anchor prediction
- Anchor와 ground truth를 매칭하는 hand-crafted rule
- Grid sensitivity와 scale range 조정

동시에 학계에서는 FCOS 같은 anchor-free detector, OTA 같은 dynamic assignment, NMS-free 연구가 발전하고 있었다. YOLOX는 이미 강하게 최적화된 YOLOv4/5 대신 널리 배포된 YOLOv3-SPP를 출발점으로 삼아 각 기술의 기여를 확인한다.

"Anchor-free"는 spatial anchor point나 grid까지 없앤다는 뜻이 아니다. **미리 정한 width-height anchor box template를 제거**하고 각 grid location에서 하나의 box를 예측한다는 뜻이다.

## YOLOv3 baseline

Baseline은 DarkNet53 backbone과 SPP를 사용한다. 원 YOLOv3에 다음 training trick을 먼저 추가한다.

- EMA weight update
- Cosine learning-rate schedule
- IoU loss와 IoU-aware branch
- Classification/objectness에 BCE loss
- Box regression에 IoU loss
- Random horizontal flip, color jitter, multi-scale training

RandomResizedCrop은 이후 Mosaic과 역할이 겹친다고 보고 제외한다. 이 baseline은 COCO val AP 38.5다. 비교표의 ultralytics YOLOv3 44.3과 차이가 크므로, YOLOX의 최종 +8.8 AP를 원 논문 YOLOv3의 단순 architecture gain으로만 해석하면 안 된다. Baseline implementation과 recipe 선택이 출발점에 영향을 준다.

## 전체 architecture

```text
image [B,3,H,W]
 -> DarkNet53 or modified CSPNet backbone
 -> SPP + PAN/FPN multi-scale neck
 -> {P3, P4, P5}
 -> level별 1×1 channel projection to 256
 -> classification branch: 3×3 conv ×2 -> C logits
 -> regression branch:     3×3 conv ×2 -> 4 box + 1 objectness
 -> decode all locations
 -> score filtering + NMS
```

S/M/L/X는 YOLOv5의 modified CSPNet, SiLU, PAN head와 scaling rule을 가져와 같은 detector design을 적용한다. Tiny와 Nano는 더 작게 줄이고 Nano에는 depthwise convolution을 사용한다.

## Decoupled head

기존 coupled head는 한 feature에서 classification, box, objectness를 한 번에 출력한다. YOLOX는 먼저 1×1 convolution으로 channel을 256으로 맞춘 뒤 두 branch를 분리한다.

```text
F_l [B,C_l,H_l,W_l]
 -> 1×1 conv -> [B,256,H_l,W_l]
    |-> 3×3 conv -> 3×3 conv -> class [B,K,H_l,W_l]
    `-> 3×3 conv -> 3×3 conv -> box [B,4,H_l,W_l]
                                 objectness [B,1,H_l,W_l]
```

Classification은 "무엇인가"를, regression은 "어디인가"를 최적화한다. 두 task가 같은 feature transform을 마지막까지 공유하면 gradient 방향이 충돌할 수 있다. Branch를 분리하면 head 계산은 늘지만 convergence가 빨라지고 최종 AP가 올라간다.

YOLOv3 baseline에서 coupled 38.5 AP를 decoupled로 바꾸면 39.6 AP다. V100 latency는 `10.5 -> 11.6 ms`, GFLOPs는 `157.3 -> 186.0`으로 증가한다. 즉 decoupling은 공짜가 아니라 AP +1.1과 latency +1.1 ms의 교환이다.

Optional end-to-end YOLO에서는 효과가 더 크다.

| Setting | Coupled head | Decoupled head |
| --- | ---: | ---: |
| Vanilla YOLO | 38.5 | 39.6 |
| End-to-end YOLO | 34.3 | 38.8 |

One-to-one assignment을 쓸 때 coupled head의 AP 손실은 -4.2지만 decoupled head는 -0.8이다. 논문은 이를 classification-localization conflict의 증거로 본다.

## Anchor-free prediction

Anchor-based YOLO는 한 location에서 3개 anchor를 예측한다. YOLOX는 이를 한 prediction으로 줄이고 grid 기준 두 offset과 box width/height를 직접 예측한다.

```math
N_{pred}=\sum_l H_lW_l
```

Anchor 3개를 쓰면 같은 feature grid에서 prediction 수가 대략 `3N_pred`다. Anchor clustering과 template dimension이 없어지고 decoding과 output transfer가 단순해진다.

논문의 anchor-free 전환은 parameter `63.86 -> 63.72M`, GFLOPs `186.0 -> 185.3G`, latency `11.6 -> 11.1 ms`로 조금 줄면서 AP는 `42.0 -> 42.9`로 올라간다.

## Batch 1 tensor shape와 output memory

다음은 `640×640`, stride `{8,16,32}`, COCO `K=80`인 **reviewer 계산**이다. 논문 Figure 2와 일반적인 YOLOX pyramid를 shape로 풀어 쓴 것이다.

| Level | Spatial shape | Location 수 |
| --- | ---: | ---: |
| P3 | `80×80` | 6,400 |
| P4 | `40×40` | 1,600 |
| P5 | `20×20` | 400 |
| 합계 | - | **8,400** |

Anchor-free raw outputs는 다음과 같다.

```math
P_{cls}\in\mathbb{R}^{1\times8400\times80}
```

```math
P_{box}\in\mathbb{R}^{1\times8400\times4},
\qquad
P_{obj}\in\mathbb{R}^{1\times8400\times1}
```

총 element 수는 다음과 같다.

```math
8400\times(80+4+1)=714{,}000
```

FP16 payload는 약 `1.362 MiB`다. 같은 출력 encoding을 location당 anchor 3개에 반복한다고 단순 비교하면 `2,142,000` element, 약 `4.086 MiB`로 3배다. 논문이 말한 NPU-CPU prediction transfer 병목을 구체화하는 계산이다.

Head 내부 256-channel feature 하나는 세 level 합계가 다음과 같다.

```math
8400\times256=2{,}150{,}400\text{ elements}
```

FP16 약 `4.102 MiB`다. Classification과 regression branch의 activation이 동시에 live하면 이보다 커지고, 각 branch의 두 3×3 conv intermediate와 workspace도 추가된다. Buffer reuse와 fusion 여부에 따라 peak가 달라지므로 이 값은 activation payload 예시이지 profiler peak가 아니다.

## Strong augmentation

YOLOX는 Mosaic와 MixUp을 사용하고 마지막 15 epoch에는 둘을 끈다.

```text
epoch 1..285: Mosaic + MixUp + other augmentation
epoch 286..300: Mosaic/MixUp off, target distribution에 적응
```

Strong augmentation을 적용하면 ImageNet pretraining 이득이 사라졌다고 보고 이후 model은 scratch에서 학습한다. 이는 "pretraining은 항상 무의미하다"는 일반 결론이 아니라 COCO, 이 recipe, 300 epoch에서의 경험적 결과다.

Model size에 따라 적정 augmentation이 다르다.

- YOLOX-L: scale jitter `[0.1,2.0]`, MixUp으로 `48.6 -> 49.5 AP`
- YOLOX-Nano: 강한 `[0.1,2.0]` + MixUp은 24.0 AP
- YOLOX-Nano: 약한 `[0.5,1.5]`, MixUp 제거는 25.3 AP

큰 model은 noisy augmented distribution을 활용할 capacity가 있지만 작은 model에는 과도한 regularization이 될 수 있다.

## Multi positives

Anchor-free로 바꾼 직후에는 object center 한 location만 positive다. YOLOX는 FCOS center sampling처럼 center 주변 3×3 영역을 candidate positive로 확장한다.

```text
one GT -> center 1 point
       -> center 3×3 candidate points
```

Positive gradient가 늘어 극단적인 positive-negative imbalance를 줄인다. 이 단순 변경으로 AP가 `42.9 -> 45.0`, +2.1 올라간다. 그러나 모든 3×3 point를 최종 positive로 고정하는 것이 아니라 다음 SimOTA가 quality에 따라 동적으로 선택한다.

## SimOTA label assignment

OTA는 assignment를 optimal transport 문제로 보고 Sinkhorn-Knopp algorithm으로 푼다. 정확하지만 저자들의 실험에서 training time을 약 25% 늘렸다. SimOTA는 핵심 원칙을 유지하며 dynamic top-k로 근사한다.

Prediction `p_j`와 ground truth `g_i`의 pair cost는 다음과 같다.

```math
c_{ij}=L^{cls}_{ij}+\lambda L^{reg}_{ij}
```

- Loss/quality aware cost
- Object center prior
- Ground truth마다 다른 positive 수, dynamic-k
- 한 GT의 local heuristic만이 아니라 후보 전체를 보는 global view

GT `g_i`마다 center region 안에서 cost가 작은 top-k prediction을 positive로 고른다. `k`는 ground truth마다 pairwise IoU를 이용해 동적으로 정한다. 논문은 정확한 dynamic-k estimation을 OTA 설명으로 넘기므로 구현 재현에는 official code가 필요하다. 이 보고서 본문은 `lambda`의 숫자도 명시하지 않는다.

### Assignment tensor shape 예

Image당 ground truth가 `G`, prediction이 `N=8400`이면 cost는 다음 shape다.

```math
C\in\mathbb{R}^{G\times8400}
```

Reviewer 예시 `G=20`이면 168,000 pair다. FP32 cost payload는 약 `0.641 MiB`지만 classification cost, IoU matrix, center mask, sorting/top-k buffer가 추가된다. 이는 training-only cost이므로 inference graph에는 들어가지 않는다.

SimOTA로 AP는 `45.0 -> 47.3`, +2.3 올라간다. Sinkhorn solver를 제거하므로 OTA보다 training 부담이 낮지만 assignment 자체가 free인 것은 아니다.

## Loss 구조

YOLOX baseline에서 논문이 직접 명시한 loss는 다음과 같다.

```math
L=L_{cls}^{BCE}+L_{obj}^{BCE}+L_{reg}^{IoU}
```

실제 scale coefficient와 IoU loss variant의 세부는 config를 확인해야 한다. 또한 SimOTA의 `c_ij`는 positive를 선택하는 matching cost이고, 선택 후 network를 업데이트하는 최종 loss와 역할이 다르다.

Optional end-to-end variant는 additional convolution 2개, one-to-one assignment, stop-gradient를 사용한다. AP 46.5, 13.5 ms로 standard SimOTA + NMS model의 47.3, 11.1 ms보다 불리해 최종 model에는 포함하지 않는다.

## Training recipe

- Dataset: COCO train2017, evaluation on val/test-dev
- Epoch: 300
- Warm-up: 5 epoch
- Optimizer: SGD
- Base learning rate: `0.01 × BatchSize / 64`
- Default batch: 128 on 8 GPUs
- Momentum: 0.9
- Weight decay: 0.0005
- Schedule: cosine
- Multi-scale input: 448부터 832까지, 32 stride 단위로 균등 sampling
- Mosaic/MixUp: 마지막 15 epoch에 종료
- Timing: Tesla V100, FP16, batch 1

Training resolution이 매 iteration 변해도 deployment benchmark는 fixed 640 또는 Tiny/Nano의 416이다. Dynamic-shape runtime을 사용했다는 뜻이 아니다.

## 구현 pseudocode

### Decoupled anchor-free head

```python
def yolox_head(feature, num_classes):
    # feature: [B, C_l, H, W]
    x = conv1x1(feature, out_channels=256)

    cls = conv3x3(x, 256, activation="silu")
    cls = conv3x3(cls, 256, activation="silu")
    cls_logits = conv1x1(cls, num_classes)

    reg = conv3x3(x, 256, activation="silu")
    reg = conv3x3(reg, 256, activation="silu")
    box_raw = conv1x1(reg, 4)
    obj_logits = conv1x1(reg, 1)
    return cls_logits, box_raw, obj_logits
```

### SimOTA 개념 구현

```python
def simota_assign(pred_cls, pred_box, gt_cls, gt_box, center_mask, lam):
    # pairwise tensors: [G, N]
    iou = pairwise_iou(gt_box, pred_box)
    cls_cost = pairwise_classification_loss(gt_cls, pred_cls)
    reg_cost = pairwise_iou_loss(gt_box, pred_box)
    cost = cls_cost + lam * reg_cost
    cost = cost.masked_fill(~center_mask, float("inf"))

    dynamic_k = estimate_k_from_top_ious(iou)
    matches = select_lowest_cost(cost, k_per_gt=dynamic_k)
    matches = resolve_prediction_conflicts(matches, cost)
    return matches
```

실제 구현에서는 mixed precision의 `inf`, IoU clamp, empty-GT case, 여러 GT가 같은 prediction을 고르는 conflict resolution을 검증해야 한다.

## Step-by-step ablation

YOLOX-DarkNet53의 논문 Table 2는 다음과 같다. 모두 `640×640`, V100 FP16, batch 1이며 post-processing 제외다.

| 단계 | AP | Params | GFLOPs | Latency |
| --- | ---: | ---: | ---: | ---: |
| YOLOv3 baseline | 38.5 | 63.00M | 157.3 | 10.5 ms |
| + decoupled head | 39.6 | 63.86M | 186.0 | 11.6 ms |
| + strong augmentation | 42.0 | 63.86M | 186.0 | 11.6 ms |
| + anchor-free | 42.9 | 63.72M | 185.3 | 11.1 ms |
| + multi positives | 45.0 | 63.72M | 185.3 | 11.1 ms |
| + SimOTA | **47.3** | 63.72M | 185.3 | 11.1 ms |
| + NMS-free optional | 46.5 | 67.27M | 205.1 | 13.5 ms |

Augmentation, positive sampling, assignment처럼 inference compute를 늘리지 않는 training-time 변화가 총 +6.6 AP를 만든다. Decoupled head는 inference cost를 늘리지만 anchor-free 전환이 일부를 회수한다.

## Model family 결과

### YOLOX S/M/L/X

| Model | AP | Params | GFLOPs | V100 latency |
| --- | ---: | ---: | ---: | ---: |
| S | 39.6 | 9.0M | 26.8 | 9.8 ms |
| M | 46.4 | 25.3M | 73.8 | 12.3 ms |
| L | 50.0 | 54.2M | 155.6 | 14.5 ms |
| X | 51.2 | 99.1M | 281.9 | 17.3 ms |

모두 `640×640`, FP16, batch 1이다. GFLOPs는 논문 표의 명칭을 유지했다. MACs로 변환하려면 profiler가 multiply-add를 어떻게 세는지 확인해야 한다.

Reviewer가 계산한 순수 FP16 weight payload 하한은 S 약 17.2 MiB, M 48.3 MiB, L 103.4 MiB, X 189.0 MiB다. Engine metadata, BN, alignment, workspace, activation은 제외했다.

### Tiny와 Nano

| Model | Resolution | AP | Params | GFLOPs |
| --- | ---: | ---: | ---: | ---: |
| YOLOX-Tiny | 416×416 | 32.8 | 5.06M | 6.45 |
| YOLOX-Nano | 416×416 | 25.3 | 0.91M | 1.08 |

Nano의 FP16 weight payload 하한은 약 1.74 MiB, INT8은 약 0.87 MiB다. 하지만 논문은 모바일 device의 latency, peak memory, quantized accuracy를 보고하지 않는다. Figure 1의 "mobile devices" 맥락은 size-accuracy comparison이지 특정 phone benchmark가 아니다.

### Val과 test-dev를 구분

Step table의 YOLOX-DarkNet53은 COCO val AP 47.3이고 SOTA Table 6의 test-dev 값은 AP 47.4다. 서로 다른 split의 0.1 차이를 improvement로 해석하면 안 된다.

### Object scale별 결과

Table 6의 COCO test-dev 세부 지표는 다음과 같다. 이 값은 model scale을 키울 때 주로 어느 object 크기에서 gain이 나는지 보여준다.

| Model | AP | AP50 | AP75 | APS | APM | APL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOX-DarkNet53 | 47.4 | 67.3 | 52.1 | 27.5 | 51.5 | 60.9 |
| YOLOX-M | 46.4 | 65.4 | 50.6 | 26.3 | 51.0 | 59.9 |
| YOLOX-L | 50.0 | 68.5 | 54.5 | 29.8 | 54.5 | 64.4 |
| YOLOX-X | 51.2 | 69.6 | 55.7 | 31.2 | 56.1 | 66.1 |

L에서 X로 갈 때 전체 AP는 +1.2, APS는 +1.4, APM은 +1.6, APL은 +1.7이다. 큰 model의 추가 capacity가 특정 크기에만 국한되지는 않지만 absolute small-object AP는 여전히 낮다. Component별 ablation에는 APS/AP75가 없으므로 SimOTA나 decoupled head가 small object에 얼마나 기여했는지는 분리할 수 없다.

### YOLOv5 counterpart와의 비교

논문은 같은 V100 FP16 batch-1 setting에서 YOLOv5 counterpart와 비교한다.

| Scale | YOLOv5 AP / latency | YOLOX AP / latency | AP 변화 | Latency 변화 |
| --- | ---: | ---: | ---: | ---: |
| S | 36.7 / 8.7 ms | 39.6 / 9.8 ms | +2.9 | +1.1 ms |
| M | 44.5 / 11.1 ms | 46.4 / 12.3 ms | +1.9 | +1.2 ms |
| L | 48.2 / 13.7 ms | 50.0 / 14.5 ms | +1.8 | +0.8 ms |
| X | 50.4 / 16.0 ms | 51.2 / 17.3 ms | +0.8 | +1.3 ms |

작은 scale에서 AP gain이 더 크고 X에서는 +0.8로 줄어든다. 모든 scale에서 YOLOX latency가 조금 더 길므로 "동일 속도에서 항상 우월"보다는 **조금 더 많은 head compute로 더 높은 AP를 얻는 family**라고 읽는 편이 정확하다. 또한 YOLOv5 version과 training recipe가 논문 시점 구현에 묶여 있으므로 현재 release와 직접 비교하면 안 된다.

## 연산량과 실제 latency가 어긋나는 이유

YOLOX-S에서 X로 갈 때 논문 GFLOPs는 `26.8 -> 281.9`, 약 10.5배지만 latency는 `9.8 -> 17.3 ms`, 약 1.77배다. 반대로 S의 연산량이 작아도 9.8 ms 아래로 크게 내려가지 않는다. 이는 GPU에서 다음 고정 비용과 utilization 차이가 있음을 시사한다.

- Small convolution의 kernel launch와 memory access
- PAN resize/concat 같은 비-GEMM 연산
- 작은 model의 낮은 SM utilization
- Head와 tensor layout conversion의 고정 overhead

따라서 GFLOPs만으로 target-device latency를 선형 예측하면 안 된다. 특히 mobile NPU는 SRAM tiling과 supported channel multiple에 따라 순위가 달라질 수 있다.

## 자주 헷갈리는 해석

### Anchor-free이면 assignment가 필요 없는가?

아니다. Anchor box template를 없애도 어떤 grid prediction을 어느 GT에 학습시킬지는 정해야 한다. YOLOX에서 이 역할을 center prior와 SimOTA가 맡는다.

### SimOTA는 inference를 느리게 하는가?

SimOTA는 training-time label assignment다. Exported inference graph에는 들어가지 않는다. Accuracy gain을 얻으면서 inference operator는 늘리지 않는다는 점이 장점이다.

### YOLOX는 NMS-free인가?

기본 YOLOX family는 NMS를 사용한다. NMS-free는 optional 실험이며 AP와 latency가 불리해 최종 model에 포함되지 않는다.

### Strong augmentation이면 pretraining이 언제나 불필요한가?

논문이 보여준 것은 이 COCO 300-epoch recipe에서 ImageNet pretraining이 더 이상 유리하지 않았다는 결과다. 작은 private dataset, 짧은 schedule, domain shift가 있는 경우에는 다시 검증해야 한다.

### GFLOPs가 작으면 activation memory도 작은가?

보장되지 않는다. Decoupled head는 class/regression branch를 따로 유지하고 PAN은 high-resolution feature를 concatenate한다. Weight와 연산량이 작아도 activation lifetime과 workspace가 peak를 만들 수 있다.

## Streaming Perception Challenge

저자들은 30 FPS stream에서 inference가 33 ms 이하인 강한 model이 streaming accuracy에 유리하다고 보고 TensorRT YOLOX-L 하나로 WAD 2021 challenge 1위를 기록했다. 이 결과는 frame-by-frame AP만 아니라 computation 중 지나가는 frame을 고려해야 한다는 점을 강조한다.

그러나 challenge metric과 COCO AP는 동일하지 않다. Camera pipeline에서는 stale prediction, queueing, dropped frame까지 함께 평가해야 한다.

## 장점

- Baseline에서 final model까지 개선 순서와 AP/latency를 투명하게 제시한다.
- Decoupled head, anchor-free, assignment의 역할을 분리해 이해하기 쉽다.
- 위치당 prediction 수 감소가 edge device transfer에 미치는 영향을 명시적으로 지적한다.
- Nano부터 X까지 같은 design을 scale해 capacity별 trade-off를 제공한다.
- ONNX, TensorRT, NCNN, OpenVINO deployment를 공식 repository에서 지원한다.
- Model 크기에 따라 augmentation strength를 바꿔야 한다는 실용적 결과가 있다.

## 한계와 비판적 관점

### 1. NMS 제외 latency

주요 latency 표는 post-processing을 제외한다. Dense 8,400 prediction을 device 밖으로 옮기고 decode/NMS하는 시간은 edge에서 무시하기 어렵다.

### 2. 실제 mobile benchmark 부재

Nano는 mobile용으로 설계됐지만 paper의 latency는 V100뿐이다. Depthwise convolution이 target NPU에서 잘 최적화되는지, CPU fallback이 있는지 알 수 없다.

### 3. Activation memory 부재

Parameter와 GFLOPs는 있지만 decoupled branch, PAN concat, NMS workspace를 포함한 peak memory가 없다. Head 분리는 intermediate activation lifetime을 늘릴 수 있다.

### 4. Optional NMS-free는 미완성

One-to-one variant는 AP -0.8, latency +2.4 ms다. YOLOX를 기본적으로 end-to-end NMS-free detector라고 소개하면 틀린다.

### 5. Recipe 영향이 크다

Final gain 상당 부분은 augmentation과 assignment에서 온다. 다른 codebase의 YOLOv5 checkpoint와 단순 architecture comparison을 하면 통제가 깨진다.

### 6. 통계적 불확실성

Random seed별 variance와 confidence interval을 보고하지 않는다. 0.8-1 AP 차이는 training variance와 recipe 차이까지 고려해야 한다.

### 7. Small-object와 tail latency 분석 부족

Table 6에는 final model의 APS가 있지만 component ablation별 APS, AP75가 없다. p95 latency, 전력, 온도도 없다.

## 온디바이스 operator와 병목

YOLOX는 대부분 정형 CNN operator로 구성돼 deployment 친화적이다.

- Convolution/depthwise convolution
- BN folding
- SiLU
- Resize와 PAN concat
- Sigmoid/box decode
- Score filter, top-k, NMS

하지만 다음을 확인해야 한다.

### Decoupled head activation

두 256-channel branch는 parameter보다 activation traffic을 늘린다. Compiler가 branch를 순차 schedule하고 buffer를 재사용하는지 확인해야 한다.

### Anchor-free output transfer

Location당 prediction을 3개에서 1개로 줄인 것은 NPU-CPU transfer에 직접 유리하다. 다만 80 class logits 전체를 host로 보내면 여전히 약 1.36 MiB FP16 raw output이므로 device-side sigmoid, threshold, top-k 지원이 중요하다.

### NMS

NMS가 CPU에서 수행되면 scene complexity에 따라 latency variance가 커진다. Average만 말고 candidate count와 p95를 함께 기록해야 한다.

### SiLU와 INT8

Backend가 SiLU를 native fuse하지 못하면 sigmoid와 multiply로 나뉜다. INT8 calibration에서 objectness와 class score의 tail distribution, box decode의 민감도를 별도로 확인해야 한다.

### Nano depthwise convolution

1.08 GFLOPs가 낮아도 depthwise kernel launch와 memory bandwidth가 지배하면 예상보다 느릴 수 있다. Nano, Tiny, S를 실제 device에서 모두 측정해 가장 작은 model이 반드시 가장 효율적인지 확인해야 한다.

## 재현 체크리스트

- [ ] arXiv v2와 official repository commit을 고정했는가?
- [ ] DarkNet53 step ablation과 CSPNet S/M/L/X를 구분했는가?
- [ ] COCO val과 test-dev 결과를 구분했는가?
- [ ] 300 epoch, 5 epoch warm-up, SGD momentum 0.9를 적용했는가?
- [ ] `lr=0.01×batch/64`, weight decay 0.0005, cosine schedule인가?
- [ ] Multi-scale 448-832, stride 32 sampling을 재현했는가?
- [ ] Mosaic/MixUp을 마지막 15 epoch에 끄는가?
- [ ] Nano/Tiny/S에서 augmentation을 약화했는가?
- [ ] Head가 1×1 projection 뒤 class/reg branch로 분리되는가?
- [ ] Location당 prediction이 1개인지 export output으로 확인했는가?
- [ ] Center candidate와 SimOTA dynamic-k를 구분했는가?
- [ ] Empty-GT와 conflict resolution을 unit test했는가?
- [ ] FP16 batch 1 latency에 post-processing 포함 여부를 표시했는가?
- [ ] Fixed resolution에서 p50/p95, peak memory, output bytes를 측정했는가?
- [ ] NMS를 device와 CPU에서 각각 profile했는가?
- [ ] INT8 calibration 뒤 AP/AP50/AP75/APS를 다시 측정했는가?

## 로드맵에서의 연결

YOLOX는 FCOS의 anchor-free dense prediction과 OTA의 dynamic assignment를 YOLO engineering에 연결한다. 다음 RTMDet는 YOLOX를 직접 baseline으로 삼아 block, neck capacity, SepBN head와 soft assignment를 개선한다.

```text
RetinaNet: dense one-stage + class imbalance
 -> FCOS: anchor-box template 제거
 -> OTA/SimOTA: dynamic positive assignment
 -> YOLOX: real-time YOLO pipeline으로 통합
 -> RTMDet: architecture와 assignment를 더 체계적으로 재균형
```

DETR 계열과 비교할 때는 YOLOX가 one-to-many dense training과 NMS를 유지한다는 점이 핵심이다. 동일 device 실험에서는 다음을 기록하는 것이 좋다.

- YOLOX-S/Nano의 decode+NMS 포함 p50/p95
- RT-DETRv2-S의 decoder layer별 p95
- Raw output transfer bytes
- COCO APS와 crowded-scene candidate 수
- FP16/INT8 accuracy와 thermal steady-state FPS

## 최종 평가

YOLOX는 anchor-free, decoupled head, SimOTA를 실시간 YOLO family에 성공적으로 통합했고, 각 단계의 AP와 latency를 공개해 실용 detector 설계의 좋은 기준점이 됐다. 특히 anchor template를 없애면서 prediction 수를 줄이고, training-only assignment로 정확도를 크게 높인 점은 온디바이스에도 직접 연결된다.

다만 paper의 속도는 V100, FP16, batch 1, post-processing 제외 조건이다. Mobile deployment에서는 decoupled head activation, PAN memory traffic, raw output transfer, NMS p95가 1.08G 또는 26.8G 같은 연산량 숫자보다 더 중요할 수 있다. 따라서 YOLOX는 "가벼운 FLOPs baseline"이 아니라 **training recipe와 device-side post-processing까지 함께 재현해야 하는 end-to-end system baseline**으로 다뤄야 한다.

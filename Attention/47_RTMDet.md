# 47. RTMDet: An Empirical Study of Designing Real-Time Object Detectors

## 논문 정보

- 제목: **RTMDet: An Empirical Study of Designing Real-Time Object Detectors**
- 저자: Chengqi Lyu, Wenwei Zhang, Haian Huang, Yue Zhou, Yudong Wang, Yanyi Liu, Shilong Zhang, Kai Chen
- 공개: arXiv:2212.07784v2, 2022-12-16
- 논문: [arXiv](https://arxiv.org/abs/2212.07784) / [PDF](https://arxiv.org/pdf/2212.07784)
- 원본 파일: `47_RTMDet.pdf`
- 공식 구현: [MMDetection RTMDet configs](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/rtmdet)
- 핵심 키워드: one-stage detector, CSP-PAFPN, large-kernel depthwise convolution, dynamic soft label assignment, SepBN head, instance segmentation, rotated detection

## 한눈에 보는 요약

RTMDet는 단일 신기술 하나를 제안하기보다 real-time detector의 **architecture, label assignment, augmentation, optimizer를 통제된 ablation으로 다시 조합**한 연구다. 출발점은 YOLOX이며, 최종 설계는 다음 네 축으로 정리된다.

```text
shallow-wide CSP backbone
 + 5×5 depthwise large-kernel block
 + backbone과 neck의 균형 조정
 + scale 간 conv weight 공유 + scale별 BN
 + IoU soft target 기반 dynamic assignment
 + cached Mosaic/MixUp -> LSJ 2-stage training
 + AdamW flat-cosine optimization
 = RTMDet
```

COCO `640×640`, 300 epoch 조건에서 RTMDet-tiny는 4.8M parameter, 8.1G FLOPs, 0.98 ms, AP 41.1을 보고한다. RTMDet-x는 94.9M, 141.7G FLOPs, 3.10 ms, AP 52.8이다. 속도는 NVIDIA RTX 3090, TensorRT 8.4.3, FP16, batch 1에서 측정했고 **object detection 표에는 NMS 시간이 포함되지 않는다**.

논문의 가장 재사용 가능한 교훈은 "더 큰 backbone"보다 **neck 용량, layer depth, 실제 kernel latency, label quality를 함께 맞춰야 한다**는 점이다. 또한 instance segmentation과 rotated detection으로 최소 수정 확장해 versatile real-time framework를 지향한다.

## 문제 설정: 실시간 detector는 무엇을 최적화해야 하는가

One-stage detector는 backbone, multi-scale neck, dense head, label assignment, data augmentation이 강하게 결합된다. 같은 FLOPs라도 layer가 지나치게 직렬이면 GPU utilization이 나쁘고, 같은 parameter 수라도 feature pyramid resize/concat이 memory traffic을 크게 만들 수 있다.

RTMDet는 다음 질문을 실험한다.

1. 큰 receptive field를 실시간 비용으로 얻으려면 어떤 block이 좋은가?
2. Backbone과 neck 중 어디에 capacity를 배치해야 하는가?
3. FPN scale별 head를 완전히 분리하지 않고도 정확도를 유지할 수 있는가?
4. Classification confidence와 localization quality가 어긋나는 assignment noise를 어떻게 줄이는가?
5. 강한 augmentation의 이득을 유지하면서 학습 말기의 distribution gap을 어떻게 줄이는가?

이 논문은 end-to-end NMS-free detector가 아니다. Anchor-free dense prediction 뒤 score filtering과 NMS를 사용하는 실시간 CNN detector다.

## 전체 architecture

```text
image
 -> CSP-style backbone with large-kernel depthwise blocks
 -> {C3, C4, C5}
 -> CSP-PAFPN top-down + bottom-up fusion
 -> {P3, P4, P5}
 -> scale-shared conv head + scale-specific BN
 -> classification / box regression
 -> threshold + top-k + NMS
```

Instance segmentation은 kernel head와 mask feature head를 추가하고, rotated detection은 regression 차원을 4에서 5로 늘리고 rotated box decoder와 Rotated IoU loss를 사용한다.

### Batch 1의 spatial shape 예

논문 Figure 2는 `C3/C4/C5`를 제시하지만 channel 수를 고정해 명시하지 않는다. 아래는 `640×640`, 일반적인 stride `{8,16,32}`를 적용한 **reviewer shape 예시**다.

| Feature | Spatial shape | 위치 수 |
| --- | ---: | ---: |
| P3 | `1×C3×80×80` | 6,400 |
| P4 | `1×C4×40×40` | 1,600 |
| P5 | `1×C5×20×20` | 400 |
| 합계 | - | **8,400** |

Anchor-free head가 각 위치에서 class `K`개와 box 4개를 출력한다고 단순화하면 COCO `K=80`에서 raw prediction shape는 다음과 같다.

```math
P_{cls}\in\mathbb{R}^{1\times8400\times80},
\qquad
P_{box}\in\mathbb{R}^{1\times8400\times4}
```

```math
8400\times(80+4)=705{,}600\text{ elements}
```

FP16 payload는 약 `1.346 MiB`다. 이는 head output만의 계산이며 intermediate convolution activation, decoded box, score sort, NMS workspace는 포함하지 않는다. 실제 official head의 encoding과 tensor layout은 사용 config와 export backend에서 확인해야 한다.

## Large-kernel depthwise basic block

기존 CSPDarkNet bottleneck의 spatial convolution을 5×5 depthwise convolution과 1×1 pointwise convolution 조합으로 바꾼다.

```text
input C channels
 -> 5×5 depthwise conv, groups=C
 -> 1×1 pointwise conv
 -> BN + SiLU
 -> residual/CSP path
```

일반 convolution의 주요 parameter 수는 `k² C_in C_out`이고 depthwise-separable block은 대략 다음과 같다.

```math
P_{standard}=k^2C_{in}C_{out}
```

```math
P_{dw+pw}=k^2C_{in}+C_{in}C_{out}
```

`C_in=C_out=C`이고 `k=5`라면 standard 5×5는 `25C²`, depthwise + pointwise는 `25C+C²`다. `C=256`인 reviewer 예시에서는 각각 1,638,400개와 71,936개 weight로 약 22.8배 차이다. 다만 parameter 감소가 latency 비율로 그대로 이어지지 않는다. Depthwise convolution은 arithmetic intensity가 낮아 mobile memory bandwidth와 kernel 최적화에 더 민감하다.

논문은 re-parameterized 3×3 convolution보다 이 block이 training memory와 low-bit quantization error 면에서 단순하다고 주장한다. 하지만 실제 INT8 결과는 표로 제공하지 않는다.

## Shallow-wide 설계

Depthwise 뒤 pointwise가 추가되면 block당 layer 수가 늘어 serial dependency가 길어진다. 저자들은 stage 2와 3의 block 수를 줄이고 width를 늘려 capacity를 보충한다.

```text
deep-narrow: 더 많은 순차 layer, 낮은 parallelism
shallow-wide: 더 적은 block, 각 layer의 matrix 폭 확대
```

Block 수를 `3-9-9-3`에서 `3-6-6-3`으로 줄인 ablation에서 latency는 `2.60 -> 2.11 ms`로 줄지만 AP도 `51.4 -> 50.9`로 0.5 떨어진다. Stage 끝에 Channel Attention을 추가한 최종 후보는 `2.40 ms`, AP 51.3이다. 즉 원래 깊은 model 대비 AP -0.1과 latency 약 -7.7%의 절충이다.

## Backbone과 neck의 capacity 균형

Object detection은 classification보다 multi-scale fusion 의존도가 높다. RTMDet는 backbone만 키우는 대신 neck의 CSP block expansion ratio를 높여 두 부분의 capacity를 비슷하게 만든다.

Table 5c에서 small model은 backbone 47%, neck 45% 구성일 때 8.54M, 15.76G, 1.21 ms, AP 43.9다. Backbone 63%, neck 29% 구성은 9.01M, 15.85G, 1.37 ms, AP 43.7이다. Large에서도 균형 구성은 50.92M, 79.70G, 2.11 ms, AP 50.9이고 backbone-heavy 구성은 57.43M, 93.73G, 2.57 ms, AP 51.0이다.

따라서 이 실험 범위에서는 backbone-heavy 설계가 AP 0.1을 위해 latency와 parameter를 더 쓴다. 이는 neck이 단순 overhead가 아니라 detection capacity의 핵심임을 보여준다.

## Scale-shared head와 separate BN

각 FPN level에 완전히 별도 head를 두면 정확하지만 parameter가 늘어난다. 모든 scale이 같은 convolution과 BN을 공유하면 P3, P4, P5의 통계 차이 때문에 AP가 크게 떨어진다. RTMDet는 convolution weight는 공유하고 BN parameter와 running statistics만 scale별로 분리한다.

```math
y_l=\mathrm{BN}_l(W*x_l),
\qquad l\in\lbrace3,4,5\rbrace
```

여기서 `W`는 공유되고 `BN_l`은 scale별이다.

| Head | Params | FLOPs | Latency | AP |
| --- | ---: | ---: | ---: | ---: |
| Shared Head | 52.32M | 80.23G | 2.44 ms | 48.0 |
| Totally Separate | 57.03M | 80.23G | 2.44 ms | 51.2 |
| Separate BN | 52.32M | 80.23G | 2.44 ms | **51.3** |

주의할 deployment 세부가 있다. BN을 conv에 fold하면 scale별 `gamma/sigma` 때문에 effective convolution weight가 달라진다. Compiler가 scale-shared weight를 보존하지 않고 세 벌로 복제할 수 있으므로, source model의 parameter 절감이 engine binary에서도 그대로 유지되는지 확인해야 한다.

## Dynamic soft label assignment

RTMDet는 SimOTA를 기반으로 prediction-ground-truth pair의 matching cost를 바꾼다.

```math
C=\lambda_1C_{cls}+\lambda_2C_{reg}+\lambda_3C_{center}
```

기본 coefficient는 다음과 같다.

```math
\lambda_1=1,\qquad\lambda_2=3,\qquad\lambda_3=1
```

### Soft classification cost

Binary target 대신 predicted box와 ground truth의 IoU를 soft label로 사용한다.

```math
Y_{soft}=\mathrm{IoU}(b_{pred},b_{gt})
```

```math
C_{cls}=\mathrm{CE}(P,Y_{soft})(Y_{soft}-P)^2
```

Classification score가 높지만 box가 나쁜 prediction을 좋은 match로 선택하는 문제를 줄인다. `(Y_soft-P)²` 항은 confidence와 localization quality가 어긋난 pair에 더 큰 cost를 준다.

### Log-IoU regression cost

Training loss와 assignment cost를 꼭 같게 둘 필요가 없다는 것이 중요한 관찰이다. Matching에서는 low-IoU pair를 더 강하게 벌주기 위해 다음을 쓴다.

```math
C_{reg}=-\log(\mathrm{IoU})
```

IoU가 1에 가까우면 cost가 0에 접근하고, 0에 가까우면 급격히 커져 좋은 match와 나쁜 match의 구분이 커진다. 수치 안정성을 위해 실제 구현에서 epsilon을 넣는지 official code를 확인해야 하며, 논문 식에는 epsilon이 적혀 있지 않다.

### Soft center prior

고정된 3×3 center region 대신 continuous distance cost를 사용한다.

```math
C_{center}=\alpha|x_{pred}-x_{gt}|-\beta
```

논문 기본값은 `alpha=10`, `beta=3`이다. 식의 좌표 정규화와 vector distance의 세부 구현은 본문만으로 충분히 규정되지 않으므로 code를 기준으로 재현해야 한다.

### Assignment cost와 training loss를 구분해야 한다

위 식들은 positive를 고르는 **matching cost**다. Table 6의 baseline 설명에서 training에는 Focal Loss와 GIoU loss를 사용한다. Matching cost를 그대로 최종 loss라고 쓰면 잘못이다.

## Assignment의 tensor shape 예

Ground truth가 `G`개이고 후보 위치가 `N=8400`이면 pairwise cost matrix는 다음 shape다.

```math
C\in\mathbb{R}^{G\times N}
```

Reviewer 예시로 한 image에 `G=20`이면 `168,000` cost element다. FP32 payload만 약 `0.641 MiB`지만 IoU, classification, center cost를 각각 materialize하고 top-k workspace를 쓰면 더 커진다. Assignment는 training-only라 inference latency에는 없지만 training peak memory에는 영향을 준다.

## Cached Mosaic/MixUp과 2-stage training

일반 Mosaic와 MixUp은 한 sample을 만들 때 여러 image를 추가로 load한다. RTMDet는 최근 image를 cache하여 data loader I/O를 줄인다.

| Augmentation | 일반 | Cached | 100 images 시간 |
| --- | --- | --- | ---: |
| Mosaic | 매번 추가 load | cache 재사용 | `87.1 -> 24.0 ms` |
| MixUp | 매번 추가 load | cache 재사용 | `19.3 -> 12.4 ms` |

첫 280 epoch에는 cached Mosaic/MixUp으로 8개 image를 섞고, 마지막 20 epoch에는 LSJ로 전환한다. Random rotation과 shear는 transformed box와 image가 어긋날 수 있어 제외한다.

Small model에서는 cache 약 10개와 FIFO popping이 repeated augmentation처럼 작동해 조금 더 좋았지만 large model에서는 동일한 이득이 없다. Augmentation strength는 model capacity에 따라 달라야 한다는 결과다.

## Optimization

- Optimizer: AdamW
- Base learning rate: 0.004
- Weight decay: 0.05, bias와 normalization에는 0
- Batch: 256
- Epoch: 300
- Warm-up: 1,000 iteration
- Schedule: Flat-Cosine
- EMA decay: 0.9998
- Input: 640×640
- Training hardware: 8 NVIDIA A100

Flat-Cosine은 앞 절반에서 learning rate를 일정하게 유지하고 뒤 절반에 cosine decay를 적용한다. Heavy augmentation에서 SGD는 논문 실험상 불안정했고, AdamW + CosineLR 43.0 AP에서 flat-cosine 43.3, norm/bias decay 제거 44.2, RSB pretrained backbone 44.5로 개선된다.

이 결과는 optimizer 하나만의 보편적 우월성을 뜻하지 않는다. Augmentation, schedule, weight decay 예외와 함께 측정한 recipe 결과다.

## Instance segmentation과 rotated detection 확장

### RTMDet-Ins

Mask feature head는 multi-level feature에서 8-channel mask feature를 만들고, kernel head는 instance마다 169차원 vector를 예측한다. 이는 세 dynamic convolution kernel용 `88 + 72 + 9` parameter로 나뉜다. Relative coordinate 2 channel을 mask feature와 concatenate하고 Dice loss로 mask를 감독한다.

```text
mask features: 8 channels
 + relative x/y: 2 channels
 -> three dynamic convs using 169 predicted parameters
 -> instance mask
```

RTMDet-Ins-x는 102.7M, 182.7G, 5.31 ms, box AP 52.4, mask AP 44.6이다. 이 instance 표의 latency는 detection과 달리 box NMS와 top-100 mask post-processing을 포함한다.

### RTMDet-R

Rotated detection은 다음 세 변경만 사용한다.

1. Regression branch에 angle용 1×1 convolution 추가
2. Rotated box coder 사용
3. GIoU loss를 Rotated IoU loss로 교체

COCO-pretrained RTMDet-R-l은 DOTA-v1.0 multi-scale에서 mAP 81.33을 보고한다. Multi-scale test 결과이므로 mobile single-scale latency와 직접 비교해서는 안 된다.

## 구현 pseudocode

### Separate-BN shared head

```python
class SepBNHead:
    def __init__(self, shared_convs, num_levels=3):
        self.shared_convs = shared_convs
        self.bn_cls = [make_bn() for _ in range(num_levels)]
        self.bn_reg = [make_bn() for _ in range(num_levels)]

    def forward_level(self, x, level):
        cls = x
        reg = x
        for conv in self.shared_convs.cls:
            cls = silu(self.bn_cls[level](conv(cls)))
        for conv in self.shared_convs.reg:
            reg = silu(self.bn_reg[level](conv(reg)))
        return self.classifier(cls), self.box_regressor(reg)
```

실제 block마다 separate BN이 필요하면 위 list는 layer와 level의 2D 구조가 된다. 핵심은 convolution parameter object는 공유하고 running mean/variance와 affine BN parameter는 공유하지 않는 것이다.

### Dynamic soft assignment

```python
def pair_cost(pred_score, pred_box, gt_box, center_distance):
    iou = pairwise_iou(pred_box, gt_box).clamp_min(1e-8)
    soft_target = iou

    cls_cost = cross_entropy_soft(pred_score, soft_target)
    cls_cost = cls_cost * (soft_target - pred_score).square()

    reg_cost = -iou.log()
    center_cost = 10.0 * center_distance.abs() - 3.0
    return cls_cost + 3.0 * reg_cost + center_cost

def assign(cost, pairwise_iou, valid_center_mask):
    cost = mask_invalid(cost, valid_center_mask)
    dynamic_k = estimate_dynamic_k(pairwise_iou)
    return select_lowest_cost_per_gt(cost, dynamic_k)
```

위 코드는 개념 pseudocode다. 충돌하는 prediction을 여러 GT가 선택했을 때의 conflict resolution과 dynamic-k 계산은 official implementation을 따라야 한다.

## 주요 benchmark

논문 Table 2의 RTMDet family는 다음과 같다.

| Model | Params | FLOPs, 논문 표기 | Latency | AP | AP50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny | 4.8M | 8.1G | 0.98 ms | 41.1 | 57.9 |
| s | 8.99M | 14.8G | 1.22 ms | 44.6 | 61.9 |
| m | 24.7M | 39.3G | 1.62 ms | 49.4 | 66.8 |
| l | 52.3M | 80.2G | 2.40 ms | 51.5 | 68.8 |
| x | 94.9M | 141.7G | 3.10 ms | 52.8 | 70.4 |

`FLOPs`는 논문 표의 명칭을 그대로 썼다. 이를 MACs로 바꿀 때 multiply-add를 1 operation으로 셌는지 2 FLOPs로 셌는지 profiler convention을 확인해야 한다.

Weight만 저장한다고 가정한 reviewer 하한은 다음과 같다.

| Model | FP32 weight | FP16 weight | INT8 weight |
| --- | ---: | ---: | ---: |
| tiny, 4.8M | 18.3 MiB | 9.2 MiB | 4.6 MiB |
| s, 8.99M | 34.3 MiB | 17.1 MiB | 8.6 MiB |
| x, 94.9M | 362.0 MiB | 181.0 MiB | 90.5 MiB |

Scale, zero-point, alignment, engine graph와 workspace는 제외했다. 논문은 quantized model size나 peak activation을 직접 보고하지 않는다.

## 핵심 ablation

### Kernel size

| Spatial kernel | Params | FLOPs | Latency | AP |
| --- | ---: | ---: | ---: | ---: |
| 3×3 | 50.80M | 79.61G | 2.10 ms | 50.0 |
| 5×5 | 50.92M | 79.70G | 2.11 ms | 50.9 |
| 7×7 | 51.10M | 80.34G | 2.73 ms | 51.1 |

5×5는 3×3 대비 latency +0.01 ms로 AP +0.9를 얻고, 7×7은 추가 AP +0.2에 latency가 크게 증가한다. 이 결론은 RTX 3090 kernel 구현에 종속될 수 있다.

### Label assignment

| 구성 | AP |
| --- | ---: |
| baseline | 39.9 |
| + soft classification cost | 40.3 |
| + soft center prior | 40.8 |
| + log-IoU cost | 41.3 |

같은 ResNet50 1x schedule에서 ATSS 39.2, PAA 40.4, OTA 40.7, TOOD without T-Head 40.7, RTMDet assignment 41.3이다. Longer RTMDet-s setting에서는 SimOTA 43.2 대비 제안 방식 44.5다.

### Data augmentation

RTMDet-s에서 cached Mosaic/MixUp을 끝까지 쓰면 AP 41.9지만 마지막 20 epoch를 LSJ로 바꾸면 43.9다. Small cache + FIFO까지 적용하면 44.2다. RTMDet-l에서는 같은 전환이 `49.8 -> 51.3`이다.

### Step-by-step YOLOX-s 개선

| 단계 | Params | FLOPs | Latency | AP |
| --- | ---: | ---: | ---: | ---: |
| YOLOX baseline | 9.0M | 13.4G | 1.20 ms | 40.2 |
| + AdamW, Flat-Cosine | 9.0M | 13.4G | 1.20 ms | 40.6 |
| + new architecture | 10.07M | 14.8G | 1.22 ms | 41.8 |
| + SepBNHead | 8.89M | 14.8G | 1.22 ms | 41.8 |
| + assignment/loss | 8.89M | 14.8G | 1.22 ms | 42.9 |
| + augmentation | 8.89M | 14.8G | 1.22 ms | 44.2 |
| + RSB pretraining | 8.89M | 14.8G | 1.22 ms | 44.5 |

Architecture alone뿐 아니라 training recipe가 최종 +4.3 AP 중 큰 부분을 차지한다. 따라서 checkpoint 비교에서 recipe를 통제하지 않으면 architecture의 기여를 과대평가한다.

## 장점

- Architecture, assignment, augmentation, optimizer를 단계별로 분리한 ablation이 풍부하다.
- Parameter/FLOPs뿐 아니라 동일 RTX 3090 TensorRT 환경의 batch-1 latency를 제공한다.
- Backbone-neck capacity 배치와 serial depth처럼 실제 latency를 좌우하는 요소를 다룬다.
- Scale-shared weight와 separate BN이라는 간단한 head 절충이 효과적이다.
- Detection에서 끝나지 않고 instance와 rotated task의 최소 수정 경로를 제시한다.
- Re-parameterization 없이도 강한 결과를 보여 quantization/deployment 논의를 열어 둔다.

## 한계와 비판적 관점

### 1. Mobile latency는 아니다

모든 주요 latency는 RTX 3090 TensorRT FP16이다. GPU에서 유리한 shallow-wide 선택이 mobile CPU/NPU에서도 같은 순서를 보장하지 않는다.

### 2. Detection latency에서 NMS가 제외된다

Score threshold는 validation에서 0.001, top 300을 유지한다. Candidate가 많은 장면에서는 NMS와 NPU-CPU transfer가 p95를 지배할 수 있다. Ablation은 계산을 빠르게 하려고 threshold 0.05, top 100을 쓰며 약 0.3 AP 저하 가능성을 명시한다. 서로 다른 표의 평가 setting을 섞으면 안 된다.

### 3. Peak activation과 bandwidth를 보고하지 않는다

PAFPN의 resize, concat, CSP branch가 만드는 temporary memory가 중요하지만 parameter/FLOPs 표만 있다. 온디바이스 OOM 위험을 판단하기 부족하다.

### 4. INT8 주장을 직접 검증하지 않는다

Re-parameterized convolution보다 quantization error가 작다고 논의하지만 PTQ/QAT accuracy, calibration, per-channel scale 결과는 없다.

### 5. 많은 개선이 recipe와 결합된다

Step-by-step 표가 이를 잘 드러내지만 최종 SOTA 비교는 pretraining과 augmentation까지 포함한다. 순수 architecture 우위로만 읽으면 안 된다.

### 6. Label assignment 식의 구현 세부가 부족하다

Center distance normalization, epsilon, tie/conflict resolution은 공식 code 확인이 필요하다. 논문 식만으로 bit-exact 재현은 어렵다.

## 온디바이스 operator와 memory 분석

RTMDet의 operator는 Transformer detector보다 전통적인 NPU 친화 연산에 가깝다.

- `Conv/DepthwiseConv`: 일반적으로 지원되지만 5×5 depthwise kernel 최적화 여부 확인
- `BN`: inference 때 fold 가능
- `SiLU`: backend에 따라 native 또는 sigmoid-multiply decomposition
- `Resize + Concat`: PAFPN에서 memory traffic과 allocation을 유발
- `TopK/NMS`: CPU fallback과 synchronization 가능성
- Dynamic mask conv: instance version에서 runtime 지원 난도가 상승
- Rotated NMS/IoU: 일반 NMS보다 지원이 제한적

특히 `P3 80×80` 고해상도 branch는 activation과 DRAM traffic 비중이 크다. Width를 늘리는 shallow-wide 설계는 GPU에서는 병렬화에 유리하지만 SRAM tile에 feature가 들어가지 않으면 mobile NPU에서는 spill이 늘 수 있다.

실험에서는 다음을 반드시 profile해야 한다.

```text
preprocess
 -> backbone
 -> PAFPN resize/concat
 -> head
 -> device-to-host output
 -> decode/top-k/NMS
```

한 덩어리 latency만 재기보다 구간별 p50/p95와 transferred bytes를 기록해야 bottleneck을 정확히 찾을 수 있다.

## 재현 체크리스트

- [ ] arXiv v2, MMDetection commit, config hash를 기록했는가?
- [ ] COCO train2017/val2017와 300 epoch를 고정했는가?
- [ ] Input 640×640, batch 256, 8 A100 training 조건 차이를 기록했는가?
- [ ] AdamW LR 0.004, weight decay 0.05, norm/bias decay 0을 적용했는가?
- [ ] Flat-Cosine, warm-up 1,000, EMA 0.9998을 재현했는가?
- [ ] Cached Mosaic/MixUp 280 epoch와 LSJ 20 epoch를 구분했는가?
- [ ] Tiny/small과 large의 augmentation scale 범위를 구분했는가?
- [ ] Conv weight를 scale 간 공유하고 BN statistics는 분리했는가?
- [ ] Assignment cost와 final training loss를 혼동하지 않았는가?
- [ ] Validation threshold 0.001/top 300과 ablation threshold 0.05/top 100을 구분했는가?
- [ ] TensorRT FP16 batch 1에서 warm-up과 반복 수를 기록했는가?
- [ ] Detection latency에 NMS 포함/제외 두 값을 모두 측정했는가?
- [ ] BN folding 후 weight sharing이 engine에서도 유지되는지 확인했는가?
- [ ] Target device의 5×5 depthwise, SiLU, resize, concat 지원을 확인했는가?
- [ ] p50/p95, peak memory, energy, thermal throttling을 기록했는가?

## 로드맵에서의 연결

RTMDet는 Faster R-CNN/FPN, RetinaNet/FCOS 다음에 읽으면 real-time one-stage detector의 구성요소가 어떻게 합쳐지는지 보여준다. YOLOX와 직접적인 관계도 명확하다.

```text
FPN/PAN: multi-scale fusion
 + FCOS: anchor-free dense prediction
 + YOLOX: decoupled head, SimOTA, strong augmentation
 -> RTMDet: large-kernel block, balanced neck, SepBN, soft assignment
```

DETR 계열과 비교할 때는 다음 차이를 중심으로 봐야 한다.

| 관점 | RTMDet | DETR/RT-DETR 계열 |
| --- | --- | --- |
| Prediction | Dense spatial points | Fixed object queries |
| Assignment | Dynamic one-to-many training | Bipartite set matching 중심 |
| Post-process | NMS 필요 | NMS-free 지향 |
| Main operators | Conv, resize, concat | Attention, sampling, LayerNorm 포함 |
| Mobile risk | Feature traffic, NMS | Unsupported attention/sampling, decoder |

로드맵 산출물에서는 RTMDet-tiny/s와 RT-DETRv2-S를 동일 device, `640×640`, batch 1로 export하고 AP뿐 아니라 NMS 포함 p95와 small-object AP를 비교하는 것이 좋다.

## 최종 평가

RTMDet는 "최신 block 하나"보다 **전체 detector를 latency와 학습 안정성 관점에서 균형 있게 설계하는 방법**을 잘 보여준다. 5×5 depthwise block, neck capacity, SepBN head, soft assignment 각각은 단순하지만, 풍부한 ablation을 통해 어떤 개선이 얼마만큼 기여했는지 설명한다.

동시에 논문의 real-time 주장은 RTX 3090 TensorRT FP16과 NMS 제외 조건에 한정된다. 온디바이스에서는 5×5 depthwise kernel, PAFPN의 resize/concat, BN folding 뒤 weight duplication, NMS fallback이 결과를 뒤집을 수 있다. 따라서 RTMDet를 모바일 baseline으로 채택할 때는 parameter/AP 표보다 **operator partition, activation peak, end-to-end p95**를 다시 측정해야 한다.

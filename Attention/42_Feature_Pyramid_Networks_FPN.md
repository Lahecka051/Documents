# 42. Feature Pyramid Networks for Object Detection

## 논문 정보

- 원본 파일: `42_Feature_Pyramid_Networks_FPN.pdf`
- 제목: **Feature Pyramid Networks for Object Detection**
- 저자: Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie
- 발표: CVPR 2017
- 공개: arXiv:1612.03144, 제공된 PDF는 v2(2017)
- 링크: [https://arxiv.org/abs/1612.03144](https://arxiv.org/abs/1612.03144)
- 핵심 키워드: Feature Pyramid Network, multi-scale detection, top-down pathway, lateral connection, semantic hierarchy, RPN, Faster R-CNN, small objects

## 한눈에 보는 요약

Feature Pyramid Network(FPN)는 CNN backbone이 이미 계산한 서로 다른 해상도의 stage feature를 이용해, **모든 scale에서 semantic이 강한 feature pyramid**를 만드는 구조다. 입력 image pyramid를 여러 번 backbone에 통과시키지 않고도 작은 객체는 높은 해상도에서, 큰 객체는 낮은 해상도에서 처리할 수 있게 한다.

```text
bottom-up ResNet:
C2 ------ C3 ------ C4 ------ C5
 |         |         |         |
1x1       1x1       1x1       1x1
 |         |         |         |
P2 <----- P3 <----- P4 <----- P5
      2x up + add at each step

each merged map -> 3x3 conv -> final P2, P3, P4, P5
```

Bottom-up의 얕은 feature는 위치가 정밀하지만 semantic이 약하고, 깊은 feature는 semantic이 강하지만 해상도가 낮다. FPN은 깊은 feature를 2배 upsample하고 같은 해상도의 얕은 feature를 lateral `1×1` convolution으로 channel 정렬해 더한다. 그 결과 `P2-P5`는 해상도는 다르지만 모두 강한 object-level semantic을 갖는다.

FPN의 중요한 차별점은 가장 높은 해상도의 map 하나로 합치는 U-Net형 decoder가 아니라, **여러 pyramid level 각각에서 독립적으로 prediction**한다는 점이다. RPN에서는 각 level에 한 base scale의 anchor를 배치하고, Fast R-CNN에서는 RoI 크기에 따라 적절한 level 하나를 선택한다.

ResNet-50 Faster R-CNN의 COCO minival에서 FPN은 강한 single-scale baseline의 `31.6 AP`를 `33.9 AP`로, small-object AP를 `13.2`에서 `17.8`로 높였다. RPN proposal AR@1000은 `48.3`에서 `56.3`, small-object AR은 `32.0`에서 `44.9`로 올랐다. 논문은 peak memory와 모바일 latency를 보고하지 않지만, FPN의 가장 큰 activation은 고해상도 `P2`이므로 온디바이스에서는 정확도 향상만큼 memory budget을 면밀히 봐야 한다.

## 배경: 왜 image pyramid가 필요했는가

### 객체 scale 변화

같은 category라도 image에서 차지하는 크기는 크게 다르다. 작은 객체와 큰 객체를 동일 stride, 동일 receptive field의 feature에서 찾으면 어느 한쪽에 불리하다. 고전 detector는 입력 image를 여러 크기로 resize해 각 scale에서 HOG/SIFT feature를 다시 계산했다.

```text
image x 0.5 -> feature -> predict
image x 1.0 -> feature -> predict
image x 2.0 -> feature -> predict
```

이 방식은 scale 변화가 pyramid level 이동으로 흡수되는 장점이 있지만, backbone을 scale마다 반복하므로 느리고 memory가 크다. 논문은 기존 multi-scale inference가 single-scale보다 약 4배 느려질 수 있고, deep network를 image pyramid 전체로 end-to-end 학습하는 것은 당시 memory상 어렵다고 지적한다.

### CNN stage를 그대로 쓰면 왜 부족한가

CNN은 downsampling 때문에 이미 자연스러운 feature hierarchy를 만든다.

```text
C2: high resolution, shallow, weak semantics
C3: medium-high resolution
C4: medium-low resolution
C5: low resolution, deep, strong semantics
```

이를 그대로 prediction pyramid로 쓰는 SSD형 접근은 계산 재사용 면에서 매력적이다. 하지만 작은 객체를 담당할 얕은 high-resolution map이 edge와 texture 중심이고 category semantic은 약하다. 반대로 C5만 쓰면 semantic은 강하지만 작은 객체의 spatial detail이 사라진다.

FPN의 질문은 다음과 같다.

```text
깊은 layer의 semantic을 얕은 고해상도 layer로 전달하면서,
각 해상도를 독립적인 detection level로 유지할 수 있는가?
```

## FPN의 세 구성 요소

### Bottom-up pathway

ResNet의 stage 마지막 출력들을 사용한다.

```math
\{C_2,C_3,C_4,C_5\}
```

Input 대비 stride는 각각 `{4,8,16,32}`다. Conv1은 해상도가 너무 높아 memory 부담이 크므로 pyramid에 넣지 않는다. ResNet-50/101의 channel은 통상 다음과 같다.

| Feature | stride | channels |
| --- | ---: | ---: |
| `C2` | 4 | 256 |
| `C3` | 8 | 512 |
| `C4` | 16 | 1024 |
| `C5` | 32 | 2048 |

각 stage의 마지막 block을 고르는 이유는 같은 해상도 안에서 가장 깊어 semantic이 가장 강하기 때문이다.

### Top-down pathway

가장 깊은 C5를 `1×1` convolution으로 256 channel에 투영해 top-down 시작점으로 삼는다. 이후 현재 top-down feature를 nearest-neighbor로 2배 upsample한다.

```math
\begin{aligned}
M_5 &= \operatorname{Conv}_{1\times1}(C_5),\\
M_l &= \operatorname{Conv}_{1\times1}(C_l)
      + \operatorname{Up}_2(M_{l+1}),\qquad l=4,3,2.
\end{aligned}
```

여기서 `M_l`은 merge 직후의 중간 feature다. Upsampling만으로는 새 detail을 복원하지 못한다. 위치 정보는 다음 lateral connection이 공급한다.

### Lateral connection

Bottom-up `C_l`을 `1×1` convolution으로 256 channel에 맞춘 뒤 top-down feature와 element-wise addition한다. Concatenation이 아니라 addition이므로 channel 수가 늘지 않는다.

```math
P_l=\operatorname{Conv}_{3\times3}(M_l),\qquad l=2,3,4,5.
```

마지막 `3×3` convolution은 upsampling에서 생길 수 있는 aliasing을 완화한다. 모든 `P_l`은 256 channel이며, 논문의 FPN extra layer에는 별도 nonlinearity가 없다.

`1×1 lateral`과 `3×3 smoothing`의 역할을 구분해야 한다.

- `1×1`: bottom-up channel 정렬과 semantic projection
- upsample + add: 깊은 semantic과 정밀한 위치 결합
- `3×3`: merge 결과를 각 prediction level용 feature로 정제

## FPN은 U-Net과 무엇이 다른가

둘 다 top-down과 skip/lateral connection을 사용하지만 출력 사용법이 다르다.

| 관점 | U-Net형 decoder | FPN |
| --- | --- | --- |
| 주 출력 | 가장 높은 해상도의 한 map | 여러 해상도의 `P2-P5` |
| prediction | 보통 finest map에서 dense output | 각 level에서 독립 prediction |
| scale 처리 | 합쳐진 high-resolution representation | 객체 크기를 pyramid level로 분산 |
| detection head | 한 output head | level 간 공유 head 가능 |

따라서 단순히 top-down skip connection이 있다고 FPN인 것은 아니다. **Pyramid 전체를 prediction space로 사용**해야 FPN의 scale-normalized 의미가 살아난다.

## FPN을 RPN에 적용하기

### Level별 anchor

원래 RPN은 한 feature map 위치에 여러 scale의 anchor를 놓는다. FPN-RPN은 scale dimension을 pyramid level로 옮긴다.

| Level | stride | anchor area scale | ratios |
| --- | ---: | ---: | --- |
| `P2` | 4 | `32²` | `1:2, 1:1, 2:1` |
| `P3` | 8 | `64²` | 동일 |
| `P4` | 16 | `128²` | 동일 |
| `P5` | 32 | `256²` | 동일 |
| `P6` | 64 | `512²` | 동일 |

P6는 큰 anchor를 위해 P5를 stride 2로 downsample한 level이며 Fast R-CNN RoI에는 쓰지 않는다. 각 level에는 scale 하나와 ratio 세 개, 총 3 anchors/location만 둔다. 전체 pyramid 관점에서는 5 scales × 3 ratios, 15종이다.

Anchor label은 Faster R-CNN과 같다.

- 어떤 GT의 최고 IoU anchor 또는 IoU > 0.7: positive
- 모든 GT와 IoU < 0.3: negative
- 나머지: ignore

GT 크기를 보고 level을 직접 지정하지 않는다. GT는 anchor IoU matching을 통해 자연스럽게 특정 level과 연결된다.

### Shared RPN head

각 `P2-P6`에 동일한 `3×3 conv -> cls/reg 1×1 conv` head를 적용하고 parameter를 공유한다. Level별 별도 head도 비슷한 accuracy였지만, 공유 head가 잘 작동했다는 사실은 각 P level의 semantic 수준이 충분히 정렬되었다는 증거다.

공유하지 않은 bottom-up raw hierarchy에서는 C2와 C5의 feature 의미가 크게 달라 동일 classifier를 적용하기 어렵다. FPN의 top-down enrichment가 이를 줄인다.

## FPN을 Fast R-CNN에 적용하기

### RoI-to-level assignment

입력 image 좌표에서 width `w`, height `h`인 RoI를 다음 level에 할당한다.

```math
k=\left\lfloor k_0+\log_2\left(\frac{\sqrt{wh}}{224}\right)\right\rfloor,
\qquad k_0=4.
```

`sqrt(wh)`는 RoI의 대표 scale이다. `224×224` RoI는 P4에 간다.

| 정사각 RoI | 계산 | 할당 |
| ---: | ---: | ---: |
| `56×56` | `4+log2(56/224)=2` | P2 |
| `112×112` | `3` | P3 |
| `224×224` | `4` | P4 |
| `448×448` | `5` | P5 |

범위를 벗어나면 구현에서 P2-P5로 clamp한다. 작은 RoI가 high-resolution P2로 가므로 pooling 전에 더 많은 spatial sample을 유지한다.

### Detection head

각 RoI는 선택된 level에서 `7×7` RoI pooling을 거친다. 이후 `1024-d` hidden layer 두 개의 2-fc MLP가 classification과 box regression 앞에 놓인다.

ResNet 기반 기존 Faster R-CNN은 C4에서 `14×14` RoI를 뽑고 무거운 conv5 stage를 RoI마다 실행했다. FPN은 C5를 이미 pyramid 생성에 사용하므로 더 가벼운 2-fc head를 쓴다. 이 차이는 논문에서 FPN system이 baseline보다 빠른 이유 중 하나이며, 속도 향상을 FPN fusion 자체가 공짜라서라고만 해석하면 안 된다.

## Batch=1 tensor shape 계산

### 800×1024 입력 예시

논문의 COCO 입력 조건인 짧은 변 800에 맞춰 `B=1`, `800×1024` image와 ResNet을 가정한다. 실제 COCO image의 긴 변과 padding 규칙에 따라 shape가 달라진다. 아래는 **리뷰어 계산**이다.

| Tensor | NCHW shape | elements | FP16 | FP32 |
| --- | ---: | ---: | ---: | ---: |
| `C2` | `1×256×200×256` | 13,107,200 | 25.00 MiB | 50.00 MiB |
| `C3` | `1×512×100×128` | 6,553,600 | 12.50 MiB | 25.00 MiB |
| `C4` | `1×1024×50×64` | 3,276,800 | 6.25 MiB | 12.50 MiB |
| `C5` | `1×2048×25×32` | 1,638,400 | 3.13 MiB | 6.25 MiB |
| `P2` | `1×256×200×256` | 13,107,200 | 25.00 MiB | 50.00 MiB |
| `P3` | `1×256×100×128` | 3,276,800 | 6.25 MiB | 12.50 MiB |
| `P4` | `1×256×50×64` | 819,200 | 1.56 MiB | 3.13 MiB |
| `P5` | `1×256×25×32` | 204,800 | 0.39 MiB | 0.78 MiB |
| `P6` | `1×256×13×16` | 53,248 | 0.10 MiB | 0.20 MiB |

Final pyramid `P2-P6` payload만 FP16 약 `33.3 MiB`, FP32 약 `66.6 MiB`다. 여기에 동시에 live한 lateral map, upsample buffer, bottom-up C feature, RPN output과 runtime workspace가 더해진다. 특히 P2 하나가 pyramid activation의 약 75%를 차지한다.

### Anchor 수

각 위치에 ratio 세 개를 놓으면:

```math
\begin{aligned}
P2 &: 200\cdot256\cdot3=153{,}600,\\
P3 &: 100\cdot128\cdot3=38{,}400,\\
P4 &: 50\cdot64\cdot3=9{,}600,\\
P5 &: 25\cdot32\cdot3=2{,}400,\\
P6 &: 13\cdot16\cdot3=624.
\end{aligned}
```

합계는 `204,624 anchors`로 논문 Table 1의 FPN 약 `200k`와 맞는다. 그중 75%가 P2에서 생긴다. 따라서 small-object coverage의 이점과 dense candidate 비용이 같은 level에서 발생한다.

### RPN 출력 shape

2-class softmax와 3 anchors/location을 가정하면 각 level에서:

```text
objectness: B x 6 x Hl x Wl
box delta:  B x 12 x Hl x Wl
```

Anchor tensor를 명시적으로 `[204624,4]`로 materialize하면 FP32 payload는 약 `3.12 MiB`다. 하지만 IoU matching은 `#anchors × #GT` 임시 행렬을 만들 수 있어 training memory가 더 커진다.

## Parameter와 MAC 분석

### FPN extra parameter: 리뷰어 계산

ResNet C2-C5를 256 channel로 만드는 lateral `1×1` weight는 bias 제외:

```math
256(256+512+1024+2048)=983{,}040.
```

P2-P5의 smoothing `3×3, 256→256` 네 개는:

```math
4\cdot3\cdot3\cdot256\cdot256=2{,}359{,}296.
```

합계는 `3,342,336 weights`, FP16 약 `6.38 MiB`, FP32 약 `12.75 MiB`다. P6를 단순 downsample하는 논문 구성을 기준으로 했다. 이는 backbone·RPN·2-fc head를 제외한 FPN 자체다.

### FPN extra MAC: 리뷰어 계산

`800×1024`에서 ideal MAC을 계산하면 lateral 1×1 합계가 약 `6.29G MACs`, P2-P5의 3×3 smoothing 합계가 약 `40.11G MACs`, 총 약 `46.40G MACs`다.

가장 큰 항은 P2 smoothing이다.

```math
200\cdot256\cdot256\cdot(3\cdot3\cdot256)
\approx30.20\ \text{G MACs}.
```

논문의 "marginal extra cost"는 image pyramid처럼 backbone 전체를 여러 번 실행하는 비용과 비교한 표현이다. 절대 연산과 activation이 작다는 뜻은 아니다. 특히 모바일에서는 high-resolution 256-channel 3×3 conv가 상당한 부담이다.

### 논문 보고와 계산의 구분

논문이 직접 보고한 것은 전체 system AP, AR, GPU runtime이다. FPN만의 parameter/MAC/peak memory는 표로 보고하지 않는다. 위 수치는 shape에 기반한 리뷰어 계산이며, 실제 profiler FLOPs는 padding, bias, framework count convention에 따라 달라질 수 있다.

## 구현 pseudocode

```python
def build_fpn(c2, c3, c4, c5):
    m5 = lateral5(c5)                  # 1x1 -> 256
    m4 = lateral4(c4) + upsample2x(m5)
    m3 = lateral3(c3) + upsample2x(m4)
    m2 = lateral2(c2) + upsample2x(m3)

    p5 = smooth5(m5)                   # 3x3 -> 256
    p4 = smooth4(m4)
    p3 = smooth3(m3)
    p2 = smooth2(m2)
    p6 = stride2_subsample(p5)
    return [p2, p3, p4, p5, p6]


def fpn_rpn(pyramid):
    all_logits, all_delta, all_anchors = [], [], []
    for level, feat in enumerate(pyramid, start=2):
        h = relu(shared_rpn_3x3(feat))
        all_logits.append(shared_rpn_cls(h))
        all_delta.append(shared_rpn_reg(h))
        all_anchors.append(make_level_anchors(level, ratios=(.5, 1, 2)))
    return concat(all_logits), concat(all_delta), concat(all_anchors)


def assign_roi_level(boxes):
    w, h = boxes.width, boxes.height
    k = floor(4 + log2(sqrt(w * h) / 224))
    return clamp(k, min=2, max=5)


def roi_head(p2_to_p5, proposals):
    level = assign_roi_level(proposals)
    pooled = empty_in_original_order()
    for k in (2, 3, 4, 5):
        roi_k = roi_pool(p2_to_p5[k], proposals[level == k], (7, 7))
        pooled.scatter_back(roi_k)
    z = relu(fc1(pooled.flatten(1)))
    z = relu(fc2(z))
    return cls(z), box_reg(z)
```

중요한 구현 세부는 upsample 후 shape가 홀수일 때 lateral tensor와 정확히 맞추는 것, proposal 순서를 level별 gather 후 다시 원래 순서로 복원하는 것, anchor stride와 feature padding convention을 일치시키는 것이다.

## 실험 설정

- Dataset: COCO trainval35k 학습, minival 5k ablation, test-dev/test-std 최종 평가
- Backbone: ImageNet-1k pretrained ResNet-50/101
- 입력: 짧은 변 800
- RPN training: 8 GPU, GPU당 2 images, image당 256 anchors
- RPN schedule: LR 0.02 for 30k mini-batches, 0.002 for 10k
- Fast R-CNN: image당 512 RoIs, 2,000 train proposals, 1,000 test proposals
- Detector schedule: LR 0.02 for 60k, 0.002 for 20k
- Momentum 0.9, weight decay 0.0001

논문은 all architectures를 end-to-end 학습하며 RPN 실험에서는 image 밖 anchor도 training에 포함한다. 이는 원 Faster R-CNN의 기본 구현과 다른 세부이므로 재현 시 중요하다.

## 주요 결과

### RPN proposal ablation

ResNet-50, COCO minival:

| RPN feature | anchors | AR100 | AR1k | AR1k small | AR1k medium | AR1k large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single `C4` | 47k | 36.1 | 48.3 | 32.0 | 58.7 | 62.2 |
| single `C5` | 12k | 36.3 | 44.9 | 25.3 | 55.5 | 64.2 |
| full FPN | 200k | 44.0 | 56.3 | 44.9 | 63.4 | 66.2 |

FPN은 C4 baseline 대비 AR1k `+8.0`, small AR `+12.9`다. 큰 객체 개선보다 작은 객체 개선이 훨씬 크므로, FPN의 주된 효과가 단순 parameter 증가가 아니라 적절한 scale의 semantic feature 제공임을 보여 준다.

### Faster R-CNN

ResNet-50, COCO minival의 consistent backbone 비교:

| System | AP@0.5 | AP | AP small | AP medium | AP large |
| --- | ---: | ---: | ---: | ---: | ---: |
| Faster R-CNN on C4 | 53.1 | 31.6 | 13.2 | 35.6 | 47.1 |
| Faster R-CNN on C5 | 51.7 | 28.0 | 9.6 | 31.9 | 43.1 |
| Faster R-CNN on FPN | 56.9 | 33.9 | 17.8 | 37.7 | 45.8 |

C4 baseline 대비 `+2.3 AP`, `+3.8 AP@0.5`, `+4.6 AP_S`다. Large AP는 C4 baseline보다 약간 낮지만 overall과 small/medium이 개선된다.

### COCO test set

ResNet-101 FPN single model은:

- test-dev: `36.2 AP`, `59.1 AP@0.5`, `18.2 AP_S`, `39.0 AP_M`, `48.2 AP_L`
- test-std: `35.8 AP`, `58.5 AP@0.5`, `17.5 AP_S`, `38.7 AP_M`, `47.8 AP_L`

당시 competition winner 단일 모델과 비교해 test-dev AP 최고치를 0.5 높이고 AP@0.5를 3.4 높였다. Image pyramid 없이 single input scale로 얻었다는 점이 핵심이다.

### Runtime

Feature sharing 시 single NVIDIA M40에서:

- ResNet-50 FPN Faster R-CNN: `0.148 s/image`
- ResNet-101 FPN Faster R-CNN: `0.172 s/image`
- single-scale ResNet-50 baseline: `0.32 s/image`

FPN extra layer는 비용을 더하지만 2-fc head가 기존 per-RoI conv5 head보다 가벼워 전체는 빨랐다. 이 숫자는 평균 GPU runtime이며 p50/p95, batch=1 명시, peak memory, 전력은 보고되지 않았다.

## 핵심 ablation

### Top-down이 없으면

Bottom-up pyramid에 lateral projection과 prediction만 붙인 구성은 RPN AR1k `49.5`, detector AP `24.9`였다. Full FPN의 `56.3 AR1k`, `33.9 AP`보다 크게 낮다. 얕은 level의 약한 semantic을 level-specific head만으로 해결하기 어렵다는 뜻이다.

### Lateral이 없으면

Top-down upsample만 사용하면 RPN AR1k `46.1`, detector AP `31.3`이다. Semantic은 강하지만 반복 downsample/upsample된 feature의 위치 정밀도가 부족하다. Bottom-up lateral이 localization detail을 되돌려 준다.

### P2 하나만 쓰면

모든 anchor를 semantic이 강한 P2 하나에 놓으면 anchor가 약 `750k`로 늘지만 RPN AR1k는 `51.3`, full FPN은 `56.3`이다. 후보 수가 많다고 scale robustness가 자동으로 생기지 않는다. Fixed-size sliding head가 여러 pyramid level을 순회하는 것이 중요하다.

Fast R-CNN에서는 P2 하나가 `33.4 AP`, full FPN이 `33.9 AP`로 차이가 작다. RoI pooling이 영역을 고정 크기로 normalize하기 때문에 region classifier는 RPN보다 scale에 덜 민감하다. 다만 이 P2 실험도 이미 FPN-RPN proposal을 사용했으므로 pyramid 이득을 완전히 제거한 비교는 아니다.

### Feature sharing

FPN-RPN과 Fast R-CNN feature를 공유하면:

| Backbone | no sharing AP | sharing AP |
| --- | ---: | ---: |
| ResNet-50 | 33.9 | 34.3 |
| ResNet-101 | 35.0 | 35.2 |

Accuracy가 소폭 오르고 test time이 줄지만 4-step training 때문에 train time은 약 1.5배 늘었다.

## 작은 객체 성능이 좋아지는 이유

작은 객체에서 중요한 것은 단순히 anchor를 작게 만드는 것이 아니다. Stride 32의 C5에서는 16×16 object가 0.5 cell에 불과해 feature가 사라질 수 있다. FPN은 P2/P3에서 각각 stride 4/8의 spatial sampling을 유지하면서 C5에서 내려온 semantic을 결합한다.

```text
small object
 -> high-resolution C2/C3에서 boundary/location 보존
 -> top-down C5 semantic으로 objectness 강화
 -> P2/P3에서 small anchor prediction 또는 RoI pooling
```

RPN small AR의 `+12.9`가 detector small AP의 `+4.6`보다 큰 것은 proposal recall 개선이 곧 같은 크기의 final AP 개선을 보장하지 않음을 보여 준다. Second-stage classification/localization, NMS, COCO metric이 함께 작용한다.

## Activation memory와 on-device 분석

### P2가 memory hotspot이다

앞의 계산처럼 `800×1024`, 256 channel P2는 FP16 `25 MiB`다. Top-down merge 중에는 C2 lateral, upsampled P3, M2, P2가 겹쳐 live할 수 있다. Runtime의 buffer reuse 여부에 따라 peak가 크게 달라진다.

온디바이스에서는 다음 trade-off가 중요하다.

- P2를 제거하고 P3부터 시작: memory/MAC 절감, 매우 작은 객체 손실 가능
- Pyramid channel 256→128: FPN conv parameter 약 1/4, activation 약 1/2
- 입력 800→640: spatial activation과 conv MAC가 대략 area 비율로 감소
- BiFPN/PAN 반복: accuracy 가능성은 있지만 activation lifetime과 fusion 횟수 증가

### Addition은 concat보다 유리하지만 공짜가 아니다

Lateral merge가 concatenation이면 channel과 후속 conv 비용이 늘어난다. Addition은 256 channel을 유지해 효율적이다. 그러나 upsample과 element-wise add는 memory bandwidth를 소비하며, NPU가 resize-add-conv를 fuse하지 못하면 device memory 왕복이 생긴다.

### Level별 작은 kernel과 scheduling

P5/P6처럼 작은 map의 conv는 MAC가 적어도 kernel launch overhead와 낮은 utilization 때문에 비효율적일 수 있다. 반대로 P2는 큰 dense conv라 accelerator utilization은 좋지만 절대 연산과 DRAM traffic이 크다. 따라서 level별 latency breakdown을 측정해야 한다.

### p50/p95에 영향을 주는 요소

FPN conv 자체는 입력 shape가 고정되면 deterministic하다. Tail latency는 후단의 다음 요소에서 커질 수 있다.

- 각 level의 candidate 수와 top-k
- pyramid 전체 NMS
- level별 RoI gather/scatter
- dynamic proposal distribution
- CPU fallback과 synchronization

논문은 p50/p95를 제공하지 않는다. 실제 target 장치에서는 batch=1, 고정 resize/padding, 동일 proposal cap으로 최소 수백 번 측정해야 한다.

## 장점과 핵심 기여

1. Image pyramid 없이 한 번의 backbone pass로 multi-scale representation을 만든다.
2. High-resolution localization과 deep semantic을 간단한 top-down/lateral 구조로 결합한다.
3. 모든 pyramid level을 256 channel과 유사한 semantic space로 정렬해 head parameter를 공유한다.
4. RPN, Fast/Faster R-CNN, mask proposal 등 여러 task에 붙는 generic feature extractor다.
5. 특히 small-object recall과 AP를 크게 개선한다.
6. 이후 거의 모든 현대 detector/segmenter의 neck 설계에 직접 영향을 주었다.

## 한계와 비판적 관점

### 1. High-resolution activation 비용

Image pyramid보다 효율적이지만 P2의 256-channel activation과 3×3 conv는 절대적으로 크다. "Marginal"이라는 표현은 서버 GPU와 multi-pass baseline 맥락에서 읽어야 한다.

### 2. 고정된 hand-designed scale assignment

Anchor scale을 level에 고정하고 RoI level을 식으로 정한다. 객체의 실제 appearance나 context에 따라 여러 level을 adaptive하게 조합하지 않는다.

### 3. Addition에서 정보가 섞인다

Bottom-up과 top-down을 단순 합산하므로 각 source의 중요도를 학습적으로 조절하지 않는다. 후속 PANet, BiFPN, NAS-FPN은 양방향 경로와 weighted fusion을 탐색한다.

### 4. 매우 작은 객체의 정보는 복원할 수 없다

FPN은 C2부터 시작한다. Backbone 초기에 이미 사라진 sub-pixel object detail은 top-down semantic으로 되살릴 수 없다. 입력 해상도와 early-stage design이 여전히 중요하다.

### 5. 실험의 system-level confound

FPN detector는 2-fc head, 800 input, 더 많은 RoI와 anchor scale 등 baseline과 여러 구현 차이가 있다. 논문은 controlled baseline을 제시하지만 전체 runtime 차이를 FPN 단독 효과로 분리하기는 어렵다.

## 자주 헷갈리는 지점

### FPN은 backbone인가 neck인가

논문은 generic feature extractor라 부르지만 현대 용어로는 backbone stage output을 detection head에 연결하는 `neck`에 가깝다. ResNet과 결합한 전체를 ResNet-FPN backbone이라 부르기도 한다.

### P2와 C2는 같은가

해상도는 같지만 feature는 다르다. P2는 C2 lateral projection에 P3에서 내려온 semantic을 더하고 3×3 conv로 정제한 결과다.

### Top-down만 있으면 충분한가

아니다. Top-down만으로 semantic은 전달되지만 정확한 위치가 약하다. Lateral connection ablation이 이를 보여 준다.

### 모든 객체가 모든 level에서 예측되는가

RPN anchor는 level별 scale이 다르고 IoU matching으로 연결된다. Fast R-CNN RoI는 식으로 level 하나에 할당된다. 원 논문은 여러 level feature를 한 RoI에 동시에 합치지 않는다.

### P6도 Fast R-CNN에 쓰는가

아니다. 원 논문에서 P6는 RPN의 `512²` anchor를 위한 level이며 detector RoI pooling은 P2-P5를 쓴다.

### FPN이 NMS를 없애는가

아니다. Level별 prediction을 합친 뒤 proposal/detection NMS가 여전히 필요하다.

## 재현 체크리스트

- [ ] 제공 PDF의 CVPR 2017/v2 결과를 기준으로 한다.
- [ ] C2-C5가 각 ResNet stage의 마지막 block인지 확인한다.
- [ ] Stride `{4,8,16,32}`와 channel `{256,512,1024,2048}`를 검증한다.
- [ ] Lateral은 1×1, top-down은 nearest 2× upsample, merge는 addition이다.
- [ ] Merge 후 P2-P5 각각 3×3 smoothing conv를 적용한다.
- [ ] 모든 P level channel을 256으로 맞춘다.
- [ ] RPN P2-P6 anchor scale `{32,64,128,256,512}`와 ratio 3개를 쓴다.
- [ ] RPN head가 level 간 parameter를 공유하는지 확인한다.
- [ ] Fast R-CNN RoI assignment의 floor, 224, k0=4, clamp를 맞춘다.
- [ ] P6는 RPN만 사용한다.
- [ ] Input short side 800과 padding/long-side rule을 기록한다.
- [ ] Training proposal 2000, test proposal 1000을 재현 protocol에 맞춘다.
- [ ] Image 밖 anchor 포함 여부를 기록한다.
- [ ] AP, AP50, AP_S/M/L와 RPN AR100/AR1k를 모두 기록한다.
- [ ] FPN-only와 total parameter/MAC을 구분한다.
- [ ] Peak memory는 P2와 merge 시점의 live tensor를 profiler로 측정한다.
- [ ] Batch=1 p50/p95와 level별 latency, NMS/RoI fallback을 기록한다.

## 로드맵에서의 연결

FPN은 이후 논문의 공통 기반이 된다.

- **RetinaNet**: P3-P7에 dense classification/regression subnet을 붙이고 focal loss로 imbalance를 해결한다.
- **FCOS**: 같은 multi-level feature에서 anchor 대신 위치별 `(l,t,r,b)`를 회귀한다.
- **Mask R-CNN**: FPN과 RoIAlign을 결합해 작은 instance의 box와 mask를 개선한다.
- **Deformable DETR**: C3-C5와 추가 C6의 multi-scale feature를 attention이 직접 교환해 명시적 FPN top-down 경로를 생략한다.
- **YOLO/RTMDet 계열**: PAN/FPN류 neck으로 bottom-up과 top-down feature를 반복 결합한다.

온디바이스 로드맵에서 FPN은 정확도 부품인 동시에 activation-memory 실험의 좋은 사례다. 동일 backbone과 입력에서 `P2 포함/제외`, channel 128/256, nearest resize fusion 여부를 ablation하고 AP_S, peak memory, p95를 함께 비교해야 한다.

## 최종 평가

FPN의 우아함은 새로운 detector loss나 복잡한 routing 없이, CNN hierarchy의 두 상반된 성질을 정확히 결합했다는 데 있다. Deep low-resolution feature의 semantic과 shallow high-resolution feature의 localization을 top-down upsampling과 lateral addition이라는 최소 구조로 묶고, 그 결과를 여러 scale의 prediction surface로 사용했다.

논문 실험은 small-object proposal recall과 final AP 모두에서 강한 개선을 보인다. 다만 서버 GPU의 "작은 추가 비용"을 모바일에 그대로 적용하면 안 된다. `800×1024` 예시에서 P2만 FP16 25 MiB이고 smoothing 3×3가 약 30G MAC이므로, target device에서는 channel·해상도·P2 사용 여부를 정확히 budget해야 한다. FPN을 이해한다는 것은 pyramid 그림을 암기하는 것이 아니라, 각 level의 tensor shape와 activation lifetime, 객체 scale assignment가 accuracy와 latency를 어떻게 교환하는지 이해하는 것이다.

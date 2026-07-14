# 52. Mask R-CNN

## 논문 정보

- 원본 파일: `52_Mask_R_CNN.pdf`
- 제목: **Mask R-CNN**
- 저자: Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick
- 발표: ICCV 2017
- 공식 링크: [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)
- 태스크: instance segmentation, object detection, human keypoint detection
- 핵심 키워드: Faster R-CNN, RoIAlign, mask branch, FPN, multi-task learning, instance-first segmentation

## 한눈에 보는 요약

Mask R-CNN은 Faster R-CNN의 두 번째 stage에 **각 instance의 binary mask를 예측하는 FCN branch**를 추가한 모델이다. 구조만 보면 작은 확장이지만, pixel 단위 출력을 가능하게 만든 결정적 요소는 `RoIAlign`이다. 기존 RoIPool은 floating-point box 좌표와 bin 경계를 정수 grid로 반올림해 feature와 원본 pixel 사이를 어긋나게 했다. RoIAlign은 좌표를 양자화하지 않고 bilinear interpolation으로 sample하여 이 misalignment를 제거한다.

```text
image
  -> shared ResNet-FPN backbone
  -> RPN proposals
  -> RoIAlign
       +-> classification head
       +-> box regression head
       +-> FCN mask head
  -> box NMS
  -> run/select masks for top detections
```

논문의 핵심 결과는 다음과 같다.

- ResNet-101-FPN은 COCO test-dev에서 mask AP `35.7`, ResNeXt-101-FPN은 `37.1`을 기록한다.
- ResNet-50-C4 ablation에서 RoIPool `26.9 AP`가 RoIAlign `30.3 AP`로 개선되고, 높은 IoU의 `AP75`는 `26.4 -> 31.5`로 더 크게 오른다.
- Class별 mask를 pixel softmax로 경쟁시키는 방식은 `24.8 AP`, 독립 sigmoid binary mask는 `30.3 AP`다.
- ResNet-101-FPN 공유 모델은 Tesla M40에서 image당 `195 ms` GPU 연산과 원본 크기 복원 `15 ms` CPU 시간을 보고하며 약 5 fps다.

Mask R-CNN은 instance segmentation을 "먼저 semantic map을 만들고 instance로 자르는 문제"가 아니라 **먼저 object instance를 찾고 각 RoI 내부를 분할하는 문제**로 정식화했다. 이 instance-first 설계는 강력하지만 RPN, RoIAlign, NMS, dynamic RoI batch, mask paste라는 복잡한 배포 pipeline을 남긴다.

## 연구 배경과 문제의식

### Semantic segmentation과 instance segmentation

Semantic segmentation은 같은 class의 모든 pixel에 같은 label을 준다. Instance segmentation은 같은 class라도 서로 다른 object를 구분해야 한다. 겹친 두 사람을 하나의 `person` 영역으로 합치면 semantic 결과로는 맞을 수 있지만 instance 결과로는 틀리다.

당시 접근은 크게 두 흐름이었다.

- **Segmentation-first**: 먼저 dense semantic output을 만들고 연결 성분이나 boundary로 instance를 나눈다.
- **Instance-first**: detector가 instance box를 찾고 각 box 안에서 mask를 예측한다.

Mask R-CNN은 Faster R-CNN이라는 강한 instance detector를 출발점으로 삼는다. Box classification과 regression은 공간 구조를 짧은 vector로 압축해도 되지만, mask는 RoI 안의 pixel-to-pixel 대응을 보존해야 한다. 따라서 단순히 fully connected output 하나를 더하는 것으로는 충분하지 않다.

### Faster R-CNN에서 무엇을 유지하는가

첫 stage인 RPN은 그대로다. Anchor를 분류하고 box offset을 회귀해 candidate proposal을 만든다. 두 번째 stage에서는 proposal마다 feature를 추출해 class와 box를 예측한다. Mask R-CNN은 이 두 branch와 **병렬**로 mask branch를 추가한다.

중요한 점은 mask가 class prediction의 입력이 아니며 class prediction도 mask pixel의 softmax로 결정되지 않는다는 것이다. 어떤 class인지와 그 instance 내부의 foreground 모양을 의도적으로 분리한다.

## Multi-task objective

Positive RoI 하나에 대한 학습 loss는 다음과 같다.

```math
L = L_{cls} + L_{box} + L_{mask}
```

`L_cls`와 `L_box`는 Fast/Faster R-CNN과 같다. Mask head가 `K x m x m` logits를 출력하고 ground-truth class가 `k`라면 `k`번째 channel에만 binary cross entropy를 적용한다.

```math
L_{mask}
=-\frac{1}{m^2}\sum_{u,v}
\left[y_{uv}\log\sigma(z_{k,uv})
+(1-y_{uv})\log(1-\sigma(z_{k,uv}))\right]
```

나머지 `K-1` mask channel은 이 RoI의 mask loss에 기여하지 않는다. 또한 negative RoI에는 mask loss를 적용하지 않는다.

이 설계에서 class 사이에는 pixel-level 경쟁이 없다. Box/class head가 object class를 결정하고 mask head는 선택된 class channel에서 "이 RoI pixel이 해당 instance에 속하는가"만 판단한다. 논문의 softmax 대 sigmoid ablation이 이 decoupling의 중요성을 직접 보여 준다.

## RoIPool의 좌표 오차

Feature stride가 `s=16`이고 원본 좌표가 `x=100.7`이라고 하자. 정확한 feature 좌표는 `6.29375`다. RoIPool은 RoI 경계와 각 bin을 정수 grid로 반올림한다. 여러 단계의 rounding이 누적되면 실제 object 경계와 추출 feature가 한 pixel 이상 어긋날 수 있다.

Classification은 작은 translation에 비교적 둔감하지만, `28x28` mask에서 한두 cell의 차이는 높은 IoU 성능에 큰 영향을 준다. Feature stride가 32라면 원본 좌표 한 cell의 의미가 더 커져 문제가 심해진다.

## RoIAlign

RoIAlign은 다음 원칙을 사용한다.

1. RoI 경계를 feature 좌표로 나눌 때 반올림하지 않는다.
2. RoI를 `h x w` floating-point bin으로 나눈다.
3. 각 bin의 정규 sample point에서 bilinear interpolation을 수행한다.
4. Sample을 average 또는 max로 합친다.

좌표 `(x,y)`의 bilinear sample은 주변 네 grid point `q`를 이용해 다음처럼 쓸 수 있다.

```math
F(x,y)=\sum_{q\in\mathcal N(x,y)}
(1-|x-x_q|)(1-|y-y_q|)F(x_q,y_q)
```

유효 이웃에 대해서만 weight가 남는다. 핵심은 bilinear interpolation 자체보다 **RoI와 bin 좌표를 먼저 정수화하지 않는 것**이다. 논문의 RoIWarp 비교는 interpolation을 사용해도 RoI를 양자화하면 RoIPool과 비슷한 성능에 머문다는 점을 보여 준다.

현대 구현에는 `aligned=True`나 `-0.5` coordinate shift처럼 원 논문 이후 정리된 convention이 존재한다. API 이름이 RoIAlign이라고 해서 서로 pixel-level로 동일하다는 보장은 없다. Reproduction에서는 spatial scale, sampling ratio, alignment flag를 함께 기록해야 한다.

## Architecture 전체 흐름

### 1. Backbone과 FPN

논문은 ResNet-50/101, ResNeXt-101과 C4 또는 FPN feature를 비교한다. FPN은 bottom-up backbone과 top-down lateral connection으로 `P2`부터 `P5`의 pyramid를 만들고 object scale에 맞는 level에서 RoI feature를 가져온다.

C4 head는 각 RoI에 무거운 ResNet stage 5를 반복 적용하므로 느리다. FPN은 `res5`를 shared backbone 안에서 한 번 계산하고 각 RoI에는 더 얕은 head를 적용해 정확도와 속도가 모두 좋다.

### 2. RPN과 proposal

RPN anchor는 5 scale과 3 aspect ratio를 사용한다. 학습에서 IoU `>=0.5`인 RoI는 positive, 그 이하는 negative로 취급한다. Mask target은 positive RoI와 연결된 ground-truth mask의 교집합을 RoI 좌표계로 변환한 것이다.

### 3. Box/class head

FPN variant는 RoIAlign으로 보통 `7x7x256` feature를 만들고 fully connected head로 class score와 class별 box offset을 예측한다. Box head 출력은 NMS의 입력이 된다.

### 4. Mask head

FPN mask head의 기본 shape는 다음과 같다.

```text
RoIAlign:         14 x 14 x 256
3x3 conv x4:      14 x 14 x 256
2x2 deconv s=2:   28 x 28 x 256
1x1 output conv:  28 x 28 x K
```

Mask branch는 spatial layout을 끝까지 유지하는 FCN이다. Inference에서는 predicted class `k`의 `28x28` mask만 선택해 detection box 크기로 resize하고 threshold `0.5`로 이진화한다.

### 5. Training과 inference의 차이

학습에서는 box/class/mask branch를 positive RoI에 병렬 적용한다. 추론에서는 먼저 box prediction과 NMS를 실행한 뒤, 높은 점수의 detection 100개에만 mask branch를 적용한다. 논문은 이것이 속도를 높이고 더 정확한 RoI에 mask 계산을 집중한다고 설명한다.

## Tensor shape: `batch=1`, 800x1280 입력 예시

논문은 짧은 변을 800으로 resize한다. 아래는 설명을 위한 `1x3x800x1280` 입력과 256-channel FPN을 가정한 리뷰어 예시다. 원 image aspect ratio에 따라 width는 달라진다.

| Tensor | Shape | FP16 storage | FP32 storage |
| --- | --- | ---: | ---: |
| P2 | `1x256x200x320` | 31.25 MiB | 62.50 MiB |
| P3 | `1x256x100x160` | 7.81 MiB | 15.63 MiB |
| P4 | `1x256x50x80` | 1.95 MiB | 3.91 MiB |
| P5 | `1x256x25x40` | 0.49 MiB | 0.98 MiB |
| FPN P2-P5 합계 | - | **41.50 MiB** | **83.02 MiB** |

이는 feature tensor storage만 계산한 값이다. Backbone의 lateral input, top-down upsample, convolution workspace와 allocator fragmentation은 제외했다. FPN의 parameter가 크지 않더라도 고해상도 P2가 activation memory를 지배한다.

### RoI batch의 memory

Inference에서 top 100 detection을 한 번에 mask head로 보낸다고 가정한다.

| Mask head tensor | Shape | FP16 storage |
| --- | --- | ---: |
| RoIAlign/conv feature | `100x256x14x14` | 9.57 MiB |
| Deconv feature | `100x256x28x28` | **38.28 MiB** |
| 80-class mask logits | `100x80x28x28` | 11.96 MiB |

`28x28x256` deconvolution output은 mask branch의 강한 peak 후보다. 여러 conv activation이 동시에 살아 있거나 backend workspace가 필요하면 더 커진다. Class-agnostic output은 마지막 logits를 약 80배 줄이지만 256-channel deconv activation은 줄이지 않는다.

RoI를 작은 chunk로 나누면 peak memory를 낮출 수 있지만 kernel launch와 반복 overhead로 latency가 늘 수 있다. 온디바이스에서는 `roi_batch_size`를 accuracy와 무관한 실행 정책으로 따로 튜닝할 가치가 있다.

## Parameter와 MAC 관점

FPN mask head의 네 개 `3x3, 256 -> 256` convolution만 보면 parameter는 다음과 같다.

```math
4\times(3\times3\times256\times256)
=2{,}359{,}296
```

80-class output `1x1`은 `256x80=20,480` weight다. Deconvolution까지 포함해도 전체 ResNet backbone에 비해 parameter 증가는 작다. 그러나 각 RoI에 head를 반복하므로 연산량은 detection 수에 선형으로 비례한다.

```math
C_{mask}\propto N_{roi}\cdot h\cdot w\cdot C^2
```

따라서 "Mask R-CNN은 Faster R-CNN보다 parameter가 조금만 늘어난다"와 "mask stage latency가 작다"는 같은 명제가 아니다. 논문은 top 100 RoI 전략에서 전형적인 모델에 약 20% inference overhead를 보고한다.

## 최소 구현 의사코드

```python
def forward(image, targets=None):
    pyramid = backbone_with_fpn(image)
    proposals, rpn_losses = rpn(pyramid, targets)

    box_roi = roi_align(
        pyramid, proposals, output_size=7,
        sampling_ratio=2, aligned=True,
    )
    class_logits, box_deltas = box_head(box_roi)

    if targets is not None:
        positive_rois, labels, mask_targets = sample_positive_rois(
            proposals, targets, positive_iou=0.5
        )
        mask_roi = roi_align(pyramid, positive_rois, output_size=14,
                             sampling_ratio=2, aligned=True)
        all_mask_logits = mask_head(mask_roi)       # [N_pos, K, 28, 28]
        selected = all_mask_logits[arange(len(labels)), labels]
        return rpn_losses + box_losses(...) + binary_mask_loss(
            selected, mask_targets
        )

    detections = decode_clip_and_nms(class_logits, box_deltas, proposals)
    detections = top_k(detections, k=100)
    mask_roi = roi_align(pyramid, detections.boxes, output_size=14,
                         sampling_ratio=2, aligned=True)
    all_masks = sigmoid(mask_head(mask_roi))
    masks = select_predicted_class(all_masks, detections.labels)
    return paste_masks_into_image(masks, detections.boxes, image.shape[-2:])
```

실제 graph에서 `proposals`, NMS 이후 detection 수, level별 RoI 개수는 동적이다. 정적 shape만 허용하는 NPU에서는 padding과 validity mask가 필요하며, 이때 padded RoI가 mask head 연산을 낭비하지 않는지도 확인해야 한다.

## 학습 설정

- COCO `trainval35k`: train 80k와 val 35k subset의 합집합
- Ablation: 남은 `minival` 5k
- Resize: 짧은 변 800
- 8 GPU, GPU당 image 2장, effective batch 16
- FPN: image당 sampled RoI 512, positive:negative = `1:3`
- C4: image당 sampled RoI 64
- Iteration: 160k
- Initial LR: `0.02`, 120k에서 10배 감소
- Momentum: `0.9`
- Weight decay: `1e-4`
- RPN anchor: 5 scale x 3 aspect ratio

Inference proposal 수는 C4 300, FPN 1000이다. Box NMS 후 mask branch는 score가 높은 100개 detection에만 적용한다. 이 수는 accuracy뿐 아니라 mask latency와 peak memory를 직접 결정한다.

## Main result: instance segmentation

COCO test-dev의 single-model 결과다.

| Backbone | Mask AP | AP50 | AP75 | APS | APM | APL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-101-C4 | 33.1 | 54.9 | 34.8 | 12.1 | 35.6 | 51.1 |
| ResNet-101-FPN | **35.7** | 58.0 | 37.8 | 15.5 | 38.1 | 52.4 |
| ResNeXt-101-FPN | **37.1** | 60.0 | 39.4 | 16.9 | 39.9 | 53.5 |

FPN은 C4보다 전체 AP를 2.6 높이고 small AP를 `12.1 -> 15.5`로 개선한다. 그러나 small AP는 large AP `52.4`보다 여전히 크게 낮다. `28x28` RoI mask가 object 크기에 맞춰 resize되더라도 작은 object는 backbone과 proposal 단계에서 이미 정보가 부족할 수 있다.

## 핵심 ablation

### Backbone

| Backbone | Mask AP | AP50 | AP75 |
| --- | ---: | ---: | ---: |
| ResNet-50-C4 | 30.3 | 51.2 | 31.5 |
| ResNet-101-C4 | 32.7 | 54.2 | 34.3 |
| ResNet-50-FPN | 33.6 | 55.2 | 35.3 |
| ResNet-101-FPN | 35.4 | 57.3 | 37.5 |
| ResNeXt-101-FPN | 36.7 | 59.5 | 38.9 |

Depth, FPN, ResNeXt가 모두 이득을 준다. Architecture 효과를 비교할 때 backbone과 feature pyramid를 고정하지 않으면 mask head 자체의 기여를 과대평가하게 된다.

### Independent sigmoid 대 multinomial softmax

| Mask objective | AP | AP50 | AP75 |
| --- | ---: | ---: | ---: |
| Per-pixel softmax | 24.8 | 44.1 | 25.1 |
| Per-class independent sigmoid | **30.3** | **51.2** | **31.5** |

차이는 `+5.5 AP`다. Instance class는 이미 classification head가 결정했으므로 mask pixel에서 다시 80개 class를 경쟁시킬 필요가 없다는 설계를 강하게 지지한다. Class-agnostic mask도 `29.7 AP`로 class-specific `30.3`에 가깝다.

### RoI layer

ResNet-50-C4, stride 16 기준이다.

| RoI operator | 좌표 양자화 제거 | AP | AP75 |
| --- | :---: | ---: | ---: |
| RoIPool | - | 26.9 | 26.4 |
| RoIWarp | - | 27.2 | 27.1 |
| RoIAlign average | O | **30.3** | **31.5** |

Stride 32 C5에서는 RoIPool `23.6 AP`에서 RoIAlign `30.9 AP`로 `+7.3`, AP75는 `21.6 -> 32.1`로 `+10.5`다. Feature stride가 클수록 좌표 반올림 오차가 더 치명적이라는 설명과 일치한다.

### FCN mask head 대 MLP

ResNet-50-FPN에서 spatial layout을 유지하는 FCN head는 `33.6 AP`, fully connected MLP head는 `31.5 AP`다. Mask는 vector classification보다 pixel correspondence 문제라는 논문의 관점을 뒷받침한다.

## Bounding box와 multi-task 효과

COCO test-dev box AP는 다음과 같다.

| 모델 | Backbone | Box AP | APS | APL |
| --- | --- | ---: | ---: | ---: |
| Faster R-CNN + FPN | ResNet-101-FPN | 36.2 | 18.2 | 48.2 |
| Faster R-CNN + RoIAlign | ResNet-101-FPN | 37.3 | 19.8 | 48.8 |
| Mask R-CNN | ResNet-101-FPN | **38.2** | 20.1 | 50.2 |
| Mask R-CNN | ResNeXt-101-FPN | 39.8 | 22.1 | 51.2 |

Mask R-CNN의 mask 출력을 inference에서 무시해도 box AP가 Faster R-CNN + RoIAlign보다 `0.9` 높다. 이는 mask auxiliary task가 shared representation에 regularization 또는 localization supervision을 제공한 multi-task 이득이다.

## Keypoint로의 확장

사람의 `K`개 keypoint를 각각 `m x m` one-hot location map으로 표현한다. Keypoint head는 `3x3, 512` convolution 8개, deconvolution, 2배 bilinear upsample로 `56x56` 출력을 만든다.

- Keypoint-only: COCO test-dev `62.7 APkp`
- Keypoint + mask: `63.1 APkp`
- Minival에서 RoIPool `59.8`, RoIAlign `64.2 APkp`

Pixel alignment가 mask뿐 아니라 keypoint에도 중요하다는 추가 증거다. 반면 keypoint task를 box/mask와 함께 학습하면 box/mask AP가 약간 낮아지는 행도 있어 multi-task가 항상 모든 task에 이득인 것은 아니다.

## Latency와 학습 비용

논문의 ResNet-101-FPN 공유 모델은 Nvidia Tesla M40에서 다음을 보고한다.

- GPU inference: `195 ms/image`
- CPU mask resize/paste: 추가 `15 ms`
- 약 5 fps
- ResNet-101-C4: 약 `400 ms`
- Mask branch overhead: 전형적인 모델에서 Faster R-CNN 대비 약 `20%`

이는 `batch=1`인지 명시적으로 정리된 현대 mobile benchmark가 아니며, M40 결과를 모바일 NPU latency로 변환할 수 없다. 특히 CPU 15 ms 후처리는 전체 pipeline의 약 7%로 이미 눈에 띈다. 모바일에서는 NMS, RoI level assignment, mask resize/paste가 accelerator 밖에서 실행되면 비율이 더 커질 수 있다.

ResNet-50-FPN은 synchronized 8 GPU에서 32시간, ResNet-101-FPN은 44시간 학습했다고 보고한다. 현재 framework와 hardware에서는 절대 시간이 의미가 달라지므로 iteration, batch, resize, schedule을 재현 기준으로 삼아야 한다.

## 장점과 핵심 기여

1. **Faster R-CNN을 최소 변경으로 instance segmentation으로 확장했다.** Detector의 강점을 그대로 재사용한다.
2. **RoIAlign으로 좌표 정렬 문제를 정확히 짚었다.** 작은 구현 차이가 높은 IoU와 keypoint 성능에 미치는 영향을 강한 ablation으로 증명한다.
3. **Class와 mask를 분리했다.** Independent sigmoid formulation은 단순하고 성능 차이도 크다.
4. **Mask를 FCN으로 표현했다.** Spatial structure를 vector로 붕괴시키지 않는다.
5. **범용 multi-task framework다.** Box, mask, keypoint를 같은 backbone과 RoI interface로 학습할 수 있다.
6. **강한 실험 통제**가 있다. Backbone, RoI operator, mask objective, head type을 각각 비교한다.

## 한계와 비판적 관점

### 1. Two-stage pipeline이 복잡하다

RPN, proposal decode, level assignment, RoIAlign, box NMS, top-k, mask head, mask paste가 순차 의존한다. Dense single-stage model보다 graph compile과 heterogeneous execution이 어렵다.

### 2. NMS가 필요하다

DETR/Mask2Former 계열과 달리 end-to-end set prediction이 아니다. NMS threshold와 proposal 수가 accuracy와 p95 latency를 동시에 바꾼다.

### 3. Mask resolution이 고정적이다

기본 mask는 RoI마다 `28x28`이다. 큰 object의 세밀한 경계나 가느다란 구조에는 부족할 수 있으며, box 크기로 확대해도 새 detail이 생기지 않는다.

### 4. Small object 성능이 낮다

ResNet-101-FPN에서도 APS `15.5`, APL `52.4`로 격차가 크다. FPN과 RoIAlign이 완화하지만 proposal recall과 저해상도 feature 한계를 없애지는 못한다.

### 5. Peak memory는 작은 overhead가 아닐 수 있다

Weight 증가는 작지만 FPN P2와 batched mask head activation은 크다. Parameter 수만 보고 edge 배포 적합성을 판단하면 안 된다.

### 6. Class-specific output은 closed vocabulary다

기본 mask head는 고정된 `K` class channel을 출력한다. 새로운 class나 text prompt를 바로 처리하지 못한다. Class-agnostic head가 거의 같은 성능이라는 결과는 후속 open-vocabulary 설계에 중요한 단서지만 논문은 이를 확장하지 않는다.

### 7. 논문의 latency 조건이 제한적이다

M40 한 장의 평균 시간은 proposal 수 분포, p95, peak memory, 전력, thermal throttling을 알려 주지 않는다.

## 자주 헷갈리는 지점

### Mask R-CNN은 Faster R-CNN 뒤에 semantic segmentation을 붙인 것인가

전체 image semantic map을 붙인 것이 아니다. 각 positive RoI 좌표계에서 instance별 binary mask를 예측한다.

### RoIAlign은 단순한 bilinear resize인가

아니다. 핵심은 floating-point RoI와 bin 좌표를 양자화하지 않고 feature map에서 sample하는 것이다. RoIWarp처럼 interpolation을 써도 RoI 경계를 먼저 반올림하면 같은 효과가 없다.

### K개의 mask는 pixel softmax를 하는가

아니다. 각 channel에 독립 sigmoid를 적용하고 ground-truth 또는 predicted class의 한 channel만 사용한다.

### Inference에서 1000 proposal 모두 mask head를 통과하는가

논문 구현은 먼저 box prediction과 NMS를 수행하고 top 100 detection에만 mask branch를 적용한다.

### Mask AP와 box AP를 직접 비교할 수 있는가

둘 다 COCO AP 형식이지만 IoU 대상이 mask와 box로 다르다. 수치 차이를 task 난이도의 절대 척도로만 해석하면 안 된다.

### Mask R-CNN은 NMS-free인가

아니다. RPN과 최종 detection 모두 proposal filtering/NMS에 의존한다.

## 온디바이스 관점

### 가장 큰 병목

- 고해상도 FPN P2의 activation과 bandwidth
- Dynamic proposal 수와 level별 RoI scatter/gather
- RoIAlign의 accelerator 지원 및 coordinate convention
- 100개 RoI의 `28x28x256` mask feature
- Box NMS와 top-k의 CPU fallback
- Mask를 원 image 좌표로 resize/paste하는 후처리

### 상시 detector와 요청형 mask의 분리

사용자의 통합 프로젝트 구조와 Mask R-CNN은 자연스럽게 연결된다. Camera frame마다 backbone, RPN, box head만 실행하고, 사용자가 object를 선택하거나 event가 발생했을 때 해당 box 몇 개에만 mask head를 실행할 수 있다.

```text
every frame: backbone/FPN -> proposals -> boxes
on demand: selected boxes -> RoIAlign -> mask head -> paste
```

이렇게 하면 top 100 전체 mask 계산을 피할 수 있다. 다만 backbone feature를 cache하려면 수십 MiB의 FPN activation을 유지해야 하므로 재계산과 cache 사이의 energy-memory trade-off를 측정해야 한다.

### 배포 측정 항목

- `N_proposal`, NMS 후 detection 수, mask 요청 RoI 수를 로그로 남긴다.
- Detector-only와 detector+mask latency를 분리한다.
- p50뿐 아니라 crowded scene의 p95를 측정한다.
- NMS, RoIAlign, mask paste의 CPU 시간을 별도 표시한다.
- Peak memory를 backbone/FPN, box head, mask head 단계별로 측정한다.
- FP16/INT8에서 box AP와 mask AP75, boundary quality를 따로 본다.
- 10분 이상 camera stream에서 전력, 온도, throttling을 기록한다.

INT8 quantization에서 RoIAlign interpolation과 mask logit의 작은 차이는 threshold `0.5` 주변 pixel을 뒤집을 수 있다. 전체 AP뿐 아니라 AP75와 boundary metric을 확인해야 한다.

## 재현 계획

### 최소 실험

- Dataset: COCO train2017/val2017 또는 논문 `trainval35k/minival` split
- 입력: 짧은 변 800, batch 1 inference
- Backbone: ResNet-50-FPN
- 기준 proposal: pre-NMS/post-NMS 수를 고정
- Mask inference: top 100
- 지표: mask AP/AP50/AP75/APS/APM/APL, box AP, p50/p95 latency, peak memory

### 핵심 ablation 하나

RoIPool과 RoIAlign을 비교하는 것이 가장 교육적이다. Backbone, seed, proposal을 동일하게 두고 AP와 AP75를 측정한다. 가능하면 stride 16과 32 feature에서 각각 비교해 좌표 오차가 feature stride에 따라 커지는지 재현한다.

### 온디바이스 ablation

Mask head RoI 수를 `1, 5, 20, 100`으로 바꿔 다음을 기록한다.

- Mask AP 또는 선택된 GT RoI의 mask IoU
- Mask head p50/p95 latency
- Peak activation
- CPU-NPU synchronization 횟수
- Energy per frame

Mask 정확도가 아니라 실행 정책에 따른 system trade-off를 분리해서 볼 수 있다.

## 구현 체크리스트

- [ ] RoI 좌표와 spatial scale 단위가 일치하는가?
- [ ] RoIAlign에서 좌표를 사전에 round하지 않는가?
- [ ] `aligned`, sampling ratio, pooling mode를 기록했는가?
- [ ] Mask loss를 positive RoI와 해당 class channel에만 적용하는가?
- [ ] Per-class sigmoid를 사용하고 pixel softmax를 피했는가?
- [ ] Training mask target crop/resize convention이 일치하는가?
- [ ] Inference에서 box NMS 후 mask head를 실행하는가?
- [ ] Top detection 수와 mask threshold를 기록했는가?
- [ ] FPN P2와 mask deconv activation을 peak memory에 포함했는가?
- [ ] NMS/RoIAlign/mask paste의 device placement를 확인했는가?
- [ ] Box AP와 mask AP를 모두 보고했는가?
- [ ] Small-object AP와 AP75를 별도로 확인했는가?
- [ ] Batch 1 p50/p95 latency와 지속 실행 온도를 측정했는가?

## 후속 연구와의 연결

- **FPN**은 multi-scale RoI feature와 small-object 성능의 기반이다.
- **FCOS/RetinaNet**은 proposal 중심 two-stage 구조를 dense one-stage로 단순화한다.
- **DETR**은 anchor와 NMS를 set prediction으로 대체한다.
- **Mask2Former**는 masked attention과 query를 이용해 semantic, instance, panoptic segmentation을 통합한다.
- **SAM/EdgeSAM**은 고정 class instance head 대신 prompt에 반응하는 class-agnostic mask decoder를 사용한다.

Mask R-CNN의 class-agnostic mask가 class-specific mask와 거의 비슷하다는 결과는 "무엇인지"와 "어디까지인지"를 분리하는 후속 prompt segmentation의 방향을 미리 보여 준다.

## 최종 평가

Mask R-CNN의 가장 큰 기여는 mask branch를 하나 추가했다는 사실보다 **정확한 좌표 정렬과 task decoupling을 올바른 interface로 만든 것**이다. RoIAlign은 rounding처럼 사소해 보이는 구현이 pixel task에서 얼마나 큰 차이를 만드는지 보여 주며, 독립 sigmoid mask는 classification과 shape prediction의 역할을 명확히 나눈다.

온디바이스 관점에서는 그대로 매 frame 실행하기 무거운 구조다. FPN activation, dynamic proposal processing, NMS, RoIAlign, mask paste가 parameter 수보다 더 큰 병목이다. 그러나 detector를 상시 실행하고 mask를 선택된 RoI에만 요청형으로 실행한다면 사용자 로드맵의 통합 architecture와 잘 맞는다. 이때 반드시 detector-only와 mask-on-demand의 latency와 memory를 분리해 측정해야 한다.

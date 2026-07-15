# 41. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

## 논문 정보

- 원본 파일: `41_Faster_R_CNN.pdf`
- 제목: **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks**
- 저자: Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
- 공개: arXiv:1506.01497, 초판 2015; 제공된 PDF는 v3(2016), 확장 논문
- 발표: NeurIPS 2015의 선행 버전을 확장한 IEEE TPAMI 논문
- 링크: [https://arxiv.org/abs/1506.01497](https://arxiv.org/abs/1506.01497)
- 핵심 키워드: two-stage detector, Region Proposal Network, anchor, objectness, bounding-box regression, shared convolution, RoI pooling, NMS

## 한눈에 보는 요약

Faster R-CNN은 그전까지 별도 알고리즘이 담당하던 region proposal 생성을 **Region Proposal Network(RPN)**라는 작은 완전 합성곱 네트워크로 바꾸고, 이 RPN과 Fast R-CNN detector가 이미지 전체의 convolution feature를 공유하게 만든 two-stage detector다.

```text
image
 -> shared CNN feature map
 -> RPN: class-agnostic object proposals + objectness
 -> proposal NMS / top-N
 -> RoI pooling
 -> Fast R-CNN: class prediction + class-specific box refinement
 -> class-wise NMS
```

핵심은 단순히 proposal도 신경망으로 만들었다는 데 있지 않다. 동일한 dense feature map 위에서 RPN은 "어디를 볼지"를 정하고, Fast R-CNN은 선택된 영역이 "무엇인지"를 판별한다. VGG-16 실험에서 RPN의 추가 연산은 K40 GPU 기준 약 `10 ms`였고, 전체 pipeline은 `198 ms/image`, 즉 약 `5 FPS`였다. Selective Search + Fast R-CNN의 `1830 ms/image`보다 크게 빠르면서 VOC 2007 test에서 더 높은 mAP를 냈다.

RPN은 각 feature-map 위치에 여러 크기와 종횡비의 reference box, 즉 anchor를 놓는다. 각 anchor마다 다음 두 값을 예측한다.

- 해당 anchor 주변에 어떤 종류든 객체가 존재할 확률인 objectness
- anchor를 실제 객체 box로 옮기고 크기를 바꾸는 네 개의 regression offset

이 구조는 이후 anchor 기반 detector의 표준 문법이 되었다. 동시에 논문의 한계도 분명하다. anchor 크기·비율, IoU threshold, proposal 수, 두 번의 NMS와 RoI 연산이 필요하며, 작은 객체는 stride 16의 단일 feature map에서 불리하다. 후속 FPN, RetinaNet, FCOS와 DETR 계열은 각각 이 약점을 다른 방식으로 해결한다.

## 연구 배경: proposal이 병목이 된 이유

### R-CNN에서 Fast R-CNN까지

Object detection은 이미지 분류와 달리 객체의 category와 위치를 동시에 맞혀야 한다. R-CNN 계열의 기본 전략은 가능한 객체 후보 영역을 먼저 많이 만들고, 각 영역을 분류하는 것이다.

```text
R-CNN:
Selective Search proposals
 -> proposal마다 CNN 실행
 -> SVM classification + box regression

Fast R-CNN:
image당 CNN 한 번
 -> shared feature map
 -> proposal마다 RoI pooling
 -> classification + box regression
```

Fast R-CNN은 proposal마다 CNN을 반복하는 낭비를 제거했지만, proposal 자체는 여전히 Selective Search 같은 CPU 알고리즘이 만들었다. 논문이 보고한 당시 속도는 Selective Search가 평균 약 `1.5 s/image`, EdgeBoxes가 약 `0.2 s/image`였다. detector가 빨라지자 proposal 생성이 전체 latency의 주된 병목으로 드러난 것이다.

단순히 Selective Search를 GPU로 옮기는 것은 계산 장치만 바꾸는 해결책이다. Faster R-CNN이 택한 더 근본적인 해결은 **proposal과 detection이 같은 시각 feature를 재사용하도록 구조를 통합하는 것**이다.

### One-stage와 two-stage의 차이

RPN 자체는 dense sliding-window predictor라 one-stage detector처럼 보인다. 그러나 Faster R-CNN 전체는 명백한 two-stage cascade다.

1. RPN은 category를 구분하지 않고 객체일 가능성이 높은 영역을 압축한다.
2. Fast R-CNN은 정렬된 RoI feature에서 category를 판별하고 box를 다시 보정한다.

논문의 ZF 실험에서 dense window를 바로 class-specific detector에 넣은 one-stage 모사는 `53.9 mAP`, RPN + Fast R-CNN two-stage는 `58.7 mAP`였다. 두 번째 단계의 adaptive RoI feature가 단순 dense window보다 더 정확한 region representation을 제공했다는 근거다.

## 전체 아키텍처

### 1. Shared convolution backbone

입력 이미지는 ZF 또는 VGG-16 backbone을 통과한다. 논문의 VGG-16 구성에서는 13개 convolution layer를 RPN과 Fast R-CNN이 공유하며, 마지막 공유 feature map의 total stride는 16이다.

입력의 짧은 변을 `600 px`로 resize하면 대략 다음 흐름이 된다.

```text
input image:         B x 3 x H x W
shared conv feature: B x 512 x ceil(H/16) x ceil(W/16)   # VGG-16
```

실제 framework의 padding과 resize된 긴 변에 따라 마지막 한두 칸은 달라질 수 있다. 중요한 점은 RPN과 Fast R-CNN이 이 큰 feature tensor를 각각 계산하지 않고 공유한다는 것이다.

### 2. Region Proposal Network

마지막 feature map 위로 `3×3` window를 sliding한다. VGG-16에서는 각 위치를 512차원 hidden feature로 변환한 뒤 두 sibling `1×1` convolution으로 분기한다.

```text
shared feature C x Hf x Wf
 -> 3x3 conv, 512 channels + ReLU
    -> cls 1x1 conv: 2k channels
    -> reg 1x1 conv: 4k channels
```

기본 anchor 수는 위치당 `k=9`다. 따라서 VGG-16 RPN의 출력 channel은 다음과 같다.

- classification: `2 × 9 = 18`, anchor별 object/background 두 logit
- regression: `4 × 9 = 36`, anchor별 `(t_x,t_y,t_w,t_h)`

오늘날 구현은 2-class softmax 대신 anchor당 binary sigmoid logit 하나를 쓰기도 한다. 이는 출력 표현의 차이이지 RPN의 개념적 역할은 같다.

### 3. Proposal filtering

RPN regression을 anchor에 decode하고 image boundary로 clip한다. 너무 작은 box를 제거한 뒤 objectness로 정렬하고 NMS를 적용한다. 논문의 RPN NMS IoU threshold는 `0.7`이다.

- 학습 시 Fast R-CNN에 제공: 2,000 RPN proposals
- 평가 시 최종 detector에 제공: 상위 300 proposals
- NMS 전에는 한 이미지에 수만 개 anchor가 있지만, NMS와 top-N이 region-wise 연산량을 크게 줄인다.

### 4. Fast R-CNN second stage

각 proposal을 shared feature map 좌표로 투영한 뒤 RoI pooling으로 고정 크기 feature를 만든다. VGG-16 Fast R-CNN에서는 통상 `7×7×512` RoI tensor가 fc6/fc7을 지나 다음 두 출력으로 분기한다.

- `C+1` category softmax: foreground classes + background
- class별 box regression: 보통 `4C`

RPN regression과 Fast R-CNN regression은 중복이 아니다. RPN은 class-agnostic proposal을 만들고, second stage는 더 정렬된 RoI feature를 사용해 class-specific detection box를 정밀 보정한다.

## Anchor의 의미와 동작

### 위치당 9개 reference box

논문의 기본 anchor 설정은 다음 Cartesian product다.

- area scale: `128²`, `256²`, `512²` pixels
- aspect ratio: `1:1`, `1:2`, `2:1`

각 anchor의 중심은 현재 feature-map 위치에 대응하는 input 좌표에 고정된다. RPN은 box 좌표를 처음부터 직접 생성하지 않고 이 reference를 이동·확대·축소한다.

Anchor는 **최종 예측 box 후보를 고정하는 격자**가 아니다. 예를 들어 `128², 1:1` anchor도 regression을 통해 논문 Table 1의 평균 `113×114` proposal로 바뀌며, 다른 형태로도 자유롭게 이동한다. Anchor는 regression이 시작하는 좌표계이자 학습 sample을 정의하는 장치다.

### Translation invariance

동일한 anchor pattern과 convolution weight를 모든 위치에 공유하므로, 입력 객체가 한 stride만큼 이동하면 feature 위치와 proposal도 함께 이동한다. 논문은 이를 translation-invariant anchor라고 부른다. 다만 pooling과 stride 때문에 pixel 단위의 완전한 equivariance를 보장한다는 뜻은 아니다.

MultiBox의 800개 global output box와 비교하면 RPN은 위치마다 같은 9개 anchor predictor를 공유한다. 논문 계산에서 VGG-16 RPN output layer는 약 `2.8×10^4` parameter로, MultiBox의 약 `6.1×10^6`보다 훨씬 작았다.

## Anchor assignment

각 training anchor는 positive, negative, ignore 중 하나다.

### Positive

다음 중 하나를 만족하면 positive다.

1. 어떤 ground-truth box와의 IoU가 그 GT에 대해 가장 높은 anchor
2. 어떤 GT와든 IoU가 `0.7`보다 큰 anchor

첫 조건은 어떤 GT도 0.7을 넘는 anchor를 갖지 못하는 드문 경우에도 최소 한 개의 positive를 보장한다.

### Negative와 ignore

- 모든 GT와의 IoU가 `0.3`보다 낮고 positive가 아니면 negative
- `0.3 <= IoU <= 0.7` 부근에서 위 조건에 들지 않으면 ignore

Ignore anchor는 classification과 regression 어느 loss에도 참여하지 않는다. 이 회색 구간은 애매한 sample이 학습을 흔드는 것을 줄이지만, threshold 자체는 수작업 hyperparameter다.

### Mini-batch sampling

한 이미지의 수만 anchor를 모두 사용하면 background가 압도한다. RPN은 이미지당 256개 anchor를 sampling하며 positive:negative를 최대 `1:1`로 맞춘다. Positive가 128개보다 적으면 나머지를 negative로 채운다.

이 sampling은 뒤의 Focal Loss와 대비된다. RetinaNet은 쉬운 negative를 버리는 대신 focal factor로 연속적으로 낮게 가중하고 약 100k anchor를 모두 loss에 넣는다.

## Bounding-box parameterization

Anchor를 `(x_a,y_a,w_a,h_a)`, 예측 box를 `(x,y,w,h)`, 연결된 GT를 `(x^*,y^*,w^*,h^*)`라 하자. 중심 좌표와 크기를 사용하면 target은 다음과 같다.

```math
\begin{aligned}
t_x &= \frac{x-x_a}{w_a}, &
t_y &= \frac{y-y_a}{h_a},\\
t_w &= \log\frac{w}{w_a}, &
t_h &= \log\frac{h}{h_a},\\
t_x^* &= \frac{x^*-x_a}{w_a}, &
t_y^* &= \frac{y^*-y_a}{h_a},\\
t_w^* &= \log\frac{w^*}{w_a}, &
t_h^* &= \log\frac{h^*}{h_a}.
\end{aligned}
```

위치 이동을 anchor width/height로 정규화하므로 서로 다른 크기의 anchor가 비슷한 scale의 target을 갖는다. Width와 height의 비율은 log로 바꿔 곱셈적 크기 변화를 덧셈 공간에서 회귀한다.

Inference에서 역변환은 다음과 같다.

```math
\begin{aligned}
x &= t_x w_a+x_a, & y &= t_y h_a+y_a,\\
w &= e^{t_w}w_a, & h &= e^{t_h}h_a.
\end{aligned}
```

## RPN multi-task loss

논문의 이미지별 RPN objective는 다음과 같다.

```math
L(\lbrace p_i\rbrace,\lbrace t_i\rbrace)=
\frac{1}{N_{cls}}\sum_i L_{cls}(p_i,p_i^*)
+\lambda\frac{1}{N_{reg}}\sum_i p_i^*L_{reg}(t_i,t_i^*).
```

- `p_i`: anchor `i`가 object일 예측 확률
- `p_i^* in {0,1}`: anchor label
- `t_i`, `t_i^*`: 예측/target box offset
- `L_cls`: binary log loss
- `L_reg`: `smooth L1(t_i-t_i^*)`
- `p_i^* L_reg`: positive anchor만 regression에 참여

논문의 구현은 `N_cls=256`, `N_reg≈2400` anchor locations, `lambda=10`을 쓴다. 두 항의 수치 규모를 비슷하게 만드는 normalization이다. Table 9에서 `lambda=0.1,1,10,100`의 VOC 2007 mAP는 각각 `67.2, 68.9, 69.9, 69.1`이었다. 넓은 범위에서 크게 민감하지 않지만, normalization convention이 달라지면 같은 lambda를 그대로 복사하면 안 된다.

Smooth L1을 `d=t-t^*`에 대해 쓰면 일반적으로 다음 형태다.

```math
\mathrm{smooth}_{L1}(d)=
\begin{cases}
\frac{1}{2}d^2,& |d|\lt 1,\\
|d|-\frac{1}{2},& \text{otherwise}.
\end{cases}
```

작은 오차에는 L2처럼 부드럽고 큰 오차에는 L1처럼 gradient 폭주를 줄인다.

## Batch=1 tensor shape 계산

### 가정

논문 조건과 가까운 `600×1000` RGB 입력, VGG-16, stride 16, `k=9`를 가정한다. Padding 세부를 단순화해 feature map을 `38×63`으로 둔다. 아래는 **리뷰어 계산**이며 논문이 그대로 보고한 tensor 표가 아니다.

| 단계 | NCHW shape | element 수 | FP16 payload | FP32 payload |
| --- | ---: | ---: | ---: | ---: |
| input | `1×3×600×1000` | 1,800,000 | 3.43 MiB | 6.87 MiB |
| shared conv5 feature | `1×512×38×63` | 1,225,728 | 2.34 MiB | 4.68 MiB |
| RPN 3×3 hidden | `1×512×38×63` | 1,225,728 | 2.34 MiB | 4.68 MiB |
| cls logits | `1×18×38×63` | 43,092 | 0.082 MiB | 0.164 MiB |
| reg offsets | `1×36×38×63` | 86,184 | 0.164 MiB | 0.329 MiB |

공간 위치는 `38×63=2,394`, raw anchor는 `2,394×9=21,546`개다. 논문의 "보통 약 2,400 locations"와 일치하는 규모다.

300개 proposal을 `7×7×512` RoI feature로 materialize하면:

```math
300\times7\times7\times512=7{,}526{,}400\ \text{elements}
```

순수 payload만 FP16 약 `14.36 MiB`, FP32 약 `28.71 MiB`다. 이는 inference의 한 순간에 대한 값이며 allocator workspace, backbone의 동시 live tensor, fc workspace는 포함하지 않는다. 학습에서 2,000 RoI를 한꺼번에 저장한다고 단순 가정하면 이 값은 약 6.67배가 되므로, proposal 수는 accuracy뿐 아니라 activation memory와 region-wise latency에 직접 영향을 준다.

### 작은 box decode 예시

`256×256` anchor의 중심이 `(400,300)`이고 RPN이

```text
(tx, ty, tw, th) = (0.10, -0.05, log(1.2), log(0.8))
```

를 냈다고 하자. Decode 결과는 다음과 같다.

```math
\begin{aligned}
x&=400+0.10\cdot256=425.6,\\
y&=300-0.05\cdot256=287.2,\\
w&=1.2\cdot256=307.2,\\
h&=0.8\cdot256=204.8.
\end{aligned}
```

즉 anchor는 중심과 크기를 정규화하는 reference일 뿐, 출력 모양을 고정하지 않는다.

## Parameter, MAC, activation memory 분석

### 논문이 보고한 것

- VGG-16 RPN proposal layer 전체: 약 `2.4M` parameter
- RPN output layer만: 약 `2.8×10^4` parameter
- K40에서 shared VGG RPN 추가 latency: `10 ms`
- 전체 VGG Faster R-CNN: `198 ms`, `5 FPS`
- 논문은 peak activation memory, p50/p95 latency, 전력, 온도를 보고하지 않는다.

### 리뷰어 계산: VGG RPN head parameter

Bias를 제외하면:

```math
\begin{aligned}
\text{3x3 hidden} &= 3\cdot3\cdot512\cdot512=2{,}359{,}296,\\
\text{cls 1x1} &= 512\cdot18=9{,}216,\\
\text{reg 1x1} &= 512\cdot36=18{,}432,\\
\text{total} &= 2{,}386{,}944.
\end{aligned}
```

논문의 약 `2.4M`과 맞는다. FP16 weight payload는 약 `4.55 MiB`, FP32는 약 `9.11 MiB`다. 이는 backbone과 Fast R-CNN fc layer를 제외한 값이다.

### 리뷰어 계산: RPN head MAC

위 `38×63` feature에서 3×3 hidden conv의 이상적 MAC은:

```math
38\cdot63\cdot512\cdot(3\cdot3\cdot512)
\approx 5.65\ \text{G MACs}
```

두 1×1 output conv는 합쳐 약 `66.2M MACs`다. 따라서 RPN head의 arithmetic 대부분은 3×3 `512→512` conv에 있다. 그러나 실제 latency는 MAC만으로 결정되지 않는다. Backbone feature가 cache에 남아 있는지, conv kernel fusion, RoI 연산, NMS의 CPU/GPU 이동, 작은 kernel launch가 영향을 준다.

### Peak memory를 단순 합산하면 안 되는 이유

Tensor payload 표를 모두 더한 값은 peak memory의 상한도 하한도 정확히 아니다.

- inference engine은 더 이상 쓰지 않는 buffer를 재사용할 수 있다.
- NMS, sorting, RoI pooling은 별도 workspace를 요구할 수 있다.
- training은 backward용 activation, gradient, optimizer state를 보존한다.
- VGG의 앞단 high-resolution feature가 실제 peak를 만들 수 있다.

따라서 peak memory는 target runtime에서 측정해야 한다. 논문에는 이 수치가 없다.

## Training: feature를 어떻게 공유하는가

### 논문의 4-step alternating training

1. ImageNet pretrained backbone에서 RPN을 end-to-end fine-tune한다.
2. 1단계 RPN proposal로 별도의 Fast R-CNN을 학습한다.
3. 2단계 detector backbone으로 RPN을 초기화하고 shared conv는 고정한 채 RPN 고유 layer만 학습한다.
4. Shared conv를 계속 고정하고 Fast R-CNN 고유 layer만 fine-tune한다.

마지막에는 두 module이 동일한 convolution weight를 사용한다. "공유"는 두 network가 서로 다른 conv를 우연히 같은 구조로 갖는다는 뜻이 아니라, 실제 동일 parameter와 동일 feature tensor를 쓴다는 의미다.

### Approximate joint training

한 forward에서 RPN proposal을 만들고 Fast R-CNN loss까지 계산한 뒤, shared layer에는 두 loss의 gradient를 합칠 수 있다. 다만 논문의 approximate joint training은 proposal coordinate가 바뀔 때 RoI pooling output이 어떻게 바뀌는지에 대한 gradient를 무시한다. Proposal을 현재 iteration에서는 상수처럼 취급하는 셈이다.

논문은 이 방식이 alternating training과 비슷한 결과를 내면서 training time을 약 `25-50%` 줄였다고 보고한다. Coordinate까지 완전히 미분하려면 당시 RoI pooling 대신 coordinate-differentiable RoI warping 같은 연산이 필요했다. 후속 RoIAlign과 differentiable sampling 계열이 이 문제를 더 깔끔하게 다룬다.

### 학습 recipe

VOC에서 논문의 기본 RPN 설정은 다음과 같다.

- ImageNet pretrained ZF/VGG-16
- 한 mini-batch는 한 이미지에서 256 anchors
- SGD, momentum `0.9`, weight decay `0.0005`
- learning rate `0.001` for 60k mini-batches, 이후 `0.0001` for 20k
- 입력 짧은 변 `600`
- 단일 image scale

이 recipe의 iteration 수를 현대 batch size에 그대로 옮기면 seen-image 수가 달라진다. 재현 시 epoch 또는 총 image 수로 환산해야 한다.

## 구현 pseudocode

```python
def faster_rcnn(image, training=False):
    feat = backbone(image)                    # [B, 512, H/16, W/16]

    # RPN
    h = relu(conv3x3(feat, out_channels=512))
    obj_logits = conv1x1(h, out_channels=2 * K)
    box_delta = conv1x1(h, out_channels=4 * K)
    anchors = make_anchors(feat.shape[-2:], scales, ratios, stride=16)

    boxes = decode(anchors, box_delta)
    scores = softmax_objectness(obj_logits)
    boxes = clip_to_image(boxes)
    boxes, scores = remove_small_boxes(boxes, scores)
    keep = pre_nms_topk(scores)
    keep = nms(boxes[keep], scores[keep], iou_threshold=0.7)
    proposals = topk(boxes[keep], 2000 if training else 300)

    # Fast R-CNN
    roi = roi_pool(feat, proposals, output_size=(7, 7))
    z = fc7(relu(fc6(roi.flatten(1))))
    class_logits = cls_head(z)
    class_box_delta = bbox_head(z)

    if training:
        return rpn_loss(obj_logits, box_delta, anchors, gt_boxes), \
               roi_loss(class_logits, class_box_delta, proposals, targets)

    det_boxes = decode_per_class(proposals, class_box_delta)
    det_scores = softmax(class_logits)
    return classwise_nms(det_boxes, det_scores)
```

재현에서 자주 빠지는 세부 사항은 image 밖 anchor 처리, ignore label, regression normalization, pre/post-NMS top-k의 순서, 좌표 convention의 inclusive/exclusive 차이다.

## 주요 실험 결과

### PASCAL VOC

VGG-16, VOC 2007 test 결과는 다음과 같다.

| Proposal / training data | proposals | mAP |
| --- | ---: | ---: |
| Selective Search, VOC07 | 2000 | 66.9 |
| RPN unshared, VOC07 | 300 | 68.5 |
| RPN shared, VOC07 | 300 | 69.9 |
| RPN shared, VOC07+12 | 300 | 73.2 |
| RPN shared, COCO+VOC07+12 | 300 | 78.8 |

VOC 2012 test에서는 RPN shared VGG-16이 VOC07++12 학습으로 `70.4 mAP`, COCO까지 pretrain/fine-tune하면 `75.9 mAP`였다.

### MS COCO

VGG-16, COCO test-dev에서:

| Method | proposals | AP@0.5 | AP@[.5:.95] |
| --- | ---: | ---: | ---: |
| Fast R-CNN + SS, paper implementation | 2000 | 39.3 | 19.3 |
| Faster R-CNN, COCO train | 300 | 42.1 | 21.5 |
| Faster R-CNN, COCO trainval | 300 | 42.7 | 21.9 |

오늘날 기준으로 절대 AP가 낮아 보여도, 이 표의 목적은 같은 시대·backbone·protocol에서 learned proposal이 external proposal보다 localization을 개선했음을 보이는 데 있다.

### 속도

논문 Table 5의 K40 측정은 다음과 같다.

| System | conv | proposal | region-wise | total | rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| VGG SS + Fast R-CNN | 146 ms | 1510 ms | 174 ms | 1830 ms | 0.5 FPS |
| VGG RPN + Fast R-CNN | 141 ms | 10 ms | 47 ms | 198 ms | 5 FPS |
| ZF RPN + Fast R-CNN | 31 ms | 3 ms | 25 ms | 59 ms | 17 FPS |

SS는 CPU, 나머지는 GPU라는 측정 조건을 함께 읽어야 한다. 또한 평균 한 값이지 p50/p95 분포가 아니다.

## Ablation 해석

### Anchor scale과 ratio

VOC 2007 VGG-16 결과:

| 설정 | mAP |
| --- | ---: |
| `128²`, ratio 1:1 하나 | 65.8 |
| `256²`, ratio 1:1 하나 | 66.7 |
| `128²`, ratios 3개 | 68.8 |
| `256²`, ratios 3개 | 67.9 |
| scales 3개, ratio 1개 | 69.8 |
| scales 3개, ratios 3개 | 69.9 |

한 anchor보다 multi-scale/multi-ratio가 분명히 낫지만, 이 데이터에서는 scale 3개와 ratio 1개만으로도 전체 기본 설정과 거의 같았다. 따라서 "9 anchors가 이론적으로 필수"가 아니라, 다양한 regression reference가 coverage를 높인다는 해석이 정확하다.

### Classification과 regression의 역할

고정된 SS-trained ZF detector에 RPN proposal을 넣은 ablation에서:

- 300 RPN proposals: `56.8 mAP`
- cls 없이 무작위 300 proposal: `51.4`
- reg 없이 anchor 300개: `52.1`
- NMS 없이 6000 proposal: `55.2`

Objectness는 적은 top-N에서도 좋은 proposal을 위로 올리고, regression은 anchor를 실제 경계에 맞춘다. NMS는 중복을 줄이지만 이 실험에서는 최종 mAP를 거의 해치지 않았다.

### Proposal 수

RPN은 2000개에서 300개로 줄여도 recall이 Selective Search나 EdgeBoxes보다 완만하게 감소했다. 그러나 proposal recall은 detector mAP와 완전히 같은 metric이 아니다. 논문도 Recall-to-IoU는 proposal 진단용이며 최종 detection accuracy의 대리 지표로 과신하지 말라고 지적한다.

## 작은 객체 분석

Faster R-CNN 원형은 작은 객체에 구조적으로 불리하다.

1. VGG-16의 마지막 공유 map이 stride 16이라 작은 물체가 한두 cell에 불과할 수 있다.
2. 기본 최소 anchor area가 `128²`라 COCO의 작은 객체와 scale 차이가 크다.
3. RoI pooling의 양자화는 작은 RoI에서 위치 오차 비율을 키운다.

COCO 실험은 작은 객체 대응을 위해 `64²` anchor scale을 추가했다. 하지만 논문은 COCO의 `AP_S/AP_M/AP_L` breakdown을 제공하지 않는다. 따라서 이 논문만으로 작은 객체 성능 향상 폭을 수치화할 수 없다.

FPN은 이 문제에 대한 직접적인 후속 답이다. 작은 객체를 stride 4/8의 high-resolution이면서 semantic이 강화된 level에서 처리한다. RoIAlign은 pooling quantization을 줄인다.

## NMS가 두 번 필요한 이유

Faster R-CNN에는 보통 서로 다른 목적의 NMS가 두 번 있다.

1. **RPN NMS**: class-agnostic proposal 중복을 줄여 second-stage workload를 제한한다. 논문 threshold는 0.7.
2. **Final class-wise NMS**: detector가 같은 객체에 낸 중복 class box를 제거한다.

RPN NMS를 통과한 proposal이 최종 detection은 아니다. 반대로 final NMS만 남기고 수만 proposal을 모두 RoI head에 보내면 region-wise latency와 memory가 급증한다. DETR 계열은 Hungarian set prediction으로 이 중복 제거 규칙 자체를 없애려 한다.

## 장점과 핵심 기여

1. External proposal 병목을 학습 가능한 RPN으로 대체했다.
2. Proposal과 detection의 가장 비싼 convolution을 공유했다.
3. Multi-scale anchor를 단일 feature map 위의 regression reference로 사용했다.
4. 300개 proposal만으로 높은 recall과 detector accuracy를 유지했다.
5. RPN objectness와 box regression을 end-to-end loss로 학습했다.
6. Detection pipeline을 거의 하나의 network로 통합해 후속 two-stage detector의 표준 기반을 만들었다.

## 한계와 비판적 관점

### 1. 완전한 end-to-end는 아니다

논문의 주된 4-step 학습은 alternating optimization이고, approximate joint training은 proposal coordinate gradient를 끊는다. Inference에도 top-k, clipping, NMS 같은 discrete operation이 남는다.

### 2. Anchor hyperparameter 의존

Scale, ratio, stride, IoU threshold, positive/negative sampling, NMS threshold가 데이터 분포에 영향을 받는다. 새로운 센서 해상도나 작은 객체 중심 데이터에서는 재설계가 필요하다.

### 3. 단일 coarse feature map

모든 scale을 stride-16 map에서 처리한다. 큰 anchor를 여러 개 놓아 scale coverage는 얻지만, 이미 downsample되어 사라진 작은 객체 detail은 anchor만으로 복원할 수 없다.

### 4. Region-wise irregular workload

RoI pooling과 per-RoI fc는 proposal 수에 비례한다. 동적 proposal 수, sorting, NMS는 정적 graph accelerator에서 불리할 수 있다.

### 5. 현대적인 latency 증거가 아니다

K40의 평균 latency는 역사적으로 중요하지만 모바일 NPU/CPU/GPU의 p50/p95, energy, thermal throttling을 말해 주지 않는다. Framework와 kernel 구현도 현재와 다르다.

## 자주 헷갈리는 지점

### RPN은 detector인가

RPN은 objectness와 box를 예측하므로 class-agnostic detector로 볼 수 있지만, Faster R-CNN에서의 역할은 category prediction이 아니라 proposal 생성이다.

### Anchor와 proposal은 같은가

Anchor는 고정 reference, proposal은 RPN delta를 decode한 결과다. Regression을 끄면 proposal이 anchor로 돌아가며 성능이 떨어진다.

### 300 proposals는 300 detections인가

아니다. 300개는 Fast R-CNN에 들어가는 최대 RoI 수다. Second-stage score와 box refinement, final NMS 뒤의 detection 수는 더 적다.

### Feature sharing은 loss sharing인가

아니다. RPN과 Fast R-CNN은 각자 classification/regression loss를 갖고 shared backbone parameter에 gradient를 보낸다. Head parameter는 서로 다르다.

### RPN의 "attention"은 Transformer attention인가

논문은 RPN이 detector에게 볼 위치를 알려 준다는 기능적 비유로 attention이라 표현한다. Query-key softmax를 계산하는 Transformer attention과는 다른 연산이다.

### NMS는 학습 중에도 loss 안에 들어가는가

NMS는 proposal 선택/post-processing 단계이며 미분 가능한 학습 loss가 아니다. RPN은 anchor-level objectness와 regression target으로 학습한다.

## 온디바이스 배포 관점

### Operator 지원

Backbone과 RPN conv는 모바일 accelerator가 잘 처리하는 dense convolution이다. 문제는 다음 후반부다.

- anchor generation/decode와 clipping
- score sort와 dynamic top-k
- NMS
- 동적 개수 RoI pooling/RoIAlign
- per-RoI head

NPU가 이 연산을 지원하지 않으면 CPU fallback과 device synchronization이 발생해 평균 MAC에 비해 latency가 커질 수 있다.

### Activation memory

VGG backbone 자체가 무겁고, proposal별 RoI tensor가 추가된다. `batch=1`이어도 proposal dimension이 사실상의 batch처럼 작동한다. FP16/INT8 quantization만 볼 것이 아니라 post-NMS proposal cap과 RoI head width를 함께 줄여야 한다.

### p50과 p95

논문은 평균 latency만 보고한다. 실제 장치에서는 다음을 별도로 기록해야 한다.

- p50: steady-state 대표 속도
- p95: NMS 후보 수, CPU scheduling, memory allocation에 따른 tail
- cold start와 warm-up 제외 여부
- RPN/ROI/NMS stage별 latency
- 10분 이상 연속 실행 후 thermal throttling

### 경량화 우선순위

1. VGG-16을 MobileNetV2/V3/V4 또는 efficient hybrid backbone으로 교체
2. FPN을 추가하되 channel 수와 high-resolution activation budget 통제
3. RPN pre-NMS/top-N과 second-stage proposal cap 축소
4. RoI head를 큰 fc 대신 작은 conv/MLP로 교체
5. 지원된다면 fused decode+top-k+NMS 사용
6. INT8 calibration에서 classification뿐 아니라 box regression 오차 확인

FLOPs가 낮아도 NMS와 RoI가 CPU fallback이면 p95가 나빠질 수 있다. 반대로 RPN의 3×3 conv가 MAC 대부분을 차지하더라도 NPU에서 잘 fuse되면 wall-clock 병목은 다른 곳일 수 있다.

## 재현 체크리스트

- [ ] 제공 PDF가 arXiv v3/확장 버전인지 기록한다.
- [ ] Dataset split: VOC07, VOC07+12, COCO train/val/test-dev를 정확히 구분한다.
- [ ] 짧은 변 600 resize와 긴 변 처리 규칙을 기록한다.
- [ ] Backbone stage와 total stride를 확인한다.
- [ ] Anchor scales/ratios와 좌표 convention을 고정한다.
- [ ] Image 밖 anchor를 ignore할지 포함할지 기록한다.
- [ ] Positive `max-IoU or >0.7`, negative `<0.3`, ignore 규칙을 검증한다.
- [ ] RPN batch 256, positive 최대 128을 확인한다.
- [ ] Smooth L1, normalization, lambda convention을 구현과 맞춘다.
- [ ] Train/test pre-NMS 및 post-NMS top-k를 기록한다.
- [ ] RPN NMS 0.7과 final class-wise NMS를 혼동하지 않는다.
- [ ] Shared/unshared, 4-step/approximate joint training을 명시한다.
- [ ] AP metric이 VOC IoU 0.5인지 COCO AP@[.5:.95]인지 표기한다.
- [ ] Parameter/MAC은 backbone, RPN, RoI head를 분리해 센다.
- [ ] Peak memory는 실제 target runtime profiler로 측정한다.
- [ ] Latency는 batch=1, warm-up 후 p50/p95와 proposal 수를 함께 남긴다.
- [ ] Accuracy와 함께 small-object AP/recall을 기록한다.

## 로드맵에서의 연결

Faster R-CNN은 object detection 단계의 출발점이다.

- **FPN**: 단일 stride-16 feature의 작은 객체 약점을 multi-level semantic feature로 해결한다.
- **RetinaNet**: RPN의 dense anchor 문법을 one-stage 전체로 확장하고 focal loss로 imbalance를 다룬다.
- **FCOS**: anchor assignment를 없애고 location에서 네 변 거리를 직접 회귀한다.
- **DETR**: proposal/NMS/anchor 규칙을 set prediction과 Hungarian matching으로 대체한다.
- **Deformable DETR**: DETR에 sparse multi-scale sampling을 넣어 작은 객체와 수렴 문제를 개선한다.

통합 프로젝트 관점에서는 Faster R-CNN의 중요한 교훈이 여전히 유효하다. 비싼 visual feature를 task 사이에서 공유하고, dense frame 처리에서 얻은 후보로 더 비싼 후속 model의 실행 영역을 줄여야 한다. 다만 모바일 실시간 pipeline에서는 dynamic RoI와 두 번의 NMS가 operator 병목이 될 수 있으므로 anchor-free one-stage detector와 반드시 동일 장치에서 비교해야 한다.

## 최종 평가

Faster R-CNN의 가장 큰 기여는 detector의 정확도를 조금 올린 특정 수치가 아니라, **proposal을 feature-sharing neural module로 바꾸어 detection pipeline의 계산 구조를 재정의한 것**이다. RPN의 objectness는 수만 anchor를 수백 proposal로 압축하고, second stage는 그 제한된 영역에 더 강한 분류와 localization을 적용한다.

오늘날에는 anchor-free detector와 NMS-free set predictor가 등장했지만, proposal quality, candidate budget, feature sharing, coarse-to-fine cascade라는 사고방식은 여전히 중요하다. 온디바이스 연구에서는 논문의 `5 FPS`를 그대로 재현하는 것보다, 같은 입력과 backbone budget에서 RPN/ROI/NMS가 만드는 activation memory와 p95 tail을 계측하고 FCOS 또는 Deformable DETR 계열과 비교하는 것이 더 의미 있다.

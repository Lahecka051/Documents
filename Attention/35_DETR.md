# 35. End-to-End Object Detection with Transformers (DETR)

## 논문 정보

- 제목: **End-to-End Object Detection with Transformers**
- 저자: Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko
- 발표: ECCV 2020
- 핵심 키워드: set prediction, Hungarian matching, object queries, Transformer encoder-decoder, anchor-free, NMS-free

## 한눈에 보는 요약

DETR은 object detection을 anchor/proposal 분류 문제가 아니라 **고정 개수 prediction slot이 object set을 직접 출력하는 문제**로 재정의한다.

```text
CNN feature map
 -> Transformer encoder
 -> learned object queries + Transformer decoder
 -> N parallel class/box predictions
 -> Hungarian one-to-one matching loss
```

핵심은 Transformer만이 아니다. Ground-truth object와 prediction을 Hungarian algorithm으로 일대일 matching해 각 object가 정확히 하나의 slot에 할당되도록 한다. 중복 prediction이 loss에서 불리하므로 hand-designed anchor assignment와 NMS가 필요 없다.

ResNet-50 DETR은 COCO에서 42.0 AP, 28 FPS로 strong Faster R-CNN-FPN과 비슷한 성능·속도를 보였다. Large object AP는 61.1로 훨씬 높지만 small object AP는 20.5로 낮고, 500 epoch라는 매우 긴 학습이 필요하다.

![DETR의 image memory·object query·Hungarian set loss](https://github.com/user-attachments/assets/53eda2d5-0b7a-4b11-9569-99bdac51f06d)

## 기존 detection pipeline의 문제

Faster R-CNN, RetinaNet 등은 많은 후보를 만든 뒤 ground truth에 할당하고 중복을 제거한다.

```text
anchors/proposals
 -> heuristic positive/negative assignment
 -> class + box regression
 -> confidence threshold
 -> NMS duplicate removal
```

Anchor size/aspect ratio, matching IoU threshold, NMS threshold 같은 task-specific prior와 hyperparameter가 많다. Detection output은 순서 없는 set인데 학습은 수많은 local candidate의 independent prediction으로 우회한다.

DETR은 output set 전체에 대한 global loss로 이 pipeline을 단순화한다.

## Fixed-size prediction set

Ground-truth object 수보다 충분히 큰 `N`개의 prediction slot을 둔다. COCO 실험은 `N=100`이다. 각 slot `i`는

```text
p_i(c): class distribution including no-object empty
b_i    : normalized box (center_x, center_y, width, height)
```

를 출력한다. 실제 object가 적으면 나머지 slot은 no-object class를 예측한다. Output 순서는 의미가 없다.

## Bipartite matching

Ground-truth set을 no-object로 padding해 크기 N으로 맞추고, prediction과 일대일 대응하는 permutation `σ`를 찾는다.

```math
\hat\sigma=\operatorname*{arg\,min}_{\sigma\in\mathfrak{S}_N}
\sum_i\mathcal{L}_{\mathrm{match}}\!\left(y_i,\hat y_{\sigma(i)}\right)
```

Object ground truth의 pairwise matching cost는 class와 box를 함께 본다.

```math
\mathcal{L}_{\mathrm{match}}
=-\hat p_{\sigma(i)}(c_i)
+\mathbf{1}_{\{c_i\ne\varnothing\}}
\mathcal{L}_{\mathrm{box}}\!\left(b_i,\hat b_{\sigma(i)}\right)
```

Hungarian algorithm이 global minimum matching을 찾는다. 한 prediction은 한 target에만, 한 target도 한 prediction에만 대응한다.

### 왜 NMS가 사라지는가

같은 object를 여러 slot이 예측하면 matching에서 하나만 positive target과 연결되고 나머지는 no-object loss를 받는다. Decoder self-attention도 slot끼리 정보를 교환해 중복을 피하도록 학습한다.

## Hungarian loss

Matching이 정해진 후 최종 loss는

```math
\mathcal{L}=\sum_i\left[
-\log\hat p_{\hat\sigma(i)}(c_i)
+\mathbf{1}_{\{c_i\ne\varnothing\}}
\mathcal{L}_{\mathrm{box}}\!\left(b_i,\hat b_{\hat\sigma(i)}\right)
\right]
```

이다. No-object class가 훨씬 많으므로 class loss weight를 낮춰 imbalance를 조절한다.

## Bounding-box loss

Absolute coordinate L1만 쓰면 큰 box와 작은 box에 scale effect가 다르고 overlap quality를 직접 반영하지 못한다. DETR은 L1과 generalized IoU를 결합한다.

```math
\mathcal{L}_{\mathrm{box}}
=\lambda_{\mathrm{iou}}\mathcal{L}_{\mathrm{GIoU}}(b,\hat b)
+\lambda_{L1}\lVert b-\hat b\rVert_1
```

Box는 image 크기로 normalize한 center/width/height로 예측한다. Matching cost와 training loss에 같은 종류의 box term을 사용하되 weight는 다를 수 있다.

## Architecture 전체 흐름

### CNN backbone

ResNet이 input image를 high-level feature `f ∈ R^{C×H×W}`로 변환한다. 기본 stride 32, DC5 variant는 마지막 stage dilation으로 stride 16을 사용해 더 높은 resolution을 유지한다.

### 1×1 projection과 flatten

`C=2048` channel을 `d=256`으로 줄이고 spatial 축을 sequence로 펼친다.

```math
z_0\in\mathbb{R}^{d\times HW}
```

2D sine positional encoding을 각 spatial token에 더한다.

### Transformer encoder

6 layer, 8-head self-attention이 feature map의 모든 spatial position을 global하게 연결한다. Encoder가 object instance를 context로 분리하고 장거리 scene relation을 모델링한다.

### Transformer decoder와 object queries

`N=100`개의 learned object query embedding이 decoder slot 역할을 한다. Decoder self-attention은 slot끼리 관계와 중복을 조절하고, cross-attention은 각 slot이 encoder image memory에서 object evidence를 가져온다.

모든 object query는 autoregressive가 아니라 병렬로 decode된다.

### Prediction heads

각 decoder output에 shared FFN을 적용한다.

```text
class head: linear -> K+1 classes
box head  : 3-layer MLP -> sigmoid -> 4 normalized coords
```

## Object query의 의미

Object query는 특정 class나 anchor 위치에 고정되지 않은 learned positional embedding이다. 학습 후 slot마다 위치/크기 경향이 생길 수 있지만 한 slot이 항상 같은 object category를 담당하는 것은 아니다.

Query content는 decoder layer에서 image cross-attention으로 채워지고 query embedding은 slot identity와 initial prior를 제공한다.

## Auxiliary decoder losses

마지막 decoder layer에만 matching loss를 주면 초기 layer gradient가 약하고 올바른 object 수를 학습하기 어렵다. DETR은 각 decoder layer output에 같은 prediction head와 Hungarian loss를 적용한다.

```math
\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{final}}
+\sum_{l=1}^{L_{\mathrm{dec}}-1}\mathcal{L}_{\mathrm{aux}}^l
```

Prediction FFN parameter는 layer 간 공유한다. Auxiliary supervision은 convergence에 중요한 요소다.

## Position encoding

Encoder spatial token에는 fixed sine 2D position을 각 layer에서 사용하고, decoder에는 learned object query를 반복 제공한다. Position 정보를 제거하면 AP가 7.8 이상 크게 하락한다.

CNN feature만으로도 boundary/locality 정보가 있지만 Transformer attention은 순서 자체를 모르므로 explicit position이 필요하다.

## 계산 복잡도

Encoder token 수 `S=HW`에 대해 self-attention은 `O(S²d)`다. Backbone stride 32에서는 token 수가 작지만 DC5 stride 16은 S가 4배이고 attention cost는 약 16배가 된다.

Decoder query 수 `N=100`은 image token보다 작다.

```math
\begin{aligned}
\text{decoder self-attention}&:\ O(N^2d),\\
\text{cross-attention}&:\ O(NSd).
\end{aligned}
```

따라서 high-resolution에서는 encoder가 지배한다.

## COCO 결과

| 모델 | GFLOPs/FPS | Params | AP | AP50 | AP75 | APS | APM | APL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Faster R-CNN-FPN+ R50 | 180/26 | 42M | 42.0 | 62.1 | 45.5 | **26.6** | 45.4 | 53.4 |
| DETR R50 | 86/28 | 41M | 42.0 | 62.4 | 44.2 | 20.5 | **45.8** | **61.1** |
| DETR-DC5 R50 | 187/12 | 41M | 43.3 | 63.1 | 45.9 | 22.5 | 47.3 | 61.1 |
| DETR R101 | 152/20 | 60M | 43.5 | 63.8 | 46.4 | 21.9 | 48.0 | 61.8 |
| DETR-DC5 R101 | 253/10 | 60M | **44.9** | **64.7** | **47.7** | 23.7 | **49.5** | **62.3** |

R50 DETR은 같은 42 AP와 비슷한 FPS지만 small object에서 -6.1, large object에서 +7.7 AP다. Global attention은 large object와 scene-level reasoning에 강하지만 stride 32 feature가 작은 object detail을 잃는다. DC5가 resolution을 높여 APS를 개선하지만 FLOPs와 latency가 크게 늘어난다.

## 긴 학습 schedule

Ablation은 300 epoch, main comparison은 500 epoch를 사용한다. 300→500 epoch가 약 1.5 AP를 추가한다. R50 baseline 300 epoch는 16 V100에서 약 3일이다.

기존 Faster R-CNN의 1×/3× schedule보다 훨씬 느리게 수렴한다. One-to-one matching과 object query가 초기에는 안정적인 assignment를 만들기 어려운 것이 주요 약점이다.

## Encoder layer ablation

| Encoder layers | Params | AP | APS | APM | APL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 33.4M | 36.7 | 16.8 | 39.6 | 54.2 |
| 3 | 37.4M | 40.1 | 18.5 | 43.8 | 58.6 |
| 6 | 41.3M | 40.6 | 19.9 | 44.3 | 60.2 |
| 12 | 49.2M | **41.6** | 19.8 | **44.9** | **61.9** |

Encoder가 없으면 AP가 3.9, large AP가 6.0 하락한다. Global image self-attention이 instance disentangling과 large object reasoning에 중요하다.

## Decoder depth와 NMS

첫 decoder layer에서 마지막 layer까지 AP/AP50이 각각 약 `+8.2/+9.5` 개선된다. 초기 layer output에는 NMS가 도움되지만 depth가 늘수록 이득이 줄고 마지막 layer에서는 NMS가 true positive를 지워 AP가 약간 하락한다.

즉 one-to-one set prediction이 decoder refinement를 거치며 중복 제거를 내부화한다는 empirical 증거다.

## Loss ablation

Class loss가 대부분 성능을 만들고 L1/GIoU가 localization을 보완한다. GIoU는 특히 medium/large object에 유리하고 L1과 함께 쓸 때 best AP를 낸다. Matching cost와 final loss의 scale balance가 assignment 안정성에 직접 영향을 준다.

## Panoptic segmentation

Decoder object embedding과 encoder feature를 결합해 query별 binary mask를 예측하는 mask head를 추가한다. Thing과 stuff class를 동일한 set prediction으로 처리하고 pixel별 mask score argmax로 overlap 없는 panoptic output을 만든다.

R101 panoptic DETR은 약 46 PQ로 PanopticFPN 계열을 능가했다. Box matching을 그대로 사용해 mask query를 target과 연결하므로 별도 thing/stuff pipeline을 단순화한다.

## 장점과 기여

- Detection을 direct set prediction으로 재정의했다.
- Hungarian one-to-one loss로 anchors, proposal assignment, NMS를 제거했다.
- Object query와 encoder-decoder attention으로 모든 detection을 병렬 출력했다.
- Large object와 global scene reasoning에서 strong result를 보였다.
- 같은 framework를 panoptic segmentation으로 확장했다.

## 한계와 비판적 관점

### 1. 매우 느린 convergence

500 epoch가 필요해 training cost가 크다. 이후 Deformable DETR, DN-DETR 등이 sampling과 denoising으로 개선한 핵심 문제다.

### 2. Small object 성능

Single-scale stride-32 feature가 detail을 잃고 global attention은 고해상도에서 quadratic이다. Feature pyramid를 쓰는 detector보다 APS가 낮다.

### 3. 고정 query 수

Image의 object가 N=100을 넘으면 모두 예측할 수 없다. Dataset object-count distribution에 맞춰 upper bound를 정해야 한다.

### 4. Matching instability

Training 초기에 prediction이 나쁘면 Hungarian assignment가 epoch마다 크게 바뀔 수 있다. Loss weight와 initialization에 민감하다.

### 5. 완전히 prior-free는 아니다

Anchor/NMS는 없지만 CNN backbone, fixed query count, positional encoding, L1/GIoU weight, no-object weight라는 설계 prior는 남는다.

## 후속 연구 연결

- **Deformable DETR**: Sparse multi-scale sampling으로 convergence와 small object를 개선한다.
- **DAB/DN/DINO**: Query box prior와 denoising training으로 matching을 안정화한다.
- **Mask2Former**: Query-based set prediction을 universal segmentation으로 확장한다.
- **RT-DETR 등**: 실시간성과 efficient encoder를 강화한다.

## 구현 체크리스트

- Ground truth를 no-object로 padding하고 일대일 matching하는가?
- Matching cost와 final loss의 class/box weight가 구분되는가?
- Box가 normalized center format인지 GIoU용 corner format 변환이 정확한가?
- Padding image token이 encoder/cross-attention에서 mask되는가?
- 2D position encoding과 flatten 순서가 일치하는가?
- Auxiliary loss가 모든 decoder layer에 적용되는가?
- Inference에서 NMS 없이 no-object class를 제외하는가?

## 온디바이스 관점

DETR은 NMS/anchor generation을 제거해 post-processing graph가 단순하고 query 수가 고정되어 static compilation에 유리하다. 반면 CNN backbone과 6+6 Transformer, global encoder attention은 compute가 크고 small object를 위해 resolution을 올리면 quadratic 비용이 증가한다.

온디바이스에서는 lightweight backbone, reduced encoder, efficient/deformable attention, 적은 query 수, quantized MLP를 사용해야 한다. End-to-end latency 측정에는 NMS 제거 이득과 Transformer kernel 비용을 모두 포함해야 한다.

## 최종 평가

DETR의 가장 큰 공헌은 Transformer를 detector에 넣은 것보다 **object detection을 one-to-one set prediction으로 바꾸어 중복 제거와 assignment를 학습 안으로 흡수**한 데 있다. 구조가 놀랄 만큼 단순하고 large object와 panoptic에 강하지만, 느린 convergence와 small-object 약점도 분명하다. 이후 query-based detector와 segmentation model의 출발점이 된 전환점이다.

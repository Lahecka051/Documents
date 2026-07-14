# 45. Deformable DETR: Deformable Transformers for End-to-End Object Detection

## 논문 정보

- 원본 파일: `45_Deformable_DETR.pdf`
- 제목: **Deformable DETR: Deformable Transformers for End-to-End Object Detection**
- 저자: Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai
- 발표: ICLR 2021
- 공개: arXiv:2010.04159, 제공된 PDF는 v4(2021)
- 링크: [https://arxiv.org/abs/2010.04159](https://arxiv.org/abs/2010.04159)
- 핵심 키워드: Deformable DETR, multi-scale deformable attention, reference point, sparse sampling, set prediction, Hungarian matching, iterative box refinement, two-stage variant

## 한눈에 보는 요약

Deformable DETR은 DETR의 두 약점, 즉 **느린 수렴**과 **작은 객체 성능 부족**을 attention이 image feature 전체를 dense하게 보는 방식에서 찾는다. Standard attention은 초기에는 모든 spatial key에 거의 균일한 weight를 주므로, 객체 경계처럼 소수의 유용한 위치에 집중하기까지 긴 학습이 필요하다. High-resolution feature를 넣으면 key 수의 제곱에 비례하는 encoder self-attention memory도 급증한다.

이 논문은 각 query가 reference point 주변의 소수 sampling point만 보도록 하는 deformable attention을 제안한다.

```math
\operatorname{DeformAttn}(z_q,p_q,x)
=\sum_{m=1}^{M}W_m
\left[\sum_{k=1}^{K}A_{mqk}W'_m x(p_q+\Delta p_{mqk})\right].
```

Query feature에서 sampling offset과 attention weight를 예측하고, fractional coordinate의 value는 bilinear interpolation으로 읽는다. Multi-scale version은 네 feature level에서 head마다 `K=4` point씩 sample한다. 기본 `M=8`, `L=4`, `K=4`이면 한 query가 dense한 모든 pixel 대신 총 `8×4×4=128` head-level sample을 사용한다.

Encoder에서는 C3-C5와 추가 C6 multi-scale feature의 모든 pixel이 query가 되고, 각 pixel 자신이 reference point다. Decoder cross-attention에서는 300 object query마다 학습된 reference point 주변을 sample한다. Decoder query 사이의 self-attention은 standard attention을 유지한다.

COCO val에서 50 epoch의 Deformable DETR은 `43.8 AP`, `26.4 AP_S`를 기록했다. DETR은 같은 `42.0 AP`에 500 epoch가 필요했고 `AP_S=20.5`였다. Iterative box refinement를 더하면 `45.4 AP`, two-stage variant까지 쓰면 `46.2 AP`다. 논문 보고 기준 ResNet-50 model은 `40M params`, `173G FLOPs`, V100 `19 FPS`다.

핵심은 sparse attention이 단지 FLOPs를 줄이는 기술이 아니라는 점이다. Reference point와 multi-scale sampling이라는 강한 spatial prior가 optimization search space를 줄여 10배 짧은 schedule에서도 작은 객체를 잘 학습하게 한다. 반면 bilinear gather와 불규칙 memory access는 모바일 NPU에서 지원이 어렵고, 논문도 같은 FLOPs의 convolution보다 deformable attention이 조금 느릴 수 있음을 인정한다.

## DETR이 해결한 것과 남긴 문제

### Set prediction

전통 detector는 anchor/proposal assignment와 NMS로 중복을 다룬다. DETR은 고정 수 object query가 예측한 set과 GT set을 Hungarian bipartite matching해 one-to-one correspondence를 만든다.

```text
N predictions + G ground truths
 -> find minimum-cost one-to-one matching
 -> matched: class + box loss
 -> unmatched: no-object/background
```

같은 GT에 여러 prediction이 positive가 되지 않으므로 final NMS가 필요 없다. Deformable DETR도 이 set-based training과 NMS-free output을 유지한다.

### 문제 1: 느린 수렴

Standard attention의 초기 projected query/key가 평균 0, variance 1 정도라면 key가 매우 많을 때 softmax weight가 거의 `1/N_k`로 시작한다. 어떤 pixel을 봐야 하는지 모르는 상태에서 전체 image를 평균낸다. 학습이 진행되며 attention이 객체 extremity 같은 sparse 위치로 바뀌어야 하는데, DETR은 COCO에서 500 epoch가 필요했다.

### 문제 2: 작은 객체

DETR 원형은 계산량 때문에 주로 backbone의 낮은 해상도 feature를 사용했다. 작은 객체를 위해 stride 8/16 feature를 모두 standard self-attention에 넣으면 token 수와 attention matrix가 급증한다.

### Standard multi-head attention 복잡도

Query 수 `N_q`, key 수 `N_k`, channel `C`라 하면 논문의 식은:

```math
O(N_qC^2+N_kC^2+N_qN_kC).
```

Encoder image self-attention은 `N_q=N_k=HW`이므로 dominant term이:

```math
O(H^2W^2C)
```

이다. Decoder cross-attention은 object query 수가 작아 spatial size에 선형이지만, 초기 query가 전체 feature를 검색해야 하는 optimization 문제는 남는다.

## Deformable attention

### 입력과 출력

- Query content: `z_q in R^C`
- Reference point: `p_q=(p_x,p_y)`
- Input feature: `x in R^{C×H×W}`
- Heads: `M`
- Samples/head: `K`

각 head와 sample마다 query에서 다음을 예측한다.

- `Delta p_mqk in R^2`: reference에서의 2D offset
- `A_mqk in [0,1]`: attention weight, head별 K개 합이 1

```math
\sum_{k=1}^{K}A_{mqk}=1.
```

Offset의 범위는 제한되지 않는다. `p_q+Delta p`가 fractional coordinate면 네 이웃 pixel을 bilinear interpolation한다.

### Query projection shape

Single-scale module에서 query 하나는:

- offsets: `2MK`
- weights: `MK`
- 합계: `3MK`

channel의 linear projection을 낸다. Multi-scale에서는 `L`이 들어가 `2MLK + MLK = 3MLK`다. 기본 `M=8,L=4,K=4`이면 query마다 256 offset 값과 128 weight logit, 총 384개를 예측한다.

### Reference point의 역할

Reference point는 attention search의 좌표 원점이다. Standard cross-attention은 query가 image 전체 어디든 처음부터 비교하지만, deformable attention은 우선 "어디 근처를 볼지"를 정하고 그 주변 offset만 학습한다.

이는 anchor box와 같지 않다.

- Reference point는 처음에는 2D point이며 고정 scale/ratio가 없다.
- Sampling offset은 content-dependent하고 head/query마다 다르다.
- Prediction matching은 anchor IoU가 아니라 Hungarian matching이다.

## Multi-scale deformable attention

Feature level `l`의 map을 `x^l in R^{C×H_l×W_l}`, normalized reference를 `p_hat_q in [0,1]^2`라 하자.

```math
\operatorname{MSDeformAttn}(z_q,\hat p_q,\{x^l\})
=\sum_{m=1}^{M}W_m
\left[
\sum_{l=1}^{L}\sum_{k=1}^{K}
A_{mlqk}W'_m x^l(\phi_l(\hat p_q)+\Delta p_{mlqk})
\right].
```

`phi_l`은 `[0,1]` normalized coordinate를 각 level의 pixel 좌표로 rescale한다. Attention weight는 한 head 안의 모든 level/sample에 대해:

```math
\sum_{l=1}^{L}\sum_{k=1}^{K}A_{mlqk}=1
```

로 normalize한다. 한 query가 같은 normalized location의 fine/coarse feature 주변을 동시에 볼 수 있어 명시적 FPN top-down fusion 없이 scale 간 정보가 교환된다.

### Deformable convolution과의 관계

논문은 `L=1`, `K=1`, value projection을 identity로 고정하면 deformable convolution으로 축약된다고 설명한다. 반대로 sampling point가 모든 spatial location을 순회하면 standard attention에 가까워진다.

차이는 다음과 같다.

| 관점 | Deformable convolution | Deformable attention |
| --- | --- | --- |
| Sampling | local learned offsets | reference 주변 learned offsets |
| Weight | kernel/content modulation 계열 | query-dependent softmax weights |
| Scale | 주로 single feature map | multi-level을 한 module에서 집계 |
| Relation modeling | 제한적 | query-key relation 형태 유지 |

## 복잡도

논문 Appendix의 single-scale deformable attention 세부 complexity는:

```math
O\left(
N_qC^2+min(HWC^2,N_qKC^2)+5N_qKC+3N_qCMK
\right).
```

기본 `M=8`, `K<=4`, `C=256`에서 `5K+3MK<C`이므로 다음처럼 요약한다.

```math
O\left(2N_qC^2+\min(HWC^2,N_qKC^2)\right).
```

- Encoder: `N_q=HW`, spatial size에 선형
- Decoder cross-attention: `N_q=N_object`, sampling aggregation은 image spatial size와 무관

Big-O가 작아져도 bilinear sampling의 주소가 연속적이지 않다. Arithmetic보다 memory gather와 interpolation이 실제 latency를 지배할 수 있다. 논문은 image용 sparse attention이 같은 FLOPs의 convolution보다 적어도 3배 느렸던 선행 사례를 언급하고, 자신들의 module도 전통 convolution보다 약간 느리다고 보고한다.

## Deformable Transformer encoder

### Multi-scale input 생성

ResNet의 C3, C4, C5를 각각 `1×1 conv`로 256 channel에 투영한다. C5에는 `3×3 stride-2 conv`를 적용해 C6를 추가한다.

```text
C3: stride 8,  512 ch -> 1x1 -> 256
C4: stride 16, 1024 ch -> 1x1 -> 256
C5: stride 32, 2048 ch -> 1x1 -> 256
C6: stride 64, C5 -> 3x3 s2 -> 256
```

FPN의 top-down/lateral pathway는 쓰지 않는다. Encoder의 multi-scale deformable self-attention이 level 간 정보를 직접 교환하기 때문이다.

### Encoder query와 reference

모든 level의 모든 pixel이 query이자 key/value source다. 각 query pixel의 reference point는 자기 위치다. Position embedding에 더해 level identity를 구분하는 learnable scale-level embedding `e_l`을 추가한다.

Input과 output은 같은 네 해상도, 같은 256 channel을 유지한다. Encoder를 지나도 하나의 flat low-resolution map으로 합치지 않고 multi-scale layout을 보존한다.

## Deformable Transformer decoder

Decoder에는 두 attention이 있다.

1. Object query 사이의 self-attention: standard multi-head attention 유지
2. Object query가 image memory를 읽는 cross-attention: multi-scale deformable attention으로 교체

Deformable module은 convolutional feature map을 key로 읽도록 설계되었으므로 query-to-query self-attention까지 바꾸지 않는다.

기본 object query 수는 DETR의 100에서 `300`으로 늘린다. 각 query embedding에서 linear + sigmoid로 2D normalized reference point를 예측한다. Detection head는 이 reference를 초기 box center로 보고 상대 offset을 예측한다.

```math
\hat b_q=
\left{
\sigma(b_{qx}+\sigma^{-1}(\hat p_{qx})),
\sigma(b_{qy}+\sigma^{-1}(\hat p_{qy})),
\sigma(b_{qw}),
\sigma(b_{qh})
\right}.
```

Inverse sigmoid space에서 offset을 더한 뒤 sigmoid로 `[0,1]`에 넣는다. Reference와 box center가 구조적으로 연결되어 attention 위치와 detection output의 상관이 강해진다.

## Set matching, loss와 NMS

### Hungarian matching

Deformable DETR은 300 prediction과 GT set 사이의 bipartite matching을 사용한다. Matching cost는 DETR의 classification 및 box distance/overlap 항을 따른다. Matched query는 해당 class와 box를 학습하고, unmatched query는 background/no-object가 된다.

Box loss는 DETR에서 이어받은 L1 coordinate loss와 generalized IoU 계열을 사용한다. 제공된 논문은 나머지 hyperparameter를 DETR에 따른다고 하고, classification은 **Focal Loss weight 2**로 바꾸었다고 명시한다. 모든 matching/loss coefficient를 본문에 다시 표로 제시하지 않으므로 구현 재현 시 공식 code/config와 version을 함께 기록해야 한다.

### NMS가 없는 이유

One-to-one matching은 같은 GT를 여러 query의 positive target으로 두지 않는다. 따라서 inference는 300 query의 score/box를 바로 ranking하며 RPN NMS나 final class-wise NMS가 없다.

Two-stage variant도 첫 stage proposal을 second-stage decoder에 넣기 전에 **NMS를 적용하지 않는다**. 여기서 "two-stage"는 Faster R-CNN의 RPN+RoI head와 구조가 다르다.

## Iterative bounding-box refinement

Decoder layer `d-1`의 box를 다음 layer가 inverse-sigmoid space에서 보정한다.

```math
\hat b_q^d=\sigma\left(\Delta b_q^d+\sigma^{-1}(\hat b_q^{d-1})\right)
```

식은 center x/y와 width/height 네 component에 각각 적용한다. 기본 decoder layer 수는 `D=6`이며 layer별 prediction head는 parameter를 공유하지 않는다.

초기 box는:

```math
\hat b^0=(\hat p_x,\hat p_y,0.1,0.1).
```

논문은 초기 width/height를 0.05, 0.1, 0.2, 0.5로 바꿔도 비슷했다고 보고한다. Training 안정화를 위해 이전 box의 inverse-sigmoid 경로에서는 gradient를 stop하고 현재 delta로만 역전파한다.

다음 decoder layer의 sampling reference는 이전 layer box center로 바뀌고 offset은 box width/height로 modulation된다. Refinement가 진행될수록 attention sampling 영역도 예측 box에 맞춰진다.

## Two-stage Deformable DETR

원 DETR의 learned object query는 현재 image와 무관하게 시작한다. Two-stage variant는 encoder output의 모든 pixel에서 foreground score와 box를 예측해 image-conditioned proposal을 만든다.

```text
multi-scale encoder pixels
 -> binary foreground + box head
 -> top-scoring boxes, no NMS
 -> their coordinates become decoder query positional embeddings / initial boxes
 -> iterative decoder refinement
```

첫 stage의 모든 pixel을 decoder query로 넣으면 decoder self-attention이 pixel 수의 제곱이 되므로, encoder-only head에서 top score만 선택한다. Faster R-CNN처럼 RoI pooling하지 않고 proposal coordinate가 decoder query 초기화에 쓰인다.

첫 stage box width/height는 level `l_i`와 base scale `s=0.05`에 대한 prior를 inverse-sigmoid offset으로 보정한다. 이 변형도 Hungarian loss로 학습한다.

## Batch=1 tensor shape 계산

### 800×1024 input

`B=1`, 256 channel, input이 stride에 맞게 padding되었다고 가정한다. 아래는 **리뷰어 계산**이다.

| Level | projected shape | tokens |
| --- | ---: | ---: |
| C3 | `1×256×100×128` | 12,800 |
| C4 | `1×256×50×64` | 3,200 |
| C5 | `1×256×25×32` | 800 |
| C6 | `1×256×13×16` | 208 |
| 합계 | flattened `[1,17008,256]` | 17,008 |

Multi-scale memory payload:

```math
17{,}008\cdot256=4{,}354{,}048\ \text{elements}
```

- FP16 약 `8.30 MiB`
- FP32 약 `16.61 MiB`

이는 final projected feature만의 값이다. Backbone stage tensor, positional/level embedding, encoder layer activation은 제외한다.

### Sparse attention weight와 offset

기본 `M=8,L=4,K=4`에서 encoder 한 layer의 query별 weight는 128개, offset scalar는 256개다.

```math
\begin{aligned}
\text{weights} &:17{,}008\cdot128=2{,}177{,}024,\\
\text{offset scalars} &:17{,}008\cdot256=4{,}354{,}048.
\end{aligned}
```

FP16 payload는 각각 약 `4.15 MiB`, `8.30 MiB`다. 한 encoder attention layer에서 두 tensor만 약 `12.45 MiB`다. Training에서는 여러 layer와 backward activation이 보존된다.

Decoder cross-attention의 query가 300개면:

- weights: `300×128=38,400`, FP16 약 0.073 MiB
- offsets: `300×256=76,800`, FP16 약 0.146 MiB
- query self-attention weights: `8×300×300=720,000`, FP16 약 1.37 MiB

Decoder에서는 sparse cross-attention보다 300-query self-attention matrix가 더 클 수 있지만 여전히 image token 전체의 quadratic matrix보다 작다.

### Dense attention과 비교

17,008 token 전체에 8-head self-attention을 직접 적용하면 attention score element는:

```math
8\cdot17{,}008^2\approx2.31\times10^9
```

FP16 payload만 약 `4.31 GiB`다. Deformable attention weight 약 `4.15 MiB`와 세 자릿수 배 차이다. 다만 standard DETR은 이 네 high-resolution level을 그대로 모두 사용한 것이 아니므로, 이는 동일 multi-scale token set에서 dense vs sparse를 비교한 reviewer calculation이다.

### Sampled value materialization 주의

Naive하게 `[N_q,M,L,K,C/M]` sampled value를 완전히 펼치면 encoder 한 layer에서 약 6,966만 element, FP16 약 133 MiB가 된다. 최적화 kernel은 sampling과 weighted sum을 tile/fuse해 이 전체 buffer를 저장하지 않을 수 있다. 따라서 이론적 sparse weight 크기만으로 peak memory를 추정하지 말고 실제 operator implementation을 profile해야 한다.

## Core module parameter와 연산

### Reviewer calculation

256 channel, `M=8,L=4,K=4`인 일반적인 multi-scale deformable attention 한 module의 주요 dense projection weight는:

- value projection `256→256`: 65,536
- sampling offset `256→256`: 65,536
- attention weight `256→128`: 32,768
- output projection `256→256`: 65,536

합계 `229,376 weights`, bias 제외 FP16 약 `0.44 MiB`다. Encoder/decoder layer마다 module parameter는 별도이며, FFN·decoder self-attention·detection head는 포함하지 않는다.

Encoder query 17,008개에서 offset/weight/output projection과 full value projection만 계산해도 수 G MAC 규모다. Bilinear interpolation과 gather cost는 MAC 표에 온전히 표현되지 않는다. 논문이 보고한 ResNet-50 전체 model 수치는 다음과 같다.

- `40M params`
- `173G FLOPs`
- V100 `19 FPS`

이 값이 paper-reported이고, 위 projection breakdown은 reviewer-calculated다.

## 초기화

Sampling offset/attention projection weight는 zero로 초기화한다. Attention bias는 모든 `LK` sample weight가 `1/(LK)`가 되게 한다. Offset bias는 8개 head가 reference 주위의 서로 다른 방향을 보도록 초기화한다.

```text
head directions:
(-k,-k), (-k,0), (-k,k), (0,-k),
(0,k), (k,-k), (k,0), (k,k)
```

`k=1,...,K`로 반경을 늘린다. 완전히 같은 점에서 시작하지 않게 spatial prior를 주는 것이다. Iterative refinement decoder에서는 초기 sample이 이전 predicted box 안에 머물도록 offset bias를 추가 scaling한다.

이 초기화는 sparse sampling의 학습 안정성에 중요하다. Random offset이 image 밖이나 무관한 지점에 몰리면 gradient가 약해질 수 있다.

## 구현 pseudocode

```python
def ms_deform_attn(query, reference, multi_scale_value):
    # query: [B, Nq, C]
    # reference: [B, Nq, L, 2] normalized coordinates
    offsets = offset_proj(query).reshape(B, Nq, M, L, K, 2)
    attn = weight_proj(query).reshape(B, Nq, M, L * K)
    attn = softmax(attn, dim=-1).reshape(B, Nq, M, L, K)

    values = value_proj(flatten_levels(multi_scale_value))
    output = 0
    for l in range(L):
        # convert normalized reference + normalized offsets for level l
        coords = make_sampling_grid(reference[:, :, l], offsets[:, :, :, l],
                                    H=height[l], W=width[l])
        sampled = bilinear_sample(values[l], coords)  # custom fused op in practice
        output += (attn[:, :, :, l, :, None] * sampled).sum(sample_axis)
    return output_proj(merge_heads(output))


def deformable_detr(image):
    c3, c4, c5 = resnet(image)
    levels = [proj3(c3), proj4(c4), proj5(c5), stride2_c6(c5)]
    memory = deformable_encoder(levels, positional_and_level_embeddings)

    query = learned_object_queries(300)
    reference = sigmoid(reference_head(query))
    boxes_per_layer, logits_per_layer = [], []

    for decoder_layer in decoder_layers:
        query = self_attention(query)
        query = ms_deform_cross_attention(query, reference, memory)
        delta = box_head[decoder_layer](query)
        box = sigmoid(delta + inverse_sigmoid(reference_or_previous_box))
        logits = class_head[decoder_layer](query)
        boxes_per_layer.append(box)
        logits_per_layer.append(logits)
        reference = stop_gradient(box)  # refinement reference

    return logits_per_layer, boxes_per_layer
```

Production implementation은 Python loop/grid-sample 조합보다 fused CUDA/accelerator kernel이 필요하다. Masked padding image에서는 level별 valid ratio를 reference coordinate에 반영해야 한다.

## 학습 설정

- Dataset: COCO 2017 train, val ablation, test-dev final
- Backbone: ImageNet pretrained ResNet-50 ablation
- Encoder/decoder hidden `C=256`
- Deformable heads `M=8`, samples/head/level `K=4`
- Object queries `300`
- 기본 50 epochs, epoch 40에서 LR 0.1배
- Adam, LR `2×10^-4`, beta1 0.9, beta2 0.999, weight decay `10^-4`
- Reference point와 sampling offset projection LR은 0.1배
- Classification은 focal loss, weight 2
- 다른 training strategy는 DETR을 주로 따름
- Runtime: NVIDIA Tesla V100

Reference/offset projection에 작은 LR을 쓰는 것은 sampling geometry가 초기에 급격히 무너지는 것을 막는 역할로 해석할 수 있다.

## DETR 및 Faster R-CNN 비교

COCO 2017 val, ResNet-50 계열:

| Method | epochs | AP | AP50 | AP75 | AP small | AP medium | AP large | params | FLOPs | train GPU h | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Faster R-CNN + FPN | 109 | 42.0 | 62.1 | 45.5 | 26.6 | 45.4 | 53.4 | 42M | 180G | 380 | 26 |
| DETR | 500 | 42.0 | 62.4 | 44.2 | 20.5 | 45.8 | 61.1 | 41M | 86G | 2000 | 28 |
| DETR-DC5 | 500 | 43.3 | 63.1 | 45.9 | 22.5 | 47.3 | 61.1 | 41M | 187G | 7000 | 12 |
| DETR-DC5+ | 50 | 36.2 | 57.0 | 37.4 | 16.3 | 39.2 | 53.9 | 41M | 187G | 700 | 12 |
| Deformable DETR | 50 | 43.8 | 62.6 | 47.7 | 26.4 | 47.1 | 58.0 | 40M | 173G | 325 | 19 |
| + iterative refinement | 50 | 45.4 | 64.7 | 49.0 | 26.8 | 48.3 | 61.7 | 40M | 173G | 325 | 19 |
| + two-stage | 50 | 46.2 | 65.2 | 50.0 | 28.8 | 49.2 | 61.7 | 40M | 173G | 340 | 19 |

핵심 비교는 다음과 같다.

- DETR과 비슷하거나 높은 AP를 500→50 epoch로 단축
- DETR 대비 AP_S `20.5→26.4`
- FLOPs는 Faster R-CNN과 비슷하지만 FPS는 26보다 낮은 19
- DETR-DC5보다 1.6배 빠르지만 convolution detector보다 약 25% 느림

FLOPs가 DETR 86G보다 큰데도 학습과 작은 객체가 좋아졌다. 효율을 단순 FLOPs 감소가 아니라 high-resolution multi-scale feature를 감당 가능한 sparse pattern으로 쓴 결과로 봐야 한다.

## Deformable attention ablation

COCO val:

| Multi-scale inputs | Multi-scale attention | K | AP | AP small | AP medium | AP large |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 아니오 | 아니오 | 1 | 39.7 | 21.2 | 44.3 | 56.0 |
| 예 | 아니오 | 1 | 41.4 | 24.1 | 44.6 | 56.1 |
| 예 | 아니오 | 4 | 42.3 | 24.8 | 45.1 | 56.3 |
| 예 | 예 | 4 | 43.8 | 26.4 | 47.1 | 58.0 |

해석:

1. Multi-scale input만 추가: `+1.7 AP`, `+2.9 AP_S`
2. K 1→4: `+0.9 AP`
3. Level 간 deformable attention: `+1.5 AP`

Full module 위에 FPN을 추가하면 `43.8 AP`, BiFPN은 `43.9 AP`로 사실상 개선이 없다. Cross-level exchange가 이미 attention 안에 있기 때문이다. 이는 모든 task에서 FPN이 불필요하다는 뜻이 아니라 이 architecture와 setting에서 중복이었다는 결과다.

## COCO test-dev 결과

Iterative refinement와 two-stage를 모두 사용한 model:

| Backbone | TTA | AP | AP50 | AP75 | AP small | AP medium | AP large |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-50 | 없음 | 46.9 | 66.4 | 50.8 | 27.7 | 49.7 | 59.9 |
| ResNet-101 | 없음 | 48.7 | 68.1 | 52.9 | 29.1 | 51.5 | 62.0 |
| ResNeXt-101 | 없음 | 49.0 | 68.5 | 53.2 | 29.7 | 51.7 | 62.8 |
| ResNeXt-101 + DCN | 없음 | 50.1 | 69.7 | 54.6 | 30.6 | 52.8 | 64.7 |
| ResNeXt-101 + DCN | 사용 | 52.3 | 71.9 | 58.1 | 34.4 | 54.4 | 65.6 |

TTA는 horizontal flip과 multi-scale testing을 포함한다. Single-scale result와 직접 latency 비교하면 안 된다.

## 작은 객체 분석

Deformable DETR이 작은 객체를 개선하는 구조적 이유는 세 가지다.

1. Stride 8 C3를 포함한 multi-scale feature를 encoder에 넣는다.
2. Dense quadratic attention 대신 고정 수 sampling point로 high-resolution map 비용을 제어한다.
3. Object query가 fine level의 reference 주변을 직접 읽을 수 있다.

DETR `AP_S=20.5`, Deformable DETR `26.4`, two-stage `28.8`은 이를 지지한다. Multi-scale input ablation의 AP_S `+2.9`도 직접 근거다.

하지만 sparse sampling은 작은 객체를 놓칠 위험도 있다. Reference point가 빗나가고 K가 너무 작으면 object pixel을 하나도 sample하지 못한다. Multi-head/level, radial initialization, iterative refinement가 이 위험을 줄인다. K를 무작정 늘리면 memory gather와 compute가 증가하고 dense attention에 가까워진다.

## Matching과 중복 예측 동작

### Anchor-based와의 차이

- Faster R-CNN/RetinaNet: 한 GT에 여러 positive anchor 가능, NMS로 중복 제거
- FCOS: 한 GT 내부 여러 positive location, NMS로 중복 제거
- Deformable DETR: Hungarian matching으로 한 GT에 query 하나, NMS 없음

Two-stage Deformable DETR의 encoder proposal도 anchor IoU assignment로 학습하지 않는다. Pixel-based initial proposal 중 top score를 고르지만 NMS 없이 decoder와 set loss가 중복을 정리한다.

Object query 수 300은 maximum detection capacity처럼 작동한다. 매우 crowded image에서 GT 수가 query 수에 가까워지면 recall upper bound가 생긴다. 반대로 query를 크게 늘리면 decoder self-attention과 classification output이 증가한다.

## 장점과 핵심 기여

1. DETR의 느린 수렴과 small-object 문제를 attention 구조의 관점에서 정확히 진단했다.
2. Reference 기반 sparse sampling으로 spatial complexity를 선형화했다.
3. Multi-scale feature exchange를 attention 내부에 통합해 FPN 없이 강한 scale robustness를 얻었다.
4. 500→50 epoch로 수렴을 크게 단축했다.
5. Iterative refinement와 image-conditioned two-stage query 초기화를 제시했다.
6. Hungarian set prediction과 NMS-free inference를 유지했다.
7. 이후 DINO, RT-DETR 등 실용 DETR 계열의 핵심 building block이 되었다.

## 한계와 비판적 관점

### 1. Custom operator 의존

Multi-scale bilinear sampling과 weighted gather는 일반 GEMM/conv보다 framework와 hardware 지원이 약하다. Official CUDA kernel이 없는 backend에서는 매우 느릴 수 있다.

### 2. Irregular memory access

Sampling offset이 query마다 달라 연속 memory access와 cache reuse가 어렵다. FLOPs가 비슷한 Faster R-CNN보다 실제 FPS가 낮은 이유다.

### 3. 여전히 복잡한 Transformer stack

6 encoder + 6 decoder, 300 query, FFN, self-attention, auxiliary prediction을 포함한다. 단순 one-stage CNN보다 graph와 memory lifetime이 복잡하다.

### 4. Sparse sampling의 miss 가능성

Reference가 부정확하거나 K가 너무 작으면 중요한 pixel을 보지 못한다. 초기화와 iterative refinement에 성능이 의존한다.

### 5. Two-stage라는 이름의 혼동

Faster R-CNN식 proposal+RoI classifier가 아니다. Encoder proposal로 decoder query를 초기화하는 two-stage Transformer다.

### 6. 시스템 측정 부족

논문은 V100 FPS와 training GPU hours를 제공하지만 peak memory, p50/p95, batch 조건의 상세 분포, energy와 thermal throttling은 없다.

### 7. End-to-end에도 선택 규칙은 남는다

Anchor/NMS는 없지만 object query 수, top-scoring encoder proposal 선택, decoder layer 수, K/L/M 같은 design choice는 존재한다.

## 자주 헷갈리는 지점

### Deformable attention은 deformable convolution인가

특정 제한에서 축약 관계가 있지만 일반 module은 multi-head, query-dependent attention weight와 multi-scale sampling을 사용한다.

### K=4면 query가 네 점만 보는가

Level과 head마다 4점이다. 기본은 `M×L×K=8×4×4=128` head-level samples다. 같은 coordinate가 겹칠 수는 있다.

### FPN을 전혀 쓰지 않는가

Top-down/lateral FPN은 쓰지 않지만 C3-C6 multi-scale feature 자체는 사용한다. "Single-scale" 모델이 아니다.

### Encoder attention과 decoder attention이 모두 deformable인가

Image feature를 key로 쓰는 encoder self-attention과 decoder cross-attention이 deformable이다. Decoder object-query self-attention은 standard 방식이다.

### Two-stage variant는 NMS를 쓰는가

아니다. Encoder top boxes를 decoder에 넣기 전 NMS를 적용하지 않는다.

### Reference point가 최종 box center인가

초기 guess이며 head가 inverse-sigmoid offset으로 보정한다. Iterative refinement에서는 이전 predicted box center가 다음 reference가 된다.

### Sparse attention이면 무조건 모바일에서 빠른가

아니다. Custom bilinear gather가 unsupported이면 CPU fallback으로 dense conv보다 훨씬 느릴 수 있다.

## 온디바이스 관점

### Operator 지원이 첫 번째 질문

배포 전 target SDK에서 다음을 확인해야 한다.

- multi-scale deformable attention native op 존재 여부
- `grid_sample`/bilinear sampler의 dynamic coordinate 지원
- gather + weighted sum fusion
- inverse sigmoid, reference normalization
- padding mask와 valid ratio 처리

Native op가 없으면 graph가 작은 elementwise/gather op로 분해되거나 CPU로 fallback한다. 이 경우 paper FLOPs는 latency 예측에 거의 쓸 수 없다.

### Memory bandwidth

Dense attention matrix는 제거했지만 query마다 128개의 흩어진 위치를 읽는다. DRAM transaction, cache miss, coordinate calculation과 interpolation이 중요하다. Weight/offset tensor도 encoder layer마다 수 MiB다.

### Decoder layer 수와 latency

Iterative refinement는 각 decoder layer를 순차 실행하므로 layer 수가 critical path다. 6 layer를 3/4 layer로 줄일 때 AP, AP_S와 latency를 함께 ablation해야 한다. RT-DETR 계열이 decoder/query selection을 효율화하는 이유와 연결된다.

### 정적 shape의 장점

Object query가 300으로 고정되고 NMS가 없어 post-processing latency variance는 anchor detector보다 낮을 가능성이 있다. 그러나 sparse kernel의 scheduling과 memory allocation에 따라 p95가 달라질 수 있으므로 실제 측정이 필요하다.

### Mixed precision

- Backbone/linear/FFN: INT8 후보
- Sampling offset/reference/inverse sigmoid: FP16 유지가 안전할 수 있음
- Bilinear interpolation weight: precision 감소가 localization AP75에 미치는 영향 측정
- Class head: focal-trained logit calibration 확인

Box coordinate와 sampling offset까지 무조건 INT8로 만들면 작은 객체의 sub-pixel sampling 오차가 커질 수 있다.

## 재현 체크리스트

- [ ] 제공 PDF가 ICLR 2021/arXiv v4인지 기록한다.
- [ ] C3-C5 1×1 projection과 C6 3×3 stride-2를 구현한다.
- [ ] FPN top-down/lateral을 기본 model에 넣지 않는다.
- [ ] 모든 level channel 256, level embedding을 확인한다.
- [ ] `M=8,L=4,K=4`, weights의 LK softmax 축을 확인한다.
- [ ] Reference coordinate와 level별 normalization/valid ratio를 검증한다.
- [ ] Fractional sampling이 bilinear interpolation인지 확인한다.
- [ ] Offset/attention radial initialization을 재현한다.
- [ ] Encoder self와 decoder cross만 deformable인지 확인한다.
- [ ] Object queries 300, encoder/decoder layer 수를 기록한다.
- [ ] Focal classification weight 2와 DETR matching/box loss config version을 기록한다.
- [ ] 50 epoch, LR drop 40, projection LR multiplier 0.1을 맞춘다.
- [ ] Iterative refinement head 비공유와 stop-gradient를 확인한다.
- [ ] Two-stage encoder proposal에 NMS를 넣지 않는다.
- [ ] Basic, refinement, two-stage 결과를 분리한다.
- [ ] AP/AP50/AP75/AP_S/M/L, params/FLOPs/FPS를 모두 비교한다.
- [ ] Custom op와 fallback 여부를 target runtime에서 확인한다.
- [ ] Batch=1 peak memory, p50/p95, layer별 latency, 전력·온도를 측정한다.

## 로드맵에서의 연결

- **DETR**의 set prediction과 Hungarian matching을 유지하면서 실용적인 수렴·small-object 성능을 만든 핵심 후속작이다.
- **FPN**과 같은 multi-scale 목표를 top-down conv가 아니라 cross-level sparse attention으로 달성한다.
- **Faster R-CNN/FCOS**와 비교하면 NMS가 없지만 custom attention operator와 decoder latency가 새 시스템 비용이다.
- **RT-DETRv2**는 decoder/query selection과 training recipe를 더 실시간 친화적으로 다듬는다.
- **DINO 계열**은 denoising training과 query initialization을 개선하면서 deformable attention을 핵심 연산으로 사용한다.

로드맵 실험에서는 anchor-free CNN detector와 Deformable DETR을 동일 input/backbone/장치에서 비교하고, NMS 비용과 deformable-attention custom-op 비용을 모두 포함해야 한다. Decoder layer 수, query 100/300, K 2/4, C3 사용 여부를 ablation하면 accuracy-latency-memory trade-off를 명확히 볼 수 있다.

## 최종 평가

Deformable DETR은 DETR의 철학을 유지하면서 실제 detection에 필요한 multi-scale locality를 attention 안에 넣었다. Reference point와 소수 learned sample은 attention을 객체 주변으로 유도해 10배 짧은 학습에서도 더 높은 AP와 small-object 성능을 만든다. Iterative refinement와 encoder proposal 초기화는 query가 box를 찾는 과정을 더 명시적인 coarse-to-fine optimization으로 바꾼다.

그러나 sparse는 곧 hardware-efficient를 뜻하지 않는다. 논문 자체에서도 173G FLOPs의 model이 V100에서 Faster R-CNN보다 느렸고, 핵심 원인은 불규칙 memory access다. 온디바이스 연구에서 이 논문의 진짜 질문은 "quadratic attention을 줄였는가"에서 끝나지 않는다. Target accelerator가 bilinear gather를 어떻게 실행하는지, 6-layer decoder가 p95와 peak memory를 어떻게 만드는지까지 측정해야 한다. 그 조건을 충족할 때 Deformable DETR은 NMS-free end-to-end detector와 multi-scale small-object 성능을 함께 제공하는 강력한 기준점이 된다.

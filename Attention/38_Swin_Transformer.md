# 38. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

## 논문 정보

- 제목: **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**
- 저자: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo
- 소속: Microsoft Research Asia
- 발표: ICCV 2021
- arXiv: [https://arxiv.org/abs/2103.14030](https://arxiv.org/abs/2103.14030)
- PDF 기준 버전: arXiv v2, 2021-08-17
- 원본 파일: `38_Swin_Transformer.pdf`
- 핵심 키워드: hierarchical vision Transformer, window attention, shifted window, patch merging, relative position bias, dense prediction

## 한눈에 보는 요약

Swin Transformer는 원형 ViT가 가진 두 가지 실용적 한계를 해결한다.

1. 모든 patch가 같은 scale과 resolution을 유지해 FPN 같은 dense prediction 구조에 바로 넣기 어렵다.
2. Global self-attention의 token-pair 비용이 image 면적의 제곱으로 증가한다.

해결책은 CNN과 비슷한 4-stage hierarchy와 고정 크기 local window attention이다.

```text
image
 -> 4×4 patch partition + linear embedding
 -> Stage 1: H/4 × W/4, C
 -> patch merging
 -> Stage 2: H/8 × W/8, 2C
 -> patch merging
 -> Stage 3: H/16 × W/16, 4C
 -> patch merging
 -> Stage 4: H/32 × W/32, 8C
```

각 block은 `7×7` window 안에서만 self-attention한다. 이대로면 서로 다른 window가 영원히 통신하지 못하므로, 다음 block에서 partition을 `(3,3)` patch만큼 이동한다. 구현은 실제로 불규칙한 작은 window를 만들지 않고 **cyclic shift -> 고정 window partition -> attention mask -> reverse shift**로 처리한다.

Swin-T는 ImageNet-1K에서 `81.3%` top-1, `29M` parameters, `4.5G` FLOPs를 보고했다. Cascade Mask R-CNN에서는 `50.5 box AP / 43.7 mask AP`, UperNet ADE20K에서는 `46.1 mIoU`다. ImageNet-22K로 pretrain한 Swin-L은 ImageNet `87.3%`, COCO test-dev `58.7 box AP / 51.1 mask AP`, ADE20K `53.5 mIoU`에 도달했다.

논문의 핵심 기여는 local attention 하나가 아니다. **고정 window로 계산을 regular하게 유지하면서, shift로 cross-window 연결을 만들고, patch merging으로 CNN-compatible feature pyramid를 제공한 것**이다.

## 연구 배경과 문제의식

### ViT를 dense vision backbone으로 쓸 때의 문제

ViT-B/16은 `224×224` image를 196 patch token으로 만들고 모든 layer에서 같은 resolution을 유지한다. Classification에서는 마지막 class token만 쓰면 되지만 detection과 segmentation에는 다음이 필요하다.

- 작은 물체를 위한 high-resolution feature
- 큰 물체와 semantic context를 위한 low-resolution, high-channel feature
- FPN/U-Net이 기대하는 `1/4, 1/8, 1/16, 1/32` scale hierarchy
- 고해상도 입력에서도 감당 가능한 attention memory

Dense task 입력은 짧은 변이 `800` 이상일 수 있다. Patch 수가 늘면 global attention matrix는 `N×N`으로 커진다. 단일 low-resolution feature map에서 deconvolution으로 pyramid를 사후 구성할 수도 있지만, hierarchy를 backbone 안에서 학습하는 것과 같지 않고 추가 비용이 든다.

### Local window만 쓰면 생기는 단절

Feature map을 겹치지 않는 window로 나누면 같은 layer의 계산은 효율적이다. 그러나 항상 같은 경계로 partition하면 window A의 token과 옆 window B의 token은 아무리 layer를 쌓아도 직접 정보를 교환하지 못한다.

Sliding-window attention은 각 query 주위의 local neighborhood를 사용해 경계 단절을 피하지만 query마다 key set이 달라 irregular memory access가 발생한다. 논문은 **모든 query가 window 안에서 같은 key set을 공유하는 batched attention**을 유지하면서 layer마다 window 경계를 바꾸는 방식을 택한다.

## 전체 architecture

### Patch partition과 linear embedding

입력을

```math
X\in\mathbb{R}^{B\times H\times W\times3}
```

라 하자. `4×4` non-overlapping patch를 flatten하면 각 token은 `4×4×3=48`차원이다.

```math
X_p\in\mathbb{R}^{B\times(H/4)\times(W/4)\times48}
```

공유 linear layer로 첫 stage channel `C`에 투영한다.

```math
Z_1\in\mathbb{R}^{B\times(H/4)\times(W/4)\times C}
```

ViT의 `16×16` patch보다 훨씬 작은 `4×4` patch로 시작하므로 dense detail을 더 오래 보존한다. 대신 high-resolution token 수가 많아져 global attention은 불가능해지고 local window가 필수가 된다.

### Patch merging

`2×2` 이웃 token 네 개를 channel 방향으로 concatenate한다.

```math
\mathbb{R}^{B\times H_s\times W_s\times C_s}
\rightarrow
\mathbb{R}^{B\times H_s/2\times W_s/2\times4C_s}
```

그 뒤 LayerNorm과 linear projection으로 `4C_s -> 2C_s`로 줄인다.

```math
\mathbb{R}^{B\times H_s/2\times W_s/2\times2C_s}
```

따라서 spatial token 수는 1/4, channel은 2배가 된다. Activation 원소 수는 stage output 기준으로 절반이 된다.

```math
\frac{H_sW_s}{4}\times2C_s
=\frac{1}{2}H_sW_sC_s
```

이는 CNN의 stride-2 convolution과 비슷한 hierarchy를 만들지만, local `2×2` token을 명시적으로 concat한 뒤 projection한다는 차이가 있다.

### Swin-T stage schedule

`224×224` 입력의 Swin-T는 다음 shape를 갖는다.

| Stage | Output stride | Spatial | Channel | Heads | Blocks | Window |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | `56×56` | 96 | 3 | 2 | `7×7` |
| 2 | 8 | `28×28` | 192 | 6 | 2 | `7×7` |
| 3 | 16 | `14×14` | 384 | 12 | 6 | `7×7` |
| 4 | 32 | `7×7` | 768 | 24 | 2 | `7×7` |

Head dimension은 모든 stage에서 `32`로 고정된다. MLP expansion ratio는 4다. Classification은 stage 4 feature의 global average pooling 뒤 linear classifier를 사용하며 class token을 쓰지 않는다.

### Model variants

| Model | Base `C` | Blocks per stage | 특징 |
| --- | ---: | --- | --- |
| Swin-T | 96 | `{2,2,6,2}` | 약 ResNet-50/DeiT-S 규모 |
| Swin-S | 96 | `{2,2,18,2}` | stage 3 depth 증가 |
| Swin-B | 128 | `{2,2,18,2}` | ViT-B/DeiT-B와 유사 규모 |
| Swin-L | 192 | `{2,2,18,2}` | Base의 약 2배 규모 |

## Window multi-head self-attention

### Global MSA와 W-MSA 복잡도

Feature map이 `h×w` tokens, channel이 `C`, window 한 변이 `M`이라 하자. 논문은 softmax 비용을 생략하고 다음 식을 제시한다.

```math
\Omega(\mathrm{MSA})
=4hwC^2+2(hw)^2C
```

```math
\Omega(\text{W-MSA})
=4hwC^2+2M^2hwC
```

첫 항은 Q/K/V와 output projection이다. 두 번째 항은 attention logits `QK^T`와 weighted value `AV`다. Global attention에서는 `hw`의 제곱이지만, window size `M`이 고정되면 W-MSA의 두 번째 항은 image token 수 `hw`에 선형이다.

"Linear complexity"는 model width와 window가 고정될 때 input image 면적에 대해 선형이라는 뜻이다. 모든 연산이 `O(N)`이거나 device latency가 해상도에 정확히 비례한다는 뜻은 아니다. Window partition, padding, mask, kernel launch도 존재한다.

### Window 내부 attention

Window 하나에는 `M²` token이 있다. 한 head의 계산은

```math
\mathrm{Attention}(Q,K,V)
=\mathrm{softmax}
\left(\frac{QK^\top}{\sqrt d}+B+M_{attn}\right)V
```

이다.

- `Q,K,V∈R^{M²×d}`
- `B∈R^{M²×M²}`: relative position bias
- `M_attn`: shifted-window에서 서로 인접하지 않은 cyclic sub-region을 막는 mask

Window 간 attention을 계산하지 않으므로 attention matrix는 block diagonal 구조와 같다.

## Shifted window의 작동 원리

### 두 block을 한 쌍으로 본다

첫 block은 정렬된 window partition인 W-MSA를 사용한다. 다음 block은 window를 `floor(M/2)`만큼 이동한 SW-MSA를 사용한다. 기본 `M=7`에서는 shift가 `(3,3)`이다.

```math
\begin{aligned}
\hat z^l
&=\text{W-MSA}(\mathrm{LN}(z^{l-1}))+z^{l-1},\\
z^l
&=\mathrm{MLP}(\mathrm{LN}(\hat z^l))+\hat z^l,\\
\hat z^{l+1}
&=\text{SW-MSA}(\mathrm{LN}(z^l))+z^l,\\
z^{l+1}
&=\mathrm{MLP}(\mathrm{LN}(\hat z^{l+1}))+\hat z^{l+1}.
\end{aligned}
```

두 번째 block의 window는 첫 block의 여러 window 일부를 함께 포함한다. 따라서 이전 window 경계를 넘어 정보가 전달된다. 여러 block을 지나면 receptive field가 점진적으로 확장된다. 한 block에서 image 전체를 보는 global attention과는 다르다.

### Naive shifted partition의 비용

예를 들어 regular partition이 `2×2` window라면 단순히 경계를 옮길 때 작고 불완전한 window가 생겨 `3×3`개가 될 수 있다. 모두 `M×M`으로 padding하면 window 수가 `4 -> 9`, 즉 `2.25배`가 된다.

### Cyclic shift와 mask

효율적 구현은 다음 순서다.

```text
1. feature map을 왼쪽 위로 cyclic roll
2. regular M×M window partition
3. 같은 batch의 각 window에서 MSA
4. 원래 image에서 떨어져 있던 sub-region 사이 score에 -∞ mask
5. window reverse
6. cyclic roll을 반대로 복원
```

Cyclic roll만 하고 mask를 빼면 image의 오른쪽 끝과 왼쪽 끝이 이웃인 torus topology가 된다. Mask는 roll 때문에 우연히 같은 window에 들어온 비인접 region을 분리한다. 이 mask는 causal mask가 아니며, 시간적 미래를 막는 목적도 아니다.

## Relative position bias

`M×M` window에서 두 token의 상대 offset은 각 축마다 `[-M+1,M-1]` 범위다. 따라서 head마다 학습할 table은

```math
\hat B\in\mathbb{R}^{(2M-1)\times(2M-1)}
```

이면 충분하다. `M=7`이면 `13×13=169`개 값이다. Token pair의 `(Δrow,Δcol)`로 table을 lookup해 실제 `49×49` bias matrix `B`를 만든다.

이 방식은 위치마다 별도 absolute vector를 더하는 것이 아니라 **attention logit에 상대 offset별 scalar 선호**를 더한다. 같은 상대 offset은 window 위치가 달라도 bias를 공유하므로 dense vision에 유리한 translation-related prior를 준다.

Window size를 바꿔 fine-tuning할 때는 learned table을 bicubic interpolation할 수 있다. 그러나 이는 다른 window size에서 정확한 geometric equivariance를 보장하는 것은 아니다.

## Batch=1 tensor shape와 activation memory

### Swin-T stage 1 예시

`B=1`, `224×224`, stage 1 channel `C=96`, heads 3, `M=7`이다.

```text
image                         [1, 224, 224, 3]
4×4 patch partition           [1, 56, 56, 48]
linear embedding              [1, 56, 56, 96]
window partition              [64, 7, 7, 96]
flattened windows             [64, 49, 96]
Q/K/V per head                [64, 3, 49, 32]
attention logits              [64, 3, 49, 49]
window outputs                [64, 49, 96]
window reverse                [1, 56, 56, 96]
```

`56/7=8`이므로 window 수는 `8×8=64`다.

### Attention map memory - 리뷰어 계산값

Stage 1 window attention의 score 원소 수는

```math
64\times3\times49\times49
=460{,}992
```

이다. FP16 materialization은

```math
460{,}992\times2
=921{,}984\ \text{bytes}
\approx900.4\ \text{KiB}
```

다. 같은 `3136` token에 global attention을 적용하면

```math
3\times3136^2
=29{,}503{,}488\ \text{elements}
```

로 FP16 약 `56.3 MiB`다. Window attention은 score 원소 수를 정확히 `64배` 줄인다. 이 값은 한 attention tensor의 논리적 크기이며 실제 runtime peak는 아니다.

Stage별 attention score 원소 수는 다음과 같다.

| Stage | Windows | Heads | Score elements | FP16 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 64 | 3 | 460,992 | 900.4 KiB |
| 2 | 16 | 6 | 230,496 | 450.2 KiB |
| 3 | 4 | 12 | 230,496 | 450.2 KiB |
| 4 | 1 | 24 | 57,624 | 112.5 KiB |

Stage 3는 block이 6개라 tensor 하나의 크기는 작아도 누적 계산 비중이 크다. Training에서는 backward를 위해 여러 activation을 저장하므로 inference 표보다 훨씬 큰 memory가 필요하다.

### Stage feature memory - 리뷰어 계산값

Swin-T stage output 하나의 FP16 logical size는 다음과 같다.

| Stage | Shape | Elements | FP16 |
| --- | --- | ---: | ---: |
| 1 | `1×56×56×96` | 301,056 | 588.0 KiB |
| 2 | `1×28×28×192` | 150,528 | 294.0 KiB |
| 3 | `1×14×14×384` | 75,264 | 147.0 KiB |
| 4 | `1×7×7×768` | 37,632 | 73.5 KiB |

Hierarchy 덕분에 stage output 원소 수가 절반씩 줄어든다. 하지만 patch merging의 concat buffer, QKV, FFN `4C`, residual, window copy가 동시에 live할 수 있으므로 이 표의 최대 `588 KiB`가 end-to-end peak는 아니다. 논문은 mobile peak memory를 보고하지 않는다.

## 계산량과 parameter reasoning

### Stage 1 한 W-MSA - 리뷰어 계산값

논문 식에 `h=w=56`, `C=96`, `M=7`을 넣으면

```math
4hwC^2=115{,}605{,}504
```

```math
2M^2hwC=29{,}503{,}488
```

이므로 약 `145.1M`이다. Global MSA라면 attention 항만 `1,888,223,232`로 전체 약 `2.00G`가 된다. Stage 1 한 module에서 이론 복잡도가 약 `13.8배` 차이 난다.

### Patch merging parameter

입력 channel이 `C`일 때 `4C -> 2C` projection weight는

```math
4C\times2C=8C^2
```

개다. Swin-T 첫 patch merging은 `C=96`이므로 `73,728`개 weight다. Spatial downsampling이 공짜인 것은 아니며 concat/layout과 linear projection을 수행한다.

### Model-level paper-reported values

| Model | Params | FLOPs at 224 | V100 throughput | ImageNet-1K top-1 |
| --- | ---: | ---: | ---: | ---: |
| Swin-T | 29M | 4.5G | 755.2 image/s | 81.3 |
| Swin-S | 50M | 8.7G | 436.9 image/s | 83.0 |
| Swin-B | 88M | 15.4G | 278.1 image/s | 83.5 |
| Swin-B at 384 | 88M | 47.0G | 84.7 image/s | 84.5 |

이 table의 FLOPs와 parameter는 논문 보고값이다. 위 stage별 activation과 W-MSA 계산은 리뷰어 계산값이다. Throughput은 [68] repository 구현과 V100 GPU 기준이며 batch=1 또는 모바일 latency가 아니다.

## 핵심 구현 pseudocode

```python
def shifted_window_attention(x, qkv, proj, rel_bias, window=7, shift=3):
    # x: [B, H, W, C], H/W는 window 배수가 되도록 우하단 padding
    b, h, w, c = x.shape

    if shift > 0:
        x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))

    # [B*nW, M*M, C]
    xw = window_partition(x, window).reshape(-1, window * window, c)

    q, k, v = split_heads(qkv(xw))
    score = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    score = score + rel_bias                 # [heads, M*M, M*M]

    if shift > 0:
        score = score + precomputed_region_mask(h, w, window, shift)

    aw = score.softmax(dim=-1) @ v
    yw = proj(merge_heads(aw))
    y = window_reverse(yw, window, h, w)

    if shift > 0:
        y = torch.roll(y, shifts=(shift, shift), dims=(1, 2))
    return y
```

Fixed resolution에서는 region mask와 relative-position index를 초기화 때 precompute해야 한다. 매 inference마다 Python indexing과 mask 생성을 수행하면 architecture의 효율을 망칠 수 있다.

## 실험 결과

### ImageNet classification

Regular ImageNet-1K training에서 Swin-T는 `29M`, `4.5G`, `755.2 image/s`, `81.3%`다. 유사 계산량의 DeiT-S는 `22M`, `4.6G`, `940.4 image/s`, `79.8%`다. Swin-T가 `+1.5%p` 높지만 classification throughput은 DeiT-S가 더 빠르다. Swin의 강점은 dense task까지 포함한 general-purpose backbone이라는 점이다.

ImageNet-22K pretraining 결과는 다음과 같다.

| Model | Size | Params | FLOPs | V100 throughput | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViT-B/16 | 384 | 86M | 55.4G | 85.9 | 84.0 |
| ViT-L/16 | 384 | 307M | 190.7G | 27.3 | 85.2 |
| Swin-B | 224 | 88M | 15.4G | 278.1 | 85.2 |
| Swin-B | 384 | 88M | 47.0G | 84.7 | 86.4 |
| Swin-L | 384 | 197M | 103.9G | 42.1 | **87.3** |

Pretraining data가 다른 regular ImageNet-1K 결과와 섞이지 않도록 구분해야 한다.

### COCO detection과 instance segmentation

Cascade Mask R-CNN에서 backbone만 비교한 주요 결과다.

| Backbone | Box AP | Mask AP | Params | FLOPs | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeiT-S + deconv pyramid | 48.0 | 41.4 | 80M | 889G | 10.4 |
| ResNet-50 | 46.3 | 40.1 | 82M | 739G | **18.0** |
| Swin-T | **50.5** | **43.7** | 86M | 745G | 15.3 |
| ResNeXt-101-32×4d | 48.1 | 41.6 | 101M | 819G | 12.8 |
| Swin-S | **51.8** | **44.7** | 107M | 838G | 12.0 |
| ResNeXt-101-64×4d | 48.3 | 41.7 | 140M | 972G | 10.4 |
| Swin-B | **51.9** | **45.0** | 145M | 982G | 11.6 |

Swin-T는 DeiT-S보다 `+2.5 box AP`, `+2.3 mask AP`이며 FPS도 `10.4 -> 15.3`이다. DeiT는 별도 deconvolution으로 pyramid를 만들어야 한다. ResNet-50과 비교하면 Swin-T 정확도는 높지만 FPS는 낮으므로 모든 metric에서 일방적으로 우월한 것은 아니다.

최고 system-level 결과는 ImageNet-22K pretraining, HTC++, 강한 multi-scale training, soft-NMS 등을 포함한다. Multi-scale test의 Swin-L은 COCO test-dev `58.7 box AP`, `51.1 mask AP`다. 단순 backbone ablation과 같은 training/inference 조건이 아니다.

### ADE20K segmentation

| Method/backbone | mIoU | Params | FLOPs | FPS |
| --- | ---: | ---: | ---: | ---: |
| UperNet DeiT-S + deconv | 44.0 | 52M | 1099G | 16.2 |
| UperNet Swin-T | 46.1 | 60M | 945G | **18.5** |
| UperNet Swin-S | 49.3 | 81M | 1038G | 15.2 |
| UperNet Swin-B, IN-22K | 51.6 | 121M | 1841G | 8.7 |
| UperNet Swin-L, IN-22K | **53.5** | 234M | 3230G | 6.2 |

Swin-T/S는 `512×512`, Swin-B/L은 ImageNet-22K pretraining과 `640×640` training을 사용한다. Inference는 `[0.5,0.75,1.0,1.25,1.5,1.75]` multi-scale test다. 최고 mIoU와 작은-model FPS를 같은 조건으로 오해하면 안 된다.

## 핵심 ablation

### Shifted window - 정확한 논문 값

모두 Swin-T다.

| 설정 | ImageNet top-1 | COCO box AP | COCO mask AP | ADE20K mIoU |
| --- | ---: | ---: | ---: | ---: |
| Shift 없음 | 80.2 | 47.7 | 41.5 | 43.3 |
| Shifted windows | **81.3** | **50.5** | **43.7** | **46.1** |
| 향상 | `+1.1%p` | `+2.8` | `+2.2` | `+2.8` |

Classification보다 dense task의 이득이 더 크다. Cross-window information flow가 localization과 pixel labeling에 특히 중요하다는 해석과 일치한다.

### Position encoding - 정확한 논문 값

| Position 방식 | Top-1 | Box AP | Mask AP | mIoU |
| --- | ---: | ---: | ---: | ---: |
| 없음 | 80.1 | 49.2 | 42.6 | 43.8 |
| Absolute | 80.5 | 49.0 | 42.4 | 43.2 |
| Absolute + relative | 81.3 | 50.2 | 43.4 | 44.0 |
| Relative bias, content term 없음 | 79.3 | 48.2 | 41.9 | 44.1 |
| Relative bias | **81.3** | **50.5** | **43.7** | **46.1** |

Absolute position은 no-position보다 ImageNet을 `+0.4%p` 높이지만 COCO와 ADE20K는 오히려 낮춘다. Relative bias가 세 task에서 가장 일관적이다. `rel. pos. w/o app.`은 scaled dot-product content term을 빼고 bias만 쓴 설정이므로, relative bias만으로 attention을 대체할 수 없다는 것도 보여준다.

### Cyclic implementation latency - 정확한 논문 값

V100에서 Swin-T/S/B 전체 architecture FPS는 다음과 같다.

| Attention 구현 | Swin-T | Swin-S | Swin-B |
| --- | ---: | ---: | ---: |
| Window, shift 없음 | 770 | 444 | 280 |
| Shift + padding | 670 | 371 | 236 |
| Shift + cyclic mask | **755** | **437** | **278** |
| Sliding window custom kernel | 488 | 283 | 187 |
| Performer | 638 | 370 | 241 |

Cyclic shift는 naive padding보다 T/S/B에서 각각 약 `13%/18%/18%` 빠르며, shift 없는 regular window와 거의 같은 throughput을 유지한다. Algorithmic idea뿐 아니라 batch-friendly implementation이 논문의 중요한 기여다.

### Resolution ablation

Swin-T는 `224: 81.3%, 755.2 image/s`에서 `384: 82.2%, 219.5 image/s`가 된다. 정확도는 `+0.9%p`지만 throughput은 약 `70.9%` 감소한다. Swin-B는 `83.3 -> 84.5%`, `278.1 -> 84.7 image/s`다. Window attention이 global attention의 폭발을 막더라도 high-resolution 비용이 사라지는 것은 아니다.

## 장점과 핵심 기여

- Transformer backbone에 CNN-compatible multi-scale hierarchy를 만들었다.
- 고정 `7×7` window로 attention 비용을 image 면적에 선형으로 제한했다.
- Shifted partition으로 window 경계 단절을 해소했다.
- Cyclic shift와 mask를 통해 irregular window padding 없이 batched kernel을 유지했다.
- Relative position bias로 작은 parameter로 2D spatial prior를 주었다.
- Classification, detection, instance segmentation, semantic segmentation을 한 backbone family로 검증했다.
- FLOPs뿐 아니라 attention 구현별 V100 stage latency와 architecture FPS를 비교했다.

## 한계와 비판적 관점

### 1. "Linear"이어도 가볍다는 뜻은 아니다

Swin-T는 29M parameter, 4.5G FLOPs이고 LayerNorm/softmax/window transform이 있다. MobileNet급 상시 camera backbone과는 비용 범주가 다르다.

### 2. Global context가 즉시 생기지 않는다

한 block은 `7×7`만 본다. Shift와 hierarchy를 여러 번 거쳐야 먼 token이 연결된다. 매우 긴-range relation이 중요한 task에서는 global block이나 별도 neck가 필요할 수 있다.

### 3. Window plumbing이 hardware에 따라 비싸다

V100의 `roll`, partition, mask, reverse가 mobile NPU에서도 효율적이라는 보장은 없다. Gather/scatter 또는 transpose가 CPU fallback되면 FLOPs 이점이 사라질 수 있다.

### 4. 모바일 실측 부재

논문은 V100 throughput과 detection/segmentation FPS를 보고하지만 mobile CPU/DSP/NPU의 `batch=1` latency, p95, peak memory, 전력은 보고하지 않는다.

### 5. 최고 성능은 서로 다른 recipe를 포함한다

Swin-L 최고치는 ImageNet-22K pretraining, 큰 resolution, HTC++/multi-scale test 등을 사용한다. Swin-T regular ImageNet ablation과 직접적인 architecture-only 비교가 아니다.

### 6. Patch merging의 detail loss

매 stage에서 spatial resolution을 절반으로 줄인다. Hierarchy는 효율적이지만 작은 object와 boundary 정보가 얼마나 손실되는지는 AP_S나 boundary metric 중심으로 충분히 분석하지 않는다.

### 7. Fixed window의 경계와 padding

Input 크기가 window 배수가 아니면 bottom-right padding과 mask가 필요하다. Aspect ratio와 dynamic resolution에서 window occupancy가 달라지고 wasted compute가 생길 수 있다.

## 자주 헷갈리는 지점

### Shifted window는 sliding window인가

아니다. Sliding window는 query마다 주변 key set이 달라진다. Swin은 한 layer 안에서는 non-overlapping fixed window를 쓰고, 다음 layer에서 전체 partition offset을 바꾼다.

### Cyclic shift가 wrap-around attention을 허용하는가

아니다. Roll로 양 끝이 붙지만 mask가 원래 떨어진 region 사이 score를 `-∞`로 막는다.

### SW-MSA mask는 causal mask인가

아니다. 미래 token을 막는 것이 아니라 cyclic batch window 안에서 원래 인접하지 않았던 spatial region을 분리한다.

### Relative position bias는 Q/K에 더하는가

아니다. `QK^T/√d` attention logit에 scalar bias를 더한다.

### Swin은 완전한 translation equivariance를 보장하는가

아니다. Relative bias와 shared windows가 유리한 prior를 주지만 patch grid, window boundary, shift, padding 때문에 exact equivariance는 아니다.

### Stage 4의 `7×7`은 global attention인가

224 입력의 Swin-T에서는 stage 4 feature가 정확히 `7×7`이라 하나의 window가 전체 stage를 본다. 더 큰 입력에서는 stage 4에도 여러 window가 생긴다.

## 온디바이스 구현 관점

### 실행 graph에서 확인할 병목

```text
LayerNorm
 -> QKV projection
 -> reshape/transpose
 -> window partition 또는 view
 -> batched QK matmul
 -> relative bias gather/add
 -> mask add + softmax
 -> AV matmul
 -> window reverse + roll
 -> projection + residual
```

Window를 단순 view로 표현할 수 있는 layout인지, 실제 memory copy가 필요한지가 중요하다. NHWC/NCHW conversion과 small batched GEMM utilization도 확인해야 한다.

### Fixed-shape 최적화

On-device라면 input resolution과 aspect ratio를 고정하고 다음을 compile-time constant로 만드는 것이 유리하다.

- attention mask
- relative-position index
- window 개수
- padding 크기
- roll offset

Dynamic shape를 허용하면 mask 생성과 graph recompilation이 생길 수 있다.

### Activation memory

Hierarchy는 later-stage feature memory를 빠르게 줄이지만 peak는 stage 1의 FFN `4C`, QKV, attention score에서 발생할 가능성이 높다. Fused window attention이 score를 전부 materialize하지 않는지, FFN activation을 tile하는지 확인한다. 논문 값 대신 target runtime profiler의 allocator peak를 사용해야 한다.

### Quantization

Linear weight는 INT8로 잘 양자화할 수 있지만 LayerNorm, softmax, relative bias, mask의 precision 경계가 까다롭다. Mixed precision 실험에서는 다음을 분리한다.

- QKV/MLP INT8, softmax FP16
- LayerNorm FP16 유지 여부
- Relative bias와 attention scale의 calibration
- Residual add의 activation scale 정렬

### MobileNetV2와의 공정 비교

동일 `224`, `batch=1`, precision에서 parameter와 MAC만 비교하지 않는다. MobileNetV2는 depthwise/pointwise conv 중심이고 Swin은 LayerNorm, batched matmul, softmax, window data movement 중심이다. CPU와 NPU에서 순위가 다를 수 있다.

## 재현 계획

### 1단계: shape와 mask 단위 테스트

1. `8×8`, window 4의 논문 toy example을 구현한다.
2. Regular partition이 4 window, naive shifted partition이 9 window가 되는지 확인한다.
3. Cyclic 방식은 4 batched window를 유지하는지 확인한다.
4. Mask가 wrap-around pair를 정확히 차단하는지 attention matrix를 시각화한다.
5. Shift 후 reverse했을 때 token 위치가 원상복구되는지 test한다.

### 2단계: Swin-T backbone

1. Stage output이 `56²×96, 28²×192, 14²×384, 7²×768`인지 assert한다.
2. Block 수가 `{2,2,6,2}`, head가 `{3,6,12,24}`인지 확인한다.
3. Parameter `29M`, FLOPs `4.5G` 근처인지 동일 convention으로 검증한다.
4. Official checkpoint와 preprocessing으로 ImageNet top-1을 평가한다.
5. No-shift와 relative-bias 제거 ablation을 같은 recipe로 재현한다.

### 3단계: dense task

FPN/UperNet에 네 stage feature를 직접 연결한다. DeiT baseline은 deconvolution pyramid 비용을 포함한다. COCO에서는 AP/AP_S/AP_M/AP_L, ADE20K에서는 mIoU와 boundary metric을 함께 기록한다.

### 4단계: on-device benchmark

| 축 | 설정 |
| --- | --- |
| Resolution | 224, 256, 320, 384 |
| Precision | FP16, INT8 mixed |
| Attention | no shift, cyclic shift, fused kernel |
| Backend | CPU, GPU, NPU |
| Metric | top-1/mAP, p50/p95, peak memory, energy |

Profiler에서 `roll/partition/reverse`, softmax, LayerNorm, matmul 시간을 별도 집계한다. 10분 이상 지속 실행해 thermal throttling도 본다.

## 구현 체크리스트

- [ ] Patch size가 4이고 stage 1 output stride가 4인가?
- [ ] Patch merging이 `2×2 concat -> 4C -> 2C`인가?
- [ ] Window size가 기본 7인가?
- [ ] W-MSA와 SW-MSA가 block마다 교대하는가?
- [ ] Shift size가 `floor(M/2)=3`인가?
- [ ] Input이 window 배수가 아닐 때 bottom-right padding하는가?
- [ ] Cyclic wrap-around region을 mask로 차단하는가?
- [ ] Relative bias table이 head마다 `(2M-1)^2`인가?
- [ ] Relative-position index와 mask를 cache하는가?
- [ ] Stage별 head dimension이 32인가?
- [ ] Classification에 class token 대신 global average pooling을 쓰는가?
- [ ] Paper FLOPs와 profiler FLOPs convention을 구분하는가?
- [ ] V100 throughput을 모바일 latency로 오인하지 않는가?

## 로드맵에서의 위치와 후속 연결

ViT/DeiT가 image를 flat token sequence로 처리했다면 Swin은 Transformer를 detection과 segmentation의 **general-purpose hierarchical backbone**으로 바꿨다. 이후 FPN, Mask R-CNN, SegFormer, Mask2Former를 읽을 때 multi-scale feature가 어디서 생성되고 어느 resolution에서 attention하는지 보는 기준이 된다.

효율적 backbone 관점에서는 다음 연결이 중요하다.

- MobileNetV2: convolutional hierarchy와 thin bottleneck으로 activation을 줄인다.
- DeiT: flat global attention을 training recipe로 강화한다.
- Swin: local window와 patch merging으로 high-resolution complexity를 제어한다.
- EfficientViT: global receptive field를 linear attention으로 유지하면서 window partition과 softmax를 피하려 한다.
- MobileNetV4: mobile-friendly convolution과 attention을 device-aware search로 결합한다.

따라서 Swin을 온디바이스 후보로 평가할 때는 "quadratic이 linear가 되었다"에서 멈추지 말고, **window data movement와 early high-resolution activation이 target accelerator에서 실제로 싸게 실행되는가**를 측정해야 한다.

## 최종 평가

Swin Transformer는 vision Transformer가 classification 전용 flat encoder에서 벗어나 dense recognition의 표준 backbone 역할을 할 수 있음을 설득력 있게 보여줬다. Hierarchy, locality, relative position이라는 vision prior를 다시 도입하면서도 attention의 content-dependent interaction을 유지했다.

가장 뛰어난 부분은 shifted window의 아이디어와 cyclic-mask 구현을 함께 제시한 점이다. 같은 이론 복잡도라도 irregular sliding window와 regular batched window의 실제 V100 latency가 크게 다름을 보였다. 반면 모바일 실측과 peak memory는 없고, roll/partition/mask가 NPU에서 효율적이라는 보장도 없다. 이 논문은 **algorithmic complexity와 hardware-friendly dataflow를 동시에 설계해야 한다**는 좋은 사례이자, 이후 효율적 vision backbone을 평가하는 강한 기준선이다.

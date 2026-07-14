# 39. EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction

## 논문 정보

- 제목: **EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction**
- 저자: Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
- 소속: MIT, Zhejiang University, Tsinghua University, MIT-IBM Watson AI Lab
- 발표: ICCV 2023
- arXiv: [https://arxiv.org/abs/2205.14756](https://arxiv.org/abs/2205.14756)
- PDF 기준 버전: arXiv v6, 2024-02-06
- 코드: [https://github.com/mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit)
- 원본 파일: `39_EfficientViT_Multi_Scale_Linear_Attention.pdf`
- 핵심 키워드: ReLU linear attention, multi-scale token aggregation, high-resolution dense prediction, hardware efficiency, semantic segmentation, EfficientViT-SAM

## 한눈에 보는 요약

이 EfficientViT는 고해상도 dense prediction을 위해 다음 세 가지를 결합한다.

1. Softmax attention 대신 **ReLU kernel linear attention**을 사용해 global receptive field를 `O(N)` memory/compute 형태로 계산한다.
2. Linear attention이 sharp local relation을 잘 표현하지 못하는 약점을 depthwise convolution과 multi-scale Q/K/V aggregation으로 보완한다.
3. Hierarchical CNN-style backbone과 단순한 additive pyramid head를 사용해 실제 mobile CPU, edge GPU, cloud GPU에서 latency를 측정한다.

```text
Q/K/V linear projection
 -> raw Q/K/V branch
 -> nearby-token aggregation branch(es): small DWConv + grouped 1×1 conv
 -> 각 scale에서 ReLU linear attention
 -> concatenate
 -> final linear projection
 -> FFN + depthwise convolution
```

핵심 수학은 `N×N` attention matrix를 만들지 않고 결합 순서를 바꾸는 것이다.

```math
\operatorname{ReLU}(Q)
\left(\operatorname{ReLU}(K)^TV\right)
```

`K^TV`를 먼저 계산하면 token 수 `N` 대신 head dimension `d`에 대한 `d×d` global summary를 만든다. 따라서 high-resolution에서 softmax attention보다 계산과 activation memory가 크게 줄어든다.

Cityscapes `1024×2048`에서 EfficientViT-B0는 `0.7M` parameters, `4.4G` MACs, `75.7 mIoU`, Jetson AGX Orin `9.9 ms`를 보고한다. EfficientViT-B3는 `83.0 mIoU`, `179G` MACs, Orin `81.8 ms`다. ImageNet에서는 EfficientViT-L2 `384×384`가 `86.0%` top-1을 달성한다. EfficientViT-SAM-XL1은 box-prompted COCO `47.8 mAP`로 SAM-ViT-H의 `46.5`를 넘으면서 A100 throughput이 `182` 대 `11 image/s`다.

이 논문의 가치는 FLOPs 표에만 있지 않다. Batch 1, data transfer 포함, TFLite/TensorRT 조건을 명시하고 여러 hardware에서 측정해 **softmax 제거와 regular operator 선택이 실제 latency로 이어지는지**를 검증했다.

## 같은 이름의 EfficientViT와 구분

`EfficientViT`라는 이름은 서로 다른 논문에 사용된다. 이 리뷰의 대상은 arXiv `2205.14756`, Han Cai/Song Han 연구팀의 **Multi-Scale Linear Attention for High-Resolution Dense Prediction**이다.

다른 계열인 **EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention**은 별개의 architecture다. Model 이름만 보고 block이나 성능표를 섞으면 안 된다. 이 리뷰의 EfficientViT-B/L 및 EfficientViT-SAM은 모두 multi-scale ReLU linear attention 계열을 뜻한다.

## 연구 배경과 문제의식

### High-resolution dense task의 두 요구

Semantic segmentation, super-resolution, prompt segmentation은 pixel-level output을 생성한다. Classification보다 훨씬 큰 feature map을 오래 유지해야 하며 다음 두 능력이 모두 필요하다.

- **Global receptive field**: 멀리 떨어진 영역과 scene context를 연결한다.
- **Multi-scale/local detail**: boundary, texture, small object를 포착한다.

SegFormer의 softmax attention은 global context에 강하지만 token 수에 대해 quadratic이다. SegNeXt의 large-kernel multi-branch convolution은 local/multi-scale feature에 강하지만 kernel이 최대 21까지 커져 일반 hardware에서 최적화가 어렵다. 복잡한 topology도 graph scheduling과 memory access에 불리할 수 있다.

논문은 이 두 능력을 다음처럼 역할 분담한다.

```text
ReLU linear attention  -> global context
small-kernel conv      -> local detail와 multi-scale context
```

### 이론 FLOPs가 실제 latency가 되려면

연산량이 적어도 softmax, large-kernel convolution, irregular data movement가 target runtime에서 비효율적이면 실제 속도는 느릴 수 있다. 저자들은 다음처럼 비교적 범용적인 operator를 의도적으로 선택한다.

- linear/`1×1` convolution
- small-kernel depthwise convolution
- grouped `1×1` convolution
- ReLU
- matrix multiplication
- bilinear/bicubic upsampling
- elementwise addition

다만 group convolution과 depthwise convolution도 모든 NPU에서 동일하게 빠른 것은 아니므로, "hardware-efficient"는 논문에서 측정한 platform 범위 안에서 검증된 주장으로 읽어야 한다.

## Generalized attention에서 ReLU linear attention까지

### Generalized normalized attention

Input token을

```math
x\in\mathbb{R}^{N\times f}
```

라 하고

```math
Q=xW_Q,\quad K=xW_K,\quad V=xW_V,
\qquad
W_Q,W_K,W_V\in\mathbb{R}^{f\times d}
```

라 하자. Similarity kernel `Sim`을 쓰는 normalized attention은 query `i`에 대해

```math
O_i=
\frac{
\sum_{j=1}^{N}\operatorname{Sim}(Q_i,K_j)V_j
}{
\sum_{j=1}^{N}\operatorname{Sim}(Q_i,K_j)
}
```

이다. Softmax attention은

```math
\operatorname{Sim}(Q,K)
=\exp\left(\frac{QK^T}{\sqrt d}\right)
```

를 사용한다.

### ReLU kernel

EfficientViT는 similarity를 다음처럼 바꾼다.

```math
\operatorname{Sim}(Q,K)
=\operatorname{ReLU}(Q)\operatorname{ReLU}(K)^T
```

`Q'=ReLU(Q)`, `K'=ReLU(K)`라 쓰면

```math
O_i=
\frac{
\sum_j(Q'_iK_j'^T)V_j
}{
Q'_i\sum_jK_j'^T
}
```

이다. ReLU 때문에 similarity는 non-negative가 되어 normalized weighted average로 해석할 수 있다.

### Associative trick

분자의 곱셈 순서를 바꾼다.

```math
\sum_j(Q'_iK_j'^T)V_j
=Q'_i\left(\sum_jK_j'^TV_j\right)
```

전체 matrix 형태는

```math
S=K'^TV\in\mathbb{R}^{d\times d},
\qquad
z=K'^T\mathbf{1}\in\mathbb{R}^{d}
```

```math
O=\frac{Q'S}{Q'z}
```

이다. 분모 `Q'z`는 `N×1`이며 각 output row에 broadcast한다. 실제 구현에서는 분모가 0 또는 매우 작을 수 있으므로 작은 `ε`를 더하는 numerical safeguard가 필요하지만 논문 식에는 별도로 표시되지 않는다.

### 복잡도

한 head에서 핵심 계산은 다음과 같다.

- `K'^T V`: `O(Nd²)`
- `Q'S`: `O(Nd²)`
- `K'` sum과 denominator: `O(Nd)`
- Activation: Q/K/V의 `O(Nd)`와 accumulator `O(d²)`

Softmax attention의 `QK^T`와 `AV`는 `O(N²d)`, score memory는 `O(N²)`다. `d`가 고정이고 `N`이 큰 dense task에서 차이가 커진다.

중요한 조건은 **ReLU kernel이 separable feature map**이라는 점이다. 일반 softmax `exp(QK^T)`는 같은 방식으로 정확히 분해할 수 없다. 따라서 결합 순서만 바꾼 동일 softmax가 아니라 similarity function 자체를 바꾼 근사/대체 attention이다.

## ReLU linear attention의 한계

Softmax의 exponential은 큰 logit 차이를 매우 sharp한 distribution으로 증폭한다. ReLU dot-product는 상대적으로 평평한 attention을 만들기 쉽다. 논문의 Figure 3에서도 softmax attention은 object의 국소 영역에 강하게 집중하지만 ReLU linear attention은 더 diffuse하다.

이로 인해 linear attention 하나만 사용하면 다음 문제가 생길 수 있다.

- edge와 boundary 같은 local pattern에 약함
- 작은 object에 필요한 sharp selection이 어려움
- 하나의 `d×d` global summary가 token-specific pair relation을 압축함
- head dimension이 작을 때 effective interaction rank가 제한될 수 있음

EfficientViT는 이를 부정하지 않고 convolution을 명시적으로 추가한다. 따라서 이 architecture의 성공을 "linear attention만으로 softmax와 완전히 같은 표현력을 얻었다"고 해석하면 안 된다.

## Multi-scale linear attention

### Local Q/K/V aggregation

Input에서 Q/K/V를 만든 뒤, raw branch와 nearby-token aggregation branch를 구성한다. Figure 2는 raw, `3×3`, `5×5` 예를 보여준다. Section 3.1의 실제 실험 설정은 효율과 성능의 균형을 위해 **two-branch design**을 사용하고, 한 branch에서 `5×5` 이웃을 aggregate한다고 명시한다.

각 head의 Q/K/V에 대해 small-kernel depthwise-separable convolution을 독립적으로 적용한다.

```text
Q/K/V at original scale
 -> ReLU linear attention

Q/K/V -> 5×5 local aggregation
       -> ReLU linear attention

outputs concatenate -> linear fusion
```

Spatial downsampling으로 scale을 만드는 것이 아니라, 같은 resolution에서 서로 다른 local receptive field로 token을 aggregate한다. 따라서 output alignment가 단순하다.

### GPU-friendly aggregation fusion

Q, K, V와 head별 aggregation을 각각 별도 kernel로 실행하면 작은 operation이 많아진다. 저자들은 다음처럼 묶는다.

- 모든 depthwise convolution을 하나의 DWConv로 합친다.
- 모든 pointwise convolution을 하나의 `1×1` group convolution으로 합친다.
- Group 수는 `3 × number_of_heads`다.
- 각 group channel 수는 head dimension `d`다.

이는 mathematical branch 수를 줄이는 것이 아니라 여러 독립 연산을 한 regular grouped operator로 배치한다. 실제 speedup은 runtime의 group-conv kernel 품질에 의존한다.

### FFN + depthwise convolution

EfficientViT module은 multi-scale linear attention 뒤에 depthwise convolution을 포함한 FFN을 둔다.

```text
input
 -> multi-scale linear attention + residual
 -> FFN + DWConv + residual
 -> output
```

Attention은 global context, FFN 내부 DWConv는 local feature extraction을 담당한다. 정확한 normalization/activation과 channel config는 official implementation을 함께 확인해야 한다. 논문 본문은 B0-B3/L series의 모든 `C_i`, `L_i` 상세 table을 싣지 않고 repository로 연결한다.

## Macro architecture

Cityscapes `1024×2048` 입력을 예로 든 Figure 5의 spatial schedule은 다음과 같다.

| 구간 | Spatial shape | 내용 |
| --- | --- | --- |
| Input | `3×1024×2048` | RGB image |
| Stem | `C0×512×1024` | Conv + DSConv |
| Stage 1 | `C1×256×512` | MBConv 반복 |
| Stage 2, `P2` | `C2×128×256` | MBConv 반복 |
| Stage 3, `P3` | `C3×64×128` | MBConv + EfficientViT module |
| Stage 4, `P4` | `C4×32×64` | MBConv + EfficientViT module |
| Head | `C5×128×256` | P3 2×, P4 4× upsample 후 add, MBConv |

Downsampling은 stride-2 MBConv를 사용한다. Linear attention은 token 수가 매우 큰 stage 1/2가 아니라 stage 3/4에 배치한다. Earlier stages는 convolution으로 local feature를 추출하고, 충분히 줄어든 spatial resolution에서 global context를 넣는 hybrid 설계다.

P2/P3/P4는 `1×1` convolution으로 channel을 맞추고 standard upsampling으로 모두 stride 8 resolution에 정렬한 뒤 concatenate가 아니라 **addition**으로 fuse한다. Head는 MBConv와 prediction/upsample layer로 단순하게 유지한다.

## Batch=1 tensor shape와 activation 계산

### 논문 macro shape의 symbolic memory

Paper가 본문에 B0-B3의 실제 `C_i` 전체를 적지 않으므로 channel 값을 추정하지 않는다. FP16에서 feature tensor 하나의 크기는 다음처럼 channel당 계산할 수 있다.

| Feature | Spatial elements/channel | FP16 bytes/channel |
| --- | ---: | ---: |
| P2 `128×256` | 32,768 | 65,536 = 64 KiB |
| P3 `64×128` | 8,192 | 16,384 = 16 KiB |
| P4 `32×64` | 2,048 | 4,096 = 4 KiB |
| Head `128×256` | 32,768 | 65,536 = 64 KiB |

예를 들어 실제 config의 `C2`를 확인한 뒤 P2 logical size는 `64 KiB × C2`로 얻는다. FPN-style head가 P2/P3/P4를 동시에 참조하므로 backbone 단일 tensor 크기만으로 peak를 정하면 안 된다.

### 리뷰어 예시: stage 4 크기의 generic linear attention

공식 특정 variant 값이 아니라 연산 이해를 위한 예로 다음을 두자.

- `B=1`, `H=32`, `W=64`, `N=2048`
- model channel `C=128`
- heads `h=4`, head dimension `d=32`

```text
input feature       [1, 2048, 128]
Q/K/V               [1, 4, 2048, 32]
K^T V summary       [1, 4, 32, 32]
K sum z             [1, 4, 32]
Q S numerator       [1, 4, 2048, 32]
denominator         [1, 4, 2048, 1]
output              [1, 2048, 128]
```

Softmax score를 materialize하면

```math
4\times2048^2
=16{,}777{,}216\ \text{elements}
```

로 FP16 `32 MiB`다. Linear attention의 `K^TV` accumulator는

```math
4\times32^2=4{,}096\ \text{elements}
```

로 FP16 `8 KiB`, `z`는 256 bytes에 불과하다. Q/K/V 전체는 FP16 약 `1.5 MiB`다. 이 비교는 score/accumulator의 logical size이며, convolution branch와 FFN, workspace를 포함한 runtime peak는 아니다.

논문은 asymptotic memory 감소를 설명하지만 model별 측정 peak memory 표는 제공하지 않는다.

### Core attention MAC - 리뷰어 계산값

위 예에서 softmax attention의 두 matmul은 대략

```math
2hN^2d
=2\times4\times2048^2\times32
=1{,}073{,}741{,}824\ \text{MAC}
```

이다. Linear attention의 `K^TV`와 `QS`는

```math
2hNd^2
=2\times4\times2048\times32^2
=16{,}777{,}216\ \text{MAC}
```

으로 약 64배 작다. QKV/output projection, multi-scale convolution, FFN은 두 비교에 포함하지 않았다. Model 전체 speedup은 이 비율보다 작으며 operator fusion과 memory traffic에 좌우된다.

## 핵심 구현 pseudocode

```python
def relu_linear_attention(q, k, v, eps=1e-6):
    # q, k, v: [B, heads, N, d]
    q = F.relu(q)
    k = F.relu(k)

    # token 축을 먼저 축약한다. N×N score를 만들지 않는다.
    kv = torch.einsum("bhnd,bhne->bhde", k, v)  # [B,h,d,d]
    k_sum = k.sum(dim=2)                         # [B,h,d]

    numerator = torch.einsum("bhnd,bhde->bhne", q, kv)
    denominator = torch.einsum("bhnd,bhd->bhn", q, k_sum)
    return numerator / (denominator[..., None] + eps)


def multi_scale_linear_attention(x):
    # x: [B, C, H, W]
    qkv = qkv_projection(x)                      # grouped Q/K/V map

    raw = reshape_heads(qkv)
    local = grouped_pointwise(depthwise_5x5(qkv))
    local = reshape_heads(local)

    raw_out = relu_linear_attention(*split_qkv(raw))
    local_out = relu_linear_attention(*split_qkv(local))

    y = torch.cat([raw_out, local_out], dim=1)
    return output_projection(merge_heads(y))
```

잘못 구현해 `q @ k.transpose(-2,-1)`를 먼저 계산하면 이미 `N×N`을 materialize했으므로 linear attention의 이점이 사라진다. Einsum의 실제 lowering이 `K^TV`를 먼저 수행하는지도 compiler trace로 확인해야 한다.

## 실험 설정

### Latency protocol

논문은 다음 조건을 명시한다.

- Mobile CPU: Qualcomm Snapdragon 8 Gen 1, TensorFlow Lite, batch 1, FP32
- Linear-attention microbenchmark: Snapdragon 855 CPU, TFLite, batch 1, FP32
- Edge/cloud GPU: TensorRT, FP16
- Data transfer time 포함

Jetson Nano, AGX Xavier, AGX Orin, A100을 폭넓게 비교한다. 좋은 점은 batch 1을 명시하고 data transfer를 포함한 것이다. 다만 p50/p95, warm-up 횟수, thread affinity, 전력, 온도와 peak memory는 논문에 없다.

### Training

PyTorch로 구현하고 AdamW와 cosine learning-rate decay를 사용한다. Semantic segmentation backbone은 ImageNet pretrained weight로 초기화하고 head는 random initialization한다. Super-resolution은 scratch에서 학습한다.

본문은 dataset과 high-level optimizer는 제시하지만 각 model의 full epoch, batch, learning rate, augmentation table은 제한적이다. Exact reproduction에는 official repository config가 필요하다.

## 실험 결과

### Softmax 대 ReLU linear attention microbenchmark

Figure 4는 input feature-map size 24, 32, 40에서 유사 계산량의 module을 비교한다. Snapdragon 855 CPU, TFLite, batch 1, FP32에서 ReLU linear attention이 softmax attention보다 **3.3-4.5배 빠르다**고 caption이 보고한다.

이 결과는 end-to-end network speedup이 아니라 attention module microbenchmark다. Softmax 제거, `N×N` materialization 회피, kernel 구성이 함께 작용한다.

### ImageNet backbone - 논문 보고값

| Model | Resolution | Top-1 | Params | MACs | Nano bs1 | Orin bs1 | A100 throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EfficientViT-B2 | 256 | 82.7 | 24M | 2.1G | 58.5 ms | 2.8 ms | 5325 image/s |
| EfficientViT-B3 | 224 | 83.5 | 49M | 4.0G | 101 ms | 4.4 ms | 3797 image/s |
| EfficientViT-B3 | 288 | 84.2 | 49M | 6.5G | 141 ms | 5.6 ms | 2372 image/s |
| EfficientViT-L1 | 224 | 84.5 | 53M | 5.3G | 미보고 | 2.6 ms | 6207 image/s |
| EfficientViT-L2 | 288 | 85.6 | 64M | 11G | 미보고 | 4.3 ms | 3102 image/s |
| EfficientViT-L2 | 384 | **86.0** | 64M | 20G | 미보고 | 미보고 | 1784 image/s |

비교 대상으로 Swin-B는 `83.5%`, `88M`, `15G`, Nano `240 ms`, Orin `6.0 ms`, A100 `2236 image/s`다. 같은 top-1에서 EfficientViT-B3 224가 model size, MACs, latency에서 우수하다. 다만 implementation과 training recipe 차이까지 포함된 system comparison이다.

### Cityscapes `1024×2048`

| Model | mIoU | Params | MACs | Nano bs1 | Orin bs1 | A100 image/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepLabV3+ MobileNetV2 | 75.2 | 15M | 555G | 미보고 | 83.5 ms | 102 |
| EfficientViT-B0 | **75.7** | **0.7M** | **4.4G** | 0.28 s | **9.9 ms** | **263** |
| SegFormer-B1 | 78.5 | 14M | 244G | 5.6 s | 146 ms | 49 |
| SegNeXt-T | 79.8 | 4.3M | 51G | 2.2 s | 93.2 ms | 95 |
| EfficientViT-B1 | **80.5** | 4.8M | **25G** | **0.82 s** | **24.3 ms** | **175** |
| SegNeXt-S | 81.3 | 14M | 125G | 3.4 s | 127 ms | 70 |
| EfficientViT-B2 | **82.1** | 15M | **74G** | **1.7 s** | **46.5 ms** | **112** |
| SegNeXt-B | 82.6 | 28M | 276G | 미보고 | 228 ms | 41 |
| EfficientViT-B3 | **83.0** | 40M | **179G** | 미보고 | **81.8 ms** | **70** |
| SegNeXt-L | 83.2 | 49M | 578G | 미보고 | 374 ms | 26 |
| EfficientViT-L2 | 83.2 | 53M | 396G | 미보고 | **60.0 ms** | **102** |

모델을 유사 mIoU끼리 비교하면 MAC 감소가 실제 Orin/A100 이득으로 이어진다. 하지만 B와 L series의 latency 순서가 parameter/MAC 순서와 완전히 같지 않다. EfficientViT-L2는 B3보다 MACs가 크지만 Orin에서 더 빠르다. Architecture와 kernel utilization이 FLOPs보다 중요하다는 실제 사례다.

### ADE20K

| Model | mIoU | Params | MACs | Nano | Orin | A100 image/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SegFormer-B1 | 42.2 | 14M | 16G | 389 ms | 12.3 ms | 542 |
| EfficientViT-B1 | **42.8** | **4.8M** | **3.1G** | **110 ms** | **4.0 ms** | **1142** |
| SegNeXt-S | 44.3 | 14M | 16G | 428 ms | 17.2 ms | 592 |
| EfficientViT-B2 | **45.9** | 15M | **9.1G** | **212 ms** | **7.3 ms** | **846** |
| SegFormer-B2 | 46.5 | 28M | 62G | 920 ms | 24.3 ms | 345 |
| EfficientViT-B3 | **49.0** | 39M | **22G** | **411 ms** | **12.5 ms** | **555** |
| EfficientViT-L2 | **50.7** | 51M | **45G** | 미보고 | **9.0 ms** | **758** |

Input은 관례에 따라 shorter side를 512로 resize한다. Cityscapes의 fixed `1024×2048`과 같은 input 조건이 아니다.

### Super-resolution

FFHQ `512² -> 1024²`에서 Restormer는 `43.43 dB`, A100 `92.0 ms`; EfficientViT w0.75는 `43.54 dB`, `14.3 ms`로 `6.4배` speedup이다. Full EfficientViT는 `43.58 dB`, `17.8 ms`다.

BSD100 `160×240 -> 320×480`에서 EfficientViT는 `32.33 dB`, `3.2 ms`; Restormer는 `32.31 dB`, `15.1 ms`다. Dense task 종류가 바뀌어도 linear-attention backbone의 latency 이점이 유지되는지 보여준다.

### EfficientViT-SAM

| Model | Params | MACs | A100 image/s | COCO mAP | LVIS mAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAM-ViT-H | 641M | 2973G | 11 | 46.5 | 44.2 |
| EfficientViT-SAM-L0 | 35M | 35G | **762** | 45.7 | 41.8 |
| EfficientViT-SAM-L2 | 61M | 69G | 538 | 46.6 | 42.7 |
| EfficientViT-SAM-XL0 | 117M | 185G | 278 | 47.5 | 43.9 |
| EfficientViT-SAM-XL1 | 203M | 322G | 182 | **47.8** | **44.4** |

Box prompt는 ViTDet predicted box를 사용하며 throughput에는 image encoder와 SAM head가 모두 포함된다. 학습은 먼저 SAM image encoder를 teacher로 distill하고, 이후 전체 SA-1B로 end-to-end training한다. 따라서 architecture 효율은 뛰어나지만 data/training budget은 작지 않다.

## 핵심 ablation - 정확한 논문 값

Cityscapes `1024×2048`, scratch training이다. 저자들은 모든 variant가 `4.4G MACs`가 되도록 width를 rescale한다.

| Multi-scale | Global attention | mIoU | Params | MACs |
| :---: | :---: | ---: | ---: | ---: |
| - | - | 68.1 | 0.7M | 4.4G |
| ✓ | - | 72.3 | 0.7M | 4.4G |
| - | ✓ | 72.2 | 0.7M | 4.4G |
| ✓ | ✓ | **74.5** | 0.7M | 4.4G |

Baseline 대비 multi-scale만 추가하면 `+4.2%p`, global attention만 추가하면 `+4.1%p`, 둘 다 쓰면 `+6.4%p`다. Multi-scale model에 global을 더하면 `+2.2%p`, global model에 multi-scale을 더하면 `+2.3%p`다. 두 능력이 상호 보완적이라는 논문의 핵심 주장을 직접 뒷받침한다.

다만 width를 각 variant에서 다시 조정했으므로 parameter/MAC가 같아도 channel configuration은 동일하지 않을 수 있다. Component를 하나 제거한 동일 graph의 순수 ablation과는 다르다.

## 장점과 핵심 기여

- ReLU kernel과 associative trick으로 high-resolution global attention을 선형화했다.
- Linear attention의 local-detail 약점을 숨기지 않고 convolution으로 명시적으로 보완했다.
- Multi-scale Q/K/V aggregation을 small DWConv와 grouped `1×1` conv로 regular하게 구현했다.
- Hierarchical backbone과 additive head로 dense task의 activation과 topology를 단순화했다.
- Classification뿐 아니라 segmentation, super-resolution, Segment Anything까지 검증했다.
- Snapdragon, Jetson Nano/Xavier/Orin, A100에서 batch와 precision/runtime을 명시해 실측했다.
- Component ablation에서 global context와 multi-scale learning의 독립적 기여를 정량화했다.

## 한계와 비판적 관점

### 1. Linear attention은 softmax의 정확한 재배열이 아니다

Similarity function을 exponential softmax에서 ReLU feature dot-product로 바꾼다. Sharp attention과 pair-specific relation 표현력이 줄 수 있으며 convolution이 이를 보완한다.

### 2. Peak memory 측정값이 없다

논문은 asymptotic memory footprint 감소를 설명하지만 end-to-end peak activation 표는 없다. P2/P3/P4 동시 보관, upsampling head, convolution workspace까지 포함한 실제 peak는 profiler로 측정해야 한다.

### 3. Full model config와 training detail이 본문에 충분히 없다

B0-B3/L series의 모든 channel/depth table과 task별 full hyperparameter는 repository 의존적이다. 논문 PDF만으로 exact architecture를 복원하기 어렵다.

### 4. Hardware별 precision이 다르다

Mobile CPU는 FP32 TFLite, GPU는 FP16 TensorRT다. 한 table 안에서 platform을 비교할 수는 있지만 precision 효과와 hardware 효과를 분리한 것은 아니다. INT8 결과도 없다.

### 5. Latency 분포와 sustained behavior가 없다

Data transfer는 포함했지만 p95, cold-start, compile time, 전력, 온도, thermal throttling을 보고하지 않는다.

### 6. Group/depthwise convolution 지원 의존

GPU에서는 branch fusion이 효율적일 수 있지만 일부 mobile accelerator는 많은 groups나 `5×5` depthwise를 잘 지원하지 않을 수 있다. CPU fallback 여부가 중요하다.

### 7. SAM의 data budget

EfficientViT-SAM은 SA-1B end-to-end training과 SAM teacher distillation을 사용한다. Inference 효율과 training/data 효율을 구분해야 한다.

### 8. Single-point LVIS 약점

EfficientViT-SAM-XL1은 LVIS 1-click에서 `56.6`, SAM-ViT-H는 `59.2`다. 저자들은 end-to-end 단계에 interactive training setup이 없었던 점을 원인 후보로 제시한다.

## 자주 헷갈리는 지점

### Linear attention은 attention matrix를 만든 뒤 압축하는가

아니다. 올바른 구현은 `K^TV`를 먼저 계산해 `N×N` matrix 자체를 만들지 않는다.

### ReLU는 attention output에만 적용하는가

아니다. Similarity feature map으로 Q와 K에 적용한다. V는 그대로 weighted aggregation에 들어간다.

### Softmax가 없으면 normalization도 없는가

아니다. 분모 `Q'(sum K')`로 각 query의 weight 합을 normalize한다.

### Multi-scale branch는 feature map resolution을 바꾸는가

기본 설명에서는 small-kernel convolution으로 같은 grid의 nearby token을 aggregate한다. Pyramid downsampling은 별도의 macro backbone stage가 담당한다.

### 이 모델은 순수 Transformer인가

아니다. Stem/earlier stage의 DSConv/MBConv, attention branch의 convolution, FFN depthwise convolution, MBConv head를 적극적으로 사용한다. 효율을 위한 convolution-attention hybrid다.

### 모든 EfficientViT 논문이 이 구조인가

아니다. 앞서 설명한 cascaded group attention 계열과 구분해야 한다.

## 온디바이스 관점

### 좋은 점

- `N×N` score와 softmax를 제거한다.
- Fixed-shape matmul과 convolution이 중심이다.
- Global attention을 spatially 줄어든 stage 3/4에만 배치한다.
- P2/P3/P4 fuse가 concat이 아니라 add라 head channel 폭을 억제한다.
- 실제 Snapdragon/Jetson benchmark를 제공한다.

### 확인할 operator

```text
QKV 1×1 projection
ReLU
5×5 depthwise convolution
1×1 grouped convolution, groups=3h
K^T V / Q S matmul
denominator reduction/division
MBConv
bilinear upsample
feature add
```

특히 reduction과 division이 NPU에서 CPU로 fallback되는지, grouped convolution이 분할 kernel로 쪼개지는지 확인한다. `einsum`을 export했을 때 compiler가 transpose와 intermediate copy를 만들 수도 있다.

### Numerical stability와 quantization

ReLU 뒤 Q/K가 sparse하면 denominator가 작아질 수 있다. FP16/INT8에서 `ε`, accumulation precision, activation scale가 중요하다.

- `K^TV` accumulator는 FP16보다 FP32 accumulation이 필요할 수 있다.
- INT8 Q/K/V 뒤 global sum에서 overflow 범위를 확인한다.
- Division과 reciprocal approximation의 error를 측정한다.
- Softmax가 없다는 이유만으로 attention 전체가 쉽게 INT8화된다고 가정하지 않는다.

### Activation memory 측정

Profiler에서 다음을 분리한다.

- Q/K/V와 raw/multi-scale branch의 동시 lifetime
- `K^TV` accumulator
- FFN expanded activation
- P2/P3/P4 cache
- Upsampled P3/P4와 head add buffer
- Convolution workspace

Model file size와 peak activation을 별도 column으로 기록한다.

## 재현 계획

### 1단계: linear attention unit test

1. 작은 `N=16,d=8`에서 explicit ReLU attention matrix 방식과 associative 방식 output이 일치하는지 검증한다.
2. Denominator에 `ε`를 넣었을 때 all-negative Q/K 입력의 NaN을 막는지 확인한다.
3. Autograd gradient가 두 구현에서 tolerance 안에 일치하는지 확인한다.
4. Graph에 `N×N` tensor가 생성되지 않는지 profiler로 확인한다.

### 2단계: multi-scale block

1. Raw-only, 5×5-only, two-branch를 구현한다.
2. 독립 DW/pointwise와 fused DW/grouped-1×1 output이 같은지 확인한다.
3. Multi-scale/global ablation을 Table 1 조건처럼 동일 MAC budget에서 비교한다.
4. Boundary와 small-object metric을 추가해 convolution 보완 효과를 검증한다.

### 3단계: macro backbone/head

1. Official config에서 B0/B1 하나를 선택한다.
2. `P2/P3/P4` stride가 `8/16/32`인지 assert한다.
3. Upsampling 후 spatial/channel shape가 같아 add 가능한지 확인한다.
4. Official parameter/MAC와 동일 counting convention으로 대조한다.
5. Cityscapes single-scale mIoU를 먼저 재현한다.

### 4단계: device benchmark

| 축 | 설정 |
| --- | --- |
| Backend | TFLite CPU, GPU delegate, NPU, TensorRT |
| Precision | FP32, FP16, INT8 mixed |
| Batch | 1 고정 |
| Input | 512 short side, 1024×2048 |
| Metric | mIoU, p50/p95, peak memory, power, temperature |

Softmax baseline과 ReLU linear attention을 **같은 channel, 같은 surrounding block, 같은 compiler**에서 비교한다. Data transfer 포함/제외를 둘 다 기록하고 10분 이상 sustained test를 수행한다.

## 구현 체크리스트

- [ ] 대상 논문이 arXiv `2205.14756`인지 확인했는가?
- [ ] Q/K에 ReLU를 적용했는가?
- [ ] `K^TV`를 `QK^T`보다 먼저 계산하는가?
- [ ] Denominator normalization과 `ε`가 있는가?
- [ ] `N×N` intermediate가 graph에 없는가?
- [ ] Multi-scale experimental setting이 raw+5×5 two-branch인가?
- [ ] Group 수가 `3 × heads`인가?
- [ ] FFN에 depthwise convolution이 있는가?
- [ ] EfficientViT module이 stage 3/4에 배치되는가?
- [ ] P2/P3/P4가 stride `8/16/32`인가?
- [ ] Pyramid feature를 add하기 전에 channel을 맞췄는가?
- [ ] Paper-reported MAC/latency와 reviewer-calculated activation을 구분했는가?
- [ ] Mobile CPU FP32와 GPU FP16 결과를 같은 precision으로 오인하지 않는가?
- [ ] Peak memory, p95, energy가 논문 미보고임을 명시했는가?

## 로드맵에서의 위치와 후속 연결

MobileNetV2는 convolutional hierarchy와 activation memory를, DeiT는 data-efficient training을, Swin은 local window hierarchy를 보여준다. EfficientViT는 그 다음 질문에 답한다.

> Global context가 필요한 고해상도 dense task에서 window로 잘라버리지 않고도, 실제 edge hardware에서 감당 가능한 attention을 만들 수 있는가?

답은 ReLU linear attention과 convolutional local branch의 결합이다. 이후 SegFormer, SAM/EdgeSAM을 읽을 때 image encoder의 token 수와 high-resolution feature가 어디서 줄어드는지, attention score가 materialize되는지 비교하는 좋은 기준이다.

통합 프로젝트에서는 EfficientViT-SAM처럼 heavy teacher의 image encoder를 작은 student로 distill하는 경로가 직접 연결된다. 다만 상시 detector와 요청형 segmenter를 분리하고, high-resolution encoder를 이벤트 때만 실행해야 전력/thermal budget을 지킬 수 있다.

## 최종 평가

EfficientViT의 가장 강한 부분은 linear-attention 수식 하나가 아니라 **표현력 손실을 인정하고 local convolution으로 보완한 균형**, 그리고 이를 실제 hardware latency로 검증한 점이다. Softmax와 `N×N` score를 제거해 high-resolution dense prediction의 병목을 직접 겨냥했고, Cityscapes/ADE20K/SR/SAM에서 폭넓게 효과를 보였다.

반면 본문만으로 full config와 training recipe를 완전히 재구성하기 어렵고, peak memory·p95·energy·INT8 결과가 없다. ReLU attention의 sharpness 한계와 grouped/depthwise operator 의존성도 남는다. 온디바이스 연구에서는 논문의 평균 latency를 그대로 인용하는 데서 끝내지 말고, **target NPU가 `K^TV` reduction과 group conv를 어떻게 lower하는지, P2/P3/P4 cache가 실제 peak memory를 얼마나 차지하는지**를 반드시 측정해야 한다.

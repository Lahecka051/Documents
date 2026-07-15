# 80. DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification

## 논문 정보

- 원본 파일: `80_DynamicViT.pdf`
- 제목: DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification
- 저자: Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, Cho-Jui Hsieh
- 소속: Tsinghua University, UCLA, University of Washington
- 발표: NeurIPS 2021
- 제공된 PDF: arXiv:2106.02034v2, 2021년 10월 26일
- 논문 링크: https://arxiv.org/abs/2106.02034
- 공개 코드: https://github.com/raoyongming/DynamicViT
- 핵심 키워드: dynamic token pruning, vision transformer, Gumbel-Softmax, attention masking, progressive sparsification, input-adaptive inference

## 한눈에 보는 요약

DynamicViT는 image마다 중요한 patch가 다르다는 점을 이용해 ViT의 token을 세 지점에서 점진적으로 제거한다. 각 prediction module은 현재 token의 local feature와 살아 있는 token 전체의 global average를 결합해 patch마다 `drop/keep` 확률을 낸다. 한 번 제거된 token은 뒤 stage에서 다시 나타나지 않으며 class token은 항상 남긴다.

12-block Transformer의 대표 배치는 다음과 같다.

```text
patch embedding
 -> blocks 1-3
 -> predictor 1, keep rho * N patches
 -> blocks 4-6
 -> predictor 2, keep rho^2 * N patches
 -> blocks 7-9
 -> predictor 3, keep rho^3 * N patches
 -> blocks 10-12
 -> classifier from class token
```

$\rho=0.7$이면 마지막에는 $0.7^3=0.343$, 즉 원래 patch token의 약 34%만 남는다. 논문 표현으로는 66%를 제거한다. DeiT-S에서 4.6 GFLOPs를 2.9 GFLOPs로 37% 줄이고, RTX 3090 batch 32 throughput을 1337.7 images/s에서 2062.1 images/s로 54% 높이면서 ImageNet top-1은 79.8%에서 79.3%로 0.5 point 낮아졌다.

이 방법에서 반드시 구분해야 할 두 실행 경로가 있다.

- training: sample마다 남는 token 수가 달라 batch tensor를 compact하게 만들기 어려우므로 원래 $N\times N$ attention을 유지하고 attention masking으로 영향만 차단한다. Gumbel-Softmax로 decision을 학습하지만 training FLOPs와 activation memory는 token 수만큼 줄지 않는다.
- inference: 각 stage에서 keep probability 상위 정확히 $\lfloor\rho^sN\rfloor$개를 골라 실제 tensor에서 제거한다. 뒤 block은 짧은 dense sequence를 처리하므로 실제 FLOPs와 throughput이 줄어든다.

논문은 `unstructured token`이라고 부르지만 inference tensor 자체는 gather 후 compact한 dense matrix다. 공간 위치가 불규칙할 뿐 sequence length는 batch 전체에서 stage마다 고정할 수 있어 GPU 병렬화가 가능하다. 다만 top-k, sort, gather, dynamic shape 및 memory copy가 mobile NPU에서 지원되지 않으면 paper의 GPU speedup이 재현되지 않는다.

## 연구 배경과 핵심 가설

ViT는 $P\times P$ patch를 token으로 바꾸고 모든 token을 거의 같은 계산량으로 12개 block에 통과시킨다. 하지만 classification은 object의 일부 patch만으로 충분한 경우가 많고 background token의 최종 class 결정 기여는 작을 수 있다. CNN의 spatial downsampling은 고정된 grid 구조로 전체 image에 동일하게 적용되지만 DynamicViT는 input마다 다른 patch를 남긴다.

이 설계의 가설은 세 부분이다.

1. classification에 필요한 정보가 일부 patch에 집중된다.
2. early feature만으로 모든 중요도를 한 번에 판단하기보다 깊이가 증가하며 점진적으로 제거하는 편이 안전하다.
3. self-attention은 arbitrary set의 token에도 적용되므로 spatially irregular patch를 compact sequence로 만든 뒤 dense GEMM을 계속 사용할 수 있다.

Figure 5의 시각화는 stage가 진행될수록 foreground object 근처 token이 남는 경향을 보여 준다. 그러나 이것은 causal explanation이나 segmentation mask가 아니다. ImageNet의 center bias도 존재한다. Figure 6의 validation-set 평균 keep probability에서 중앙 patch가 더 자주 선택되는 현상이 이를 보여 준다.

## 전체 구조와 tensor shape

### Backbone

DynamicViT는 별도의 새 Transformer block을 만들지 않고 DeiT 또는 LV-ViT backbone 사이에 prediction module을 삽입한다. 12-layer 예에서는 4, 7, 10번째 block 앞, 즉 앞선 3개 block의 feature를 받은 지점에 세 module을 둔다.

patch token을 $x\in\mathbb{R}^{N\times C}$, 누적 decision mask를 $\hat D\in\lbrace0,1\rbrace^N$라고 두자. 논문 수식은 class token을 생략하지만 구현에서는 class token decision을 항상 1로 고정한다.

초기 상태는

```math
\hat D_i=1,\qquad i=1,\ldots,N
```

이다. stage decision $D$가 나오면

```math
\hat D\leftarrow\hat D\odot D
```

로 갱신한다. 따라서 과거에 0이 된 위치는 이후에도 0이다.

### Prediction module

먼저 token마다 channel을 절반으로 줄인 local feature를 만든다.

```math
z^{local}=\mathrm{MLP}(x)
\in\mathbb{R}^{N\times C'},
\qquad C'=C/2.
```

살아 있는 token만 평균해 global context를 만든다.

```math
z^{global}
=\mathrm{Agg}(\mathrm{MLP}(x),\hat D)
=\frac{\sum_{i=1}^N\hat D_i u_i}
{\sum_{i=1}^N\hat D_i}
\in\mathbb{R}^{C'}.
```

각 token에 global vector를 broadcast하고 local vector와 연결한다.

```math
z_i=[z_i^{local},z^{global}]
\in\mathbb{R}^{2C'}=\mathbb{R}^{C}.
```

두 class 확률은

```math
\pi=\mathrm{Softmax}(\mathrm{MLP}(z))
\in\mathbb{R}^{N\times2},
```

이며 $\pi_{i,0}$은 drop, $\pi_{i,1}$은 keep 확률이다.

appendix의 구체적인 architecture는 다음과 같다.

- local/global projection: 각각 `LayerNorm -> Linear(C,C/2) -> GELU`
- decision head: `Linear(C,C/2) -> GELU -> Linear(C/2,C/4) -> GELU -> Linear(C/4,2) -> Softmax`
- 세 stage는 같은 architecture를 쓰되 각자의 parameter를 갖는다.

### Predictor parameter 손계산

DeiT-S의 $C=384$를 예로 들고 bias와 LayerNorm parameter를 제외하면 module 하나의 linear weight는 reviewer 계산으로

```math
\begin{aligned}
&2\times(384\times192) &&\text{local/global projections}\\
&+(384\times192) &&\text{decision layer 1}\\
&+(192\times96) &&\text{decision layer 2}\\
&+(96\times2) &&\text{decision output}\\
&=239{,}616.
\end{aligned}
```

세 module이면 약 718,848 weight, FP16 payload 약 1.37 MiB다. paper Table 2에서 LV-ViT-S는 26.2M에서 DynamicViT-LV-S 26.9M으로 약 0.7M 증가하고, LV-ViT-M은 55.8M에서 57.1M으로 1.3M 증가한다. token pruning은 weight compression이 아니며 predictor 때문에 parameter memory는 소폭 늘어난다.

## Training: Gumbel-Softmax와 attention masking

### Gumbel-Softmax decision

binary sample은 미분할 수 없으므로 training decision은

```math
D=\mathrm{GumbelSoftmax}(\pi)_{:,1}
\in\lbrace0,1\rbrace^{N}
```

로 만든다. forward에서는 one-hot keep/drop sample을 사용하고 reparameterized surrogate를 통해 prediction module로 gradient를 전달한다.

### 왜 token vector를 0으로 만들면 안 되는가

일반 attention은

```math
A=\mathrm{Softmax}\left(\frac{QK^T}{\sqrt C}\right)
```

이다. dropped token vector를 0으로 만들어도 그 key의 logit은 0일 뿐 softmax denominator에 $e^0=1$로 참여한다. 다른 token의 attention normalization을 바꾸므로 완전히 제거한 것과 같지 않다.

DynamicViT는 score $P=QK^T/\sqrt C$에 graph mask를 적용한다.

```math
G_{ij}=
\begin{cases}
1,&i=j,\\
\hat D_j,&i\ne j,
\end{cases}
```

```math
\tilde A_{ij}
=\frac{\exp(P_{ij})G_{ij}}
{\sum_{k=1}^{N}\exp(P_{ik})G_{ik}}.
```

$\hat D_j=0$인 dropped token은 다른 모든 query의 key/value source에서 빠진다. 다만 자기 자신으로 가는 self-loop는 1로 남겨 denominator가 0이 되는 것을 막는다. 해당 dropped query의 output은 뒤에서 사용되지 않으므로 kept token만 계산한 결과와 kept-token 관점에서 동등하다.

### Training masking은 acceleration이 아니다

Equation 11은 constant $N\times N$ shape를 유지한다. dropped token의 Q/K/V와 attention row/column을 실제로 제거하지 않으므로 다음은 그대로 계산된다.

- full-length QKV projection
- $N\times N$ score matrix
- masked softmax
- full-length FFN
- Gumbel sample과 predictor

이는 batch-parallel differentiable training을 위한 장치다. 논문의 31%에서 37% FLOPs 절감과 throughput 향상은 inference에서 compact token을 실제로 제거했을 때의 결과다.

## Training objectives

### Classification loss

```math
\mathcal L_{cls}=\mathrm{CrossEntropy}(y,\bar y).
```

### Token distillation

원본 pretrained backbone을 frozen teacher로 두고 마지막 stage에서 남은 student token $t_i$를 teacher token $t_i'$와 맞춘다.

```math
\mathcal L_{distill}
=\frac{\sum_{b=1}^{B}\sum_{i=1}^{N}
\hat D_{i}^{b,S}\|t_i-t_i'\|^2}
{\sum_{b=1}^{B}\sum_{i=1}^{N}\hat D_i^{b,S}}.
```

### Prediction KL loss

```math
\mathcal L_{KL}=\mathrm{KL}(y\|y').
```

$y'$는 teacher prediction이다.

### Ratio loss

각 sample과 stage의 실제 keep fraction이 target $\rho^{(s)}$에 가깝도록 한다.

```math
\mathcal L_{ratio}
=\frac{1}{BS}\sum_{b=1}^{B}\sum_{s=1}^{S}
\left(\rho^{(s)}-\frac{1}{N}\sum_{i=1}^{N}
\hat D_i^{b,s}\right)^2.
```

최종 objective는

```math
\mathcal L=\mathcal L_{cls}
+0.5\mathcal L_{KL}
+0.5\mathcal L_{distill}
+2\mathcal L_{ratio}.
```

teacher forward와 token matching 때문에 training compute와 memory는 baseline fine-tuning보다 커질 수 있다. inference artifact에는 teacher가 없다.

## Inference: 실제 top-k token 제거

stage $s$의 target token 수는

```math
m_s=\left\lfloor\rho^sN\right\rfloor
```

이다. keep probability를 정렬해

```math
I^s=\mathrm{argsort}(\pi_{:,1})
```

상위 $m_s$ index만 gather한다. classification token은 별도로 항상 유지한다.

```text
tokens, spatial_ids = patch_embed(image)
keep_mask = all_true(N)

for stage s in {1, 2, 3}:
    tokens = run_three_backbone_blocks(tokens)
    keep_prob = predictor_s(tokens, keep_mask)
    m = floor((rho ** s) * original_N)
    ids = topk(keep_prob, m)
    tokens = gather(tokens, ids)
    spatial_ids = gather(spatial_ids, ids)

tokens = run_remaining_blocks(tokens)
logits = classifier(tokens[class_index])
```

모든 image가 같은 stage에서 같은 $m_s$를 가지므로 batch tensor는 rectangular하다. content-dependent한 것은 어떤 spatial index를 남기는지다. 논문의 `unstructured지만 hardware friendly` 주장은 이 dense compaction에 근거한다.

그러나 target runtime에는 `topk/argsort -> gather -> dynamic sequence attention`이 필요하다. static-shape NPU라면 최대 길이로 padding하여 계산 절감이 사라지거나, 세 길이별 compiled subgraph를 준비해야 할 수 있다.

## Token 수, FLOPs, activation memory 손계산

### 224 image, patch 16, rho 0.7

$14\times14=196$ patch token이고 class token을 포함하면 첫 길이는 197이다. absolute target을 사용하면 patch token 수는

| 구간 | Patch token | CLS 포함 sequence |
| --- | ---: | ---: |
| Blocks 1-3 | 196 | 197 |
| Blocks 4-6 | $\lfloor0.7\times196\rfloor=137$ | 138 |
| Blocks 7-9 | $\lfloor0.7^2\times196\rfloor=96$ | 97 |
| Blocks 10-12 | $\lfloor0.7^3\times196\rfloor=67$ | 68 |

마지막에 patch 129개, 약 65.8%가 제거된다. paper의 `66%`와 일치한다.

### Hidden activation

DeiT-S의 $C=384$, FP16, batch 1로 reviewer가 tensor 하나만 계산하면

| Sequence | $N\times C\times2$ | 크기 |
| ---: | ---: | ---: |
| 197 | 151,296 byte | 147.8 KiB |
| 138 | 105,984 byte | 103.5 KiB |
| 97 | 74,496 byte | 72.8 KiB |
| 68 | 52,224 byte | 51.0 KiB |

Q, K, V를 동시에 materialize하면 이 값의 약 3배이고 FFN expansion은 별도다. 실제 peak는 allocator, fusion, residual, workspace에 따라 다르다.

### Attention matrix

DeiT-S의 6 heads, FP16 attention probability 전체를 materialize한다고 가정하면

```math
M_{attn}=6N^2\times2\;bytes.
```

| Sequence | Attention matrix |
| ---: | ---: |
| 197 | 약 454.8 KiB |
| 138 | 약 223.2 KiB |
| 97 | 약 110.3 KiB |
| 68 | 약 54.2 KiB |

12개 block 전체에 각 구간이 3 block씩 있다고 단순 합산하면 baseline attention element traffic 대비 약

```math
\frac{3(197^2+138^2+97^2+68^2)}{12(197^2)}
\approx0.463
```

으로, attention score/probability 부분은 약 54% 줄어든다. 전체 model FLOPs 감소가 37%인 이유는 QKV/FFN의 선형항, 첫 세 block, patch embedding, predictor overhead가 남기 때문이다.

### Peak memory는 FLOPs처럼 37% 줄지 않는다

최대 sequence 197의 early block은 baseline과 동일하다. buffer를 layer별로 재사용한다면 single-layer peak가 early block에서 이미 결정될 수 있어 later token pruning이 peak activation을 크게 낮추지 못할 수 있다. 반면 cumulative memory traffic과 average live tensor는 줄어든다. paper는 peak memory를 보고하지 않으므로 `37% FLOPs 감소 = 37% peak RAM 감소`라고 쓰면 안 된다.

### Weight와 KV cache

DynamicViT는 backbone weight를 제거하지 않고 predictor parameter를 추가한다. 따라서 weight memory는 줄지 않는다. 또한 이것은 bidirectional image encoder라 autoregressive LLM의 persistent KV cache가 없다. 각 layer의 K/V activation은 token과 함께 줄지만 layer 사이에 누적 저장하는 KV cache와는 다른 개념이다.

## 학습 설정

- ImageNet-1K train 1.28M, validation 50k, single-crop top-1
- pretrained backbone에서 시작
- sparsification stage $S=3$
- target sequence $[\rho,\rho^2,\rho^3]$
- 30 epochs joint training
- backbone은 첫 5 epochs freeze
- prediction module LR: $(\text{batch size}/1024)\times0.001$
- backbone LR: prediction-module LR의 0.01배
- single machine, 8 NVIDIA GTX 1080 Ti
- backbone별 batch size는 GPU memory에 따라 조정

학습 batch가 모델마다 다를 수 있으므로 training throughput 비교는 제공되지 않는다. main inference throughput은 single RTX 3090, batch 32다.

## 주요 결과

### ImageNet accuracy, GFLOPs, throughput

| Backbone | $\rho$ | Top-1 | GFLOPs | RTX 3090 throughput |
| --- | ---: | ---: | ---: | ---: |
| DeiT-S | 1.0 | 79.8 | 4.6 | 1337.7 im/s |
| DeiT-S | 0.9 | 79.8 | 4.0 (-14%) | 1524.8 (+14%) |
| DeiT-S | 0.8 | 79.6 | 3.4 (-27%) | 1774.6 (+33%) |
| DeiT-S | 0.7 | 79.3 | 2.9 (-37%) | 2062.1 (+54%) |
| LV-ViT-S | 1.0 | 83.3 | 6.6 | 993.3 im/s |
| LV-ViT-S | 0.7 | 83.0 | 4.6 (-31%) | 1417.6 (+43%) |
| LV-ViT-M | 1.0 | 84.0 | 12.7 | 589.5 im/s |
| LV-ViT-M | 0.7 | 83.8 | 8.5 (-33%) | 888.2 (+50%) |

paper abstract의 `31%에서 37% FLOPs, 40% 이상 throughput, 0.5% 이내 accuracy drop`은 $\rho=0.7$ 세 backbone에 해당한다. batch 32 GPU throughput이므로 batch 1 mobile latency 주장으로 바꾸면 안 된다.

### Accuracy-complexity trade-off

| Model | Params | GFLOPs | Resolution | Top-1 |
| --- | ---: | ---: | ---: | ---: |
| LV-ViT-S | 26.2M | 6.6 | 224 | 83.3 |
| DynamicViT-LV-S/0.5 | 26.9M | 3.7 | 224 | 82.0 |
| DynamicViT-LV-S/0.7 | 26.9M | 4.6 | 224 | 83.0 |
| LV-ViT-M | 55.8M | 12.7 | 224 | 84.0 |
| DynamicViT-LV-M/0.7 | 57.1M | 8.5 | 224 | 83.8 |
| DynamicViT-LV-M/0.8 | 57.1M | 9.6 | 224 | 83.9 |

DynamicViT-LV-M/0.7은 83.8%로 paper가 비교한 EfficientNet-B5와 NFNet-F0의 83.6%를 넘지만 resolution, parameter, training recipe가 다르다. 동일 recipe의 direct ablation은 base LV-ViT와의 비교가 더 강한 evidence다.

### 큰 model과 큰 resolution

| Model | GFLOPs | Top-1 |
| --- | ---: | ---: |
| DeiT-B | 17.5 | 81.8 |
| DynamicViT-B/0.7 | 11.2 (-36%) | 81.3 (-0.5) |
| DeiT-S 384 | 15.5 | 81.6 |
| DynamicViT-S/0.7 384 | 9.5 (-39%) | 81.4 (-0.2) |
| DynamicViT-S/0.5 384 | 7.0 (-55%) | 80.3 (-1.3) |

token 수가 큰 384 input에서 absolute savings가 커지고 $\rho=0.7$ accuracy drop은 0.2 point다. $\rho=0.5$까지 줄이면 FLOPs는 55% 줄지만 1.3 point 하락한다.

## Ablation

### Dynamic, structural, static

DeiT-S, 모두 2.9 GFLOPs로 맞춘 결과다.

| Strategy | Top-1 | Baseline 대비 |
| --- | ---: | ---: |
| Structural 2x2 average pooling after block 6 | 78.2 | -1.6 |
| Static, input-independent token selection | 73.4 | -6.4 |
| Dynamic prediction | **79.3** | **-0.5** |

input-dependent selection의 효과가 크다. static의 구체적 최적화 capacity가 dynamic과 완전히 같지 않을 수 있지만 동일 FLOPs에서 중요한 비교다.

### Token importance criterion

| Criterion | Top-1 |
| --- | ---: |
| Random | 77.5 (-2.3) |
| Class-token attention score | 78.1 (-1.7) |
| Local-global prediction module | **79.3 (-0.5)** |

attention score만으로 pruning하는 것보다 학습된 predictor가 1.2 point 높다.

### Stage 수

동일 2.9 GFLOPs에서 target을 맞춘다.

| Sparsification | Target | Top-1 |
| --- | --- | ---: |
| Single stage | $[0.25]$ | 77.4 (-2.4) |
| Two stages | $[0.60,0.60^2]$ | 79.2 (-0.6) |
| Three stages | $[0.70,0.70^2,0.70^3]$ | **79.3 (-0.5)** |

early에 한 번에 75%를 버리는 것보다 feature가 성숙하면서 점진적으로 제거하는 편이 안전하다. appendix는 3개보다 더 늘려도 개선이 작고 predictor compute가 늘어 3개를 선택했다고 설명한다.

### Distillation과 ratio loss

DeiT-S $\rho=0.7$에서 $\lambda_{KL}=\lambda_{distill}=0$은 79.17, 0.5는 79.32, 1은 79.23이다. $\lambda_{ratio}=1/2/4$는 각각 79.15/79.32/79.29다. distillation 이득은 약 0.15 point로 작지만 일관되며 ratio loss는 desired compute budget을 맞추는 기능도 한다.

main Table 3에서 DeiT-S는 full loss 79.3, distill 제거 79.3, KL 제거 79.2, 둘 다 제거 79.2다. LV-ViT-S는 각각 83.0, 82.7, 82.9, 82.5로 teacher losses의 효과가 더 크다.

### Keeping ratio sweep

| $\rho$ | DeiT-S top-1 | GFLOPs | Last-stage patch fraction |
| ---: | ---: | ---: | ---: |
| 0.9 | 79.8 | 4.0 | 72.9% |
| 0.8 | 79.6 | 3.4 | 51.2% |
| 0.7 | 79.3 | 2.9 | 34.3% |
| 0.6 | 78.5 | 2.5 | 21.6% |
| 0.5 | 77.5 | 2.2 | 12.5% |

$\rho\lt 0.7$에서 information loss가 급격히 커진다는 appendix 해석과 일치한다.

## 실제 디바이스 관점

### 왜 GPU에서는 빨라졌는가

- gather 후 token tensor가 dense contiguous layout이 된다.
- stage별 output length가 고정이라 batch 32 GEMM을 유지한다.
- 뒤 block의 QKV, attention, FFN이 모두 짧아진다.
- attention은 quadratic이라 token 감소의 효과가 크다.

### 모바일에서 확인할 위험

- `topk` 또는 `argsort` 지원이 없으면 CPU fallback이 발생할 수 있다.
- gather/scatter가 NPU와 CPU 사이 copy를 만들 수 있다.
- compiler가 세 sequence length를 static subgraph로 만들지 못할 수 있다.
- batch 1에서는 predictor와 kernel launch overhead 비율이 커진다.
- 작은 GEMM은 accelerator utilization이 낮아져 FLOPs 감소만큼 latency가 줄지 않을 수 있다.
- token index를 보존해야 positional information과 dense-task 좌표를 복원할 수 있다.
- input content는 다르지만 keep count가 같아 compute variance는 제한적이다. 그래도 top-k 및 cache behavior의 p95는 측정해야 한다.

논문 실측은 RTX 3090 batch 32 throughput뿐이다. mobile CPU, NPU, GPU의 batch 1 p50/p95, peak memory, power, temperature는 보고하지 않는다.

## Detection과 segmentation으로 확장할 때

paper는 ImageNet classification만 평가하고 dense prediction은 future work로 남긴다. classification에서는 background를 버려도 class token만 정확하면 되지만 detection/segmentation에는 작은 object와 boundary token이 중요하다.

- 작은 object가 몇 patch뿐이면 early predictor가 모두 제거할 수 있다.
- FPN 또는 decoder가 원래 grid를 요구하면 sparse token을 scatter해야 한다.
- mask loss가 없는 classification-trained keep map은 boundary를 보존한다는 보장이 없다.
- detector ROI가 주어지면 ROI 안 token의 minimum quota를 두는 방식이 더 안전하다.
- dense task에서는 token recall, small-object AP, boundary IoU를 별도 ablation해야 한다.

## 강점

1. input-adaptive spatial sparsity를 실제 dense Transformer 연산 감소로 연결한다.
2. progressive three-stage design이 single-shot보다 정확도가 높다는 ablation이 강하다.
3. Gumbel-Softmax와 attention masking으로 end-to-end training 문제를 해결한다.
4. prediction module이 local과 global context를 모두 사용한다.
5. DeiT와 LV-ViT의 여러 크기, 224와 384 resolution에서 일관된 trade-off를 보인다.
6. theoretical FLOPs뿐 아니라 GPU throughput을 함께 보고한다.
7. stage별 exact token count를 사용해 batch inference를 dense하게 유지한다.

## 한계

1. training masking은 실제 token을 제거하지 않아 training compute와 activation memory를 줄이지 않는다.
2. teacher distillation은 training cost를 더한다.
3. throughput은 RTX 3090 batch 32이며 edge batch 1 latency가 아니다.
4. peak memory, p95, power, thermal throttling이 없다.
5. predictor가 parameter와 operator overhead를 추가한다.
6. top-k/gather 및 dynamic length의 backend 지원에 민감하다.
7. ImageNet classification만 검증해 dense prediction과 video 일반화가 미확인이다.
8. visual keep map을 explanation이나 segmentation으로 간주할 수 없다.
9. weight와 first-stage peak activation은 줄지 않는다.

## 재현 체크리스트

### Architecture

- [ ] sparsification을 block 4, 7, 10 앞에 배치했다.
- [ ] class token은 항상 keep했다.
- [ ] local/global projection dimension을 $C/2$로 맞췄다.
- [ ] cumulative mask로 dropped token의 재진입을 막았다.
- [ ] stage target을 $[\rho,\rho^2,\rho^3]$로 사용했다.
- [ ] spatial index를 token과 함께 gather했다.

### Training

- [ ] Gumbel-Softmax의 forward one-hot과 backward surrogate를 확인했다.
- [ ] zero-vector가 아니라 attention denominator까지 mask했다.
- [ ] dropped token에 self-loop를 남겼다.
- [ ] ratio loss, token distillation, KL의 coefficient를 2/0.5/0.5로 맞췄다.
- [ ] pretrained backbone을 사용하고 첫 5 epochs freeze했다.
- [ ] total 30 epochs와 두 learning rate를 기록했다.
- [ ] training path가 FLOPs를 절감하지 않는다고 명시했다.

### Inference

- [ ] 각 stage에서 정확히 $\lfloor\rho^sN\rfloor$ patch를 top-k로 골랐다.
- [ ] attention mask가 아니라 실제 gather로 sequence를 줄였다.
- [ ] top-k, gather, 세 sequence length가 target backend에서 실행되는지 확인했다.
- [ ] CPU fallback과 device copy를 operator trace로 검사했다.
- [ ] predictor overhead를 포함한 end-to-end latency를 측정했다.
- [ ] baseline과 같은 precision, resolution, batch를 사용했다.

### Memory와 latency

- [ ] weight 증가분과 activation 감소분을 분리했다.
- [ ] early block이 peak memory를 결정하는지 allocator trace로 확인했다.
- [ ] batch 1 p50/p95와 batch throughput을 모두 기록했다.
- [ ] warm-up, cold start, sustained run을 분리했다.
- [ ] power, temperature, throttling을 측정했다.
- [ ] FLOPs 감소율을 peak RAM 감소율로 대체하지 않았다.

### Dense task 확장

- [ ] small-object AP와 boundary metric을 추가했다.
- [ ] ROI 및 foreground token minimum quota를 검토했다.
- [ ] decoder가 요구하는 grid scatter 비용을 포함했다.
- [ ] dropped-token visualization을 정답 mask처럼 사용하지 않았다.

## 최종 평가

DynamicViT의 중요한 공헌은 ViT의 token sparsity를 irregular sparse kernel이 아니라 `content-dependent top-k + shorter dense sequence`로 구현한 데 있다. 공간적으로는 불규칙하지만 stage별 token 수를 고정해 GPU가 잘 처리하는 dense matrix multiplication을 유지한다. 세 단계의 progressive selection과 local-global predictor는 같은 2.9 GFLOPs의 structural, static, random, attention-score baseline보다 높은 accuracy를 보였다.

동시에 training과 inference를 분리해서 읽어야 한다. attention masking은 differentiable training을 위한 constant-shape simulation이며 training acceleration이 아니다. 실제 speedup은 inference에서 token을 gather한 뒤에만 생긴다. 또한 논문의 43%에서 54% throughput 향상은 RTX 3090 batch 32 결과다. 온디바이스 채택 여부는 top-k/gather 지원, 작은 GEMM utilization, batch 1 p95, early-layer peak memory, 전력 및 thermal trace로 다시 판단해야 한다.

로드맵 관점에서는 DynamicViT가 ROI 기반 visual token 압축과 동적 해상도의 직접적인 선행 아이디어다. detector나 prompt로 중요 영역을 더 안정적으로 알려 주고, runtime이 몇 개의 정적 token budget을 선택하도록 만들면 mobile 친화적으로 확장할 수 있다. 다만 classification accuracy만으로 작은 객체와 dense boundary 보존을 가정하지 않는 것이 필수다.

# 77. LoRA: Low-Rank Adaptation of Large Language Models

## 논문 정보

- 원본 파일: `77_LoRA.pdf`
- 제목: LoRA: Low-Rank Adaptation of Large Language Models
- 저자: Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
- 소속: Microsoft Corporation
- 제공된 PDF: Version 2, arXiv:2106.09685v2, 2021년 10월 16일
- 논문 링크: https://arxiv.org/abs/2106.09685
- 공개 코드: https://github.com/microsoft/LoRA
- 핵심 키워드: parameter-efficient fine-tuning, low-rank update, frozen backbone, weight merging, task-specific adapter, large language model

## 한눈에 보는 요약

LoRA는 pretrained dense weight $W_0$를 그대로 고정하고, downstream adaptation에서 필요한 변화량 $\Delta W$만 두 작은 matrix의 곱으로 학습한다.

```math
W=W_0+\Delta W,
\qquad
\Delta W=BA,
```

```math
A\in\mathbb{R}^{r\times k},
\qquad
B\in\mathbb{R}^{d\times r},
\qquad
r\ll\min(d,k).
```

입력 $x\in\mathbb{R}^{k}$에 대한 forward는

```math
h=W_0x+\frac{\alpha}{r}BAx
```

이다. $A$는 Gaussian random으로, $B$는 0으로 초기화하므로 첫 step의 $\Delta W$는 정확히 0이다. $W_0$에는 gradient와 optimizer state를 만들지 않고 $A,B$만 Adam으로 update한다.

LoRA의 가장 중요한 deployment 특성은 linear update라는 점이다. inference 전에

```math
W_{merged}=W_0+\frac{\alpha}{r}BA
```

를 한 번 계산하면 기존 dense layer 하나로 실행할 수 있다. 이 `merged inference`에는 adapter branch, 추가 kernel launch, 추가 sequence token이 없다. 반대로 여러 task adapter를 sample마다 동적으로 골라야 하면 merge하지 않고 두 low-rank matmul을 실행할 수 있지만, 이 `unmerged inference`에는 작은 추가 연산과 latency가 있다. 논문의 `추가 inference latency 없음` 주장은 merged path에 대한 말이다.

LoRA가 줄이는 것은 downstream task별 trainable parameter, gradient, Adam state, checkpoint 저장량이다. 다음은 줄이지 않는다.

- pretrained base weight의 resident memory
- Transformer forward/backward activation
- autoregressive inference의 KV cache
- merged model의 dense matmul FLOPs
- base model 자체의 parameter 수

GPT-3 175B에서 paper는 full fine-tuning의 training VRAM 1.2 TB를 LoRA 350 GB로 낮추고, task checkpoint를 350 GB에서 35 MB로 약 10,000배 줄였다고 보고한다. 하지만 deployment에도 350 GB base model은 여전히 필요하다. footnote의 100-task 예에서는 full copy 35 TB 대신 base 350 GB와 adapter 35 MB x 100, 약 354 GB를 저장한다.

## 문제 설정

pretrained autoregressive model을 $P_\Phi(y|x)$라 하고 downstream dataset을

```math
\mathcal Z=\lbrace(x_i,y_i)\rbrace_{i=1}^{N}
```

라 하자. full fine-tuning은 pretrained parameter $\Phi_0$ 전체를 update해

```math
\max_{\Phi}
\sum_{(x,y)\in\mathcal Z}
\sum_{t=1}^{|y|}
\log P_\Phi(y_t\mid x,y_{\lt t})
```

를 최적화한다. task마다 $|\Delta\Phi|=|\Phi_0|$인 새 checkpoint가 생긴다.

LoRA는 task-specific update를 작은 parameter $\Theta$로 encode한다.

```math
\Delta\Phi=\Delta\Phi(\Theta),
\qquad |\Theta|\ll|\Phi_0|,
```

```math
\max_{\Theta}
\sum_{(x,y)\in\mathcal Z}
\sum_{t=1}^{|y|}
\log P_{\Phi_0+\Delta\Phi(\Theta)}
(y_t\mid x,y_{\lt t}).
```

GPT-3 175B 실험에서 $|\Theta|$는 base의 약 0.01%보다 작을 수 있다. 이는 downstream update가 full-rank일 필요 없이 낮은 intrinsic rank를 가진다는 가설에 기반한다.

## 핵심 reparameterization

### Shape

pretrained linear layer를

```math
W_0\in\mathbb{R}^{d\times k},
\qquad x\in\mathbb{R}^{k}
```

라고 두면 LoRA path는 다음 순서다.

```text
x[k]
 -> A[r,k] @ x[k]       = z[r]
 -> B[d,r] @ z[r]       = delta_h[d]
 -> alpha/r scaling
 -> add W0[d,k] @ x[k]  = h[d]
```

batch와 sequence를 포함한 framework convention에서는 $X\in\mathbb{R}^{B\times T\times k}$이고,

```math
Z=XA^T\in\mathbb{R}^{B\times T\times r},
\qquad
\Delta H=ZB^T\in\mathbb{R}^{B\times T\times d}.
```

matrix orientation은 library가 row-vector 입력을 쓰는지 column-vector 입력을 쓰는지에 따라 transpose가 달라지지만 parameter count와 rank는 같다.

### Parameter count

full weight는 $dk$개 parameter이고 LoRA는

```math
r(k+d)
```

개다. square projection $d=k=d_{model}$이면 $2d_{model}r$이다.

Transformer layer $L$개에서 $W_q,W_v$ 두 projection에 적용하면

```math
|\Theta|=4Ld_{model}r.
```

GPT-3 175B의 $L=96$, $d_{model}=12{,}288$을 대입하면

| Rank와 target | Trainable parameter reviewer 계산 | Paper table |
| --- | ---: | ---: |
| $r=1$, $W_q,W_v$ | $4\times96\times12288=4.72$M | 4.7M |
| $r=4$, $W_q,W_v$ | 18.87M | 18.8M |
| $r=8$, $W_q,W_v$ | 37.75M | 37.7M |
| $r=2$, Q/K/V/O 전체 | 18.87M | 18.8M |

$r=4$, Q/V의 18.87M parameter를 FP16으로 저장하면 약 37.7 MB decimal, 36.0 MiB이므로 paper의 `35 MB` 설명과 부합한다.

### Initialization과 scaling

논문은

```math
A\sim\mathcal N(0,\sigma^2),
\qquad B=0
```

로 초기화한다. 따라서 random $A$가 있어도 $BA=0$이고 첫 forward는 정확히 pretrained model이다. 두 factor를 모두 0으로 시작하면 첫 backward에서 서로에게 전달되는 gradient도 0이 될 수 있으므로 asymmetric initialization이 중요하다.

update에는 $\alpha/r$를 곱한다. paper는 첫 번째로 시도한 rank에 $\alpha$를 맞추고 rank별로 다시 tuning하지 않았다고 설명한다. RoBERTa base는 $r_q=r_v=8,\alpha=8$, RoBERTa large는 $r_q=r_v=8,\alpha=16$, GPT-2는 $r_q=r_v=4,\alpha=32$를 사용했다. $\alpha$는 universal constant가 아니라 experiment recipe의 일부다.

## Transformer에서 어디에 적용하는가

Transformer self-attention에는 $W_q,W_k,W_v,W_o$가 있고 MLP에는 두 dense matrix가 있다. LoRA는 원리상 어느 dense layer에도 적용할 수 있지만 paper의 main study는 attention weight만 대상으로 하고 MLP를 고정한다. 대부분의 experiment는 $W_q,W_v$에 LoRA를 둔다.

이는 다음을 의미한다.

- `LoRA는 항상 Q/V에만 쓴다`는 이론적 제한은 아니다.
- paper가 검증한 default가 Q/V라는 뜻이다.
- MLP, LayerNorm, bias adaptation은 이 paper에서 주된 empirical study 대상이 아니다.
- modern vision encoder나 VLM projector에 적용하는 것은 원리의 확장이며 paper 원 실험과 구분해야 한다.

attention head별 matrix로 나누지 않고 $W_q$ 등을 $d_{model}\times d_{model}$ 하나로 취급한다. 따라서 LoRA factor는 모든 head에 걸친 shared low-rank update를 표현한다.

## Training과 inference의 세 가지 상태

### 1. Training, unmerged

```text
freeze W0
train A and B
h = linear(x, W0) + (alpha/r) * linear(linear(x, A), B)
```

base weight gradient와 Adam state는 만들지 않지만 adapter gradient를 계산하려면 layer input과 upstream gradient가 필요하다. frozen backbone을 통과하는 activation까지 모두 없앨 수 있는 것은 아니다.

### 2. Inference, merged

```text
delta_W = (alpha/r) * (B @ A)
W_merged = W0 + delta_W
h = linear(x, W_merged)
```

output shape, dense weight shape, layer depth가 원래와 같다. adapter에 의한 추가 runtime FLOPs와 kernel launch가 없다. 대신 하나의 merged weight에는 한 task가 반영되어 있다.

### 3. Inference, unmerged dynamic adapters

```text
h = linear(x, W0)
h += (alpha/r) * linear(linear(x, A_task), B_task)
```

base 한 개를 공유하면서 request마다 adapter를 바꾸기 쉽다. 그러나 batch 안 sample별 task가 다르면 grouped execution이나 per-sample adapter kernel이 필요하고 latency가 0은 아니다.

## Merge가 추가 latency를 없애는 이유와 조건

merged weight는 $W_0$와 같은 $d\times k$ shape다. dense matmul 하나라는 점에서 fully fine-tuned model과 동등하다. adapter layer처럼 network depth를 늘리지 않고 prefix처럼 token을 추가하지도 않는다.

조건은 다음과 같다.

1. $BA$를 미리 계산하고 $W_0$에 더해야 한다.
2. merged weight를 base와 같은 dtype/layout으로 저장해야 한다.
3. task를 바꿀 때 merge cost가 request critical path에 없어야 한다.
4. 여러 task를 같은 batch에 넣지 않거나 unmerged path를 감수해야 한다.

paper는 task switch 때 기존 $BA$를 빼고 새 $B'A'$를 더할 수 있다고 설명한다. 반복 add/subtract의 rounding drift를 피하려면 실무에서는 immutable $W_0$에서 새 merged copy를 만들거나 높은 precision master를 유지하는 편이 안전하다.

### Unmerged FLOPs 손계산

square layer $d=4096$, rank $r=8$, token 하나를 예로 들자. dense matmul은 약 $2d^2=33.55$ MFLOPs이고 두 LoRA matmul은

```math
2rk+2dr=4dr=131{,}072\text{ FLOPs}
```

로 약 0.39%다. arithmetic 비율은 작지만 batch 1에서 두 kernel launch, adapter weight read, intermediate write가 latency에 더 큰 영향을 줄 수 있다. merged path는 이 비용 자체가 없다. 이 수치는 reviewer estimate이며 paper의 GPT-2 latency 표는 merged LoRA와 adapter layer를 비교한 실측이다.

## Training memory: 무엇이 실제로 줄어드는가

### 줄어드는 상태

full fine-tuning에서 trainable base parameter마다 보통 다음이 필요하다.

- weight
- weight gradient
- Adam first moment
- Adam second moment
- mixed-precision training이면 FP32 master weight

LoRA는 $W_0$를 frozen으로 두므로 base gradient와 base optimizer state를 제거한다. $A,B$에만 이 상태가 필요하다. paper는 GPT-3 175B training VRAM을 1.2 TB에서 350 GB로 약 3.4배 줄였다고 보고한다.

### 여전히 필요한 상태

- base weight $W_0$
- forward activation과 backward에 필요한 saved tensor
- loss 및 attention workspace
- data-parallel/model-parallel communication buffer
- adapter parameter, gradient, optimizer state

base가 frozen이어도 앞 layer의 adapter로 gradient를 전달하려면 backbone의 Jacobian을 거쳐야 한다. `requires_grad=False`가 전체 activation 저장을 자동으로 없애지 않는다. activation checkpointing이나 recomputation은 LoRA와 별도의 기술이다.

### 한 square layer의 reviewer memory budget

$d=k=4096$, $r=8$, FP16 parameter와 FP32 Adam master/moments를 가정한다.

Base weight:

```math
4096^2\times2=33{,}554{,}432\;bytes=32\;MiB.
```

LoRA parameter:

```math
r(k+d)=8(4096+4096)=65{,}536.
```

| LoRA training state | 크기 |
| --- | ---: |
| FP16 A/B | 128 KiB |
| FP16 gradient | 128 KiB |
| FP32 master | 256 KiB |
| FP32 Adam m/v | 512 KiB |
| 합계 | 약 1 MiB |

full fine-tuning에서 같은 base matrix에 gradient, FP32 master, 두 moment를 두면 weight를 포함해 약 256 MiB 수준이 될 수 있다. LoRA는 base weight 32 MiB와 adapter state 약 1 MiB만 요구한다. 실제 distributed optimizer, shard, dtype에 따라 달라지는 format estimate이며 paper의 1.2 TB/350 GB 실측과 동일한 계산은 아니다.

## Activation memory와 KV cache

### LoRA intermediate는 작지만 base activation은 그대로다

$B=1,T=2048,d=4096,r=8$, FP16에서 hidden tensor 하나는

```math
1\times2048\times4096\times2=16\;MiB.
```

LoRA down-projection intermediate는

```math
1\times2048\times8\times2=32\;KiB.
```

adapter 자체의 추가 activation은 작다. 그러나 Transformer 각 block의 base hidden, Q/K/V, attention/MLP saved tensor는 여전히 필요하다. LoRA가 activation memory를 rank 비율만큼 줄이지는 않는다.

### KV cache는 전혀 줄지 않는다

LoRA는 head 수, head dimension, layer 수, context length를 바꾸지 않는다. merged 여부와 관계없이 K/V output shape가 같으므로 autoregressive KV cache도 같다.

GPT-3 175B를 $L=96,d_{model}=12{,}288$, standard multi-head attention, FP16 KV로 가정한 reviewer 계산은 token당

```math
2\;(K,V)\times96\times12{,}288\times2\;bytes
=4{,}718{,}592\;bytes
=4.5\;MiB/token.
```

context 2048, batch 1이면 약 9 GiB다. model parallel layout과 allocator에 따라 device별 resident 양은 달라지지만 LoRA 전후 총 logical KV는 같다. LoRA의 `sequence length를 줄이지 않는다`는 장점은 prefix token처럼 사용 가능한 context를 소비하지 않는다는 뜻이지 KV cache를 압축한다는 뜻이 아니다.

## Task storage와 base model memory

GPT-3 FP16 175B base payload는

```math
175\times10^9\times2=350\;GB
```

다. LoRA adapter 35 MB는 task별 delta artifact일 뿐 standalone model이 아니다.

| 100 tasks | 저장량 |
| --- | ---: |
| Full fine-tuned copies | 약 $100\times350$ GB = 35 TB |
| Shared base + LoRA | $350$ GB + $100\times35$ MB = 약 353.5 GB |

paper footnote는 이를 약 354 GB로 반올림한다. 한 task만 실행할 때 base resident memory는 350 GB로 남는다. LoRA를 `base model compression`으로 분류하면 안 되는 이유다.

## 학습 및 merge 의사코드

```text
class LoRALinear:
    frozen W0[d, k]
    trainable A[r, k] ~ Gaussian
    trainable B[d, r] = 0
    constant scale = alpha / r

    forward(X[B, T, k]):
        base = X @ W0.T
        low_rank = (X @ A.T) @ B.T
        return base + scale * low_rank

training:
    set W0.requires_grad = false
    optimizer = AdamW([A, B], task_hyperparameters)
    for batch:
        logits = model(batch)
        loss = task_loss(logits, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

export adapter:
    save A, B, alpha, rank, target_module_names,
         base_model_id, base_model_hash, dtype

merge for one task:
    for each target module:
        delta = (alpha / r) * (B @ A)
        W_merged = cast(W0 + delta, deployment_dtype)
    run normal dense model

unmerge or switch:
    rebuild from immutable W0 and selected adapter
```

adapter manifest에 base model hash와 exact target module name이 필요한 이유는 같은 shape라도 다른 checkpoint에 잘못 merge하면 실행은 되지만 품질이 무너질 수 있기 때문이다.

## 실험 결과

### GLUE

| Model / Method | Trainable | GLUE average |
| --- | ---: | ---: |
| RoBERTa-base full FT | 125.0M | 86.4 |
| RoBERTa-base BitFit | 0.1M | 85.2 |
| RoBERTa-base LoRA | 0.3M | **87.2** |
| RoBERTa-large full FT | 355.0M | 88.9 |
| RoBERTa-large LoRA | 0.8M | **89.0** |
| DeBERTa-XXL full FT | 1500.0M | 91.1 |
| DeBERTa-XXL LoRA | 4.7M | **91.3** |

RoBERTa-base LoRA의 task별 값은 MNLI 87.5, SST-2 95.1, MRPC 89.7, CoLA 63.4, QNLI 93.3, QQP 90.8, RTE 86.6, STS-B 91.5다. restricted adapter-comparison setup에서는 RoBERTa-large LoRA 평균이 88.6으로 unrestricted 89.0과 다르다. initialization, sequence length와 batch setup이 다르므로 같은 row처럼 섞으면 안 된다.

### GPT-2 E2E NLG

| Model / Method | Trainable | BLEU | NIST | METEOR | ROUGE-L | CIDEr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-2 M full FT | 354.92M | 68.2 | 8.62 | 46.2 | 71.0 | 2.47 |
| GPT-2 M PreLayer | 0.35M | 69.7 | 8.81 | 46.1 | 71.4 | 2.49 |
| GPT-2 M LoRA | 0.35M | **70.4** | **8.85** | **46.8** | **71.8** | **2.53** |
| GPT-2 L full FT | 774.03M | 68.5 | 8.78 | 46.0 | 69.9 | 2.45 |
| GPT-2 L LoRA | 0.77M | **70.4** | **8.89** | **46.8** | **72.0** | 2.47 |

additional DART BLEU는 GPT-2 Medium full FT 46.2 대 LoRA 47.1, GPT-2 Large full FT 47.0 대 LoRA 47.5다. WebNLG all-category BLEU는 Medium 46.5 대 LoRA 55.3, Large 55.5 대 LoRA 57.0이다.

### GPT-3 175B

| Method | Trainable | WikiSQL acc | MNLI-m acc | SAMSum R1/R2/RL |
| --- | ---: | ---: | ---: | --- |
| Full fine-tuning | 175,255.8M | 73.8 | 89.5 | 52.0/28.0/44.5 |
| BitFit | 14.2M | 71.3 | 91.0 | 51.3/27.4/43.5 |
| PreEmbed | 3.2M | 63.1 | 88.6 | 48.3/24.2/40.5 |
| PreLayer | 20.2M | 70.1 | 89.5 | 50.8/27.3/43.5 |
| AdapterH | 40.1M | 73.2 | 91.5 | 53.2/29.0/45.1 |
| LoRA | 4.7M | 73.4 | **91.7** | **53.8/29.8/45.9** |
| LoRA | 37.7M | **74.0** | 91.6 | 53.4/29.2/45.1 |

paper가 밝힌 typical fluctuation은 WikiSQL 약 $\pm0.5$, MNLI 약 $\pm0.1$, SAMSum 약 $\pm0.2/0.2/0.1$이다. 작은 차이는 이 variance와 함께 해석해야 한다. 모든 task에서 더 많은 LoRA parameter가 항상 더 높은 score를 주지도 않는다.

### Low-data MNLI

| Method | 100 samples | 1k | 10k | Full 392k |
| --- | ---: | ---: | ---: | ---: |
| Full FT | 60.2 | 85.8 | 88.9 | 89.5 |
| PrefixEmbed | 37.6 | 75.2 | 79.5 | 88.6 |
| PrefixLayer | 48.3 | 82.5 | 85.9 | 89.6 |
| LoRA | **63.8** | 85.6 | **89.2** | **91.7** |

100-example regime에서 LoRA가 좋은 sample efficiency를 보이지만 한 dataset과 GPT-3의 결과다. 모든 low-data task에 대한 보장은 아니다.

## Adapter latency 비교

GPT-2 Medium, NVIDIA Quadro RTX8000, single forward 100회 평균의 paper Table 1이다.

| Batch | Sequence | Fine-Tune / merged LoRA | AdapterL | AdapterH |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 512 | 1449.4 ms | 1482.0 (+2.2%) | 1492.2 (+3.0%) |
| 16 | 256 | 338.0 ms | 354.8 (+5.0%) | 366.3 (+8.4%) |
| 1 | 128 | 19.8 ms | 23.9 (+20.7%) | 25.8 (+30.3%) |

adapter FLOPs가 작아도 sequential layer와 kernel launch 때문에 online batch 1에서 overhead가 크다. LoRA row가 fine-tune과 같은 이유는 merge했기 때문이다. 이 표는 modern mobile device latency가 아니며 p95도 아닌 평균이다.

## GPT-3 training memory와 throughput

paper가 보고한 system-level 수치는 다음과 같다.

- full fine-tuning training VRAM: 1.2 TB
- LoRA training VRAM: 350 GB
- full fine-tuning throughput: 32.5 tokens/s per V100 GPU
- LoRA throughput: 43.1 tokens/s per V100 GPU
- 본문 요약: 약 25% training speedup

32.5에서 43.1의 단순 비율 증가는 약 32.6%이지만 paper 본문은 25%라고 표현한다. measurement 조건과 effective throughput 정의가 완전히 서술되지 않았으므로 원 수치를 함께 기록하는 편이 안전하다. 중요한 원인은 vast majority weight의 gradient를 계산하지 않는다는 점이다.

## 어떤 attention weight를 adapt할 것인가

GPT-3에서 trainable budget을 약 18M으로 맞춘 Table 5다.

| Target | Rank | WikiSQL | MNLI |
| --- | ---: | ---: | ---: |
| $W_q$ | 8 | 70.4 | 91.0 |
| $W_k$ | 8 | 70.0 | 90.8 |
| $W_v$ | 8 | 73.0 | 91.0 |
| $W_o$ | 8 | 73.2 | 91.3 |
| $W_q,W_k$ | 4 | 71.4 | 91.3 |
| $W_q,W_v$ | 4 | **73.7** | 91.3 |
| Q/K/V/O 전체 | 2 | **73.7** | **91.7** |

같은 budget이면 Q 하나의 rank를 키우는 것보다 여러 useful projection에 작은 rank를 분산하는 편이 좋았다. Q/V가 overall default로 선택되었지만 MNLI만 보면 네 projection 전체가 가장 높다.

## Rank ablation과 intrinsic rank

### GPT-3

| Target / Metric | $r=1$ | 2 | 4 | 8 | 64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q/V, WikiSQL | 73.4 | 73.3 | 73.7 | 73.8 | 73.5 |
| Q/V, MNLI | 91.3 | 91.4 | 91.3 | 91.6 | 91.4 |
| Q/K/V/O, WikiSQL | 74.1 | 73.7 | 74.0 | 74.0 | 73.9 |
| Q/K/V/O, MNLI | 91.2 | 91.7 | 91.7 | 91.5 | 91.4 |

$r=1$도 매우 경쟁적이고 $r=64$가 일관되게 낫지 않다. 단 paper도 다른 언어처럼 pretraining과 크게 다른 task에서는 작은 rank가 충분하지 않을 수 있다고 명시한다.

### GPT-2

GPT-2 Medium E2E에서 validation loss는 $r=1$의 1.23에서 $r=16$의 1.16까지 좋아지고 이후 거의 plateau다. BLEU는 $r=4$에서 70.38로 최고이며 $r=1024$는 69.37이다. 일부 hyperparameter가 $r=4$에 맞춰졌다는 confound가 있지만 rank 확대가 자동으로 성능 향상으로 이어지지 않음을 보여 준다.

### Subspace 분석

paper는 rank 8과 rank 64 adapter의 singular subspace overlap을

```math
\phi(A,B,i,j)=
\frac{\|U_A^{iT}U_B^j\|_F^2}{\min(i,j)}
\in[0,1]
```

로 측정한다. GPT-3 48번째 layer에서 Q/V update의 top direction은 rank 8과 64 사이 similarity가 0.5보다 크지만 나머지 direction overlap은 작았다. 서로 다른 random seed의 rank 64도 일부 top singular direction을 공유했고 random Gaussian pair와 달랐다.

$\Delta W_q$와 pretrained $W_q$의 관계를 본 Table 7에서 $r=4$일 때 $\|W_q\|_F=61.95$, $\|\Delta W_q\|_F=6.91$이고 $\Delta W$ subspace로 투영한 pretrained 성분 norm은 0.32였다. paper는 $6.91/0.32\approx21.5$를 task-specific direction amplification으로 해석한다. 이는 한 layer의 empirical analysis이며 모든 layer/task에 대한 low-rank theorem은 아니다.

## LoRA와 quantization을 혼동하지 않기

original LoRA paper는 base model quantization 방법이 아니다.

- $W_0$는 paper의 normal pretrained precision으로 저장된다.
- $A,B$도 ordinary trainable floating-point parameter다.
- 4-bit base, NF4, double quantization, paged optimizer는 이 paper에 없다.
- QLoRA는 훗날 quantized frozen base와 LoRA를 결합한 별도 방법이다.

low-bit base에 LoRA를 merge하면 $W_0$를 dequantize해 $BA$를 더한 뒤 다시 quantize해야 할 수 있다. 이 과정은 rounding error를 만들고 task switch가 비싸며 backend에 따라 merged zero-overhead가 깨진다. 다른 선택은 base를 quantized로 유지하고 LoRA branch를 FP16으로 unmerged 실행하는 것이지만 그때는 추가 kernel과 activation이 생긴다. 어느 쪽도 original paper의 `merged FP model` 주장과 동일하지 않다.

## 온디바이스 VLM 관점

### 적합한 사용

- vision encoder와 LLM을 고정하고 projector 또는 일부 attention Q/V만 적응
- 여러 사용자/도메인별 작은 adapter 저장
- training은 server/GPU에서 하고 mobile에는 merged artifact 배포
- 동일 base hash의 adapter만 허용
- adapter download/update를 base model update와 분리

### 주의할 점

- LoRA는 base LLM resident memory를 줄이지 않는다.
- visual token 수, TTFT, KV cache는 그대로다.
- device에서 adapter를 자주 merge하면 $BA$ 계산과 full weight rewrite가 발생한다.
- merged model 여러 개를 저장하면 task별 base copy가 다시 생길 수 있다.
- dynamic multi-adapter serving은 unmerged kernel이 필요하다.
- quantized base와 merge할 때 accuracy 및 physical packing을 다시 검증해야 한다.

mobile deployment에서 LoRA의 가장 큰 이점은 runtime acceleration보다 update distribution과 task customization storage다. latency가 목표라면 merged path는 baseline과 같고, base quantization, token reduction, KV optimization이 별도로 필요하다.

## 강점

1. 구현이 단순한 low-rank reparameterization으로 task parameter를 크게 줄인다.
2. merged inference가 기존 dense graph와 동일해 adapter layer latency를 피한다.
3. input sequence를 소비하지 않아 prefix method의 context penalty가 없다.
4. RoBERTa, DeBERTa, GPT-2, GPT-3 175B까지 규모와 task가 다양하다.
5. trainable count뿐 아니라 GPT-3 VRAM, throughput, adapter latency를 보고한다.
6. target projection과 rank ablation이 있고 update subspace도 분석한다.
7. base 한 개와 task adapter 여러 개를 분리해 operational storage를 크게 줄인다.

## 한계

1. base weight와 KV cache는 줄지 않는다.
2. training activation memory는 대부분 남는다.
3. zero additional latency는 사전 merge했을 때만 성립한다.
4. 서로 다른 task adapter를 같은 batch에서 merged 방식으로 처리하기 어렵다.
5. MLP, LayerNorm, bias에 대한 체계적 study는 future work다.
6. optimal target와 rank는 task에 따라 다르며 principled selection rule이 없다.
7. low-rank sufficiency는 empirical observation이지 보장된 theorem이 아니다.
8. original paper에는 4-bit base나 QLoRA가 없다.
9. on-device power, peak memory, TTFT, tokens/s, thermal benchmark는 없다.

## 재현 체크리스트

### Adapter 정의

- [ ] $W_0$를 freeze하고 optimizer parameter list에서 제외했다.
- [ ] $A\in\mathbb{R}^{r\times k}$, $B\in\mathbb{R}^{d\times r}$ shape를 확인했다.
- [ ] $A$는 random, $B$는 zero로 초기화했다.
- [ ] update에 $\alpha/r$를 적용했다.
- [ ] target module을 Q/V 또는 명시한 subset으로 고정했다.
- [ ] base model ID와 hash를 adapter metadata에 저장했다.

### Training memory

- [ ] trainable parameter, gradient, Adam m/v byte를 실제로 합산했다.
- [ ] base weight resident memory를 포함했다.
- [ ] activation peak와 optimizer-state 절감을 분리했다.
- [ ] frozen weight gradient가 생성되지 않는지 profiler로 확인했다.
- [ ] activation checkpointing 사용 여부를 별도 기록했다.
- [ ] full FT와 같은 batch/sequence/precision으로 throughput을 비교했다.

### Merged inference

- [ ] $W_0+(\alpha/r)BA$를 offline에 계산했다.
- [ ] graph에 LoRA branch가 남지 않았는지 operator trace로 확인했다.
- [ ] merged output과 unmerged output의 numerical tolerance를 검사했다.
- [ ] task switch 때 immutable base에서 다시 merge했다.
- [ ] merged file이 task별 full base copy를 만들지 않는 배포 전략을 세웠다.

### Unmerged inference

- [ ] 두 low-rank matmul과 add를 latency에 포함했다.
- [ ] batch 1 kernel-launch overhead를 측정했다.
- [ ] sample별 adapter batching 정책을 명시했다.
- [ ] adapter cache와 memory bandwidth를 측정했다.
- [ ] p50/p95를 merged baseline과 비교했다.

### LLM/VLM system

- [ ] KV cache가 LoRA 전후 같다는 것을 기록했다.
- [ ] visual token 수와 TTFT가 줄지 않는다고 명시했다.
- [ ] base quantization과 LoRA를 별도 ablation했다.
- [ ] quantized merge 후 accuracy를 다시 평가했다.
- [ ] model file, resident base, adapter, KV, activation peak를 분리했다.
- [ ] tokens/s, power, temperature, throttling을 기록했다.

## 최종 평가

LoRA의 핵심은 작은 adapter layer를 추가한 것이 아니라 dense weight의 `업데이트 좌표계`를 low-rank factor로 제한한 것이다. 이 때문에 training에서는 base gradient와 Adam state를 없애고, deployment에서는 factor를 base weight에 흡수해 원래 dense graph로 돌아갈 수 있다. GPT-3 175B에서 4.7M에서 37.7M trainable parameter로 full fine-tuning에 근접하거나 더 좋은 task score를 얻고, task checkpoint를 수십 MB로 줄인 결과는 이 아이디어의 실용성을 강하게 보여 준다.

그러나 효율의 범위를 정확히 말해야 한다. LoRA는 base model compression이 아니며 activation, KV cache, merged inference FLOPs를 줄이지 않는다. 350 GB GPT-3 base는 배포에 계속 필요하고, `추가 latency 없음`은 merge를 완료한 single-task dense weight에만 해당한다. unmerged dynamic adapters와 quantized base 결합은 별도의 kernel 및 accuracy 문제다.

온디바이스 연구에서는 LoRA를 `작은 비용으로 모델을 학습하고 배포하는 adaptation 기술`로 두고, runtime 효율은 W4A16 quantization, visual token compression, KV-cache 관리와 별도로 측정하는 것이 올바르다. training memory 표에는 base, activation, adapter optimizer state를 분리하고, inference 표에는 merged/unmerged, base precision, TTFT, tokens/s, KV cache를 분리해야 과장 없는 비교가 된다.

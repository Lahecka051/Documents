# 82. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models

## 논문 정보

- 제목: SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
- 저자: Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han
- 소속: MIT, NVIDIA
- 발표: ICML 2023, PMLR 202
- arXiv: [2211.10438](https://arxiv.org/abs/2211.10438)
- 코드: [mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant)
- 원본 파일: `82_SmoothQuant.pdf`

## 한눈에 보는 요약

SmoothQuant는 LLM의 weight와 activation을 모두 INT8로 계산하기 위한 post-training quantization 방법이다. LLM activation에는 일부 channel에 매우 큰 outlier가 반복적으로 나타나므로 일반적인 per-tensor W8A8 quantization이 무너진다. 반면 weight는 상대적으로 고르게 분포해 quantize하기 쉽다. SmoothQuant는 linear layer의 함숫값을 바꾸지 않는 channel별 scaling으로 activation outlier의 크기를 낮추고, 그 scale을 weight 쪽으로 미리 옮긴다.

핵심 변환은 다음 한 줄이다.

$$
Y=XW=(X\mathrm{diag}(s)^{-1})(\mathrm{diag}(s)W)=\hat X\hat W.
$$

$s$는 calibration data에서 얻은 activation channel max와 weight channel max로 계산한다.

$$
s_j=\frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}.
$$

$\alpha$가 클수록 activation의 quantization 난이도를 weight 쪽으로 더 많이 이동시킨다. 이 변환은 offline에 수행되고, $1/s$는 앞의 LayerNorm이나 linear parameter에 fuse할 수 있으므로 일반적인 경우 runtime scaling kernel을 추가하지 않는다.

논문이 보고한 대표 결과는 다음과 같다.

- OPT-175B에서 가장 공격적인 per-tensor static W8A8인 O3도 7개 zero-shot 평균 66.8%로 FP16 66.9%에 근접했다.
- OPT, BLOOM, GLM, MT-NLG, Llama 계열에서 W8A8 정확도를 검증했다.
- NVIDIA A100 80GB에서 PyTorch 구현은 최대 1.51배 speedup, 1.96배 memory saving을 보였다.
- FasterTransformer 통합은 최대 1.56배 speedup과 거의 2배 memory saving을 보였다.
- MT-NLG 530B를 FP16 16 GPU 대신 INT8 8 GPU 한 node에서 비슷한 latency로 실행했다.

중요한 점은 SmoothQuant가 weight-only quantization이 아니라 compute-heavy GEMM과 attention BMM의 양쪽 operand를 INT8로 만드는 W8A8 방법이라는 것이다.

## 문제의 핵심: LLM activation outlier

### Uniform symmetric quantization

$N$ bit signed integer quantization에서 floating tensor $X$를 단순화해 다음처럼 표현할 수 있다.

$$
\bar X=\mathrm{round}(X/\Delta),\qquad
\Delta=\frac{\max|X|}{2^{N-1}-1}.
$$

INT8이면 양의 최대 code는 127이다. dequantization은 대략 $X\approx\Delta\bar X$다. tensor 안에 크기가 매우 큰 값 하나가 있으면 $\Delta$가 커지고, 대부분의 작은 값은 소수의 code에 몰린다.

예를 들어 한 channel의 일반 값이 $[-1,1]$인데 같은 tensor에 100짜리 outlier가 있다면 $\Delta\approx100/127=0.787$이다. $[-1,1]$ 구간은 대략 $-1,0,1$ 정도의 code만 실질적으로 사용한다. 논문은 LLM activation outlier가 일반 값보다 약 100배 크며, 소수의 고정 channel에 token 전반에 걸쳐 반복된다고 관찰한다.

### 왜 per-token으로 해결되지 않는가

linear layer를 다음 shape로 쓰자.

$$
Y=XW,qquad
X\in\mathbb{R}^{T\times C_i},
W\in\mathbb{R}^{C_i\times C_o},
Y\in\mathbb{R}^{T\times C_o}.
$$

- $T$: batch와 sequence를 펼친 token 수
- $C_i$: input channel, GEMM reduction axis
- $C_o$: output channel

outlier가 특정 token이 아니라 특정 $C_i$ channel에 지속적으로 있다면 token마다 scale을 달리하는 per-token quantization은 근본 해결이 아니다. 반면 activation per-channel scale은 정확도를 거의 복원한다. 논문의 OPT 평균 결과는 다음과 같다.

| OPT 크기 | 6.7B | 13B | 30B | 66B | 175B |
|---|---:|---:|---:|---:|---:|
| FP16 | 64.9 | 65.6 | 67.9 | 69.5 | 71.6 |
| INT8 per-tensor activation | 39.9 | 33.0 | 32.8 | 33.1 | 32.3 |
| INT8 per-token activation | 42.5 | 33.0 | 33.1 | 32.9 | 31.7 |
| INT8 per-channel activation | 64.8 | 65.6 | 68.0 | 69.4 | 71.4 |

하지만 activation per-channel scale은 GEMM의 reduction axis $C_i$에 놓인다. 일반적인 고성능 INT8 GEMM은 정수 dot product를 끝낸 후 바깥 축의 scale을 적용하는 구조다.

$$
Y=\mathrm{diag}(\Delta_X)
(\bar X\bar W)
\mathrm{diag}(\Delta_W).
$$

activation의 token별 scale은 왼쪽 바깥 축 $T$, weight의 output-channel scale은 오른쪽 바깥 축 $C_o$에 있으므로 accumulator 결과에 적용할 수 있다. 반면 $C_i$별 activation scale은 sum 내부 항마다 달라져 INT8 MMA 사이에 scale 연산을 넣어야 한다. 이는 Tensor Core 같은 고처리량 kernel의 실행 구조와 맞지 않는다.

SmoothQuant의 목표는 per-channel activation quantization의 효과를 얻되 runtime GEMM에는 hardware-friendly per-tensor 또는 per-token scale만 남기는 것이다.

## 수학적으로 동등한 smoothing

### Channel scaling

$s\in\mathbb{R}^{C_i}$를 양수 channel scale이라고 하자.

$$
\hat X=X\mathrm{diag}(s)^{-1},\qquad
\hat W=\mathrm{diag}(s)W.
$$

그러면

$$
\hat X\hat W
=X\mathrm{diag}(s)^{-1}\mathrm{diag}(s)W
=XW.
$$

즉, floating-point에서 수학적 output은 동일하다. $s_j$가 큰 activation channel은 $X_j/s_j$로 작아지고, 대신 weight의 대응 input row $W_{j,:}$가 $s_j$배 커진다. quantization error의 위치만 바뀐다.

### Migration strength

activation channel max를

$$
a_j=\max_{\text{calibration tokens}}|X_j|
$$

weight input-channel max를

$$
w_j=\max_{k}|W_{j,k}|
$$

라고 하자. SmoothQuant는

$$
s_j=\frac{a_j^\alpha}{w_j^{1-\alpha}}
$$

를 쓴다.

- $\alpha=0$: weight 분포를 우선해 난이도를 activation 쪽에 둔다.
- $\alpha=1$: activation channel max를 완전히 평탄화하고 난이도를 weight에 둔다.
- $\alpha=0.5$: 두 tensor의 대응 channel max를 기하평균 관점에서 균형 있게 맞춘다.

$\alpha=0.5$라면

$$
\max|\hat X_j|=\frac{a_j}{\sqrt{a_j/w_j}}=\sqrt{a_jw_j},
$$

$$
\max|\hat W_{j,:}|=w_j\sqrt{a_j/w_j}=\sqrt{a_jw_j}.
$$

두 쪽의 최대 크기가 같아진다. 이 등식이 SmoothQuant intuition의 핵심이다.

논문은 OPT와 BLOOM에 $\alpha=0.5$, activation outlier가 더 심한 GLM-130B에 $\alpha=0.75$를 사용했다. OPT-175B ablation에서는 0.4보다 작으면 activation quantization이 어려워지고, 0.6보다 크면 weight가 어려워졌다. 최근 모델 표에서는 Llama-2 7B/13B에 0.85, 70B에 0.9, Falcon에 0.6-0.7, Mistral/Mixtral에 0.8을 사용했다. 따라서 $\alpha$는 보편 상수가 아니다.

## Offline fusion과 runtime graph

### 앞 연산에 $1/s$를 흡수

linear 입력 $X$가 LayerNorm output이라면 LayerNorm affine parameter의 scale과 bias에 $1/s$를 반영할 수 있다. 이전 linear output이라면 이전 layer의 output-channel weight와 bias를 $1/s$로 조정할 수 있다. 그 결과 runtime에는 이미 smoothed activation이 나오며 별도 element-wise division kernel이 없다.

residual add에서 직접 온 입력처럼 단일 predecessor에 흡수하기 어려운 경우에는 residual branch에 추가 scaling이 필요할 수 있다. 따라서 모델 rewrite 시 다음을 확인해야 한다.

- 해당 activation의 모든 producer에 같은 scale이 적용되는가
- residual 두 branch의 값이 같은 좌표계에서 더해지는가
- tied weight나 shared module을 한쪽 path만 수정하지 않았는가
- LayerNorm/RMSNorm의 affine parameter 유무가 고려되었는가

### Precision mapping

논문의 Transformer block mapping은 compute-heavy operation을 INT8로 둔다.

- Q/K/V projection과 output projection: W8A8 INT8 GEMM
- FFN linear layer: W8A8 INT8 GEMM
- attention의 QK와 probability-V: INT8 BMM
- LayerNorm, Softmax, ReLU 같은 가벼운 element-wise operation: FP16

즉, 모델 전체를 무조건 INT8 tensor로 유지하는 방식이 아니다. INT8 kernel 사이에는 FP16 activation이 존재할 수 있으며, quantize/dequantize와 scale fusion의 배치가 성능을 좌우한다.

## Quantization scheme O1, O2, O3

| 방법 | weight | activation |
|---|---|---|
| naive W8A8 | per-tensor | per-tensor dynamic |
| ZeroQuant | group-wise | per-token dynamic |
| LLM.int8() | per-channel | per-token dynamic + FP16 outlier |
| Outlier Suppression | per-tensor | per-tensor static |
| SmoothQuant-O1 | per-tensor | per-token dynamic |
| SmoothQuant-O2 | per-tensor | per-tensor dynamic |
| SmoothQuant-O3 | per-tensor | per-tensor static |

O1에서 O3로 갈수록 scale granularity가 거칠고 runtime 통계 계산이 줄어 latency가 낮아진다. 대신 calibration distribution과 실제 입력이 다르면 static per-tensor O3의 오차가 커질 수 있다.

Table 11의 batch 4 GPU latency는 이를 직접 보여 준다.

| 모델/길이 | FP16 | LLM.int8() | SQ-O1 | SQ-O2 | SQ-O3 |
|---|---:|---:|---:|---:|---:|
| OPT-13B, 256 | 152.6 ms | 237.1 | 124.5 | 120.5 | 112.1 |
| OPT-13B, 512 | 296.3 ms | 371.5 | 243.3 | 235.1 | 223.1 |
| OPT-30B, 256 | 343.0 ms | 387.9 | 246.7 | 240.2 | 227.6 |
| OPT-30B, 512 | 659.9 ms | 654.9 | 490.7 | 478.3 | 458.4 |

mixed FP16 outlier path를 가진 LLM.int8()는 정확하지만 kernel 분해와 mixed-precision overhead 때문에 대체로 FP16보다 느리다. SmoothQuant의 장점은 모든 heavy matmul을 하나의 INT8 path로 보낼 수 있다는 점이다.

## Calibration과 변환 절차

논문은 Pile pretraining dataset에서 무작위 512개 sentence를 뽑아 activation channel 통계를 수집했다. 이 통계로 smoothing factor와 static quantization scale을 한 번 계산하고 모든 downstream task에 같은 quantized model을 사용했다.

모델 변환 pseudocode는 다음과 같이 정리할 수 있다.

```python
def calibrate(model, calibration_batches):
    # Each entry stores max abs over batch and token axes.
    act_max = {}
    register_pre_hooks_on_quantized_linears(model, act_max)
    for tokens in calibration_batches:
        model(tokens)
    remove_hooks()
    return act_max

def smooth_linear(prev, linear, amax, alpha):
    # linear.weight convention: [Cout, Cin]
    # wmax over Cout gives one value per Cin.
    wmax = abs(linear.weight).amax(dim=0)
    s = amax.pow(alpha) / wmax.pow(1 - alpha)
    s = clamp_min(s, epsilon)

    # Make previous op emit X / s.
    fuse_inverse_scale_into_previous_op(prev, 1 / s)

    # Compensate this linear: W_hat = diag(s) W.
    linear.weight *= s[None, :]
    return s

def quantize_w8a8(model, scheme="O3"):
    for linear in target_linears(model):
        linear.weight_int8, linear.weight_scale = quantize_weight(linear.weight)
        replace_with_int8_gemm(linear, activation_scheme=scheme)
    replace_attention_bmm_with_int8(model, scheme)
```

실제 구현에서는 zero max channel, epsilon, weight layout, tensor parallel shard, bias, tied module, QKV fused weight를 처리해야 한다. calibration hook이 보는 값도 반드시 smoothing 전 원래 activation이어야 한다.

## 정확도 실험

### OPT-175B 세부 결과

| 방법 | LAMBADA | HellaSwag | PIQA | WinoGrande | OBQA | RTE | COPA | 평균 | WikiText PPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 74.7 | 59.3 | 79.7 | 72.6 | 34.0 | 59.9 | 88.0 | 66.9 | 10.99 |
| naive W8A8 | 0.0 | 25.6 | 53.4 | 50.3 | 14.0 | 49.5 | 56.0 | 35.5 | 93080 |
| ZeroQuant | 0.0 | 26.0 | 51.7 | 49.3 | 17.8 | 50.9 | 55.0 | 35.8 | 84648 |
| LLM.int8() | 74.7 | 59.2 | 79.7 | 72.1 | 34.2 | 60.3 | 87.0 | 66.7 | 11.10 |
| Outlier Suppression | 0.0 | 25.8 | 52.5 | 48.6 | 16.6 | 53.4 | 55.0 | 36.0 | 96151 |
| SmoothQuant-O1 | 74.7 | 59.2 | 79.7 | 71.2 | 33.4 | 58.1 | 89.0 | 66.5 | 11.11 |
| SmoothQuant-O2 | 75.0 | 59.0 | 79.2 | 71.2 | 33.0 | 59.6 | 88.0 | 66.4 | 11.14 |
| SmoothQuant-O3 | 74.6 | 58.9 | 79.7 | 71.2 | 33.4 | 59.9 | 90.0 | 66.8 | 11.17 |

naive W8A8와 ZeroQuant가 사실상 붕괴하는데 O3는 평균 0.1% 차이다. LLM.int8()도 정확하지만 뒤의 latency 실험에서 혼합 경로 overhead가 드러난다.

### 100B 이상 모델

| 방법 | OPT-175B | BLOOM-176B | GLM-130B |
|---|---:|---:|---:|
| FP16 | 71.6 | 68.2 | 73.8 |
| naive W8A8 | 32.3 | 64.2 | 26.9 |
| ZeroQuant | 31.7 | 67.4 | 26.7 |
| LLM.int8() | 71.4 | 68.0 | 73.8 |
| Outlier Suppression | 31.7 | 54.1 | 63.5 |
| SmoothQuant-O1 | 71.2 | 68.3 | 73.7 |
| SmoothQuant-O2 | 71.1 | 68.4 | 72.5 |
| SmoothQuant-O3 | 71.1 | 67.4 | 72.8 |

GLM은 outlier가 더 심해 O1이 가장 안전하며, BLOOM O3는 FP16보다 0.8% 낮다. 이 표는 architecture와 training recipe에 따라 quantization 난이도가 다름을 보여 준다.

### Llama와 최근 architecture

Llama-1의 WikiText-2 PPL은 다음과 같다. sequence length는 512, $\alpha=0.8$, activation은 per-token이다.

| 모델 | FP16 PPL | W8A8 SmoothQuant PPL |
|---|---:|---:|
| Llama-7B | 11.51 | 11.56 |
| Llama-13B | 10.05 | 10.08 |
| Llama-30B | 7.53 | 7.56 |
| Llama-65B | 6.17 | 6.20 |

개정판은 Llama-2, Falcon, Mistral, Mixtral에도 sequence length 2048에서 작은 PPL 차이를 보고한다. 예를 들어 Mistral-7B는 5.253에서 5.277, Mixtral-8x7B는 3.842에서 3.893이다. 이 결과는 MoE에도 channel smoothing을 적용할 수 있음을 보여 주지만 모든 downstream task를 검증한 것은 아니다.

## 성능과 memory 결과

### 측정 환경을 먼저 고정

모든 주요 performance 실험은 NVIDIA A100 80GB server에서 수행했다. PyTorch context-stage는 batch 4, 단일 GPU에서 OPT-6.7B/13B/30B를 측정했다. FasterTransformer는 tensor parallelism을 사용해 OPT-13B부터 175B까지 측정했다. 따라서 모바일 CPU/NPU latency로 직접 일반화하면 안 된다.

### PyTorch context stage

Figure 8에서 SmoothQuant-O3는 OPT-30B, sequence 256에서 FP16 343 ms 대비 228 ms 수준으로 최대 1.51배 빨랐다. memory는 OPT-30B에서 sequence에 따라 FP16 약 56.6-63.1 GB, SmoothQuant 약 28.9-33.3 GB로 거의 절반이다. LLM.int8()는 memory는 절감하지만 latency가 대부분 FP16보다 길었다.

### FasterTransformer context stage

단일 GPU의 OPT-13B/30B에서 최대 1.56배 speedup을 보고했다. 더 큰 모델에서는 GPU 수 자체를 절반으로 줄였다.

- OPT-66B: FP16 2 GPU와 SmoothQuant 1 GPU가 비슷한 latency
- OPT-175B: FP16 8 GPU보다 SmoothQuant 4 GPU가 약간 느리지만 같은 범위의 latency
- memory footprint: 전반적으로 약 절반

이는 throughput만이 아니라 tensor parallel communication과 server 배치 비용을 줄이는 효과가 있음을 의미한다.

### Autoregressive decoding

| 모델 | batch | seq | FP16 latency | SQ latency | speedup | FP16 memory | SQ memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| OPT-30B, 1 GPU | 1 | 512 | 422 ms | 314 | 1.35배 | 57 GB | 30 GB |
| OPT-30B, 1 GPU | 1 | 1024 | 559 | 440 | 1.27배 | 58 | 31 |
| OPT-30B, 1 GPU | 16 | 512 | 2488 | 1753 | 1.42배 | 69 | 44 |
| OPT-30B, 1 GPU | 16 | 1024 | OOM | 3947 | 해당 없음 | OOM | 61 |
| OPT-175B, 8 GPU | 1 | 512 | 426 | 359 | 1.19배 | 44 | 23 |
| OPT-175B, 8 GPU | 16 | 1024 | 4133 | 3231 | 1.28배 | 56 | 37 |

여기서 memory 수치는 model, activation, workspace를 포함한 측정 peak다. SmoothQuant가 KV cache를 항상 정확히 2배 압축한다고 해석해서는 안 된다. paper의 전체 memory 절감은 INT8 weight와 compute tensor가 지배하는 설정의 결과다.

### MT-NLG 530B

| sequence | FP16 GPU | FP16 latency/memory | INT8 GPU | INT8 latency/memory |
|---:|---:|---:|---:|---:|
| 128 | 16 | 232 ms / 1040 GB | 8 | 253 ms / 527 GB |
| 256 | 16 | 451 ms / 1054 GB | 8 | 434 ms / 533 GB |
| 512 | 16 | 838 ms / 1068 GB | 8 | 839 ms / 545 GB |
| 1024 | 16 | 1707 ms / 1095 GB | 8 | 1689 ms / 570 GB |

정확도 평균은 FP16과 INT8 모두 73.1%였다. INT8은 8개의 A100 80GB, 즉 한 node 안에 530B model을 넣었다.

## Weight와 activation memory 손계산

이 절은 논문 측정값이 아니라 datatype과 tensor shape에 따른 별도 계산이다.

### Weight storage

$P$개 parameter를 FP16으로 저장하면 $2P$ bytes, INT8이면 $P$ bytes다. 십진 GB로 단순 계산하면 다음과 같다.

| parameter | FP16 weight만 | INT8 weight만 |
|---:|---:|---:|
| 6.7B | 13.4 GB | 6.7 GB |
| 30B | 60 GB | 30 GB |
| 175B | 350 GB | 175 GB |
| 530B | 1060 GB | 530 GB |

실제 runtime에는 quantization scale, embedding sharing, padding, tensor parallel shard, allocator, KV cache가 추가된다. 그래서 표의 measured peak는 이 단순 weight 계산과 정확히 같지 않다.

### Linear activation

$X$가 $[T,C_i]$라면 FP16 input은 $2TC_i$ bytes, INT8 input은 $TC_i$ bytes다. $W$도 절반이다. 그러나 INT8 GEMM accumulator는 보통 INT32이므로 $[T,C_o]$ accumulator만 보면 $4TC_o$ bytes가 필요할 수 있다. 이후 FP16 output으로 requantize하면 $2TC_o$ bytes다. kernel fusion과 tiling에 따라 전체 accumulator를 global memory에 쓰지 않을 수 있으므로 이것은 upper-level tensor 크기 비교일 뿐 peak workspace 공식은 아니다.

예를 들어 $T=2048$, $C_i=C_o=4096$이면 input은 FP16 16 MiB, INT8 8 MiB다. weight는 FP16 32 MiB, INT8 16 MiB다. 전체 INT32 output을 materialize하면 32 MiB지만 fused kernel이 tile accumulator를 register에 유지하면 global traffic은 더 작다.

### Scale metadata

per-tensor scale은 tensor당 상수 하나라 미미하다. per-token activation scale은 $T$개, per-channel weight scale은 $C_o$개다. metadata 용량보다 더 중요한 비용은 dynamic max reduction, quantize kernel, scale load, GEMM fusion 가능성이다. O3가 O1보다 빠른 이유도 scale 저장 bytes보다는 runtime reduction을 제거하기 때문이다.

## 논문 증거와 이 리뷰의 해석

### 논문이 직접 검증한 것

- calibration 512 sentence로 offline smoothing했다.
- A100에서 PyTorch와 FasterTransformer의 end-to-end latency와 peak memory를 측정했다.
- OPT-175B, BLOOM-176B, GLM-130B, MT-NLG 530B의 정확도를 비교했다.
- O1/O2/O3 granularity와 $\alpha$ ablation을 제공했다.
- compute-heavy linear와 BMM을 INT8로, light op를 FP16으로 배치했다.

### 별도 해석 또는 주의점

- weight/activation byte 계산은 dtype 기반 손계산이며 measured peak가 아니다.
- 모바일 NPU에서 이득이 날 것이라는 평가는 INT8 GEMM/BMM과 fusion을 장치가 지원한다는 조건부 추론이다.
- calibration domain shift가 static O3를 해칠 수 있다는 판단은 BLOOM/GLM 결과와 static quantization 원리에서 나온다.
- 모든 KV cache가 INT8로 저장된다고 논문이 보장하지 않으므로 별도 구현 확인이 필요하다.

## 온디바이스 관점

### 유리한 조건

- target이 W8A8 GEMM과 attention BMM을 native로 지원한다.
- quantize/dequantize, scale, LayerNorm이 compiler에 의해 fuse된다.
- model이 weight bandwidth 또는 memory capacity에 제한된다.
- calibration 입력이 실제 앱의 prompt/image-text distribution을 잘 대표한다.
- static O3 graph가 delegate 안에서 end-to-end로 유지된다.

### 실패하기 쉬운 조건

- NPU가 weight INT8만 지원하고 activation/BMM은 FP16 fallback한다.
- per-token dynamic quantization이 CPU reduction으로 실행된다.
- residual branch scaling 때문에 graph partition이 쪼개진다.
- INT8 accumulator/output requantization이 별도 large tensor를 materialize한다.
- short sequence, batch=1에서 quantization launch overhead가 GEMM 절감보다 크다.
- outlier pattern이 calibration과 실제 입력에서 크게 다르다.

### 실험 계획

1. FP16, naive W8A8, SQ-O1, SQ-O3를 동일 checkpoint에서 만든다.
2. 각 linear/BMM의 delegate assignment와 fallback을 기록한다.
3. batch=1에서 prompt length를 32, 128, 512처럼 나누고 prefill과 decode를 분리한다.
4. TTFT, per-token latency, tokens/s, p50/p95, peak RSS를 기록한다.
5. weight file, mapped memory, KV cache, temporary workspace를 분리 측정한다.
6. 일반 calibration과 실제 앱 calibration을 비교한다.
7. $\alpha$를 0.4-0.9로 sweep하고 layer별 clipping error와 downstream accuracy를 본다.
8. 10분 이상 지속 generation에서 전력, 온도, throttling을 확인한다.

## 장점과 기여

1. LLM W8A8의 병목을 activation outlier와 GEMM inner-axis scale의 충돌로 명확히 정리했다.
2. retraining 없이 정확한 함수 동등 변환을 사용했다.
3. activation과 weight 사이 quantization 난이도를 $\alpha$ 하나로 제어했다.
4. offline fusion으로 runtime scaling overhead를 최소화했다.
5. per-tensor static까지 포함한 세 efficiency level을 제공했다.
6. accuracy뿐 아니라 실제 framework latency와 memory를 측정했다.
7. 수백 billion parameter 모델을 더 적은 GPU로 serving할 수 있음을 보였다.

## 한계와 비판적 관점

1. 성능 측정이 A100 중심이라 edge CPU/GPU/NPU에 대한 직접 증거가 없다.
2. calibration과 $\alpha$ 선택이 model family에 따라 달라 완전한 one-click 방법은 아니다.
3. mathematical equivalence는 floating transform에 대한 것이며 quantization 이후 output이 동일하다는 뜻은 아니다.
4. residual, shared module, fused QKV가 복잡한 모델은 rewrite 오류 가능성이 있다.
5. O3 static scale은 distribution shift에 취약하다.
6. INT8 kernel이 없거나 mixed precision graph가 분할되면 speedup이 사라질 수 있다.
7. W8A8은 W4A16 weight-only보다 weight compression 비율이 낮다. 목표가 capacity인지 compute acceleration인지에 따라 선택이 달라진다.
8. perplexity와 일부 zero-shot 평균이 실제 instruction following, safety, long-context 품질을 완전히 대변하지 않는다.

## 구현 체크리스트

- [ ] activation max를 batch와 token 축으로 모으고 channel 축은 유지했는가
- [ ] weight layout에서 input channel 축을 정확히 찾았는가
- [ ] zero max channel에 epsilon을 적용했는가
- [ ] $s$와 $1/s$가 올바른 두 연산에 상쇄되도록 적용됐는가
- [ ] bias와 LayerNorm affine parameter도 함께 변환했는가
- [ ] residual branch의 좌표계가 일치하는가
- [ ] tied/shared weight를 중복 변환하지 않았는가
- [ ] tensor parallel shard별 통계와 scale 축이 일치하는가
- [ ] QKV fused weight의 channel axis를 올바르게 처리했는가
- [ ] calibration data가 실제 deployment domain을 대표하는가
- [ ] O1/O2/O3별 dynamic reduction overhead를 측정했는가
- [ ] 모든 heavy GEMM/BMM이 실제 INT8 kernel로 실행되는가
- [ ] LayerNorm/Softmax를 FP16으로 유지했는가
- [ ] prefill과 decode 성능을 분리했는가
- [ ] accuracy, model size, peak memory, TTFT, tokens/s를 함께 기록했는가

## 후속 연구와의 연결

SmoothQuant는 GPTQ/AWQ 같은 weight-only PTQ와 목적이 다르다. weight-only W4A16은 model capacity와 bandwidth를 더 크게 줄일 수 있지만 activation이 FP16이어서 compute path가 다르다. SmoothQuant W8A8은 activation까지 INT8로 만들어 GEMM throughput을 높이려 한다. 실제 온디바이스에서는 두 축을 조합하거나 layer별 mixed precision을 선택할 수 있다.

또한 이 논문의 channel equalization 아이디어는 VLM의 vision encoder, projector, text decoder에 각각 다른 $\alpha$와 calibration domain을 적용하는 연구로 확장할 수 있다. 특히 visual token은 text token과 activation distribution이 다르므로 하나의 calibration set으로 전체 multimodal model을 quantize하면 위험하다.

## 최종 평가

SmoothQuant의 핵심은 outlier를 제거하는 것이 아니라, hardware가 처리하기 좋은 위치로 옮기는 것이다. 함수 동등성을 유지한 channel scaling 덕분에 어려운 activation per-channel quantization을 runtime에서 수행하지 않고도 INT8 GEMM 친화적인 분포를 만든다. 이 단순한 아이디어가 175B와 530B 규모에서 실제 latency와 GPU 수 절감으로 이어졌다는 점이 논문의 강점이다.

온디바이스 적용에서는 논문의 accuracy recipe만 복제해서는 충분하지 않다. target compiler가 W8A8 GEMM과 BMM, scale fusion, residual rewrite를 실제로 지원하는지 확인해야 한다. 지원이 완전하면 memory와 compute를 동시에 줄이는 강력한 PTQ가 되지만, 한 operation이라도 CPU/FP16으로 fallback하면 이론적 장점이 쉽게 사라진다. 따라서 최종 평가는 quantized accuracy와 함께 kernel trace, TTFT, decode latency, peak memory, 전력까지 포함해야 한다.

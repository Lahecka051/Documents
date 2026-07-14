# 78. AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration

## 논문 정보

- 원본 파일: `78_AWQ.pdf`
- 제목: AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
- 저자: Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han
- 소속: MIT, Shanghai Jiao Tong University, NVIDIA, Tsinghua University, MIT-IBM Watson AI Lab, UMass Amherst
- 발표: MLSys 2024, Best Paper Award
- 제공된 PDF: arXiv:2306.00978v6, 2026년 4월 25일
- 논문 링크: https://arxiv.org/abs/2306.00978
- 공개 코드: https://github.com/mit-han-lab/llm-awq
- 핵심 키워드: weight-only quantization, W4A16, activation-aware scaling, group quantization, post-training quantization, TinyChat, on-device LLM

## 한눈에 보는 요약

AWQ는 LLM의 모든 weight가 출력 품질에 똑같이 중요하지 않으며, 큰 입력 activation과 곱해지는 input channel의 weight가 특히 중요하다는 관찰에서 출발한다. 중요한 weight 0.1%에서 1%를 FP16으로 남기면 3-bit quantization의 perplexity가 크게 회복되지만, 불규칙한 mixed precision은 실제 하드웨어에서 효율적이지 않다. AWQ는 이 관찰을 다음과 같은 등가 변환으로 바꾼다.

```math
WX=(W\operatorname{diag}(s))(\operatorname{diag}(s)^{-1}X).
```

중요한 input channel의 weight를 $s>1$로 키운 뒤 activation을 $1/s$로 줄이면 부동소수점 함수는 변하지 않는다. 그러나 확대된 weight를 group-wise low-bit로 quantize할 때 그 channel의 상대적인 rounding error가 작아진다. 논문은 calibration activation의 channel별 평균 절댓값 $s_X$를 측정하고,

```math
s=s_X^\alpha,\qquad \alpha\in[0,1]
```

라는 작은 탐색 공간에서 output reconstruction error가 가장 작은 $\alpha$를 고른다. backpropagation이나 layer별 weight reconstruction은 하지 않는다. 논문의 기본 정확도 설정은 group size 128의 INT4 또는 INT3 weight-only quantization이며 activation은 FP16이다.

AWQ 알고리즘과 TinyChat 시스템은 구분해서 읽어야 한다.

- AWQ는 low-bit weight를 만드는 calibration 및 scaling 방법이다.
- TinyChat은 4-bit packed weight를 저장하고, GEMM 안에서 FP16으로 즉시 dequantize하며, device별 packing과 kernel fusion으로 실제 속도를 얻는 inference runtime이다.
- 논문의 INT3 표는 정확도 결과다. TinyChat의 실측 가속 표는 W4A16 결과이며 INT3 실측 kernel 결과가 아니다.
- W4A16은 weight traffic을 줄이지만 FP16 activation과 KV cache를 압축하지 않는다.

TinyChat은 desktop RTX 4090과 Jetson Orin에서 Hugging Face FP16 구현 대비 평균 3배가 넘는 속도를 보고한다. 그러나 이 수치는 AWQ 파일만 저장하면 자동으로 생기는 것이 아니다. packed format, fused dequantization, MM/MV kernel, KV cache 관리, LayerNorm 및 attention fusion을 모두 구현한 시스템 결과다.

## 문제 설정과 범위

### 왜 weight-only인가

autoregressive LLM inference에는 서로 다른 두 단계가 있다.

1. prefill 또는 context 단계는 여러 prompt token을 한꺼번에 처리해 matrix-matrix 연산 비중이 높다.
2. decode 또는 generation 단계는 매번 새 token 하나를 처리해 matrix-vector 연산 비중이 높고 weight를 반복해서 읽는다.

논문 Figure 3은 Llama-2-7B, RTX 4090, batch 1 예에서 200-token prompt 처리는 10 ms인데 20-token generation은 310 ms라고 제시한다. 같은 그림의 roofline 해석에서 FP16 decode의 arithmetic intensity는 약 1 FLOP/Byte이고, RTX 4090의 계산량 대 bandwidth 경계인 165 FLOP/Byte보다 훨씬 낮다. 따라서 decode는 계산량보다 weight memory traffic에 묶인다. W4A16으로 weight byte를 약 4분의 1로 줄이면 이상적인 arithmetic intensity는 약 4 FLOP/Byte로 올라간다.

Figure 3(c)의 한 layer 예에서도 paper가 집계한 traffic은 다음처럼 weight 지배적이다.

| 구성 요소 | Weight access | Activation access | 비율 |
| --- | ---: | ---: | ---: |
| Attention | 134 MB | 1.7 MB | 79배 |
| FFN | 271 MB | 0.2 MB | 1700배 |

이는 batch 1 decode에서 weight-only quantization을 택한 이유를 지지한다. 긴 prompt의 prefill, 큰 batch, 매우 긴 KV cache에서는 병목 비율이 달라질 수 있으므로 모든 상황에서 4배 빨라진다는 뜻은 아니다.

### AWQ가 quantize하는 것과 하지 않는 것

논문의 기본 설정을 정확히 적으면 `W4A16` 또는 `W3A16`이다.

- linear layer weight: group-wise INT4 또는 INT3
- activation: FP16
- accumulation 및 실제 TinyChat 계산: low-bit weight를 on-the-fly FP16으로 복원한 뒤 FP16 연산
- LayerNorm, Softmax, positional embedding: low-bit weight quantization 대상이 아님
- embedding, 일부 head 등 구현별 예외: backend가 별도 처리하며 논문 표만으로 모두 low-bit라고 단정할 수 없음
- KV cache: FP16이며 AWQ가 줄이지 않음

즉 AWQ는 integer-only inference가 아니다. 저장 및 DRAM load는 low-bit이고, 연산 loop 안에서 FP16으로 변환한다. `INT4 모델`이라는 이름만 보고 activation도 INT4이거나 native INT4 x FP16 instruction을 쓴다고 해석하면 잘못이다.

## 핵심 관찰: weight saliency는 activation으로 찾는다

weight matrix를 $W\in\mathbb{R}^{d_{out}\times d_{in}}$, calibration input을 $X\in\mathbb{R}^{d_{in}\times T}$라고 두자. $j$번째 input channel의 weight column $W_{:,j}$는 activation row $X_{j,:}$와 곱해진다. $|X_{j,:}|$가 지속적으로 크면 같은 weight error라도 output error에 더 크게 반영된다.

논문은 OPT 모델을 INT3-g128로 quantize하면서 일부 channel을 FP16으로 남기는 실험을 했다.

| Model | FP16 PPL | RTN | Activation 기준 0.1% | Activation 기준 1% | Weight 기준 1% | Random 1% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OPT-1.3B | 14.62 | 119.00 | 25.03 | 16.91 | 98.55 | 109.38 |
| OPT-6.7B | 10.86 | 23.54 | 11.58 | 11.39 | 22.37 | 24.23 |
| OPT-13B | 10.13 | 46.04 | 10.51 | 10.43 | 48.96 | 42.00 |

큰 weight norm으로 고른 channel은 random과 비슷한 반면 activation 크기로 고르면 perplexity가 거의 FP16까지 회복된다. Figure 2의 OPT-6.7B 사례도 RTN 43.2에서 1% FP16 보존 후 13.0으로 개선된다. 이 실험은 `중요도를 activation 통계로 찾는다`는 핵심 관찰을 뒷받침한다.

하지만 1% FP16 channel을 그대로 두는 방식은 두 precision의 gather, 별도 kernel, irregular memory access가 필요하다. 총 bit 수가 거의 늘지 않아도 latency는 나빠질 수 있다. AWQ는 중요한 channel을 선택적으로 FP16에 남기지 않고 모든 weight를 동일한 low-bit format에 넣기 위해 scaling을 사용한다.

## Quantizer와 activation-aware scaling

### 기본 group-wise quantizer

한 group의 weight $w$를 $N$-bit로 quantize하는 논문의 식은 다음과 같다.

```math
Q(w)=\Delta\cdot\operatorname{Round}\left(\frac{w}{\Delta}\right),
\qquad
\Delta=\frac{\max(|w|)}{2^{N-1}}.
```

실제 구현에는 representable integer 범위로 clamp하는 단계가 필요하다. 논문 실험은 별도 언급이 없으면 input dimension 방향으로 128개 weight마다 scale을 두는 `g128`을 사용한다. 따라서 이 quantizer의 granularity는 다음처럼 구분해야 한다.

- AWQ transformation의 $s$: weight matrix의 input channel마다 하나
- low-bit quantizer의 $\Delta$: 각 output row 안의 128-weight group마다 하나
- $\alpha$: 한 layer 또는 논문 구현 단위에서 grid search하는 scalar
- activation: FP16이므로 activation quantization scale은 없음

per-channel scaling과 group-wise quantization을 같은 scale로 혼동하면 안 된다. $s$는 등가 변환용이고 $\Delta$는 low-bit code를 복원하는 quantization scale이다.

### scaling이 salient weight를 보호하는 이유

한 weight 원소 $w$와 입력 $x$에 대해 원래 quantization error는 대략

```math
\operatorname{Err}(Q(w)x)
=\Delta\cdot\operatorname{RoundErr}(w/\Delta)\cdot x
```

이다. $w$에 $s>1$을 곱하고 $x$를 $s$로 나누면

```math
\operatorname{Err}(Q(ws)(x/s))
=\Delta'\cdot\operatorname{RoundErr}(ws/\Delta')\cdot x\cdot\frac{1}{s}.
```

논문은 rounding error의 평균 크기가 비슷하고, 소수 channel만 키울 때 group max가 자주 바뀌지 않아 $\Delta'\approx\Delta$라고 본다. 이 조건에서는 salient channel의 error 비율이 약

```math
\frac{\Delta'}{\Delta}\frac{1}{s}\approx\frac{1}{s}
```

로 줄어든다. 다만 $s$가 너무 크면 group maximum과 $\Delta'$가 커져 나머지 weight의 오차가 증가한다. OPT-6.7B의 상위 1% channel을 일괄 확대했을 때 논문 Table 2는 다음 trade-off를 보인다.

| $s$ | 1 | 1.25 | 1.5 | 2 | 4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| $\Delta'\ne\Delta$ group 비율 | 0% | 2.8% | 4.4% | 8.2% | 21.2% |
| 평균 $(\Delta'/\Delta)/s$ | 1.000 | 0.804 | 0.676 | 0.519 | 0.303 |
| WikiText-2 PPL | 23.54 | 12.87 | 12.48 | 11.92 | 12.36 |

$s=4$가 salient error는 더 줄여도 전체 PPL은 $s=2$보다 나쁘다. AWQ가 고정 배율 대신 output error 기반 탐색을 쓰는 이유다.

### Search objective

원래 linear output과 quantized output 사이의 차이를 최소화한다.

```math
\begin{aligned}
s^*&=\arg\min_s \mathcal{L}(s),\\
\mathcal{L}(s)&=\left\|
Q(W\operatorname{diag}(s))
(\operatorname{diag}(s)^{-1}X)-WX
\right\|.
\end{aligned}
```

여기서 $Q(W\operatorname{diag}(s))$와 $(\operatorname{diag}(s)^{-1}X)$ 사이의 인접 표기는 matrix product다. 코드로는 `Q(W @ diag(s)) @ (diag(s)^-1 @ X)`이며, 혼동을 피한 구현형은 다음과 같다.

```text
Y_ref = W @ X
Y_q   = Q(W @ diag(s)) @ (diag(1/s) @ X)
loss  = norm(Y_q - Y_ref)
```

논문은 비미분 quantizer에 STE나 backpropagation을 적용하지 않고 검색 공간을

```math
s=s_X^\alpha,
\qquad
s_X[j]=\operatorname{mean}_{t}|X_{j,t}|,
\qquad
\alpha\in[0,1]
```

로 제한한다. 실험 설정은 $\alpha$에 20-point grid를 사용한다. $\alpha=0$은 scaling 없음이고, $\alpha=1$은 activation magnitude를 그대로 강하게 반영한다. 이후 quantization MSE를 줄이기 위한 weight clipping도 적용한다.

### 등가 변환을 graph에 흡수하는 법

$X/s$를 매 inference마다 별도 elementwise kernel로 계산하면 saved bandwidth를 일부 잃는다. 논문은 보통 이 역 scale을 이전 operator에 fuse할 수 있다고 설명한다.

- linear 앞이 LayerNorm이면 LayerNorm affine parameter에 $1/s$를 흡수할 수 있다.
- 연속된 linear 관계라면 앞 layer의 output channel과 뒤 layer의 input channel scale을 함께 조정할 수 있다.
- weight 쪽에는 $W\operatorname{diag}(s)$를 offline으로 만든 후 quantize한다.

모든 graph에서 자동으로 안전한 것은 아니다. residual branch가 갈라지거나 같은 tensor가 여러 consumer로 전달되면 한 consumer만을 위한 scale을 upstream에 넣을 수 없다. 실제 exporter는 graph equivalence와 broadcasting axis를 검사해야 한다.

## Calibration 알고리즘 의사코드

```text
inputs:
    pretrained FP16 model
    small calibration sequences from pretraining-like data
    bit width b in {3, 4}
    group_size = 128
    alpha_grid = 20 points in [0, 1]

for each target linear layer W[d_out, d_in]:
    cache calibration input X[num_tokens, d_in]
    sx[j] = mean(abs(X[:, j]))
    Y_ref = X @ W.T

    best_loss = infinity
    for alpha in alpha_grid:
        s = stabilize_and_normalize(sx ** alpha)
        W_scaled = W * s[None, :]
        Wq = groupwise_quantize_dequantize(W_scaled,
                                          bits=b,
                                          group_size=128)
        Yq = (X / s[None, :]) @ Wq.T
        loss = norm(Yq - Y_ref)
        if loss < best_loss:
            best_loss = loss
            best_s = s

    search a clipping ratio around W * best_s
    fold 1 / best_s into the preceding compatible operator
    quantize and pack W * best_s by 128-weight groups

output:
    packed low-bit weights
    per-group dequantization scales and zero-point if implementation uses it
    graph with inverse activation scales folded into preceding operators
```

논문이 강조하는 것은 `no backpropagation / no regression`이다. 그래도 data-free는 아니다. calibration input을 forward하고 channel 통계를 수집하며 여러 $\alpha$ 후보의 layer output을 비교한다. label은 필요 없지만 representative token distribution은 필요하다.

## Tensor shape와 손계산

### Linear layer 예

Llama-2-7B의 대표 hidden dimension을 $d=4096$, FFN intermediate dimension을 $m=11008$이라고 두자. batch 1 decode의 한 token에서 shape는 다음과 같다.

| 연산 | Weight shape | Input shape | Output shape |
| --- | ---: | ---: | ---: |
| Q projection | $4096\times4096$ | $1\times4096$ | $1\times4096$ |
| K projection | $4096\times4096$ | $1\times4096$ | $1\times4096$ |
| V projection | $4096\times4096$ | $1\times4096$ | $1\times4096$ |
| Attention output | $4096\times4096$ | $1\times4096$ | $1\times4096$ |
| FFN up/gate | $11008\times4096$ | $1\times4096$ | $1\times11008$ |
| FFN down | $4096\times11008$ | $1\times11008$ | $1\times4096$ |

예를 들어 $4096\times11008=45,088,768$ weight다.

- FP16 payload: $45,088,768\times2=90,177,536$ byte, 약 86.0 MiB
- packed INT4 payload: $45,088,768/2=22,544,384$ byte, 약 21.5 MiB
- group 128당 FP16 scale 하나라고 가정한 reviewer estimate: $(45,088,768/128)\times2=704,512$ byte, 약 0.67 MiB
- scale overhead 포함: 약 22.2 MiB, FP16 대비 약 3.87배 감소

이는 이상적인 4배보다 조금 작다. zero-point, alignment, metadata, low-bit 적용 제외 tensor가 더해지면 실제 model file ratio는 더 낮아진다. 반대로 scale을 FP32로 저장하면 이 예의 scale overhead는 약 1.34 MiB다. 논문은 모든 backend의 scale dtype과 container overhead를 하나의 숫자로 고정하지 않으므로 실제 파일에서 확인해야 한다.

### 7B 전체 weight payload의 대략적 하한

parameter 수를 7.0B로 단순화한 reviewer estimate는 다음과 같다.

- FP16: $7.0\times10^9\times2=14.0$ GB
- ideal INT4: $7.0\times10^9\times0.5=3.5$ GB
- FP16 group scale, g128: $(7.0\times10^9/128)\times2\approx109$ MB
- 합계 하한: 약 3.61 GB plus metadata와 FP16 예외 tensor

이 계산은 논문의 특정 checkpoint 실측 파일 크기가 아니라 format budget이다. TinyChat이 8GB GPU에서 13B를 실행했다는 결과에는 runtime workspace, KV cache, model별 제외 tensor, allocator 동작이 함께 작용한다.

### KV cache는 그대로 남는다

Llama-2-7B를 32 layers, 32 KV heads, head dimension 128, FP16 KV라고 가정하면 token당 KV cache는

```math
2\;(K,V)\times32\;layers\times32\;heads
\times128\times2\;bytes
=524{,}288\;bytes
=512\;KiB/token.
```

따라서 context 2048 token은 약 1 GiB, 4096 token은 약 2 GiB다. AWQ 적용 전후가 같다. 70B처럼 grouped-query attention을 쓰면 KV head 수가 적어 이 값은 달라지지만, `AWQ 때문에` 줄어드는 것은 아니다. 긴 context에서 KV cache가 weight 절감분을 잠식할 수 있으므로 peak memory 표에는 weight와 KV를 분리해 기록해야 한다.

### Prefill activation과 attention

hidden tensor 하나의 FP16 payload는 $T=2048$, $d=4096$일 때

```math
2048\times4096\times2=16{,}777{,}216\;bytes=16\;MiB.
```

naive full attention score는 32 heads라면

```math
32\times2048\times2048\times2
=268{,}435{,}456\;bytes=256\;MiB.
```

이는 reviewer의 materialized-tensor 계산이며 TinyChat의 실제 allocator peak가 아니다. fused 또는 tiled attention은 score 전체를 DRAM에 저장하지 않을 수 있다. 중요한 점은 AWQ 자체가 이 activation을 INT4로 만들지 않는다는 것이다.

## TinyChat: low-bit 파일을 실제 속도로 바꾸기

### On-the-fly dequantization

일반 GPU와 CPU에는 논문이 원하는 INT4 x FP16 multiply instruction이 직접 제공되지 않는다. TinyChat은 packed INT4를 register로 읽고 unpack 및 scale하여 FP16으로 만든 뒤 FMA한다. 복원한 전체 weight matrix를 DRAM에 쓰지 않고 MM/MV inner loop 안에서 처리한다.

따라서 실제 경로는 다음과 같다.

```text
packed INT4 weight in DRAM
 -> vector load
 -> bit shift / mask
 -> convert to FP16 and apply group scale
 -> FP16 FMA with FP16 activation
 -> FP16 output
```

이 구조의 장점은 DRAM traffic이고, 비용은 unpack과 conversion instruction이다. matrix가 compute-bound이거나 dequantization overhead가 크면 4배 speedup에 도달하지 못한다.

### SIMD-aware packing

ARM NEON의 128-bit SIMD register에는 4-bit weight 32개가 들어간다. TinyChat은 offline에 원래 순서 $w_0,w_1,\ldots,w_{31}$를 $w_0,w_{16},w_1,w_{17},\ldots$처럼 재배치한다. runtime에는 128-bit mask, shift, AND 세 vector 연산으로 low/high nibble을 분리할 수 있다. Figure 4는 이 packing이 ARM에서 최대 1.2배 추가 향상을 준다고 설명한다.

GPU에서는 8개 weight를 `0,2,4,6,1,3,5,7` 순서로 packing한다. 즉 AWQ checkpoint의 logical quantization과 target-specific physical packing은 별도 단계다. 한 backend용 packed file을 다른 backend에서 그대로 읽을 수 있다고 가정하면 안 된다.

### Kernel fusion

TinyChat은 다음을 명시적으로 최적화한다.

- Q, K, V projection fusion
- positional embedding의 on-the-fly 계산
- LayerNorm 내부 연산 fusion
- KV cache 사전 할당과 attention kernel 안의 update
- GPU kernel launch 수 감소
- CPU에서는 전체 graph를 C++로 lowering

4090에서 작은 FP16 kernel 하나가 약 0.01 ms로 kernel launch overhead와 비슷하다는 관찰 때문에, weight GEMM만 빠르게 만들어서는 end-to-end 결과가 나오지 않는다.

## 주요 정확도 결과

### LLaMA와 Llama-2 WikiText-2 perplexity

| 설정 | Llama-2 7B | Llama-2 13B | Llama-2 70B | LLaMA 7B | LLaMA 13B | LLaMA 30B | LLaMA 65B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 | 5.47 | 4.88 | 3.32 | 5.68 | 5.09 | 4.10 | 3.53 |
| INT3-g128 RTN | 6.66 | 5.52 | 3.98 | 7.01 | 5.88 | 4.88 | 4.24 |
| INT3-g128 GPTQ-R | 6.42 | 5.41 | 3.86 | 6.53 | 5.64 | 4.74 | 4.21 |
| INT3-g128 AWQ | **6.24** | **5.32** | **3.74** | **6.35** | **5.52** | **4.61** | **3.95** |
| INT4-g128 RTN | 5.73 | 4.98 | 3.46 | 5.96 | 5.25 | 4.23 | 3.67 |
| INT4-g128 GPTQ-R | 5.63 | 4.99 | 3.43 | 5.83 | 5.20 | 4.22 | 3.66 |
| INT4-g128 AWQ | **5.60** | **4.97** | **3.41** | **5.78** | **5.19** | **4.21** | **3.62** |

INT4에서는 차이가 작지만 일관되고, INT3에서 AWQ의 이점이 더 크다. Mistral-7B-Instruct의 WikiText-2 PPL은 FP16 4.14, INT4-g128 4.30, INT3-g128 4.83이다. Mixtral-8x7B-Instruct는 각각 5.94, 6.05, 6.52다.

### VLM 결과

OpenFlamingo-9B에서 논문은 language 부분만 quantize한다. COCO caption CIDEr는 다음과 같다.

| 설정 | 0-shot | 4-shot | 8-shot | 16-shot | 32-shot | 32-shot FP16 대비 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 | 63.73 | 72.18 | 76.95 | 79.74 | 81.70 | 0 |
| INT4-g128 RTN | 60.24 | 68.07 | 72.46 | 74.09 | 77.13 | -4.57 |
| INT4-g128 GPTQ | 59.72 | 67.68 | 72.53 | 74.98 | 74.98 | -6.72 |
| INT4-g128 AWQ | **62.57** | **71.02** | **74.75** | **78.23** | **80.53** | **-1.17** |
| INT3-g128 AWQ | 56.33 | 64.73 | 68.79 | 72.86 | 74.47 | -7.23 |

이 결과는 vision encoder까지 W4라는 뜻이 아니다. paper가 명시한 대로 model size를 지배하는 language part만 quantize했다. VILA-7B/13B의 11개 benchmark에서도 INT4-g128 AWQ가 대체로 FP16에 근접했으며, 일부 지표는 소폭 오르거나 내린다. 작은 표본 차이를 `정확히 lossless`라는 절대 명제로 확대하기보다 benchmark 범위에서의 근접 성능으로 읽는 편이 안전하다.

### Programming과 math

INT4-g128에서 CodeLlama-7B MBPP pass@1/pass@10은 FP16 38.53/49.77, RTN 37.51/48.49, GPTQ 31.97/44.75, AWQ 40.64/49.25다. GSM8K accuracy는 다음과 같다.

| Model | FP16 | RTN | GPTQ | AWQ |
| --- | ---: | ---: | ---: | ---: |
| Llama-2-7B | 13.87 | 11.07 | 12.13 | 13.57 |
| Llama-2-13B | 26.16 | 21.23 | 24.26 | 25.25 |
| Llama-2-70B | 56.41 | 53.98 | 56.03 | 56.40 |

single benchmark에서 AWQ가 FP16을 약간 넘은 값은 quantization이 일반적으로 품질을 높인다는 증거가 아니라 evaluation variance와 regularization-like perturbation 가능성을 포함한다.

## Calibration ablation과 일반화

Figure 8의 OPT-6.7B, INT3-g128 실험은 AWQ가 약 16개의 2048-token sequence에서 안정화되는 반면 GPTQ는 약 192 sequence까지 개선된다고 보고한다. paper 표현으로는 약 10배 적은 calibration data다.

PubMed와 Enron을 calibration/evaluation distribution으로 교차한 결과도 다음과 같다.

| Calibration -> Evaluation | GPTQ PPL | AWQ PPL |
| --- | ---: | ---: |
| PubMed -> PubMed | 32.48 | 32.56 |
| PubMed -> Enron | 50.41 | 45.07 |
| Enron -> PubMed | 34.81 | 33.16 |
| Enron -> Enron | 45.52 | 44.57 |

distribution mismatch로 인한 증가량을 논문은 GPTQ 2.33에서 4.89, AWQ 0.50에서 0.60으로 요약한다. 이 결과는 activation mean 기반 search가 reconstruction보다 덜 overfit한다는 주장과 일치하지만, 두 corpus와 한 모델만의 controlled ablation이다. 모든 도메인, tokenizer, 언어에 대한 보장은 아니다.

INT2-g64에서는 AWQ 단독 표가 아니라 AWQ와 GPTQ를 결합한다. OPT-6.7B PPL은 RTN 7622, GPTQ 16.65, AWQ+GPTQ 15.71이다. 따라서 `AWQ가 INT2를 단독 해결했다`고 쓰면 부정확하다.

## 실제 시스템 결과와 해석

### Figure 9 throughput

논문은 RTX 4090, Jetson Orin, RTX 4070 laptop에서 Hugging Face FP16, 자체 FP16, TinyChat AWQ W4A16을 비교한다. 대표 수치는 다음과 같다.

| Device / Model | Hugging Face FP16 | TinyChat FP16 | TinyChat W4A16 |
| --- | ---: | ---: | ---: |
| RTX 4090, Llama-2-7B | 52 tok/s | 62 tok/s | 194 tok/s |
| RTX 4090, MPT-7B | 59 tok/s | 63 tok/s | 158 tok/s |
| Jetson Orin, Llama-2-7B | 11 tok/s | 12 tok/s | 39 tok/s |
| Jetson Orin, MPT-7B | 11 tok/s | 12 tok/s | 38 tok/s |
| RTX 4070 laptop, Llama-2-13B | FP16 OOM | FP16 OOM | 33 tok/s |

Figure 9는 최대 4090 3.9배, Orin 3.5배를 말한다. 결론의 평균 요약은 desktop/mobile GPU에서 Hugging Face FP16 대비 3.2배에서 3.3배다. 비교 baseline에 따라 배율이 달라지므로 `Hugging Face 대비`와 `TinyChat 자체 FP16 대비`를 분리해야 한다.

VILA throughput도 실제 W4A16 결과가 있다.

| Model | Precision | A100 | RTX 4090 | Orin |
| --- | --- | ---: | ---: | ---: |
| VILA-7B | FP16 | 81.6 | 58.5 | 11.5 |
| VILA-7B-AWQ | W4A16 | 155.3 | 168.1 | 35.6 |
| VILA-13B | FP16 | 48.5 | OOM | 6.1 |
| VILA-13B-AWQ | W4A16 | 102.1 | 99.0 | 17.5 |

### Figure 10 protocol

system 비교는 batch 1, prompt length 4, generation 200 token이며 여러 run의 median latency로 tokens/s를 계산한다. Jetson Orin 64GB에서 TinyChat은 Llama-2-7B 39.1 tok/s, 13B 21.2 tok/s, LLaMA-30B 8.8 tok/s, Llama-2-70B 3.5 tok/s를 기록한다. Raspberry Pi 4에서는 Llama-2-7B, OPT-6.7B, OPT-1.3B가 모두 약 0.7 tok/s다.

이 protocol은 decode throughput을 잘 드러내지만 TTFT, 긴 prompt prefill, p95 tail, 장시간 thermal throttling, energy/token을 보고하지 않는다. `on-device` 결론을 제품 수준으로 옮기려면 이 항목을 추가해야 한다.

## Fake quantization과 real quantization의 경계

이 논문은 두 종류의 증거를 모두 갖고 있다는 점이 강점이지만 범위를 정확히 나눠야 한다.

### Algorithm accuracy path

- INT3/INT4-g128으로 weight를 quantize하고 perplexity와 task score를 측정한다.
- 표는 low-bit rounding이 품질에 미치는 영향을 검증한다.
- 표 자체만으로 target device가 그 bit width를 packed storage와 fused kernel로 실행했다는 뜻은 아니다.
- 특히 INT3 결과는 TinyChat의 실제 INT3 speed benchmark로 연결되지 않는다.

### TinyChat real kernel path

- W4A16 packed weight를 실제 device memory에 둔다.
- target-specific bit packing을 사용한다.
- GEMM/MV kernel 내부에서 unpack/dequantize한다.
- desktop GPU, mobile GPU, ARM CPU의 실제 tokens/s를 측정한다.

따라서 deployment 검증 표에는 다음 네 열이 별도로 있어야 한다.

| 항목 | 확인 질문 |
| --- | --- |
| Logical precision | checkpoint가 W4A16인가, W3A16인가 |
| Physical storage | 실제 file과 resident memory가 4-bit packed인가 |
| Compute path | fused dequant + FP16 FMA인가, fallback FP16 GEMM인가 |
| Measured system | 어떤 device, runtime, prompt, output length, statistic인가 |

ONNX나 generic mobile NPU에 Q/DQ node만 넣는 것으로 TinyChat과 같은 실행이 보장되지 않는다. runtime이 group-128 weight-only format, per-group scale, AWQ folding, packed nibble layout을 모두 지원해야 한다.

## 강점

1. weight 중요도를 weight norm이 아니라 input activation과 연결해 간단하고 해석 가능한 criterion을 제시한다.
2. FP16 outlier channel을 남기지 않고 등가 scaling으로 uniform low-bit storage를 유지한다.
3. backpropagation과 Hessian reconstruction 없이 small calibration set으로 동작한다.
4. LLaMA, Llama-2, OPT, Mistral, Mixtral, instruction model, code/math, VLM까지 넓게 검증한다.
5. 알고리즘 정확도에서 끝나지 않고 TinyChat의 packing, fused dequantization, kernel fusion과 실제 edge throughput을 제시한다.
6. group size, calibration grid, batch 1 system protocol을 비교적 명시적으로 공개한다.

## 한계와 주의점

1. activation은 FP16이라 prefill activation과 KV cache memory를 줄이지 않는다.
2. 정확도 표의 INT3까지 실제 TinyChat speedup으로 검증된 것은 아니다.
3. 4배 weight payload 절감은 scale, zero-point, metadata, padding, FP16 예외 tensor 때문에 실제 model size와 다르다.
4. scaling을 preceding operator에 fuse하는 과정은 graph topology와 backend에 의존한다.
5. system 비교의 Hugging Face FP16 baseline은 TinyChat 자체 fused FP16보다 약할 수 있어 두 baseline을 함께 봐야 한다.
6. prompt length 4, generation 200의 median throughput은 긴 context TTFT와 p95 latency를 대표하지 않는다.
7. mobile GPU 결과가 mobile NPU, DSP, CPU에서 그대로 재현되는 것은 아니다.
8. 최신 PDF v6에는 초판 뒤에 추가된 Mistral, VILA, system 결과가 포함되어 있으므로 다른 버전의 표와 섞으면 안 된다.

## 온디바이스 재현 계획

### 1. Quantization 품질

- 동일 FP16 checkpoint와 tokenizer를 고정한다.
- RTN W4A16-g128, AWQ W4A16-g128, 필요하면 GPTQ를 비교한다.
- calibration sequence 수를 8, 16, 32, 128로 바꾼다.
- calibration domain을 일반 corpus와 target domain으로 교차한다.
- perplexity뿐 아니라 target task accuracy, hallucination, long-context degradation을 측정한다.

### 2. Artifact 검증

- original parameter count와 quantized tensor count를 기록한다.
- packed weight byte, scale byte, metadata byte, FP16 제외 tensor byte를 합산한다.
- group size와 scale dtype을 manifest에 저장한다.
- exporter 전후 FP16 graph output의 등가성을 먼저 검증한다.
- low-bit file을 load한 뒤 runtime resident memory를 별도 측정한다.

### 3. Runtime 검증

- target backend의 W4A16 groupwise kernel 지원 여부를 확인한다.
- operator trace에서 fallback FP16 GEMM과 unpacked-weight DRAM write가 없는지 본다.
- batch 1에서 prompt 4/128/512/2048, output 32/200/512 조합을 측정한다.
- TTFT, time-per-output-token, tokens/s, p50/p95를 함께 기록한다.
- weight resident, KV cache, temporary activation, kernel workspace를 분리한다.
- energy/token, 10분 이상 온도, clock, throttling을 기록한다.

### 4. VLM 적용

- vision encoder, projector, LLM 중 실제 quantization 대상을 명시한다.
- image resolution과 visual token 수를 고정한다.
- image prefill TTFT와 text decode throughput을 분리한다.
- VLM의 KV cache가 image token 때문에 커지는 양을 계산한다.
- caption/VQA 품질과 language-only perplexity를 둘 다 측정한다.

## 구현 체크리스트

### Calibration

- [ ] calibration data에 label이나 backpropagation이 필요 없음을 확인했다.
- [ ] activation mean을 input channel axis로 계산했다.
- [ ] $\alpha$를 20-point grid로 검색했다.
- [ ] layer output reconstruction loss의 norm과 token sampling을 기록했다.
- [ ] clipping search를 scaling search와 구분했다.
- [ ] calibration sequence 수와 길이, corpus를 기록했다.

### Scale와 granularity

- [ ] AWQ channel scale $s$와 quantizer group scale $\Delta$를 구분했다.
- [ ] group size 128을 사용하거나 변경 이유를 기록했다.
- [ ] scale의 dtype과 zero-point 유무를 기록했다.
- [ ] inverse scale을 preceding operator에 올바른 axis로 fold했다.
- [ ] residual 및 multi-consumer graph에서 등가성을 test했다.

### Real kernel

- [ ] 실제 weight가 nibble-packed 4-bit인지 file byte로 확인했다.
- [ ] dequantized full weight를 DRAM에 materialize하지 않는다.
- [ ] target별 packing 순서를 적용했다.
- [ ] fused MM과 decode MV kernel을 모두 지원한다.
- [ ] unsupported layer의 FP16 fallback 비율을 trace했다.
- [ ] INT3 accuracy를 INT3 kernel speed로 오해하지 않았다.

### Memory와 latency

- [ ] model weight, scales, KV cache, activation, workspace peak를 분리했다.
- [ ] context length별 KV cache 증가량을 계산했다.
- [ ] TTFT와 decode tokens/s를 분리했다.
- [ ] p50과 p95를 모두 기록했다.
- [ ] cold start, warm run, sustained run을 나눴다.
- [ ] power, temperature, throttling을 기록했다.

## 최종 평가

AWQ의 핵심 공헌은 `큰 activation channel과 연결된 weight를 보호하라`는 관찰을 hardware-friendly한 등가 scaling으로 바꾼 데 있다. 1% FP16 mixed precision이 보여 준 품질 회복을 uniform group-wise low-bit format으로 옮겼고, 작은 calibration set과 단순한 grid search로 GPTQ보다 넓은 domain 일반화를 보였다.

더 중요한 점은 TinyChat을 통해 quantization accuracy와 실제 deployment 사이의 간극을 직접 다룬 것이다. 4-bit weight를 저장하는 것만으로는 부족하며, target-specific packing, on-the-fly dequantization, MM/MV kernel, graph fusion이 있어야 3배 수준의 end-to-end throughput이 나온다.

반면 AWQ를 `모든 LLM memory를 4분의 1로 줄이는 integer-only 방법`이라고 요약하면 틀린다. activation과 KV cache는 FP16으로 남고, TinyChat 계산도 dequantized FP16 FMA다. 온디바이스 실험의 올바른 결론은 `batch 1 decode에서 weight bandwidth가 병목일 때 W4A16 packed kernel이 매우 효과적`이라는 조건부 문장이다. 실제 제품 평가는 model file뿐 아니라 KV cache, TTFT, p95, energy, thermal throttling까지 포함해야 완성된다.

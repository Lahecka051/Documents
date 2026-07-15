# 79. Learned Step Size Quantization

## 논문 정보

- 원본 파일: `79_LSQ.pdf`
- 제목: Learned Step Size Quantization
- 저자: Steven K. Esser, Jeffrey L. McKinstry, Deepika Bablani, Rathinakumar Appuswamy, Dharmendra S. Modha
- 소속: IBM Research, San Jose, California, USA
- 발표: ICLR 2020
- 제공된 PDF: arXiv:1902.08153v3, 2020년 5월 7일
- 논문 링크: https://arxiv.org/abs/1902.08153
- 핵심 키워드: quantization-aware training, learned step size, straight-through estimator, low-bit CNN, gradient scaling, integer inference

## 한눈에 보는 요약

LSQ는 quantizer의 step size $s$를 단순한 activation 통계나 quantization MSE로 정하지 않고, network의 최종 task loss가 줄어드는 방향으로 weight와 함께 학습한다. 균일 quantizer 자체는 단순하다.

```math
\begin{aligned}
\bar v &= \mathrm{round}
\left(\mathrm{clip}\left(\frac{v}{s},-Q_N,Q_P\right)\right),\\
\hat v &= s\bar v.
\end{aligned}
```

핵심은 두 가지다.

1. round에 straight-through estimator를 적용해 step size gradient가 quantization bin의 transition point와의 거리에 민감하도록 만든다.
2. 한 layer의 수많은 element에서 $s$ 하나로 gradient가 합산되어 update가 과도하게 커지는 문제를 $1/\sqrt{NQ_P}$ scale로 보정한다.

논문 설정에서는 각 weight layer와 각 activation layer가 FP32 scalar step size 하나를 가진다. 즉 오늘날 자주 쓰는 per-output-channel weight quantization이 아니라 per-layer, per-tensor granularity다. first와 last convolution/fully-connected layer는 항상 8-bit이고, 나머지는 실험에 따라 weight와 input activation을 모두 2, 3, 4 또는 8-bit로 맞춘다.

LSQ는 post-training quantization이 아니라 quantization-aware fine-tuning이다. full-precision pretrained weight를 시작점으로 2/3/4-bit 모델은 90 epoch 학습하며, 학습 중에는 FP32 master weight를 저장하고 fake-quantized $\hat v$를 matrix operation에 사용한다. 따라서 `2-bit 모델을 학습하니 training memory도 16분의 1`이라고 해석하면 틀린다. low-bit payload 절감은 integer kernel로 내보낸 inference artifact의 이야기다.

ImageNet에서 LSQ는 당시 비교한 ResNet, VGG-16bn, SqueezeNext 전반에 걸쳐 강한 2/3/4-bit 정확도를 보였다. knowledge distillation을 결합하면 ResNet-18/34/50의 3-bit top-1이 각 full-precision baseline에 도달하거나 이를 소폭 넘었다. 그러나 논문은 실제 mobile CPU/NPU latency, energy, peak memory나 packed 3-bit kernel을 측정하지 않는다. low-precision integer inference는 Figure 1의 설계 목표이지 end-to-end system benchmark가 아니다.

## 연구 배경과 문제의식

균일 quantizer의 step size는 두 종류의 error 사이를 조절한다.

- $s$가 너무 작으면 representable range $[-sQ_N,sQ_P]$가 좁아져 clipping error가 커진다.
- $s$가 너무 크면 range는 넓지만 adjacent level 간 간격이 커져 rounding error가 커진다.

기존 방법은 distribution statistics에 맞추거나 MSE 같은 local quantization error를 최소화했다. 하지만 network의 목표는 각 tensor를 원본과 최대한 같게 만드는 것이 아니라 최종 classification loss를 줄이는 것이다. 어떤 channel이나 layer의 작은 perturbation은 task에 매우 중요하고 다른 error는 거의 중요하지 않을 수 있으므로 local MSE optimum과 task-loss optimum은 다를 수 있다.

LSQ는 step size를 학습 parameter로 두어 이 차이를 직접 최적화한다. 난점은 round가 불연속이라 실제 derivative가 거의 모든 곳에서 0이고 transition에서 정의되지 않는다는 점이다. 논문은 forward의 round는 유지하면서 backward에 STE를 사용한다.

## 균일 quantizer의 정의

### Signed weight와 unsigned activation

$b$-bit activation은 nonnegative라고 가정한다.

```math
Q_N=0,\qquad Q_P=2^b-1.
```

$b$-bit weight는 signed integer range를 사용한다.

```math
Q_N=2^{b-1},\qquad Q_P=2^{b-1}-1,
```

따라서 clip 범위는 $[-2^{b-1},2^{b-1}-1]$이다. 예를 들면 다음과 같다.

| bits | Activation code | Weight code |
| ---: | --- | --- |
| 2 | 0, 1, 2, 3 | -2, -1, 0, 1 |
| 3 | 0 to 7 | -4 to 3 |
| 4 | 0 to 15 | -8 to 7 |
| 8 | 0 to 255 | -128 to 127 |

이 설정은 ReLU 계열 CNN activation에 맞는다. negative activation을 내는 SiLU, GELU, LayerNorm output에는 그대로 적용할 수 없고 signed activation quantizer로 바꿔야 한다. 논문 appendix도 pseudocode가 unsigned activation을 가정하며 signed로 수정 가능하다고 명시한다.

### Training representation과 inference representation

두 tensor를 구분해야 한다.

- $\bar v$: integer code, inference에서 low-bit matrix unit에 넣을 값
- $\hat v=s\bar v$: 원래 scale로 dequantize한 값, paper의 PyTorch training에서 사용하는 fake-quant tensor

weight와 activation step을 각각 $s_w,s_x$라고 하면 integer dot product는

```math
\bar y=\bar W\bar x
```

이고 real scale 출력은

```math
\hat y=(s_ws_x)\bar y
```

이다. $s_ws_x$ multiplication은 convolution 뒤에 적용하거나 batch normalization, 다음 layer scale과 algebraically fold할 수 있다. 이는 integer-only graph를 가능하게 하는 방향을 보여 주지만 paper가 특정 backend의 실제 folding 및 requantization kernel을 구현해 latency를 보고한 것은 아니다.

## Step size gradient

round에 대해 STE를 적용하고 나머지 연산을 미분하면 논문의 핵심 gradient가 나온다.

```math
\frac{\partial\hat v}{\partial s}=
\begin{cases}
-v/s+\mathrm{round}(v/s),
& -Q_N\lt v/s\lt Q_P,\\
-Q_N,&v/s\le -Q_N,\\
Q_P,&v/s\ge Q_P.
\end{cases}
```

clip 안에서는 `quantized code - normalized real value`, 즉 signed rounding residual이다. 예를 들어 $v/s=1.49$이면 code 1이므로 gradient는 $-0.49$이고, $v/s=1.51$이면 code 2라 $+0.49$다. transition point를 넘을 때 부호가 바뀌고 크기가 커진다. $s$가 조금 바뀌어 bin이 바뀌기 쉬운 값에 더 민감하다는 의미다.

clip 밖에서는 gradient가 $-Q_N$ 또는 $Q_P$라 range를 넓히거나 줄이는 방향의 신호를 유지한다. 반면 input $v$로 전달되는 STE gradient는

```math
\frac{\partial\hat v}{\partial v}=
\begin{cases}
1,&-Q_N\lt v/s\lt Q_P,\\
0,&\text{otherwise}.
\end{cases}
```

이다. clipping 밖 weight/activation은 upstream gradient가 차단된다. 이 역시 true derivative가 아니라 학습을 가능하게 하는 surrogate gradient다.

### 왜 QIL, PACT와 다른가

Figure 2에서 LSQ의 $\partial\hat v/\partial s$는 각 quantization transition 사이에서 기울어진 sawtooth 형태다. QIL은 clip point와의 거리에 주로 민감하고, PACT의 clipping parameter gradient는 clip point 아래에서 0이다. LSQ는 bin 내부 위치를 gradient에 남기므로 step size를 바꿀 때 어느 값이 다음 code로 넘어갈 가능성이 큰지를 더 세밀하게 반영한다.

이 설명은 LSQ gradient의 직관이지 true discrete optimization gradient를 정확히 추정한다는 증명은 아니다. STE 선택에 따른 bias가 있으며 최종 성능으로 정당화된다.

## Step size gradient scaling

### 문제: scalar 하나에 모든 element gradient가 모인다

weight tensor에 $N_W$개 element가 있으면

```math
\nabla_sL=\sum_{i=1}^{N_W}
\frac{\partial L}{\partial\hat w_i}
\frac{\partial\hat w_i}{\partial s}.
```

weight update의 norm과 달리 step size update는 scalar 하나에 합산되므로 tensor가 커질수록 상대 update가 지나치게 커질 수 있다. precision이 높아지면 $s$ 자체가 작아지는 영향도 있다. 논문은 step size와 weight의 상대 update 비율

```math
R=\frac{\nabla_sL/s}{\|\nabla_wL\|/\|w\|}
```

이 평균적으로 1 근처가 되도록 하려 한다.

appendix의 heuristic은

```math
\frac{\|w\|}{s}\approx\sqrt{N_WQ_P},
\qquad
\|\nabla_wL\|\approx |\nabla_sL|
```

를 사용해 보정 전 $R\approx\sqrt{N_WQ_P}$라고 추정한다. 따라서 gradient multiplier를 다음처럼 둔다.

```math
g_w=\frac{1}{\sqrt{N_WQ_P}},
\qquad
g_a=\frac{1}{\sqrt{N_FQ_P}}.
```

$N_W$는 layer weight 수, $N_F$는 activation feature 수다. 이는 forward의 $s$ 값은 바꾸지 않고 backward gradient만 $g$배 하는 `gradient scale`이다.

### 수치 예

$3\times3$, $C_{in}=C_{out}=256$ convolution에는

```math
N_W=3\times3\times256\times256=589{,}824
```

개의 weight가 있다. 4-bit signed weight의 $Q_P=7$이므로

```math
g_w=\frac{1}{\sqrt{589{,}824\times7}}
\approx4.92\times10^{-4}.
```

2-bit에서는 $Q_P=1$이라

```math
g_w\approx1.30\times10^{-3}.
```

precision이 높아질수록 같은 layer에서 step-size gradient를 더 줄이는 이유가 보인다. 이 숫자는 reviewer의 식 대입이며 paper가 해당 convolution에 보고한 측정값은 아니다.

## Initialization과 granularity

각 step size는 다음으로 초기화된다.

```math
s_0=\frac{2\langle|v|\rangle}{\sqrt{Q_P}}.
```

weight step은 initial pretrained weight에서, activation step은 첫 training batch에서 계산한다. 이후 task loss gradient로 계속 갱신한다.

논문의 granularity는 매우 명확하다.

- weight layer마다 FP32 scalar $s_w$ 하나
- activation layer마다 FP32 scalar $s_x$ 하나
- per-channel 또는 per-group scale이 아님
- first와 last matrix layer는 8-bit
- batch normalization parameter 및 다른 parameter는 FP32

step parameter 자체의 저장비는 layer당 4 byte라 작다. 그러나 per-tensor scale은 outlier channel 하나가 전체 layer range에 영향을 줄 수 있다. 현대 Transformer에 적용할 때는 original LSQ 식을 per-channel 또는 group granularity로 확장하는 경우가 많지만, 그 결과를 이 paper의 설정이라고 부르면 안 된다.

## Training pipeline

### Paper recipe

- pretrained full-precision model로 초기화
- full-precision master weight를 저장하고 update
- quantized weight와 activation으로 forward 및 backward
- first/last layer 8-bit, 나머지는 동일한 2/3/4/8-bit
- softmax cross entropy, momentum 0.9
- cosine learning-rate decay, restart 없음
- 2/3/4-bit: 90 epochs, initial LR 0.01
- 8-bit: 1 epoch, initial LR 0.001
- full precision: initial LR 0.1
- ImageNet, train resize 256 후 random 224 crop 및 50% horizontal mirror
- test는 224 center crop
- pre-activation ResNet, batch-normalized VGG, SqueezeNext

weight decay sweep 결과 ResNet-18의 기본값은 precision에 따라 달랐다. 4/8-bit는 $10^{-4}$, 3-bit는 $0.5\times10^{-4}$, 2-bit는 $0.25\times10^{-4}$를 후속 실험에 사용했다. low precision이 regularization처럼 작용한다는 가설에 맞춘 선택이다.

### 구현 의사코드

```text
function grad_scale(x, scale):
    # forward value is x, backward gradient is multiplied by scale
    return detach(x - x * scale) + x * scale

function round_pass(x):
    # forward value is round(x), backward derivative is 1
    return detach(round(x) - x) + x

function lsq_quantize(v, s, bits, is_activation):
    if is_activation:
        qn = 0
        qp = 2**bits - 1
        g = 1 / sqrt(num_features(v) * qp)
    else:
        qn = -(2**(bits - 1))
        qp = 2**(bits - 1) - 1
        g = 1 / sqrt(num_weights(v) * qp)

    s_for_backward = grad_scale(s, g)
    z = clip(v / s_for_backward, qn, qp)
    code = round_pass(z)
    return code * s_for_backward

for minibatch in ImageNet:
    for each quantized convolution or linear:
        W_fake = lsq_quantize(W_master, sw, bits, false)
        X_fake = lsq_quantize(X, sx, bits, true)
        Y = convolution_or_linear(X_fake, W_fake)
    loss = cross_entropy(Y, label)
    backward through STE
    SGD updates W_master and all step sizes
```

`detach(x - x*scale) + x*scale`은 forward에서는 $x$이고 backward에서는 $x\times scale$ branch만 gradient를 받는다. `round_pass`도 forward에서는 round, backward에서는 identity다.

## Fake quantization과 실제 integer inference

### 학습 중 실제로 일어나는 일

paper는 단순한 PyTorch 구현을 위해 $\hat v=s\bar v$를 float tensor로 만들고 기존 convolution/matmul에 넣는다. bit width는 representable level을 제한하지만 일반 GPU training tensor가 물리적으로 2-bit packed라는 뜻은 아니다.

- FP32 master weight가 남는다.
- weight gradient와 momentum buffer가 남는다.
- forward activation은 low-bit level 중 하나의 값을 갖지만 FP32 container일 수 있다.
- backward graph와 saved activation이 필요하다.
- step size도 FP32이며 gradient와 optimizer update를 가진다.

따라서 LSQ QAT의 training memory는 full-precision training과 비슷하거나 fake-quant graph 때문에 더 클 수 있다.

### 논문이 구상한 inference

Figure 1의 inference path는 $\bar W$, $\bar x$를 low-precision matrix multiplication unit에 넣고 결과를 $s_ws_x$로 rescale한다. 이 경로가 실제 이득을 내려면 다음이 필요하다.

- 2/3/4-bit signed weight와 unsigned activation의 packed storage
- 해당 bit-width를 처리하는 convolution/GEMM kernel
- 충분한 accumulator precision
- scale multiplication 또는 BN/requantization fusion
- first/last 8-bit와 내부 low-bit 사이의 dtype conversion

paper는 이 hardware를 benchmark하지 않는다. accuracy가 높다는 사실은 특정 mobile NPU가 3-bit를 native로 실행한다는 뜻이 아니다.

## Tensor shape와 memory 손계산

### Weight payload

앞의 $3\times3\times256\times256$ convolution은 589,824 weight다.

| Representation | Payload 계산 | 크기 |
| --- | ---: | ---: |
| FP32 master | $589{,}824\times4$ | 2.25 MiB |
| INT8 | $589{,}824\times1$ | 576 KiB |
| packed INT4 | $589{,}824\times4/8$ | 288 KiB |
| packed INT3 | $589{,}824\times3/8$ | 216 KiB |
| packed INT2 | $589{,}824\times2/8$ | 144 KiB |

step size는 per-layer FP32 하나이므로 4 byte뿐이다. alignment와 container header를 무시한 이론값이다. 특히 INT3는 byte 경계와 맞지 않아 row padding 또는 bit unpack overhead가 생길 수 있다.

### Training state 하한

같은 convolution에 SGD momentum을 쓰는 단순 reviewer estimate는 다음 상태를 요구한다.

- FP32 master weight: 2.25 MiB
- FP32 gradient: 2.25 MiB
- FP32 momentum: 2.25 MiB
- materialized fake-quant weight를 FP32로 보관하면 추가 2.25 MiB

activation과 autograd saved tensor를 빼도 최대 약 9 MiB다. INT2 packed inference의 144 KiB와 약 64배 차이다. framework가 fake weight를 recompute하거나 buffer를 재사용하면 줄어들 수 있으므로 이는 paper 실측값이 아니라 보수적인 tensor budget이다.

### Activation payload

batch 1에서 $56\times56\times64=200,704$ activation은 다음과 같다.

| Representation | 이론 payload |
| --- | ---: |
| FP32 fake-quant container | 약 0.766 MiB |
| INT8 deployment | 약 196 KiB |
| packed INT4 | 약 98 KiB |
| packed INT3 | 약 73.5 KiB |
| packed INT2 | 약 49 KiB |

training batch가 256이면 이 activation 하나만 FP32로 약 196 MiB다. QAT 중 activation memory가 bit width 비율로 줄지 않는 이유다. deployment에서는 backend가 low-bit activation을 layer 사이에 실제로 유지할 때만 표의 절감이 실현된다.

### Accumulator width

4-bit activation의 최대 code는 15, signed weight 절댓값 최대는 8이다. dot-product length $K=2304$에서 최악의 절댓값 합은

```math
15\times8\times2304=276{,}480.
```

signed 16-bit 범위를 넘으므로 INT32 accumulator가 필요하다. 2-bit라고 accumulator까지 2-bit로 줄어드는 것이 아니다. bias와 rescale 후 다음 activation range로 requantize하는 규칙도 backend에 필요하다.

## 주요 ImageNet 결과

### LSQ 단독 top-1

| Network | FP baseline | 2-bit | 3-bit | 4-bit | 8-bit |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet-18 | 70.5 | 67.6 | 70.2 | 71.1 | 71.1 |
| ResNet-34 | 74.1 | 71.6 | 73.4 | 74.1 | 74.1 |
| ResNet-50 | 76.9 | 73.7 | 75.8 | 76.7 | 76.8 |
| ResNet-101 | 78.2 | 76.1 | 77.5 | 78.3 | 78.1 |
| ResNet-152 | 78.9 | 76.9 | 78.2 | 78.5 | 78.5 |
| VGG-16bn | 73.4 | 71.4 | 73.4 | 74.0 | 73.5 |
| SqueezeNext-23-2x | 67.3 | 53.3 | 63.7 | 67.4 | 67.0 |

ResNet은 3-bit에서 baseline에 가까워지고 4-bit는 거의 같거나 일부 더 높다. 반면 SqueezeNext는 2-bit에서 14.0 point 하락한다. architecture가 parameter efficiency frontier에 가까울수록 precision reduction에 민감할 수 있다는 paper의 해석이다.

ResNet-50 비교에서 LSQ top-1은 2/3/4-bit 각각 73.7/75.8/76.7이다. PACT는 72.2/75.3/76.5, LQ-Nets는 71.5/74.2/75.1이다. 다만 training recipe, full-precision baseline, first/last precision이 논문마다 완전히 같지 않을 수 있으므로 method-only 인과 비교에는 주의가 필요하다.

### Top-5 대표 결과

| Network | FP top-5 | 2-bit | 3-bit | 4-bit | 8-bit |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet-18 | 89.6 | 87.6 | 89.4 | 90.0 | 90.1 |
| ResNet-34 | 91.8 | 90.3 | 91.4 | 91.7 | 91.8 |
| ResNet-50 | 93.4 | 91.5 | 92.7 | 93.2 | 93.4 |
| ResNet-101 | 94.1 | 92.8 | 93.6 | 94.0 | 94.0 |
| ResNet-152 | 94.3 | 93.2 | 93.9 | 94.1 | 94.2 |

### Knowledge distillation 결합

같은 architecture의 frozen FP32 teacher를 사용하고 temperature 1, standard loss와 distillation loss를 1:1로 합친다.

| Network | Precision | LSQ 단독 top-1 | LSQ+KD top-1 | FP baseline |
| --- | ---: | ---: | ---: | ---: |
| ResNet-18 | 2 | 67.6 | 67.9 | 70.5 |
| ResNet-18 | 3 | 70.2 | 70.6 | 70.5 |
| ResNet-34 | 2 | 71.6 | 72.4 | 74.1 |
| ResNet-34 | 3 | 73.4 | 74.3 | 74.1 |
| ResNet-50 | 2 | 73.7 | 74.6 | 76.9 |
| ResNet-50 | 3 | 75.8 | 76.9 | 76.9 |
| ResNet-50 | 4 | 76.7 | 77.6 | 76.9 |

paper가 말하는 `3-bit가 full precision에 도달`한 결과는 이 KD 결합 표를 포함한다. LSQ 단독의 모든 3-bit 모델이 baseline과 같다는 뜻은 아니다.

## Ablation 해석

### Gradient scale

2-bit ResNet-18에서 결과는 다음과 같다.

| Step gradient scale | LR | Top-1 |
| --- | ---: | ---: |
| $1/\sqrt{NQ_P}$ | 0.01 | **67.6** |
| $1/\sqrt N$ | 0.01 | 67.3 |
| 1 | 0.01 | 수렴 실패 |
| 1 | 0.0001 | 64.2 |
| $10/\sqrt{NQ_P}$ | 0.01 | 67.4 |
| $1/(10\sqrt{NQ_P})$ | 0.01 | 67.3 |

보정이 없으면 step size의 상대 update가 weight보다 2에서 3 orders of magnitude 크고 precision이 높을수록 불균형이 심했다. $1/\sqrt N$은 tensor size 효과는 줄이지만 precision 차이를 남기며, $Q_P$까지 포함한 scale이 가장 균형적이었다.

### Learning-rate schedule

2-bit ResNet-18에서 cosine decay는 67.6이고, 매 20 epoch마다 0.1을 곱하는 step decay는 67.2다. LSQ의 전체 우위를 gradient 하나에만 돌리면 안 되며 schedule이 0.4 point 기여했다.

### Quantization error를 최소화하지 않는다

2-bit ResNet-18에서 학습된 평균 step은 activation $0.949\pm0.206$, weight $0.025\pm0.019$였다. 학습된 $\hat s$와 local error-minimizing $s$의 평균 절대 상대 차이는 다음과 같다.

| Tensor | MAE optimum과 차이 | MSE optimum과 차이 | KL optimum과 차이 |
| --- | ---: | ---: | ---: |
| Activation layers | 50% | 63% | 64% |
| Weight layers | 47% | 28% | 46% |

LSQ가 단순 quantization reconstruction이 아니라 task-aware optimum을 찾는다는 중요한 evidence다. 동시에 한 test batch와 $0.01\hat s$에서 $20\hat s$ grid로 비교한 분석이므로 전역 최적성의 증명은 아니다.

## Model size frontier

Figure 3은 full-precision model size를 ResNet-18 45 MB, ResNet-34 83 MB, ResNet-50 97 MB, ResNet-101 170 MB, ResNet-152 230 MB, VGG-16bn 528 MB, SqueezeNext-23-2x 10 MB로 제시한다. bit 수에 비례해 단순 축소한 accuracy-size plot에서 2-bit ResNet-34와 ResNet-50은 같은 size budget의 작은 고정밀 모델보다 좋은 frontier를 만든다.

다만 이 plot은 packed bit payload 관점이다. first/last 8-bit, BN 및 bias FP32, alignment, scale metadata, runtime workspace를 정확히 합산한 배포 파일 측정으로 읽으면 안 된다. 또한 low-bit convolution이 실제 device에서 더 빠른지는 별도 문제다.

## 온디바이스 관점

### 유리한 조건

- target accelerator가 2/4-bit weight와 activation을 native 지원한다.
- layer 사이 activation도 실제 packed low-bit로 유지된다.
- accumulator와 rescale이 fused된다.
- memory bandwidth 또는 SRAM capacity가 병목이다.
- first/last 8-bit transition overhead가 전체에서 작다.

### 어려운 조건

- 모바일 NPU가 INT8만 지원하면 INT4/INT3 tensor가 INT8 container로 승격될 수 있다.
- 3-bit는 packing, address calculation, SIMD lane alignment가 특히 어렵다.
- per-layer scale은 channel outlier에 민감하다.
- unsigned activation 가정은 modern Transformer의 signed activation과 맞지 않는다.
- unsupported operator가 CPU로 fallback하면 dtype conversion과 copy가 이득을 없앤다.
- QAT training cost가 PTQ보다 훨씬 크다.

LSQ 논문의 accuracy 표만으로 3-bit가 8-bit보다 실제 mobile latency가 낮다고 결론 내릴 수 없다. `3-bit model size`와 `3-bit native throughput`을 분리해야 한다.

## 강점

1. quantizer 설정을 task loss로 직접 학습하는 목적이 명확하다.
2. transition point에 민감한 step-size gradient가 단순하고 구현하기 쉽다.
3. $1/\sqrt{NQ_P}$ heuristic을 derivation, update-ratio plot, accuracy ablation으로 검증한다.
4. ResNet-18부터 152, VGG, SqueezeNext까지 architecture 폭이 넓다.
5. 2/3/4/8-bit, top-1/top-5, KD와 model-size frontier를 모두 보고한다.
6. appendix에 detach 기반 재현 pseudocode가 있다.

## 한계

1. ImageNet classification CNN에 집중하며 detection, segmentation, Transformer는 검증하지 않았다.
2. 실제 low-bit kernel latency, peak RAM, power, energy를 측정하지 않았다.
3. training은 fake quantization이라 low-bit training-memory 절감이 없다.
4. weight와 activation 모두 per-layer scalar scale이라 channel outlier에 취약할 수 있다.
5. first/last layer 8-bit 및 다른 parameter FP32 때문에 전체 model이 균일 2/3/4-bit는 아니다.
6. 2-bit SqueezeNext처럼 architecture에 따른 민감도가 크다.
7. KD가 필요한 결과와 LSQ 단독 결과를 구분해야 한다.
8. 3-bit의 theoretical size는 좋지만 commodity hardware support가 드물다.

## 재현 체크리스트

### Quantizer

- [ ] weight는 signed, activation은 unsigned range를 사용했다.
- [ ] 각 weight/activation layer에 별도 FP32 scalar step을 만들었다.
- [ ] first와 last matrix layer를 8-bit로 유지했다.
- [ ] $s_0=2\langle|v|\rangle/\sqrt{Q_P}$로 초기화했다.
- [ ] activation step은 첫 batch 통계로 초기화했다.
- [ ] per-channel 확장을 original LSQ 결과와 구분했다.

### Gradient

- [ ] round forward는 유지하고 backward만 STE로 통과시켰다.
- [ ] clip 밖 input gradient를 0으로 했다.
- [ ] step gradient의 세 piecewise branch를 구현했다.
- [ ] weight에는 $1/\sqrt{N_WQ_P}$를 사용했다.
- [ ] activation에는 $1/\sqrt{N_FQ_P}$를 사용했다.
- [ ] gradient scale이 forward step 값을 바꾸지 않는지 test했다.

### Training

- [ ] pretrained FP32 checkpoint에서 시작했다.
- [ ] FP32 master weight와 optimizer state를 유지했다.
- [ ] 2/3/4-bit를 90 epoch fine-tune했다.
- [ ] precision별 LR과 weight decay를 기록했다.
- [ ] cosine schedule과 data augmentation을 맞췄다.
- [ ] KD 사용 여부와 teacher baseline을 별도 표기했다.

### Real deployment

- [ ] fake-quant accuracy와 packed integer kernel을 구분했다.
- [ ] target runtime의 W/A bit-width 지원을 확인했다.
- [ ] 3-bit가 INT8 container로 fallback하지 않는지 확인했다.
- [ ] accumulator width와 requantization을 검사했다.
- [ ] scale, BN, bias fusion 여부를 operator trace로 확인했다.
- [ ] model file, resident weight, activation peak, workspace를 따로 측정했다.
- [ ] batch 1 p50/p95, power, thermal throttling을 기록했다.

## 최종 평가

LSQ의 지속적인 가치는 low-bit network의 step size를 distribution-fitting hyperparameter가 아니라 task-loss parameter로 바꾼 데 있다. transition-aware surrogate gradient와 $1/\sqrt{NQ_P}$ gradient normalization은 간단하지만, 2-bit처럼 민감한 영역에서 수렴과 accuracy를 크게 바꾼다. 또한 학습된 step이 MAE, MSE, KL optimum과 상당히 다르다는 분석은 `local quantization error가 작을수록 task accuracy가 좋다`는 직관을 직접 반박한다.

온디바이스 관점에서 이 논문은 좋은 QAT 알고리즘 기준점이지만 deployment paper는 아니다. 학습에는 FP32 master weight와 fake-quant tensor가 필요하고, 실제 이득은 target이 packed W2A2/W3A3/W4A4 convolution을 지원할 때만 발생한다. 따라서 재현의 완성 조건은 ImageNet top-1뿐 아니라 실제 artifact의 bit packing, operator trace, peak activation, p95 latency, power를 확인하는 것이다. 특히 3-bit accuracy와 3-bit hardware 효율을 같은 것으로 취급하지 않는 것이 가장 중요한 독해 원칙이다.

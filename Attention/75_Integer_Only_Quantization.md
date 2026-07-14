# 75. Integer-Only Quantization

## 논문 정보

- 원본 파일: `75_Integer_Only_Quantization.pdf`
- 제목: **Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference**
- 저자: Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko
- 공개: arXiv:1712.05877, v1, 2017-12-15
- 공식 링크: [arXiv:1712.05877](https://arxiv.org/abs/1712.05877)
- 태스크: 8-bit integer-only inference, quantization-aware training, mobile classification and detection
- 핵심 키워드: affine quantization, scale, zero-point, uint8, int32 accumulator, fake quantization, batch-norm folding, gemmlowp, ARM NEON

## 한눈에 보는 요약

이 논문은 neural network의 weight와 activation을 8-bit integer로 표현하고, convolution/matrix multiplication부터 bias, activation, requantization까지 추론 경로를 integer arithmetic으로 실행하는 체계를 제시한다. 단순히 model 파일을 4배 줄이는 데 그치지 않고 실제 ARM CPU에서 latency-accuracy trade-off가 좋아지는지를 측정한다.

```text
floating-point training variables
  -> insert fake-quant nodes in forward pass
  -> train with quantization error
  -> fold batch normalization
  -> export uint8 weights/activations + int32 bias

integer inference layer
  uint8 input x uint8 weight
  -> int32 accumulation
  -> int32 bias add
  -> fixed-point multiplier + rounded shift
  -> clamp / ReLU6
  -> uint8 output
```

핵심 수식은 real value `r`과 quantized integer `q`를 잇는 affine mapping이다.

```math
r=S(q-Z)
```

`S`는 scale, `Z`는 real zero를 정확히 나타내는 zero-point다. Asymmetric zero-point를 허용하면서도 matrix multiplication의 중심 비용을 `uint8 x uint8 -> int32`로 유지하고, scale ratio는 fixed-point multiplier와 bit shift로 처리한다.

MobileNet SSD COCO에서 100% width model은 float `22.1 mAP, 778/370 ms`에서 INT8 `21.7 mAP, 687/272 ms`가 되고, 50% model은 float `16.7 mAP, 270/121 ms`에서 INT8 `16.6 mAP, 146/61 ms`가 된다. 두 latency 열은 Snapdragon 835 LITTLE/big 단일 core다.

이 리뷰는 2017년 논문이 실제 제안한 per-array uint8 QAT와 오늘날 흔한 symmetric int8, per-channel quantization, PTQ calibration을 분리해서 설명한다. 후자는 현대적 구현 관행이지 모두 이 논문의 원 기여가 아니다.

## 왜 weight-only compression으로 충분하지 않은가

Weight만 8-bit로 줄여도 저장 공간과 DRAM traffic은 줄지만 activation이 float면 convolution kernel은 여전히 floating-point 또는 mixed conversion 비용을 가진다. 논문은 weight와 activation을 함께 8-bit로 만들어 integer multiply-accumulate hardware를 직접 사용한다.

```text
weight-only quantization:
  small model file, but activation and compute may remain float

integer-only quantization:
  int8/uint8 operands + int32 accumulator + integer requantization
```

Binary/ternary network는 곱셈을 bit operation으로 바꿀 수 있지만 당시 일반 CPU의 multiply-add pipeline에서는 8-bit SIMD가 더 실용적일 수 있고, 1-bit 표현은 정확도 손실이 크다. 논문은 custom hardware가 아니라 Snapdragon ARM core에서 이를 검증하려 한다.

## Affine quantization

### 기본 mapping

Quantized integer `q`를 real value로 해석하는 식은 다음과 같다.

```math
r=S(q-Z)
```

- `q`: 저장되는 B-bit integer
- `S>0`: real step size
- `Z`: real zero에 대응하는 integer zero-point
- 8-bit uint8이면 일반적으로 `q in [0,255]`

반대 방향은 rounding과 saturation을 포함한다.

```math
q
=
\operatorname{clamp}
\left(
\operatorname{round}\left(\frac{r}{S}\right)+Z,
q_{min},q_{max}
\right)
```

### Zero가 정확히 표현되어야 하는 이유

Convolution boundary의 zero padding, ReLU의 zero, sparse activation에서 `r=0`이 자주 등장한다. Zero가 quantization grid에 없으면 padding이 실제로 작은 nonzero가 되어 convolution 결과에 bias가 누적된다. `Z`를 integer로 선택해 `q=Z`일 때 `r=0`을 보장한다.

```math
r=0\Rightarrow q=Z
```

Asymmetric range `[a,b]`에서 zero를 포함시키면 양의 activation 범위를 효율적으로 쓸 수 있다. ReLU6는 `[0,6]`이라는 자연스러운 bounded range가 있어 quantization에 유리하다.

### Per-array parameter

논문은 각 weight array와 activation array마다 하나의 `(S,Z)`를 사용한다. 같은 convolution의 모든 output channel이 한 scale을 공유한다. 이 제약 때문에 channel별 weight range가 100배 이상 다르면 작은-range channel의 상대 오차가 커지는 문제가 생긴다고 논문이 직접 지적한다.

오늘날 흔한 per-output-channel weight scale은 이 문제를 완화하지만 원문 scheme의 핵심 실험 설정과 다르다. 재현할 때 `per-tensor`와 `per-channel` 결과를 구분해야 한다.

## Integer-only matrix multiplication

### Real matrix에서 quantized matrix로

세 matrix가 `r3=r1 r2`를 만족하고 각 matrix의 quantization parameter가 `(S1,Z1)`, `(S2,Z2)`, `(S3,Z3)`라고 하자.

```math
r_\alpha^{(i,j)}
=
S_\alpha
\left(q_\alpha^{(i,j)}-Z_\alpha\right)
```

Matrix multiplication을 대입하면:

```math
S_3(q_3^{(i,k)}-Z_3)
=
\sum_{j=1}^{N}
S_1(q_1^{(i,j)}-Z_1)
S_2(q_2^{(j,k)}-Z_2)
```

정리하면:

```math
q_3^{(i,k)}
=
Z_3
+M
\sum_{j=1}^{N}
(q_1^{(i,j)}-Z_1)
(q_2^{(j,k)}-Z_2)
```

여기서:

```math
M=\frac{S_1S_2}{S_3}
```

합 내부는 integer이고 유일한 real constant는 `M`이다. `S1`, `S2`, `S3`가 fixed tensor range에서 정해지므로 `M`은 offline에 계산할 수 있다.

### Fixed-point multiplier

논문은 경험적으로 `M in (0,1)`인 경우를 다루고 다음처럼 정규화한다.

```math
M=2^{-n}M_0,
\qquad
M_0\in[0.5,1)
```

`M0`를 int32 fixed-point로 표현하면 대략 `round(2^31 M0)`이다. Multiplication은 fixed-point multiply, `2^-n`은 rounding right shift로 구현한다.

```text
int32 accumulator
  -> multiply by quantized M0
  -> rounding right shift n bits
  -> add output zero-point Z3
  -> saturation to uint8
```

Scale metadata가 float로 저장될 수 있어도 runtime inner loop에서 float arithmetic을 할 필요는 없다. Model preparation 단계에서 integer multiplier와 shift를 만든다.

## Zero-point를 효율적으로 처리하기

Naive 구현은 inner product의 매 operand에서 zero-point를 빼야 한다. 그러면 `N^3` 규모의 subtraction과 operand widening이 생긴다. 논문은 곱을 전개한다.

```math
\sum_j(q_{1j}-Z_1)(q_{2j}-Z_2)
```

```math
=
\sum_j q_{1j}q_{2j}
-Z_1\sum_j q_{2j}
-Z_2\sum_j q_{1j}
+NZ_1Z_2
```

Matrix index를 포함한 원문 형태는:

```math
q_3^{(i,k)}
=Z_3+M
\left(
NZ_1Z_2
-Z_1a_2^{(k)}
-Z_2\bar a_1^{(i)}
+\sum_j q_1^{(i,j)}q_2^{(j,k)}
\right)
```

```math
a_2^{(k)}=\sum_jq_2^{(j,k)},
\qquad
\bar a_1^{(i)}=\sum_jq_1^{(i,j)}
```

Row/column sum은 총 `O(N^2)`에 계산할 수 있고, 중심 `O(N^3)` 연산은 zero-point 없는 경우와 동일한 dot product다.

```text
main kernel:       sum uint8 * uint8 -> int32
correction terms: precomputed row/column sums and constants
```

Weights는 고정이므로 관련 sum을 model load 시 미리 계산할 수도 있다. Activation sum은 runtime에 필요하지만 core multiplication보다 낮은 차수다.

## Fused integer layer

### Data type 흐름

논문의 typical fused convolution은 다음과 같다.

```text
uint8 weights
uint8 input
  -> int32 accumulator
  -> add int32 bias
  -> fixed-point requantization
  -> saturating cast to uint8
  -> clamp for ReLU/ReLU6
  -> uint8 output
```

Core multiply-add는:

```text
int32 += uint8 * uint8
```

8-bit product를 많은 channel에 걸쳐 더하므로 accumulator는 int32가 필요하다. Kernel dimension이 매우 크면 int32 overflow 가능성을 따로 검토해야 한다.

### Bias quantization

Bias는 int32로 저장하고 zero-point는 0, scale은 input과 weight scale의 곱으로 둔다.

```math
S_{bias}=S_1S_2,
\qquad
Z_{bias}=0
```

이렇게 하면 convolution int32 accumulator와 bias가 같은 real scale이라 integer addition을 바로 할 수 있다. Bias parameter 수는 전체 weight에 비해 작으므로 int32 저장 비용이 미미하다.

Bias 오차는 많은 spatial output에 반복해서 더해져 nonzero-mean error를 만들 수 있으므로 높은 precision이 중요하다. 논문은 단순 편의를 위해 bias를 32-bit로 둔 것이 아니다.

### Requantization과 activation

Int32 accumulator를 output uint8 scale로 바꾸는 과정이 requantization이다.

```text
accumulator in S1*S2 scale
  -> multiply M=(S1*S2)/S3
  -> round
  -> add Z3
  -> clamp [0,255]
```

ReLU와 ReLU6는 output quantized range에서 clamp로 표현할 수 있다. Convolution, bias, requantization, activation을 하나의 fused operator로 실행하면 중간 int32 tensor를 memory에 쓰지 않아도 된다.

논문은 fusion을 단순 최적화 이상으로 본다. Training fake-quant node 위치가 inference fused operator의 8-bit input/output boundary와 같아야 forward simulation이 실제 arithmetic을 반영한다.

## Quantization-aware training

### 왜 naive post-training quantization이 실패하는가

논문은 큰 model은 float training 후 quantization만으로도 충분할 수 있지만 작은 model에서는 accuracy drop이 크다고 관찰한다. 대표 실패 원인은:

1. Output channel별 weight range가 100배 이상 다르지만 하나의 scale을 공유함
2. Outlier weight 하나가 range를 넓혀 나머지 weight resolution을 낮춤
3. Activation range가 training 중 변하고 calibration이 안정적이지 않음

그래서 forward pass에 quantization rounding을 시뮬레이션한다. Weight와 bias master copy, optimizer state, backward 계산은 float로 유지한다.

```text
float master weights
  -> fake quantize for forward
  -> float convolution using rounded values
  -> fake quantize activation
  -> loss
  -> straight-through style gradient to float master weights
```

논문은 오늘날 용어의 straight-through estimator를 중심 이론으로 전개하지는 않지만, non-differentiable rounding을 forward에 넣고 backpropagation은 conventional 방식으로 진행한다고 설명한다.

### Fake quantization 함수

Range `[a,b]`, level 수 `n`에서 step은:

```math
s(a,b,n)=\frac{b-a}{n-1}
```

Clamp:

```math
\operatorname{clamp}(r;a,b)
=
\min(\max(r,a),b)
```

Fake quantized real output은:

```math
q(r;a,b,n)
=
\operatorname{round}
\left(
\frac{\operatorname{clamp}(r;a,b)-a}{s(a,b,n)}
\right)
s(a,b,n)+a
```

여기서 첫 `q`는 integer code가 아니라 quantization grid로 round된 real value를 나타내는 함수 표기다. `n=2^8=256`이면 8-bit simulation이다.

Range boundary는 real zero가 정확히 representable하도록 nudging한다. 최종 scale과 zero-point는 이 boundary에서 유도된다.

### Weight range와 activation range

Weight는 기본적으로 `a=min(w)`, `b=max(w)`를 사용한다. ARM optimization을 위해 int8 변환 후 `[-127,127]`만 쓰고 `-128`은 피하도록 조정한다.

Activation은 input-dependent하므로 training 중 관찰한 min/max를 EMA로 누적한다. Smoothing parameter는 1에 가깝게 두어 수천 step을 평활화한다.

초기에는 activation distribution이 빠르게 변하므로 quantization을 50K에서 2M step까지 지연하는 것이 유용하다고 설명한다. Experimental protocol의 detection은 `500,000 steps` 지연한다.

```text
early training: float activation, collect/stabilize ranges
later training: enable activation fake quant
final: export fixed ranges
```

### 전체 algorithm

1. Float training graph 생성
2. Inference에서 downcast될 위치에 fake-quant op 삽입
3. Simulated quantized mode로 convergence까지 학습
4. Low-bit engine용 inference graph 생성 및 최적화
5. Quantized graph로 integer inference

QAT checkpoint의 float weight를 저장했다고 해서 runtime도 float인 것은 아니다. Export converter가 scale, zero-point, quantized buffer, fused operator를 올바르게 만들어야 한다.

## Batch normalization folding

Training graph에서 convolution과 BN은 분리되지만 inference에서는 BN을 weight와 bias에 fold한다. Quantization error를 정확히 시뮬레이션하려면 **fold된 weight를 quantize**해야 한다.

Convolution weight `w`, BN scale `gamma`, batch variance EMA를 쓰면:

```math
w_{fold}
=
\frac{\gamma w}
{\sqrt{EMA(\sigma_B^2)+\epsilon}}
```

Bias도 BN mean, beta와 함께 합성된다. `gamma/sqrt(var)`가 channel별 range를 크게 바꿀 수 있으므로 unfused raw weight를 quantize한 뒤 BN을 fold하면 실제 inference weight error와 다르다.

```text
wrong simulation:
  fake_quant(w) -> convolution -> BN

paper's intended simulation:
  fold convolution and BN -> fake_quant(w_fold) -> convolution
```

Residual connection에서는 addition 이후 activation이 다시 8-bit로 내려가는 위치에 fake quantization을 둔다. Graph boundary와 runtime fusion boundary의 일치가 핵심이다.

## Addition, concatenation, nonlinear function

### Addition

두 activation의 scale이 다르면 한쪽을 다른 scale로 fixed-point rescale한 뒤 더하고 output scale로 다시 rescale해야 한다. Float add보다 비쌀 수 있다.

```text
x1(S1,Z1) --rescale--+
                     +-> integer add -> output rescale
x2(S2,Z2) -----------+
```

Residual-heavy architecture는 convolution MAC만 보지 말고 rescale operator 수와 fusion을 확인해야 한다.

### Concatenation

Concatenation은 원래 lossless data movement여야 하는데 input scale이 다르면 lossy rescale이 필요하다. 논문은 concat의 모든 input과 output이 같은 quantization parameter를 가지도록 요구해 arithmetic-free concat을 만든다.

이 제약은 converter의 range propagation을 복잡하게 만들 수 있다. 현대 graph compiler가 concat 전후 requantize를 자동 삽입하면 정확도와 latency가 달라진다.

### Tanh, sigmoid, softmax

Lookup table 없이 pure fixed-point arithmetic으로 구현할 수 있다고 Appendix에서 설명한다. 그러나 typical fused convolution에는 넣지 않고 별도 operator로 다룬다. 오늘날 accelerator에서 이 operator가 CPU fallback되는지 반드시 확인해야 한다.

## ARM NEON 구현 세부

### Fixed-point rounding

Normalized fixed-point multiply는 `SQRDMULH` instruction과 대응한다. `SQDMULH`가 아니라 rounding variant가 중요하다.

Right shift의 tie-breaking도 중요하다. NEON `RSHL`은 tie를 upward로 round할 수 있어 negative value에 overall upward bias가 생긴다. 논문은 fix-up arithmetic으로 round-to-nearest behavior를 구현한다. 작은 rounding bias도 layer를 거치며 accuracy를 크게 낮출 수 있다.

### uint8에서 int8 kernel로

Operand와 zero-point에서 128을 빼 uint8을 int8로 바꾸면 core는:

```text
int32 += int8 * int8
```

Weight가 `-128`을 사용하지 않도록 학습 range를 조정했으므로 `-128 * -128` product가 없다. Product absolute value가 `2^14`보다 작아 local int16 accumulator에 두 product를 모은 뒤 int32에 accumulate할 수 있다.

논문은 SMULL, SMLAL, SADALP를 조합한 8-way SIMD 경로를 설명한다. 이는 quantization scheme이 hardware instruction과 co-design되었다는 좋은 예다.

## 실험 결과

### ResNet ImageNet

| Depth | float accuracy | integer-quantized accuracy | absolute drop |
|---|---:|---:|---:|
| 50 | 76.4% | 74.9% | 1.5 pp |
| 100 | 78.0% | 76.6% | 1.4 pp |
| 150 | 78.8% | 76.7% | 2.1 pp |

ResNet-50 비교에서 BWN 1-bit weight/float activation `68.7%`, TWN 2-bit/float `72.5%`, INQ 5-bit/float `74.8%`, FGQ 2-bit weight/8-bit activation `70.8%`, 논문 8/8-bit `74.9%`다. Bit 수만 낮추는 것이 accuracy-latency frontier를 보장하지 않음을 보여 준다.

### Inception-v3

| Activation | precision | accuracy | recall@5 |
|---|---|---:|---:|
| ReLU6 | float | 78.4% | 94.1% |
| ReLU6 | 8-bit | 75.4% | 92.5% |
| ReLU6 | 7-bit | 75.0% | 92.4% |
| ReLU | float | 78.3% | 94.2% |
| ReLU | 8-bit | 74.2% | 92.2% |
| ReLU | 7-bit | 73.7% | 92.0% |

ReLU6가 bounded activation range를 제공해 ReLU보다 quantization degradation이 작다. 7-bit와 8-bit 차이는 작지만 float 대비 drop 자체는 남는다.

### MobileNet ImageNet

논문은 다양한 width multiplier와 resolution을 Snapdragon 835 LITTLE, 835 big, 821 big에서 측정한다. 같은 latency budget에서 8-bit point가 float보다 높은 accuracy를 보인다. 835 LITTLE의 실시간 기준 약 33 ms에서 accuracy gap이 약 10 percentage points라고 해석한다.

이는 `같은 architecture의 accuracy drop`만 보는 대신 `같은 latency에서 더 큰 quantized model을 실행`하는 system-level 비교다. Quantization의 이득은 float와 int kernel의 해당 hardware 최적화 수준에 의존하며 Snapdragon 821에서는 차이가 작다.

### MobileNet SSD COCO

입력은 320 x 320, latency는 Snapdragon 835 단일 core다.

| Width | type | mAP | LITTLE ms | big ms |
|---|---|---:|---:|---:|
| 100% | float | 22.1 | 778 | 370 |
| 100% | 8-bit | 21.7 | 687 | 272 |
| 50% | float | 16.7 | 270 | 121 |
| 50% | 8-bit | 16.6 | 146 | 61 |

50% model big core는 `121 -> 61 ms`, 거의 2배 개선이고 mAP drop은 0.1이다. 100% LITTLE은 `778 -> 687 ms`로 개선폭이 작다. Model과 core에 따라 speedup이 일정하지 않다는 중요한 결과다.

### Face detection

Precision/recall은 100% float `68/76%`, INT8 `66/75%`; 50% float `65/70%`, INT8 `62/70%`; 25% float `56/64%`, INT8 `54/63%`다.

Latency 표의 일부를 보면:

| Width | type | LITTLE 1 core | LITTLE 4 cores | big 1 core | big 4 cores |
|---|---|---:|---:|---:|---:|
| 100% | float | 711 | - | 337 | - |
| 100% | 8-bit | 372 | 167 | 154 | 69 |
| 50% | float | 233 | - | 106 | - |
| 50% | 8-bit | 134 | 74 | 56 | 30 |
| 25% | float | 100 | - | 44 | - |
| 25% | 8-bit | 67 | 43 | 28 | 18 |

25% INT8은 big 1 core에서 28 ms, 약 36 FPS이고 float는 44 ms, 약 23 FPS다. Multi-core speedup은 1.5배에서 2.2배 범위다.

### Bit-depth ablation

Face attribute 결과는 weight가 activation보다 bit 감소에 더 민감하고, 8/7-bit가 float와 가깝고, 총 bit budget이 같다면 weight와 activation bit를 균형 있게 두는 편이 낫다고 해석한다.

예를 들어 average category precision drop은 weight 8/activation 8에서 `-0.9%`, w8/a4에서 `-3.5%`, w4/a8에서 `-11.4%`, w4/a4에서 `-14.0%`다. Age precision은 w8/a8 `-1.3%`, w4/a4 `-19.5%`다.

이 결과를 모든 현대 architecture에 일반화할 수는 없다. Transformer activation outlier나 depthwise convolution channel range는 다른 sensitivity를 보일 수 있다.

## 메모리와 bandwidth 계산

Parameter 수 `P`에서 raw weight 저장량은:

```math
\text{FP32 bytes}=4P,
\qquad
\text{INT8 bytes}=P
```

따라서 이상적으로 4배 절감한다. 25M parameter model이면:

```text
FP32: 100,000,000 bytes = 95.4 MiB
INT8:  25,000,000 bytes = 23.8 MiB
```

Activation도 FP32에서 INT8로 저장하면 같은 element 수에서 4배 줄어든다. 예를 들어 `[1,112,112,32]` tensor는 401,408 elements다.

```text
FP32: about 1.53 MiB
INT8: about 0.383 MiB
```

하지만 int32 accumulator tile, im2col workspace, alignment padding, scale metadata가 추가된다. Fused kernel이 중간 accumulator를 materialize하지 않으면 peak가 크게 줄고, fallback kernel이 전체 int32 tensor를 만들면 예상보다 커진다.

## 구현 의사코드

### Quantize/dequantize reference

```python
def quantize_affine(x, scale, zero_point, qmin=0, qmax=255):
    q = round(x / scale) + zero_point
    return clamp(q, qmin, qmax).astype("uint8")

def dequantize_affine(q, scale, zero_point):
    return scale * (q.astype("int32") - zero_point)
```

### Integer requantization

```python
def requantize(acc_int32, multiplier_int32, right_shift, out_zero):
    y = saturating_rounding_high_mul(acc_int32, multiplier_int32)
    y = rounding_divide_by_power_of_two(y, right_shift)
    y = y + out_zero
    return clamp(y, 0, 255).astype("uint8")
```

Reference 결과와 optimized kernel 결과를 bit-exact 또는 허용 오차 기준으로 비교해야 한다.

### QAT loop

```python
for step, (x, y) in enumerate(loader):
    w_qdq = fake_quant(fold_bn(weights, bn_state), weight_range)

    if step < activation_quant_delay:
        act = forward_with_weights(x, w_qdq)
        update_activation_ema(act)
    else:
        act = fake_quant(forward_with_weights(x, w_qdq), activation_ema)

    loss = criterion(head(act), y)
    loss.backward()          # update float master weights
    optimizer.step()
```

실제 framework에서는 graph rewrite가 BN fold와 fake-quant position을 일관되게 관리해야 한다.

## 원 논문과 현대적 관행 분리

### 원 논문의 직접 기여

- `r=S(q-Z)` affine quantization과 exact zero representation
- Per-array uint8 weight/activation, int32 accumulator/bias
- Fixed-point multiplier와 rounded shift를 이용한 integer-only requantization
- Arbitrary zero-point의 효율적 row/column-sum 처리
- Inference fusion boundary를 반영한 simulated quantization training
- Activation range EMA와 delayed quantization
- BN folding 뒤 weight quantization
- ARM NEON instruction 수준 구현과 실제 Snapdragon latency

### 오늘날 흔하지만 이 논문의 원 실험과 다른 것

- Signed int8 activation을 기본으로 하는 symmetric quantization
- Per-channel 또는 per-group weight scale
- Calibration-only PTQ, percentile/MSE/KL observer
- SmoothQuant, AWQ, GPTQ 같은 Transformer/LLM 방식
- INT4 weight-only, W4A16
- Learned step size와 trainable clipping
- NPU-specific mixed precision compiler search

현대 backend에서 이 논문의 affine 원리는 이어지지만 data type, scale granularity, operator contract는 달라질 수 있다.

## 온디바이스 배포 체크 포인트

### Quantized model인지 integer-only graph인지 구분

Model 파일에 INT8 weight가 있어도 activation 사이에 dequantize/quantize가 반복되면 integer-only가 아니다.

```text
good path:
  int8 conv -> int8 conv -> int8 add -> int8 output

fallback path:
  int8 -> dequantize -> float op -> quantize -> int8
```

Profiler에서 float fallback, conversion node, CPU-GPU copy를 확인해야 한다.

### Operator별 scale constraint

Residual add와 concat은 scale 일치가 중요하다. Detection head의 sigmoid, NMS, decode는 float에 남을 수 있다. Vision Transformer는 LayerNorm, softmax, reshape가 integer accelerator를 끊을 수 있다.

### 측정 protocol

- 동일 image resolution과 batch=1
- Float와 INT8 모두 같은 thread/core affinity
- Warm-up 후 p50/p95
- Preprocessing과 postprocessing 포함/제외 두 값
- Peak RSS와 accelerator memory
- Model load/cold start
- Sustained power, temperature, throttling
- 정확도 metric과 per-class/rare class 분석

논문처럼 device core 종류가 바뀌면 quantization 이득도 달라진다. FLOPs만으로 speedup을 예측하면 안 된다.

## 강점

1. 수학적 quantization mapping부터 ARM instruction까지 end-to-end로 연결한다.
2. Weight뿐 아니라 activation과 bias까지 integer inference contract를 명시한다.
3. Asymmetric zero-point의 비용을 core GEMM에서 효율적으로 분리한다.
4. Training graph와 fused inference graph의 일치를 강조한다.
5. MobileNet, SSD를 실제 Snapdragon CPU에서 측정한다.
6. Accuracy뿐 아니라 latency-accuracy frontier를 비교한다.
7. Addition, concat, rounding처럼 구현에서 자주 놓치는 세부를 다룬다.

## 한계와 주의점

### 오래된 hardware와 software stack

Snapdragon 821/835, 당시 TensorFlow Lite와 gemmlowp 결과다. 현대 ARM dot-product, GPU, NPU, CoreML/NNAPI backend의 상대 speedup은 다시 측정해야 한다.

### Per-array scale의 정확도 한계

Channel range 차이가 큰 문제를 인식하지만 main scheme은 per-array다. 현대 per-channel quantization보다 accuracy가 낮을 수 있다.

### 평균 latency 중심

p95, energy, thermal throttling, peak memory가 없다. Multi-core 표도 float baseline의 multi-core 값이 없어 완전한 scaling 비교가 어렵다.

### Architecture 범위

주로 CNN, ReLU/ReLU6, MobileNet/SSD다. Transformer의 LayerNorm, softmax, activation outlier, KV cache에는 별도 기법이 필요하다.

### QAT 비용

Accuracy를 회복하지만 training pipeline과 graph rewrite가 복잡해진다. 이미 배포된 model에 data 없이 즉시 적용하는 PTQ와는 다른 요구사항이다.

## 재현 체크리스트

### Quantization contract

- [ ] Weight/activation data type이 uint8인지 int8인지 기록한다.
- [ ] Tensor별 `(S,Z)`와 qmin/qmax를 dump한다.
- [ ] Zero가 정확히 representable한지 확인한다.
- [ ] Bias scale이 `S_input*S_weight`, zero-point 0인지 확인한다.
- [ ] Accumulator overflow bound를 계산한다.
- [ ] Fixed multiplier와 shift rounding이 reference와 일치하는지 검사한다.

### QAT graph

- [ ] Weight fake quant가 BN fold 뒤에 있는지 확인한다.
- [ ] Activation fake quant 위치가 fused operator boundary와 같은지 확인한다.
- [ ] Range EMA smoothing과 activation quant delay를 기록한다.
- [ ] Residual add와 concat scale constraint를 검사한다.
- [ ] Float master weight와 optimizer state를 유지한다.

### Export와 runtime

- [ ] Export 전후 float/QAT graph accuracy를 비교한다.
- [ ] Integer reference와 optimized kernel output을 비교한다.
- [ ] Quantize/dequantize node 수를 센다.
- [ ] Unsupported operator와 execution-provider fallback을 기록한다.
- [ ] NMS, resize, preprocessing이 어느 precision인지 확인한다.

### Benchmark

- [ ] FP32/FP16/INT8의 model size와 peak memory를 측정한다.
- [ ] 동일 core, thread, affinity, input에서 latency를 잰다.
- [ ] Cold/warm p50/p95와 sustained latency를 구분한다.
- [ ] Accuracy, per-class error, calibration을 함께 기록한다.
- [ ] 전력, 온도, throttling을 포함한다.

## 추천 ablation

| 축 | 설정 |
|---|---|
| Weight scale | per-tensor, per-channel |
| Activation | asymmetric uint8, symmetric int8 |
| Training | float baseline, PTQ, QAT |
| Range | min-max, percentile, learned clipping |
| Precision | W8A8, W8A16, FP16 |
| Runtime | CPU 1/4 cores, GPU/NPU if supported |
| Metric | accuracy, p50/p95, peak memory, power |

원 논문 재현 표에는 per-array uint8 QAT를 중심에 두고, 현대 extension은 별도 열로 표시하는 것이 정직하다.

## 최종 평가

이 논문의 가장 중요한 기여는 `8-bit로 저장한다`는 아이디어가 아니라, quantization parameter, accumulator, bias, scale 변환, rounding, fused activation, training simulation, ARM kernel을 하나의 일관된 inference contract로 만들었다는 점이다. `r=S(q-Z)`라는 단순한 식이 실제 integer-only convolution으로 이어지려면 zero-point correction, fixed-point requantization, BN fold, graph boundary까지 모두 맞아야 한다.

온디바이스 연구에서 얻을 교훈은 세 가지다.

1. Model 파일의 bit 수가 아니라 실제 runtime operator의 data type을 확인한다.
2. Quantization speedup은 hardware kernel과 graph fusion에 의존하므로 target device에서 측정한다.
3. Accuracy뿐 아니라 동일 latency budget에서 더 큰 model을 쓸 수 있는지 frontier로 비교한다.

오늘날 per-channel INT8, INT4 weight-only, Transformer quantization으로 기술이 확장되었어도 이 원칙은 그대로 유효하다. 좋은 quantization 연구는 수식, graph, kernel, 실제 device 지표를 끝까지 연결해야 한다.

# 76. PTQ4ViT: Post-Training Quantization for Vision Transformers with Twin Uniform Quantization

## 논문 정보

- 저자: Zhihang Yuan, Chenhao Xue, Yiqi Chen, Qiang Wu, Guangyu Sun
- 소속: Peking University, Houmo AI
- 발표: ECCV 2022
- 공개: arXiv:2111.12293v3, 2024년 6월 23일
- 원본 파일: `76_PTQ4ViT.pdf`
- 논문 링크: https://arxiv.org/abs/2111.12293
- 공개 코드: https://github.com/hahnyuan/PTQ4ViT

## 한눈에 보는 요약

PTQ4ViT는 Vision Transformer를 label 없이 적은 calibration image만으로 weight와 activation까지 양자화하는 post-training quantization 방법이다. 저자들은 일반 CNN용 uniform PTQ를 ViT에 적용할 때 문제가 되는 두 activation을 찾는다.

1. Softmax 출력은 대부분 0 근처지만 일부 중요한 attention 값은 1에 가깝다.
2. GELU 출력은 작은 음수 범위와 훨씬 큰 양수 범위가 함께 있어 대칭 uniform quantizer에 맞지 않는다.

이를 위해 한 tensor를 두 uniform range로 나누는 twin uniform quantization을 제안하고, scale 선택에는 MSE나 cosine distance 대신 task loss sensitivity를 근사하는 Hessian-guided metric을 사용한다. ImageNet에서 32장만 calibration해 ViT, DeiT, Swin의 W8A8 정확도 하락을 모두 0.5 percentage point 아래로 줄였다.

다만 이 논문이 강하게 입증한 것은 quantized accuracy와 calibration 시간이다. 실제 mobile INT kernel의 end-to-end latency, peak memory, 전력은 보고하지 않는다. 특히 twin range format과 W6A6는 표준 mobile NPU가 바로 지원하는 affine INT8 graph가 아니므로 fake quantization 정확도와 real integer deployment를 구분해야 한다.

## 문제 설정

ViT block의 주 계산은 다음 matrix multiplication이다.

- linear: $XW_Q$, $XW_K$, $XW_V$, output projection, MLP FC1/FC2
- attention score: $QK^\top$
- attention aggregation: $PV$, where $P=\operatorname{softmax}(QK^\top/\sqrt d)$

일반적인 symmetric uniform quantizer는 floating value $x$를 $k$-bit signed integer로 바꾼다.

$$
x_q=\Psi_k(x,\Delta)
=\operatorname{clamp}\left(
\operatorname{round}\left(\frac{x}{\Delta}\right),
-2^{k-1},2^{k-1}-1
\right).
$$

matrix multiplication $O=AB$를 양자화하면

$$
A_q=\Psi_k(A,\Delta_A),
\qquad
B_q=\Psi_k(B,\Delta_B),
$$

$$
\hat O=\Delta_A\Delta_B A_qB_q.
$$

base PTQ는 $O$와 $\hat O$의 local distance가 최소인 $\Delta_A,\Delta_B$를 layer별로 찾는다. 그러나 ViT에서는 local reconstruction이 작다고 final cross-entropy 변화도 작다는 보장이 약하다.

## PTQ 대상과 제외 대상

논문의 quantization coverage는 다음과 같다.

- 모든 fully-connected layer의 weight와 input activation
- patch projection과 마지막 classifier 포함
- $QK^\top$와 $PV$ matrix multiplication의 두 input
- head마다 다른 attention quantization parameter
- $W_Q$, $W_K$, $W_V$에 서로 다른 scale

Softmax와 normalization operator 자체는 양자화하지 않는다. 이를 정확히 풀어 쓰면 다음과 같다.

```text
FP LayerNorm
  -> quantize X and Wq/Wk/Wv
  -> integer-like Q/K/V matmul
  -> FP scale / Softmax
  -> quantize post-softmax P and V
  -> integer-like P @ V
  -> FP residual / LayerNorm

FP GELU
  -> quantize post-GELU activation and FC2 weight
  -> integer-like FC2 matmul
```

즉 post-softmax와 post-GELU activation은 quantization 대상이지만 Softmax, GELU, LayerNorm의 nonlinear 계산이 모두 integer-only라는 뜻은 아니다. real deployment에서는 이 경계의 quantize/dequantize와 fusion 가능성이 latency를 좌우한다.

## 왜 일반 uniform quantization이 실패하는가

### Post-softmax distribution

attention probability $P$는 $[0,1]$에 있고 row sum이 1이다. 대부분의 patch pair는 거의 무관하므로 값이 0 근처에 몰리지만, 소수의 큰 값은 중요한 attention edge를 의미한다.

- 큰 scale: 1에 가까운 값을 포화 없이 보존하지만 작은 값 대부분이 0으로 collapse한다.
- 작은 scale: 작은 probability resolution은 좋아지지만 큰 attention이 clipping되어 강도가 사라진다.

MSE만 보면 작은 값이 수적으로 많아 그쪽을 보존하는 scale이 유리할 수 있다. 그러나 task에는 소수의 큰 attention이 더 중요할 수 있다.

### Post-GELU distribution

GELU는 음수 쪽이 작은 범위에 눌려 있고 양수 쪽은 unbounded다. 하나의 symmetric scale을 쓰면 다음 trade-off가 생긴다.

- positive tail을 덮으면 작은 negative 구간의 code가 거의 없다.
- negative resolution을 높이면 positive activation이 clipping된다.

asymmetric affine quantizer도 후보지만, 논문은 두 range가 각각 uniform인 custom format을 사용해 hardware 처리 가능성을 유지하려 한다.

## Twin uniform quantization

$k$-bit code의 most significant bit를 range flag로 사용하고 나머지 $k-1$ bits를 magnitude로 사용한다.

$$
T_k(x,\Delta_{R1},\Delta_{R2})=
\begin{cases}
\Psi_{k-1}(x,\Delta_{R1}), & x\in R_1,\\
\Psi_{k-1}(x,\Delta_{R2}), & \text{otherwise}.
\end{cases}
$$

일반 signed integer의 sign bit를 range flag로 재해석하는 셈이다. 두 range 안에서는 값의 부호가 이미 정해지거나 softmax처럼 모두 nonnegative이므로 별도 sign bit가 필요 없다.

### Softmax용 twin range

- $R_1$: 작은 probability를 작은 $\Delta^s_{R1}$로 촘촘하게 표현
- $R_2$: 전체 $[0,1]$을 덮어 큰 probability 보존
- $\Delta^s_{R2}=1/2^{k-1}$로 고정
- calibration에서는 $\Delta^s_{R1}$만 탐색

softmax scale search space는

$$
\Delta^s_{R1}\in
\left\{
\frac1{2^k},\frac1{2^{k+1}},\ldots,\frac1{2^{k+10}}
\right\}.
$$

한 값이 두 range에 모두 표현될 수 있으므로 code flag가 어느 resolution을 사용할지 결정한다.

### GELU용 twin range

- $R_1=[-2^{k-1}\Delta^g_{R1},0]$: negative activation
- $R_2=[0,2^{k-1}\Delta^g_{R2}]$: positive activation
- $\Delta^g_{R1}$은 negative range 전체를 덮도록 고정
- calibration에서는 positive scale $\Delta^g_{R2}$ 탐색

positive와 negative가 별도 scale을 가지므로 작은 음수와 큰 양수를 한 대칭 grid에 억지로 넣지 않는다.

### Power-of-two scale relation

서로 다른 range의 integer를 한 accumulator에서 정렬하려면 두 scale 비를 power of two로 제한한다.

$$
\Delta_{R2}=2^m\Delta_{R1},
\qquad m\in\mathbb{Z}_{\ge0}.
$$

예를 들어

$$
a_q\Delta_{R1}+b_q\Delta_{R2}
=(a_q+(b_q\ll m))\Delta_{R1}.
$$

FP32 multiply/add 대신 integer shift로 range를 맞출 수 있다는 주장이다. 그러나 이 custom range flag decode와 shift-accumulate가 target backend kernel에 실제로 구현되어야 한다. 표준 INT8 GEMM에 scale metadata만 붙이는 것으로 자동 지원되지는 않는다.

## Scale granularity

논문의 granularity를 정리하면 다음과 같다.

| 대상 | Scale granularity | 비고 |
|---|---|---|
| FC weight | layer-wise | $\Delta_W$ 하나 |
| FC input activation | layer-wise | $\Delta_X$ 하나 |
| attention $A,B$ inputs | attention head별 | head마다 $\Delta_A,\Delta_B$ |
| $W_Q,W_K,W_V$ | projection별 | 서로 다른 scale |
| post-softmax | head별 twin range로 해석 | small scale 탐색, global range 고정 |
| post-GELU | layer별 twin range | negative scale 고정, positive 탐색 |

현대 PTQ의 per-channel weight quantization보다 coarse한 layer-wise weight scale이다. 논문의 정확도는 6/8-bit에서 충분했지만 4-bit에서 크게 무너지는 원인 중 하나가 될 수 있다.

## Base scale search

$A_{max}=\max|A|$라 하면 candidate를 다음 구간에서 선형 생성한다.

$$
\Delta_A\in
\left[
\alpha\frac{A_{max}}{2^{k-1}},
\beta\frac{A_{max}}{2^{k-1}}
\right].
$$

$B$도 같은 방식이다. PTQ4ViT 설정은 $\alpha=0$, $\beta=1.2$, candidate 100개, alternating search 3 rounds다. 먼저 $\Delta_B$를 고정하고 $\Delta_A$를 찾고, 다음에는 반대로 한다.

Appendix의 Base PTQ는 $\alpha=0.5$, $\beta=1.2$, 1 round, cosine distance다. 따라서 Table 1의 base와 PTQ4ViT 차이는 twin/Hessian뿐 아니라 search range와 rounds 변화도 일부 포함한다. 주 ablation Table 3에서 각 구성요소 효과를 따로 봐야 한다.

## Hessian-guided metric

quantization을 weight perturbation $\epsilon$으로 보고 task loss를 2차 Taylor expansion한다.

$$
\mathbb{E}[L(W+\epsilon)]-\mathbb{E}[L(W)]
\approx
\epsilon^\top\bar g(W)
+\frac12\epsilon^\top\bar H(W)\epsilon.
$$

pretrained model이 local optimum에 가까우므로 first-order term을 무시한다. full Hessian은 너무 비싸므로 layer output $O_l$에서 diagonal Fisher approximation을 사용한다.

$$
\mathcal{M}_l(\Delta)=
\mathbb{E}\left[
(\hat O_l-O_l)^\top
\operatorname{diag}\left(
\left(\frac{\partial L}{\partial O_{l,1}}\right)^2,
\ldots,
\left(\frac{\partial L}{\partial O_{l,n}}\right)^2
\right)
(\hat O_l-O_l)
\right].
$$

직관은 output error를 모두 동일하게 보지 않고 final loss에 민감한 element의 error에 더 큰 가중치를 주는 것이다. MSE는 gradient weight가 모두 1인 특수한 경우로 볼 수 있다.

### Label 없는 calibration

ground-truth $y$가 없으므로 floating model prediction $y_{FP}$를 pseudo target으로 사용한다.

$$
L=CE(\hat y,y_{FP}).
$$

이 loss를 한 번 backward해 각 layer output gradient를 얻는다. supervised QAT는 아니지만 완전히 forward-only인 min-max calibration도 아니다. calibration environment에 autograd와 backward memory가 필요하다.

## Calibration algorithm

```python
 # phase 1: floating model statistics
with save_layer_outputs_to_host():
    logits_fp, outputs = forward_all_layers(calibration_images)
    pseudo_labels = argmax(logits_fp)
    loss = cross_entropy(logits_fp, pseudo_labels)
    gradients = backward_and_capture_dL_dO(loss, outputs)

 # phase 2: layer-parallel scale search
for layer in quantizable_layers:
    O = load_host_output(layer)
    G = load_host_gradient(layer)

    delta_w = absmax(layer.weight) / qmax
    act_candidates = make_scale_grid(alpha=0.0, beta=1.2, n=100)
    weight_candidates = make_scale_grid(alpha=0.0, beta=1.2, n=100)

    for _ in range(3):
        delta_a = argmin_hessian_metric(act_candidates, delta_w, O, G)
        delta_w = argmin_hessian_metric(weight_candidates, delta_a, O, G)

    save_scale(layer, delta_a, delta_w)
    release_host_cache(layer)
```

실제 algorithm은 먼저 모든 $O_l$을 forward로 수집하고 reverse order backward로 gradient를 수집한 뒤 layer별 scale을 찾는다. candidate별 $\hat O_l$ 계산을 batch로 묶어 GPU parallelism을 활용한다.

## Calibration memory 손계산

ViT-B/224의 표준 shape를 reviewer example로 사용한다. patch+CLS token 197, hidden 768, MLP expansion 3072다.

| Tensor | Elements | FP32 | INT8 |
|---|---:|---:|---:|
| hidden $[1,197,768]$ | 151,296 | 약 591 KiB | 약 148 KiB |
| all-head attention $[1,12,197,197]$ | 465,708 | 약 1.78 MiB | 약 455 KiB |
| post-GELU $[1,197,3072]$ | 605,184 | 약 2.31 MiB | 약 591 KiB |

calibration 중 post-GELU output과 gradient를 FP32로 함께 저장한다고 가정하면 image당 layer 하나에 약 4.62 MiB다. 12 blocks와 32 calibration images만 곱해도 약 1.73 GiB이며, 이는 MLP output 한 종류만 센 하한이다. attention과 다른 FC output까지 저장하면 더 커진다.

논문이 $O_l$과 gradient를 생성 즉시 GPU에서 host memory로 옮기는 이유다. 정확한 dtype과 실제 peak host memory는 보고하지 않으므로 이 수치는 shape 기반 reviewer estimate다.

## Batch=1 activation과 weight memory

W8A8의 이론적 tensor payload는 FP32 대비 4분의 1이다. 앞 표의 post-GELU처럼 ViT activation peak 후보 하나가 2.31 MiB에서 약 591 KiB로 줄 수 있다. 그러나 다음을 주의해야 한다.

- INT32 accumulator와 rescale buffer가 추가된다.
- Softmax, LayerNorm, GELU가 float이면 FP activation이 경계에 남을 수 있다.
- kernel이 quantized activation을 바로 소비하지 않으면 Q/DQ buffer가 동시 존재한다.
- INT6은 bit-packed storage가 필요하며 많은 runtime이 int8 container에 저장한다.

이론적인 packed INT6 post-GELU는 $605,184\times6/8\approx443$ KiB지만 int8 container를 쓰면 591 KiB다. 논문의 W6 model size가 실제 target runtime resident memory와 같다고 가정하면 안 된다.

## Deployment artifact와 peak memory 추가 손계산

### Weight file은 bit 수만으로 결정되지 않는다

ViT-B/16을 약 86.6M parameter라고 두는 reviewer estimate에서 raw weight payload 하한은 다음과 같다.

| 표현 | 계산 | 이상적 payload |
| --- | ---: | ---: |
| FP32 | $86.6\text{M}\times4$ | 346.4 MB |
| packed INT8 | $86.6\text{M}\times1$ | 86.6 MB |
| packed INT6 | $86.6\text{M}\times6/8$ | 64.95 MB |
| INT6 in INT8 container | $86.6\text{M}\times1$ | 86.6 MB |

이 표는 paper checkpoint의 실측 파일 크기가 아니라 bit-packing 하한이다. 실제 artifact에는 scale, zero-point 또는 twin range metadata, alignment padding, LayerNorm parameter, bias, graph constant가 더해진다. 특히 W6 code를 byte 하나에 저장하면 accuracy는 6-bit quantizer와 같아도 storage는 INT8과 같다. 따라서 `W6A6` label과 physical 6-bit packing을 따로 확인해야 한다.

### Q/DQ 경계의 순간 peak

mobile runtime이 FP16 operator와 INT8 matmul을 섞으면 quantize/dequantize 경계에서 두 buffer가 동시에 살아 있을 수 있다. ViT-B/224의 hidden tensor $197\times768=151{,}296$ element를 예로 들면

```math
M_{FP16}=151{,}296\times2\approx295.5\;KiB,
```

```math
M_{INT8}=151{,}296\times1\approx147.8\;KiB.
```

in-place 변환이 불가능하면 순간적으로 약 443.3 KiB가 필요하다. MLP post-GELU $197\times3072=605{,}184$ element에서는 FP16 약 1.154 MiB와 INT8 약 591 KiB가 겹쳐 약 1.73 MiB다. 이 역시 tensor 하나의 reviewer 계산이며 allocator 전체 peak가 아니다. Softmax, GELU, LayerNorm 사이의 Q/DQ가 여러 개 겹치거나 workspace가 추가되면 더 커질 수 있다.

### Head-wise scale은 metadata보다 kernel interface가 문제다

12-head ViT-B에서 head마다 FP32 scale 하나를 둔다고 해도 layer당 수십 byte 수준이라 model size 영향은 작다. 병목은 fused attention kernel이 head마다 다른 scale을 받아 accumulator와 rescale에 적용할 수 있는가다. backend가 per-tensor scale만 지원하면 다음 중 하나가 필요하다.

1. 모든 head를 하나의 scale로 재보정해 정확도를 다시 측정한다.
2. head별로 matmul을 분리해 kernel launch와 memory traffic 증가를 감수한다.
3. custom fused kernel을 구현한다.

따라서 granularity는 단순한 calibration hyperparameter가 아니라 operator ABI 조건이다.

## Real integer 검증 절차

PTQ4ViT의 algorithmic accuracy를 device result로 옮길 때는 세 단계 결과를 따로 보관하는 것이 안전하다.

| 단계 | 확인 대상 | 실패 예 |
| --- | --- | --- |
| Fake quant | Q/DQ를 통과한 floating output과 ImageNet accuracy | quantizer 구현 또는 scale search 오류 |
| Packed artifact | tensor dtype, byte 수, scale/range metadata | W6가 INT8 container, twin flag 누락 |
| Runtime trace | 실제 INT kernel, fallback, conversion, latency | Softmax/GELU 주변 FP graph와 CPU fallback |

구체적인 audit 순서는 다음과 같다.

```text
1. FP model과 fake-quant model의 layer output을 저장한다.
2. weight를 target format으로 pack한 뒤 file byte를 tensor별로 합산한다.
3. runtime이 읽은 resident tensor dtype과 scale granularity를 dump한다.
4. 같은 calibration sample에서 fake-quant와 real-kernel output을 layer별 비교한다.
5. operator trace로 FC, QK^T, PV가 integer kernel인지 확인한다.
6. LayerNorm, Softmax, GELU와 Q/DQ의 device placement를 확인한다.
7. batch=1 warm p50/p95, cold start, peak allocator, sustained power를 측정한다.
```

fake-quant와 real-kernel accuracy가 다르면 rounding mode, clamp range, accumulator overflow, scale fusion, layout 순서를 먼저 의심해야 한다. latency가 개선되지 않으면 unsupported twin quantizer, head-wise scale 분할, INT6 unpack, FP fallback과 memory copy를 trace해야 한다.

## ImageNet 결과

Table 1의 전체 핵심 결과는 다음과 같다. 괄호는 FP32 대비 drop이다.

| Model | FP32 | Base W8A8 | PTQ4ViT W8A8 | PTQ4ViT W6A6 |
|---|---:|---:|---:|---:|
| ViT-S/224/32 | 75.99 | 73.61 (-2.38) | 75.58 (-0.41) | 71.90 (-4.08) |
| ViT-S/224 | 81.39 | 80.46 (-0.91) | 81.00 (-0.38) | 78.63 (-2.75) |
| ViT-B/224 | 84.54 | 83.89 (-0.64) | 84.25 (-0.29) | 81.65 (-2.89) |
| ViT-B/384 | 86.00 | 85.35 (-0.64) | 85.82 (-0.17) | 83.34 (-2.65) |
| DeiT-S/224 | 79.80 | 77.65 (-2.14) | 79.47 (-0.32) | 76.28 (-3.51) |
| DeiT-B/224 | 81.80 | 80.94 (-0.85) | 81.48 (-0.31) | 80.25 (-1.55) |
| DeiT-B/384 | 83.11 | 82.33 (-0.77) | 82.97 (-0.13) | 81.55 (-1.55) |
| Swin-T/224 | 81.39 | 80.96 (-0.42) | 81.24 (-0.14) | 80.47 (-0.91) |
| Swin-S/224 | 83.23 | 82.75 (-0.46) | 83.10 (-0.12) | 82.38 (-0.84) |
| Swin-B/224 | 85.27 | 84.79 (-0.47) | 85.14 (-0.12) | 84.01 (-1.25) |
| Swin-B/384 | 86.44 | 86.16 (-0.26) | 86.39 (-0.04) | 85.38 (-1.04) |

W8A8은 모든 모델에서 drop 0.41 이하다. W6A6 평균 drop은 논문 기준 2.1 point로, base PTQ 평균 9.8보다 크게 줄었다. Swin은 local window로 softmax가 보는 token 수가 적어 distribution imbalance가 완화된다는 가설을 제시한다.

## 다른 PTQ와 model size 비교

DeiT-S에서 PTQ4ViT는 32 calibration images로 W8A8 79.47, W6A6 76.28이다. 1,024 images를 쓰는 EasyQuant는 76.59/73.26, Liu et al.은 77.47/74.58, mixed precision Liu는 78.09/75.10이다.

DeiT-B에서는 다음과 같다.

| Method | Bits | Calibration | Reported size | Top-1 |
|---|---:|---:|---:|---:|
| EasyQuant | W8A8 | 1,024 | 86.0 | 79.36 |
| Liu mixed precision | W8A8 | 1,024 | 86.8 | 81.29 |
| PTQ4ViT | W8A8 | 32 | 86.0 | 81.48 |
| EasyQuant | W6A6 | 1,024 | 64.5 | 75.86 |
| Liu mixed precision | W6A6 | 1,024 | 64.3 | 77.47 |
| PTQ4ViT | W6A6 | 32 | 64.5 | 80.25 |

표의 size 단위는 논문 표기상 model size지만 runtime resident memory나 file packing 상세는 제공되지 않는다. W8 86.0에서 W6 64.5로 정확히 0.75배인 점은 이론적 bit packing에 가깝다.

## 4-bit에서는 실패한다

DeiT-B W4A4에서 PTQ4ViT는 60.91, bias correction을 더해도 64.39다. Liu mixed precision은 75.94다. FP32 81.80과 비교하면 큰 손실이다.

이 결과는 twin uniform과 Hessian scale search가 universal low-bit solution이 아님을 보여준다. 4-bit에서는 layer sensitivity에 따른 mixed precision, per-channel/group scale, reconstruction/QAT가 더 중요하다. 논문의 near-lossless 주장은 W8A8 범위에 해당한다.

## Ablation

ViT-S/224 결과를 보면 구성 요소의 상호작용이 선명하다.

| Hessian | Softmax twin | GELU twin | W8A8 | W6A6 |
|---|---|---|---:|---:|
| 아니오 | 아니오 | 아니오 | 80.47 | 70.24 |
| 예 | 아니오 | 아니오 | 80.93 | 77.20 |
| 예 | 예 | 아니오 | 81.11 | 78.57 |
| 예 | 아니오 | 예 | 80.84 | 76.93 |
| 아니오 | 예 | 예 | 79.25 | 74.07 |
| 예 | 예 | 예 | 81.00 | 78.63 |

Hessian metric alone은 W6A6를 6.96 point 높인다. twin quantizer는 올바른 Hessian scale search와 결합할 때 도움이 되지만, twin만 사용하면 W8A8이 base보다 1.22 point 낮아진다. quantizer 표현력이 커질수록 calibration metric도 더 중요해진다는 결과다.

ViT-B/384의 base W6A6은 46.89인데 Hessian만으로 79.99, all components로 83.19가 된다. 고해상도 global attention이 scale 선택 오류에 특히 민감함을 보여준다.

## Calibration image 수와 시간

Tesla V100 한 장에서 32 images calibration은 모델별 2분에서 25분, 128 images는 5분에서 69분이다.

| Model | 32 images W8/W6/time | 128 images W8/W6/time |
|---|---|---|
| ViT-S/224 | 81.00 / 78.63 / 3m | 80.99 / 78.44 / 7m |
| ViT-B/384 | 85.83 / 83.35 / 12m | 85.81 / 83.84 / 43m |
| DeiT-B/384 | 82.97 / 81.55 / 14m | 83.01 / 81.67 / 52m |
| Swin-B/384 | 86.39 / 85.39 / 25m | 86.36 / 85.45 / 69m |

128장으로 늘려도 accuracy 변화는 작아 32장이 실용적이다. 32장을 20번 random sampling한 W8 결과의 standard deviation은 ViT-S/32 0.055, ViT-S 0.046, ViT-B 0.068, DeiT-S 0.094, Swin-S 0.035 point다.

calibration time은 quantized model의 inference latency가 아니다. deployment 준비에 걸린 offline 시간이다.

## Fake quantization과 real integer inference

논문의 accuracy 실험은 quantize/dequantize를 통해 low-bit error를 모사하는 PTQ 평가로 읽어야 한다. 다음 실제 실행 증거는 제공되지 않는다.

- custom twin-format INT GEMM kernel의 end-to-end benchmark
- CPU/GPU에서 FP baseline 대비 latency와 energy
- mobile NPU/CoreML/TFLite/NNAPI operator mapping
- W6 packed load와 dot-product throughput
- Softmax/GELU/LayerNorm을 포함한 mixed graph의 Q/DQ 비용

저자들은 power-of-two scale 관계로 shift 처리가 가능하다고 설계했지만, kernel 가능성과 kernel 구현 및 실측은 다른 단계다. 실제 backend가 twin flag를 모르면 dequantize 후 FP matmul을 실행할 수 있고, 이 경우 accuracy만 quantized이고 속도는 quantized가 아니다.

### Real quantization 확인법

1. exported graph에 FP GEMM이 남았는지 확인한다.
2. weight가 실제 packed INT8/INT6 file로 저장되는지 확인한다.
3. activation이 Q/DQ 사이에서 int tensor로 유지되는지 trace한다.
4. accumulator dtype과 requantization kernel을 확인한다.
5. operator별 시간과 memory bandwidth를 FP16 baseline과 비교한다.

## On-device 관점

### W8A8의 장점

- FP32 대비 weight/activation payload를 이론상 4배 줄인다.
- INT8 dot-product가 있는 CPU/NPU에서 FC와 attention matmul 가속 가능성이 있다.
- calibration label이 필요 없고 32 images로 빠르게 준비할 수 있다.

### 실제 장애물

- twin uniform은 표준 per-tensor affine INT8과 다른 custom code다.
- head별 scale은 fused multi-head attention kernel과 맞지 않을 수 있다.
- INT6 multiply는 일반 mobile accelerator에서 직접 지원되지 않는 경우가 많다.
- LayerNorm, Softmax, GELU가 FP이면 빈번한 dtype transition이 생긴다.
- high-resolution ViT는 attention activation이 커서 weight 압축만으로 peak memory가 해결되지 않는다.

따라서 첫 deployment baseline은 PTQ4ViT W8A8 accuracy를 재현한 뒤, target runtime이 지원하는 standard symmetric/per-channel INT8로 mapping 가능한 부분과 twin custom kernel이 필요한 부분을 나누는 것이다.

### 측정해야 할 항목

- FP32, FP16, standard INT8, twin W8A8의 top-1
- model file과 실제 resident weight memory
- batch=1 p50/p95, cold/warm latency
- LayerNorm/Softmax/GELU/QDQ별 operator time
- activation peak와 allocator workspace
- power, temperature, 10분 이상 throttling
- INT6이 packed인지 int8 container인지

## 재현 체크리스트

### Calibration

- [ ] training label을 사용하지 않고 FP model prediction을 pseudo-label로 썼다.
- [ ] calibration image 32장을 random sampling했다.
- [ ] layer output과 $dL/dO$를 한 번 저장해 scale 후보에서 재사용했다.
- [ ] output/gradient host offload dtype과 peak RAM을 기록했다.
- [ ] $\alpha=0$, $\beta=1.2$, 100 candidates, 3 rounds를 사용했다.
- [ ] parallel calibration과 sequential calibration을 구분했다.

### Quantizer와 granularity

- [ ] FC weight/input, $QK^T$, $PV$를 모두 quantized 대상으로 했다.
- [ ] $W_Q/W_K/W_V$ scale을 분리했다.
- [ ] attention head별 scale을 적용했다.
- [ ] softmax $R_2$ scale과 GELU negative scale을 고정했다.
- [ ] twin scale ratio가 $2^m$이 되도록 제한했다.
- [ ] range flag와 magnitude bits를 실제 packed format으로 검증했다.

### Accuracy

- [ ] FP32 preprocessing와 quantized preprocessing이 동일하다.
- [ ] W8A8, W6A6, W4A4를 별도 평가했다.
- [ ] twin-only가 오히려 악화되는 ablation을 확인했다.
- [ ] calibration seed를 여러 번 바꿔 mean/std를 기록했다.
- [ ] ImageNet 외 dense prediction에서도 activation distribution을 다시 조사했다.

### Real deployment

- [ ] fake Q/DQ accuracy와 real integer kernel accuracy를 분리했다.
- [ ] target backend가 twin format과 head별 scale을 지원하는지 확인했다.
- [ ] unsupported W6가 int8 container로 fallback하는지 확인했다.
- [ ] p50/p95, peak memory, model size, 전력, 온도를 기록했다.
- [ ] operator trace에서 FP GEMM fallback이 없는지 확인했다.

## 로드맵에서의 위치

PTQ4ViT는 integer-only CNN quantization 다음에 읽을 때 transformer activation이 왜 더 어렵다는 점을 보여준다. 핵심은 bit 수 자체보다 distribution, scale search objective, hardware granularity다.

후속 LSQ와 비교하면 차이가 명확하다.

- PTQ4ViT: 32 unlabeled images, scale grid search, no weight update, 빠른 배포
- LSQ: labeled training과 STE, scale을 gradient로 학습, 더 낮은 bit 가능

AWQ와 비교할 때는 대상이 다르다.

- PTQ4ViT: vision transformer W/A quantization, softmax/GELU activation 문제
- AWQ: LLM weight-only quantization, salient input activation으로 weight scale 보호

온디바이스 실험에서는 동일 ViT를 FP16, standard INT8, PTQ4ViT-style W8A8로 만들고 accuracy뿐 아니라 실제 backend kernel, peak activation, p95 latency를 비교해야 한다.

## 최종 평가

PTQ4ViT의 가장 큰 기여는 ViT PTQ 실패를 단순 outlier 문제가 아니라 post-softmax와 post-GELU의 구조적 distribution 문제, 그리고 local metric의 scale 선택 오류로 분해한 것이다. Twin uniform quantization과 gradient-weighted Hessian metric을 결합해 32 calibration images만으로 W8A8을 거의 무손실로 만들었다.

반면 W4A4는 크게 실패하고, custom twin format의 실제 mobile kernel 성능은 검증하지 않았다. 따라서 논문의 accuracy table을 곧바로 8-bit latency 향상으로 번역하면 안 된다. 재현의 완성 조건은 `calibration accuracy + packed model + real integer operator trace + batch=1 p95 + peak memory + power`를 모두 확인하는 것이다.

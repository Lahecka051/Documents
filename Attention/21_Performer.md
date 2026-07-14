# 21. Rethinking Attention with Performers

## 논문 정보

- 제목: **Rethinking Attention with Performers**
- 저자: Krzysztof Choromanski et al.
- 발표: ICLR 2021
- 핵심 키워드: FAVOR+, random feature, softmax kernel, orthogonal random feature, linear attention

## 한눈에 보는 요약

Performer는 softmax attention을 kernel로 해석하고, 그 kernel을 positive random feature로 근사해 `[n,n]` attention matrix를 만들지 않는다. 이 알고리즘을 **FAVOR+**(Fast Attention Via positive Orthogonal Random features)라고 부른다.

표준 attention의 unnormalized kernel은

```math
A_{ij}=\exp\!\left(\frac{q_i^{\top}k_j}{\sqrt d}\right)
```

이다. `exp(q^T k)`를 feature map 내적 `phi(q)^T phi(k)`로 근사하면 계산 순서를 바꿀 수 있다.

```math
\phi(Q)\left[\phi(K)^{\top}V\right]
```

중간 값은 `[r,d_v]`이고, `r`은 random feature 수다. normalization denominator도 같은 방식으로 계산한다. 따라서 time은 `O(n r d)`, memory는 `O(nr + nd + rd)`가 된다. `r`을 고정하면 길이에 대해 선형이다.

Performer의 중요한 개선은 일반 random feature가 아니라 **양수 feature**와 **orthogonal random vector**를 사용해 softmax kernel 근사의 분산과 normalization 불안정을 줄인 점이다.

<p align="center"><img src="https://github.com/user-attachments/assets/dda250ac-3966-4284-a35d-7fd8fb0eaff4" alt="Performer FAVOR+ factorization" width="760"></p>
<p align="center"><sub>Figure 1 — FAVOR+ factorization으로 계산 순서를 바꾸는 Performer</sub></p>

## 표준 softmax attention을 kernel로 보기

한 head에서 scaled query/key를 이미 반영했다고 하자. unnormalized attention matrix와 row normalization은 다음과 같다.

```math
\begin{aligned}
A&=\exp\!\left(QK^{\top}\right)&&\in\mathbb{R}^{n\times n}
\quad\text{(element-wise exp)},\\
D&=\mathrm{diag}(A\mathbf{1}_n)
\quad\text{(row sum)},\\
Y&=D^{-1}AV.
\end{aligned}
```

`A`를 직접 만들지 않고 kernel

```math
K(q,k)=\exp\!\left(q^{\top}k\right)
```

의 저차원 feature 표현을 찾는 것이 핵심이다.

```math
K(q,k)\approx\phi(q)^{\top}\phi(k),
\qquad
\phi(x)\in\mathbb{R}^{r}
```

그러면 전체 kernel matrix는

```math
\begin{aligned}
A&\approx Q'{K'}^{\top},\\
Q'&=\phi(Q)\in\mathbb{R}^{n\times r},\\
K'&=\phi(K)\in\mathbb{R}^{n\times r}.
\end{aligned}
```

로 factorize된다.

## FAVOR+ attention 수식

Numerator는 결합 법칙으로 다음처럼 계산한다.

```math
AV\approx\left(Q'{K'}^{\top}\right)V=Q'\left({K'}^{\top}V\right)
```

Denominator도 `[n,n]` 없이 계산한다.

```math
A\mathbf{1}\approx Q'\left({K'}^{\top}\mathbf{1}\right)
```

최종 출력은 row별 scalar로 나눈다.

```math
\begin{aligned}
S&={K'}^{\top}V&&\in\mathbb{R}^{r\times d_v},\\
z&={K'}^{\top}\mathbf{1}_n&&\in\mathbb{R}^{r},\\
Y_i&=\frac{Q'_iS}{Q'_iz}.
\end{aligned}
```

Batch/head를 포함하면 shape은 다음과 같다.

```text
Q',K' : [B,H,N,R]
V     : [B,H,N,D]
S     : [B,H,R,D]
z     : [B,H,R]
Y     : [B,H,N,D]
```

분모가 매우 작아질 때 epsilon과 stable accumulation이 필요하다.

## Positive random feature의 유도

Gaussian random vector `ω ~ N(0,I)`에 대해 다음 feature를 생각할 수 있다.

```math
\phi_{\omega}(x)=\exp\!\left(\omega^{\top}x-\frac{\lVert x\rVert^2}{2}\right)
```

Gaussian moment generating function을 사용하면

```math
\mathbb{E}_{\omega}\!\left[\phi_{\omega}(x)\phi_{\omega}(y)\right]=\exp\!\left(x^{\top}y\right)
```

가 된다. `r`개의 random vector를 샘플링하고 `1/sqrt(r)`로 정규화해 finite-dimensional feature map을 만든다.

```math
\phi(x)=\frac{1}{\sqrt r}\left[\phi_{\omega_1}(x),\ldots,\phi_{\omega_r}(x)\right]
```

모든 component가 양수이므로 근사 kernel과 denominator가 음수가 되지 않는다. 이것이 일반적인 sine/cosine random Fourier feature보다 softmax attention에 안정적인 이유다.

## 왜 기존 trigonometric feature가 불안정한가

일반 random Fourier feature는 양수와 음수 term을 함께 사용한다. 기대값은 올바르더라도 실제 finite sample에서 cancellation이 크고, kernel 값이 작은 영역의 상대 오차가 커진다. attention row sum이 0 근처 또는 음수가 되면 normalization이 불안정해진다.

Positive feature는 cancellation을 줄이고 denominator의 의미를 유지한다. 하지만 exponential feature 자체가 큰 값을 만들 수 있으므로 max subtraction, input scaling, FP32 accumulation 같은 수치 안정화는 여전히 필요하다.

## Orthogonal random features

독립 Gaussian vector 대신 서로 직교하도록 구성한 random vector를 사용한다. 같은 `r`개의 방향이 중복 정보를 덜 담으므로 Monte Carlo variance가 낮아진다.

```text
independent features : 방향이 우연히 비슷할 수 있음
orthogonal features  : feature budget으로 더 다양한 방향을 탐색
```

`r <= d`이면 하나의 orthogonal block을 사용할 수 있고, 더 큰 `r`은 여러 block을 이어 붙인다. 논문은 orthogonal feature의 mean squared error가 일반 feature보다 낮음을 이론과 실험으로 보인다.

## Query/key scaling

표준 scaled dot-product의

```math
\exp\!\left(\frac{q^{\top}k}{\sqrt d}\right)
```

를 `exp(x^T y)` 형태로 맞추려면 query와 key를 `d^(-1/4)`씩 scaling할 수 있다.

```math
\begin{aligned}
x&=\frac{q}{d^{1/4}},
&y&=\frac{k}{d^{1/4}},\\
x^{\top}y&=\frac{q^{\top}k}{\sqrt d}.
\end{aligned}
```

이 scaling이 누락되면 kernel의 sharpness와 feature variance가 크게 달라진다.

## Causal Performer

Bidirectional attention에서는 `S=K'^T V`, `z=K'^T 1`을 sequence 전체에서 한 번 계산한다. causal attention에서는 위치 `i`가 `j<=i`만 봐야 하므로 prefix statistics를 사용한다.

```math
\begin{aligned}
S_i&=\sum_{j\le i}K'_j\otimes V_j&&\in\mathbb{R}^{r\times d},\\
z_i&=\sum_{j\le i}K'_j&&\in\mathbb{R}^{r},\\
Y_i&=\frac{Q'_iS_i}{Q'_iz_i}.
\end{aligned}
```

parallel training에서는 associative prefix scan으로 모든 `S_i,z_i`를 계산할 수 있다. autoregressive decoding에서는 새 key/value가 들어올 때 state를 한 번 update한다.

```math
\begin{aligned}
S&\leftarrow S+k'_t\otimes v_t,\\
z&\leftarrow z+k'_t,\\
y_t&=\frac{q'_tS}{q'_tz}.
\end{aligned}
```

이 recurrent state 크기는 sequence length와 무관한 `O(rd)`다. 하지만 정확한 KV history를 보존하지 않으므로 random-feature 근사의 정보 bottleneck이 된다.

## 계산 복잡도

| 항 | Exact attention | Performer |
| --- | ---: | ---: |
| Score/kernel | `O(n²d)` | `O(nrd)` |
| Attention memory | `O(n²)` | `O(nr + rd)` |
| Causal decode state | `O(nd)` KV cache | `O(rd)` statistics |

`r << n`이면 큰 절감이 난다. 그러나 feature map에서 exponential과 norm을 계산하는 비용, prefix scan, numerical stabilization도 포함해야 한다. 짧은 sequence에서는 exact fused attention이 더 빠를 수 있다.

## Feature redraw

Random projection을 학습 내내 고정하면 모델이 특정 feature sample의 approximation error에 과도하게 적응할 수 있다. 논문은 일정 간격으로 random features를 redraw하는 옵션을 사용한다.

Redraw는 kernel estimator의 다양한 sample을 경험하게 하지만 다음 관리가 필요하다.

- distributed replica에 같은 random matrix를 배포할지
- evaluation에서는 고정할지
- checkpoint에 projection seed/state를 저장할지
- reversible/recompute와 RNG를 일치시킬지

## 이론적 보장

논문은 positive feature estimator의 unbiased 또는 거의 unbiased 성질과 variance bound를 분석한다. 또한 bounded query/key domain에서 uniform convergence를 위해 필요한 feature 수가 sequence length `n`이 아니라 dimension, norm bound, error tolerance에 의존하는 결과를 제시한다.

이 결과의 의미는 `n`이 커진다고 반드시 `r`을 같은 비율로 늘릴 필요는 없다는 것이다. 다만 실제 Transformer의 query/key norm이 bound를 얼마나 만족하는지, 작은 error가 downstream output에 어떻게 누적되는지는 별도 문제다.

## 실험 결과

### Kernel approximation

Positive feature가 trigonometric feature보다 attention matrix와 output의 approximation error가 낮았고, orthogonal random vector를 사용하면 같은 feature 수에서 error가 더 줄었다. 특히 kernel 값이 작거나 score distribution이 sharp한 영역에서 positivity의 안정성 이득이 나타난다.

### Pretrained Transformer 변환

기존 full-attention model을 Performer attention으로 치환하면 즉시 품질 손실이 생기지만, 짧은 fine-tuning으로 상당 부분 회복한다. 이는 Reformer와 마찬가지로 모델이 approximation 구조에 적응할 필요가 있음을 보여준다.

### PG-19 language modeling

긴 책 dataset에서 positive feature와 periodic redraw가 중요했다. 단순 ReLU kernel이나 불안정 feature는 긴 training에서 품질이 떨어졌고, FAVOR+가 더 안정적으로 학습되었다.

### Protein sequence modeling

36-layer protein model에서 softmax-kernel Performer는 exact attention과 비슷한 성능을 보였다. 일부 설정에서는 ReLU kernel이 더 좋은 task metric을 내기도 해, 모든 task에서 softmax approximation 자체가 최적 kernel이라는 보장은 없음을 보여준다.

### ImageNet64

64×64 RGB를 펼친 약 12,288 길이 sequence에서 6-layer Performer가 12-layer Reformer에 경쟁적인 density modeling 성능을 보였다. sorting 없는 dense linear algebra로 긴 image sequence를 처리한 사례다.

### 매우 긴 protein concatenation

8,192 길이의 concatenated protein sequence도 full attention memory 없이 처리할 수 있음을 보였다. 논문의 주장은 특정 benchmark SOTA보다 다양한 modality에서 하나의 kernel mechanism이 작동한다는 범용성에 있다.

## 장점과 기여

- softmax attention을 kernel factorization으로 바꿔 `[n,n]` materialization을 제거했다.
- positive random feature로 normalization 안정성을 개선했다.
- orthogonal feature로 같은 `r`에서 variance를 줄였다.
- causal attention을 prefix statistics로 선형 계산하는 방법을 제시했다.
- 이론 bound와 text, image, protein 실험을 함께 제공했다.

## 한계와 비판적 관점

### 1. Approximation variance

유한한 `r`에서는 output이 random projection에 의존한다. `r`을 늘리면 품질은 좋아지지만 선형 attention의 비용 이점이 줄어든다.

### 2. Numerical stability

`exp(ω^T x - ||x||²/2)`는 overflow/underflow 가능성이 있고 denominator가 작으면 오차가 증폭된다. mixed precision 구현이 특히 어렵다.

### 3. Sharp attention과 retrieval

거의 one-hot인 attention kernel을 작은 random feature 수로 근사하기 어렵다. exact match와 needle retrieval에서 feature bottleneck이 드러날 수 있다.

### 4. Redraw와 재현성

projection 갱신은 regularization 효과가 있지만 training state와 distributed synchronization을 복잡하게 만든다.

### 5. Exact kernel의 발전

FlashAttention은 moderate context에서 exact softmax를 매우 효율적으로 계산한다. Performer의 장점은 매우 긴 context나 recurrent causal state가 중요한 조건에서 평가해야 한다.

## 관련 방법과 비교

| 방법 | Factorization 축 | 중간 크기 | 특징 |
| --- | --- | ---: | --- |
| Efficient Attention | Q/K channel | `[d_k,d_v]` | separable softmax |
| Linformer | sequence rank | `[n,k]` | learned projection |
| Performer | kernel feature | `[n,r]` | random feature |
| Nyströmformer | landmarks | `[n,m]` | pseudo-inverse core |

## 구현 체크리스트

- `d^(-1/4)` query/key scaling이 올바른가?
- random feature normalization `1/sqrt(r)`가 포함되는가?
- exponential 계산 전에 안정화가 적용되는가?
- denominator에 안전한 epsilon과 FP32 accumulation을 쓰는가?
- causal prefix sum이 미래 token을 포함하지 않는가?
- redraw seed/state가 checkpoint와 distributed training에서 재현되는가?
- feature 수 `r`별 error·quality·latency를 함께 측정했는가?

## 온디바이스 관점

Performer는 sort나 irregular sparse gather 없이 GEMM, reduction, elementwise exp로 구성할 수 있다는 장점이 있다. causal decode state가 `O(rd)`로 고정되므로 KV cache가 긴 context에 비례해 커지는 문제도 피할 수 있다. 반면 exponential feature와 FP32 accumulation이 양자화·저정밀 NPU에서 부담일 수 있다.

Vision encoder에서는 fixed resolution의 긴 patch sequence를 처리하기 쉽고 dense linear algebra 활용도가 높다. 실제 배포에서는 `r`을 vector unit tile에 맞추고, feature map 생성과 `K'^T V`를 fuse할 수 있는지가 핵심이다.

## 최종 평가

Performer는 linear attention을 단순한 `Q(K^T V)` 재배열에서 **softmax kernel의 확률적 factorization**으로 발전시킨 대표 논문이다. positivity와 orthogonality로 random-feature 근사의 실용성을 높였고, causal prefix state까지 하나의 수식으로 처리한다. 그러나 exact attention이 아니며 feature 수, random seed, 수치 안정성에 품질이 좌우된다. 긴 context에서 kernel approximation을 받아들일 수 있을 때 강력하지만, moderate length에서 exact fused attention과 반드시 비교해야 한다.

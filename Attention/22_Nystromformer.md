# 22. Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention

## 논문 정보

- 제목: **Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention**
- 저자: Yunyang Xiong et al.
- 발표: AAAI 2021
- 핵심 키워드: Nyström approximation, landmark attention, pseudo-inverse, linear attention, long sequence

## 한눈에 보는 요약

Nyströmformer는 `[n,n]` softmax attention matrix를 `m`개의 landmark를 이용한 세 개의 작은 행렬 곱으로 근사한다.

```text
token -> landmark
landmark core pseudo-inverse
landmark -> token
```

Query와 key를 sequence segment별 평균으로 압축해 landmark `Q_tilde,K_tilde ∈ R^{m×d}`를 만들고 다음 근사를 사용한다.

```math
\operatorname{softmax}\!\left(QK^{\top}\right)
\approx
\operatorname{softmax}\!\left(Q\tilde K^{\top}\right)
\left[\operatorname{softmax}\!\left(\tilde Q\tilde K^{\top}\right)\right]^+
\operatorname{softmax}\!\left(\tilde QK^{\top}\right)
```

각 factor의 shape은 `[n,m]`, `[m,m]`, `[m,n]`이다. 전체 `[n,n]`을 materialize하지 않고 오른쪽에서부터 `V`와 곱하면 memory를 줄일 수 있다. `m`을 64처럼 작은 상수로 두면 sequence length `n`에 대해 선형에 가까워진다.

핵심 trade-off는 landmark가 전체 attention structure를 얼마나 잘 대표하는지, 그리고 중앙 `m×m` matrix의 pseudo-inverse를 얼마나 안정적으로 근사하는지다.

<p align="center"><img src="https://github.com/user-attachments/assets/2af03b28-faae-4f76-936b-c076f7a736b4" alt="Nyströmformer landmark factorization" width="820"></p>
<p align="center"><sub>원 논문 Figure 2 — landmark 기반 세 행렬 곱으로 근사하는 Nyström factorization</sub></p>

## Nyström method의 직관

Nyström method는 큰 kernel matrix의 일부 column과 row를 샘플링해 전체 matrix를 재구성하는 고전적인 low-rank approximation이다. 대칭 kernel matrix `S`에서 landmark index 집합을 고르면 개념적으로 다음 형태를 사용한다.

```math
S\approx CW^{+}C^{\top}
```

- `C`: 모든 point와 landmark 사이 kernel
- `W`: landmark끼리의 kernel
- `W^+`: Moore–Penrose pseudo-inverse

Self-attention은 row-wise softmax 때문에 단순 대칭 kernel matrix가 아니고 query와 key가 다를 수 있다. Nyströmformer는 token-to-landmark와 landmark-to-token factor를 분리해 이를 attention에 맞게 변형한다.

## Landmark 생성

Sequence 길이 `n`을 `m`개의 contiguous segment로 균등 분할한다. 각 segment의 query와 key 평균을 landmark로 사용한다.

```math
\begin{aligned}
l&=\frac{n}{m},\\
\tilde Q_j&=\frac{1}{l}\sum_{i=(j-1)l+1}^{jl}Q_i,\\
\tilde K_j&=\frac{1}{l}\sum_{i=(j-1)l+1}^{jl}K_i.
\end{aligned}
```

shape은 다음과 같다.

```text
Q,K                 : [n,d]
Q_tilde,K_tilde     : [m,d]
```

Segment mean은 추가 parameter가 없고 연산이 싸며 순서를 보존한다. 하지만 중요한 token과 배경 token을 같은 segment에서 평균내면 landmark가 희석될 수 있다. `n`이 `m`으로 나누어지지 않을 때 padding과 segment 경계를 일관되게 처리해야 한다.

## Attention factorization

Scaled score를 포함해 세 factor를 정의한다.

```math
\begin{aligned}
F&=\operatorname{softmax}\!\left(\frac{Q\tilde K^{\top}}{\sqrt d}\right)
&&\in\mathbb{R}^{n\times m}
\quad\text{(token }\to\text{ landmark key)},\\
A&=\operatorname{softmax}\!\left(\frac{\tilde Q\tilde K^{\top}}{\sqrt d}\right)
&&\in\mathbb{R}^{m\times m}
\quad\text{(landmark core)},\\
B&=\operatorname{softmax}\!\left(\frac{\tilde QK^{\top}}{\sqrt d}\right)
&&\in\mathbb{R}^{m\times n}
\quad\text{(landmark query }\to\text{ every token)}.
\end{aligned}
```

전체 attention approximation은

```math
\hat S=FA^+B
```

이고 output은

```math
Y=F\left[A^+(BV)\right]
```

순서로 계산한다.

```math
\begin{aligned}
BV&:\ [m\times n][n\times d_v]=[m\times d_v],\\
A^+(BV)&:\ [m\times m][m\times d_v]=[m\times d_v],\\
F(\cdots)&:\ [n\times m][m\times d_v]=[n\times d_v].
\end{aligned}
```

따라서 `[n,n]` `S_hat`을 실제로 만들 필요가 없다.

## 왜 pseudo-inverse가 필요한가

Token-to-landmark factor `F`와 landmark-to-token factor `B`를 단순 곱하면 landmark 사이 관계가 중복되거나 왜곡된다. 중앙 core `A`가 landmark space의 metric을 나타내며, `A^+`가 이 중복을 보정한다.

`A`가 full rank square matrix라면 inverse를 쓸 수 있지만, softmax core는 singular하거나 condition number가 클 수 있다. Moore–Penrose pseudo-inverse는 rank-deficient matrix에도 정의되지만 정확한 SVD는 `O(m³)`이고 GPU pipeline에 부담이다.

## Iterative pseudo-inverse approximation

논문은 Razavi 계열의 iterative method를 사용한다. 초기값을

```math
Z_0=\frac{A^{\top}}{\lVert A\rVert_1\lVert A\rVert_{\infty}}
```

로 두고 다음 고차 iteration을 반복한다.

```math
Z_{j+1}=\frac{1}{4}Z_j\left[13I-AZ_j\left(15I-AZ_j(7I-AZ_j)\right)\right]
```

`Z_j`가 `A^+`로 수렴하도록 하며 논문 구현은 보통 6회 iteration을 사용한다. 모든 연산이 `m×m` matrix multiplication이므로 batch/head 단위 GPU 연산으로 실행할 수 있다.

### 수치 안정성

- `A`의 condition number가 크면 수렴이 느리거나 오차가 커질 수 있다.
- FP16에서 반복 matrix multiplication 오차가 누적될 수 있다.
- 초기 scaling의 norm 계산이 정확해야 한다.
- fixed iteration은 input마다 다른 수렴 상태를 남길 수 있다.

실무에서는 pseudo-inverse residual `||A Z A - A||`를 확인하거나 핵심 계산을 FP32로 수행하는 것이 안전하다.

## Value residual convolution

Landmark approximation은 global low-rank structure를 잘 포착할 수 있지만 local detail을 평균내기 쉽다. 논문은 value에 depthwise 1D convolution을 적용해 output에 residual로 더한다.

```math
Y_{\mathrm{final}}=\operatorname{Nystr\ddot{o}mAttention}(Q,K,V)+\operatorname{DWConv}(V)
```

channel마다 local neighborhood를 convolution해 근접 정보를 보강한다. 이는 attention approximation 자체의 수학에는 필요 없지만 empirical 성능에 중요한 architecture 요소다. 비교 시 convolution이 주는 이득과 landmark factorization의 이득을 분리해 봐야 한다.

## 계산량과 메모리

세 factor와 pseudo-inverse를 모두 고려하면 주요 비용은 다음과 같다.

```math
\begin{aligned}
F,B\text{ 계산}&:\ O(nmd),\\
\text{landmark core}&:\ O(m^2d),\\
\text{pseudo-inverse 반복}&:\ O(tm^3),\\
\text{factor-value 곱}&:\ O(nmd_v+m^2d_v).
\end{aligned}
```

요약하면 `O(nm² + nmd_v + m³)`로 기술할 수 있고 구현 방식에 따라 `nmd` 항이 지배한다. `m`을 고정하고 `m << n`이면 `n`에 대해 선형이지만, 정확도를 위해 `m`을 `n`과 함께 키우면 선형성이 약해진다.

메모리는 대략

```math
\begin{aligned}
F/B\text{ 또는 streaming factor}&:\ O(nm),\\
\text{core}&:\ O(m^2),\\
\text{activations}&:\ O(nd).
\end{aligned}
```

이다. `F`와 `B`를 둘 다 보관하지 않고 곱 순서와 recomputation을 조정하면 peak memory를 더 줄일 수 있다.

## Exactness 조건

Landmark가 모든 query/key를 포함해 `m=n`이고 landmark가 원본과 동일하면 원래 attention을 재구성할 수 있는 조건에 가까워진다. 일반 `m<<n`에서는 low-rank approximation이다.

논문은 landmark가 attention matrix의 주요 column/row space를 잘 span하면 근사가 좋다는 방향의 이론을 제시한다. 그러나 softmax를 세 factor에 각각 적용한 행렬은 원본 row-softmax matrix와 단순 Nyström sample이 완전히 같지 않다. 실용 알고리즘은 kernel approximation과 attention-specific normalization을 결합한 것으로 이해하는 편이 정확하다.

## 실험 결과: 효율

길이 8,192에서 논문이 보고한 한 예는 다음과 같다.

| 모델 | Landmark/window | Memory | Time |
| --- | ---: | ---: | ---: |
| Transformer | full | 10,233 MB | 155.4 ms |
| Linformer | 256 | 635 MB | 11.3 ms |
| Longformer | 257 | 455 MB | 36.2 ms |
| Nyströmformer | 64 | 450 MB | 12.3 ms |
| Nyströmformer | 32 | **383 MB** | 11.5 ms |

Nyströmformer-64는 Longformer와 비슷한 memory로 더 짧은 시간, Linformer와 비슷한 시간으로 더 적은 memory를 보였다. 다만 측정은 당시 특정 PyTorch/CUDA 구현에 해당하며, 최신 kernel과 동일 조건 비교가 필요하다.

## 실험 결과: GLUE

BERT-base 구조의 attention을 Nyströmformer로 바꾼 뒤 pretraining/fine-tuning한 결과는 여러 GLUE task에서 경쟁적이다. 예를 들어 SST-2는 BERT-base 약 `90.0`에 비해 Nyströmformer 약 `91.4`로 높았지만, QNLI는 `90.3` 대비 `88.7` 정도로 낮았다.

즉 평균적으로 근접한 품질을 보이지만 모든 task에서 exact attention을 일관되게 능가하지는 않는다. Convolution residual과 training variation이 함께 있으므로 단일 숫자보다 task별 편차를 봐야 한다.

## 실험 결과: Long Range Arena

논문의 PyTorch 재구현 기준 평균 결과는 다음과 같다.

| 모델 | LRA 평균 |
| --- | ---: |
| Standard Transformer | 58.77 |
| Reformer | 55.04 |
| Linformer | 55.59 |
| Performer | 53.63 |
| Nyströmformer | **58.95** |

Nyströmformer가 exact baseline과 비슷하거나 약간 높은 평균을 기록했다. 다만 LRA는 구현과 hyperparameter에 민감하며, 논문별 원래 code가 아니라 동일 framework 재구현 수치라는 점을 고려해야 한다.

## Landmark 수 ablation

Landmark 수를 늘리면 일반적으로 approximation error가 줄지만 다음 비용이 증가한다.

```math
\begin{aligned}
\text{token-landmark scores}&:\ O(nm),\\
\text{core}&:\ O(m^2),\\
\text{pseudo-inverse}&:\ O(m^3).
\end{aligned}
```

논문에서는 `m=64`가 여러 task에서 좋은 균형을 보였다. 그러나 optimal `m`은 input length뿐 아니라 attention rank, task, head dimension에 따라 다르다. 모든 layer/head에 같은 `m`을 쓰는 것이 최적이라는 보장은 없다.

## 장점과 기여

- 고전 Nyström low-rank approximation을 self-attention에 맞게 세 factor로 구성했다.
- segment mean landmark로 parameter 추가 없이 sequence를 압축했다.
- pseudo-inverse를 iterative matrix multiplication으로 근사해 GPU에서 실행 가능하게 했다.
- local convolution residual로 low-rank approximation의 detail 손실을 보완했다.
- BERT/GLUE, LRA, 긴 sequence의 memory/time을 함께 검증했다.

## 한계와 비판적 관점

### 1. Landmark 품질

Contiguous segment mean은 단순하고 빠르지만 content 중요도를 반영하지 않는다. 중요한 token 하나가 긴 segment 평균에서 사라질 수 있다.

### 2. Segment boundary

같은 local pattern도 segment 경계에 걸리면 서로 다른 landmark로 나뉜다. length와 padding에 따라 representation이 달라질 수 있다.

### 3. Pseudo-inverse 안정성

Core matrix가 ill-conditioned하면 fixed iteration approximation이 부정확할 수 있다. `m³` 비용도 `m`을 크게 늘리는 것을 제한한다.

### 4. Causal attention의 어려움

Segment landmark가 미래 token 평균을 포함하면 autoregressive leak가 발생한다. 위치별 causal landmark와 triangular factorization이 필요하므로 원래 bidirectional algorithm을 그대로 decoder에 적용할 수 없다.

### 5. Exact attention이 아니다

세 softmax factor의 곱은 원본 `softmax(QK^T)`와 다르다. retrieval과 sharp alignment에서 approximation error가 중요할 수 있다.

### 6. Convolution의 영향

성능 일부는 depthwise convolution의 local inductive bias에서 올 수 있다. 순수 attention approximation 비교에서는 residual 유무를 통제해야 한다.

## 관련 방법과 비교

| 방법 | Representative basis | Basis 선택 | 핵심 보정 |
| --- | --- | --- | --- |
| Linformer | `k` latent sequence slot | learned projection | 없음 |
| Performer | `r` kernel feature | random/orthogonal | normalized feature sum |
| Nyströmformer | `m` landmarks | segment mean | core pseudo-inverse |

Nyströmformer는 token-like landmark를 유지해 해석이 직관적이지만, 중앙 inverse 계산이 추가된다는 차이가 있다.

## 구현 체크리스트

- `n`이 `m`으로 나뉘지 않을 때 segment mean이 padding을 제외하는가?
- 세 softmax의 axis가 각각 마지막 token/landmark 축인가?
- `F A^+ B`를 직접 `[n,n]`으로 만들지 않고 `V`부터 곱하는가?
- pseudo-inverse 초기값의 `L1/L∞` norm이 올바른가?
- iteration을 FP32로 수행할 필요가 있는가?
- depthwise convolution의 padding이 sequence length를 보존하는가?
- landmark 수별 품질·memory·latency·inverse residual을 측정했는가?

## 온디바이스 관점

Nyströmformer는 random sort나 irregular sparse graph 없이 dense GEMM과 pooling으로 구성할 수 있어 accelerator 친화적인 면이 있다. 하지만 작은 `m×m` matrix multiplication을 여러 번 반복하는 pseudo-inverse는 mobile GPU/NPU에서 kernel launch와 synchronization overhead가 될 수 있다.

Vision patch에서는 segment mean보다 2D spatial pooling landmark가 더 자연스러울 수 있다. flatten 순서의 contiguous segment가 이미지에서 이상한 영역을 묶지 않도록 해야 한다. 고정 resolution과 고정 `m`에서는 static compilation이 쉽지만, 다양한 해상도에서는 landmark partition 규칙이 달라진다.

## 최종 평가

Nyströmformer는 attention matrix를 **token-landmark-core-landmark-token**의 세 단계로 분해해 global interaction을 낮은 비용으로 유지한다. segment mean과 iterative pseudo-inverse라는 구체적 구현 덕분에 개념이 실제 모델로 연결되었고, 긴 sequence에서 경쟁력 있는 memory/time을 보였다. 반면 landmark 대표성, pseudo-inverse 안정성, causal 적용이 주요 한계다. low-rank attention을 학습된 projection이 아닌 data-dependent landmark로 구성한다는 점에서 Linformer와 Performer 사이의 중요한 대안이다.

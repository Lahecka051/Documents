# 15. Efficient Attention: Attention with Linear Complexities

## 논문 정보

- 제목: **Efficient Attention: Attention with Linear Complexities**
- 저자: Zhuoran Shen, Mingyuan Zhang, Haiyu Zhao, Shuai Yi, Hongsheng Li
- 발표: WACV 2021
- 논문: *arXiv:1812.01243*
- 핵심 키워드: attention associativity, linear complexity, global context, separable normalization

## 한눈에 보는 요약

이 논문의 핵심은 새로운 sparse pattern을 설계하는 것이 아니라, attention의 행렬 곱 순서를 바꾸는 데 있다. 정규화가 없는 dot-product attention

```math
(QK^{\top})V
```

는 행렬 곱의 결합 법칙에 의해 정확히

```math
Q(K^{\top}V)
```

와 같다. 왼쪽 순서는 길이 `n`에 대해 `[n, n]` attention map을 만들지만, 오른쪽 순서는 `[d_k, d_v]` 크기의 작은 전역 context만 만든다. 따라서 token이나 pixel 수에 대한 메모리와 계산량을 선형으로 줄일 수 있다.

다만 이 등가성은 normalization까지 자동으로 보장하지 않는다. 단순 scaling normalization에서는 정확한 재배열이 가능하지만, 표준 row-wise softmax는 곱 전체에 결합되어 있으므로 그대로 분리할 수 없다. 저자들은 query에는 row softmax, key에는 column softmax를 적용하는 separable normalization을 사용한다. 따라서 Efficient Attention을 이해할 때는 다음 두 문장을 분리해야 한다.

- 행렬 곱 순서 변경 자체는 정확하다.
- softmax 버전은 표준 softmax attention의 근사적 변형이다.

<p align="center"><img src="https://github.com/user-attachments/assets/737dfa7a-52ec-42f1-b815-76b89af88b35" alt="Efficient Attention calculation order" width="820"></p>
<p align="center"><sub>Figure 1 — 표준 attention과 곱 순서를 바꾼 Efficient Attention 비교</sub></p>

## 문제의식

입력 feature를 `X ∈ R^{n×d}`라고 하자. 이미지에서는 `n = H×W`, 문장에서는 `n`이 sequence length이다. 표준 attention은 모든 위치 쌍의 관계를 계산한다.

```math
\begin{aligned}
Q &= XW_q &&\in \mathbb{R}^{n\times d_k},\\
K &= XW_k &&\in \mathbb{R}^{n\times d_k},\\
V &= XW_v &&\in \mathbb{R}^{n\times d_v},\\
S &= QK^{\top} &&\in \mathbb{R}^{n\times n},\\
Y &= \rho(S)V &&\in \mathbb{R}^{n\times d_v}.
\end{aligned}
```

`S`의 크기가 `n²`이므로 고해상도 vision feature에 non-local block을 넣으면 activation memory와 FLOPs가 급격히 커진다. 예를 들어 `100×100` feature map은 위치가 10,000개이고, head 하나의 score만 1억 개다. 논문은 이 병목이 attention의 의미 자체보다 **중간 행렬을 어떤 순서로 계산하느냐**에서 발생한다고 본다.

## 핵심 수식

논문은 attention을 다음처럼 정의한다.

```math
D(Q,K,V)=\rho\!\left(QK^{\top}\right)V
```

여기서 `rho`는 score normalization이다. Efficient Attention은 query와 key에 각각 적용 가능한 normalization `rho_q`, `rho_k`를 사용해 다음 형태로 계산한다.

```math
E(Q,K,V)=\rho_q(Q)\left(\rho_k(K)^{\top}V\right)
```

계산은 두 단계다.

1. `rho_k(K)^T V`로 `d_k`개의 global context vector를 만든다.
2. 각 query가 이 context들을 `rho_q(Q)`로 조합한다.

### Tensor shape 추적

```text
Q                      : [n, d_k]
K                      : [n, d_k]
V                      : [n, d_v]

K^T V                  : [d_k, d_v]
Q (K^T V)              : [n, d_v]
```

표준 순서의 가장 큰 중간 값은 `[n, n]`이고, 효율 순서의 가장 큰 추가 중간 값은 `[d_k, d_v]`다. 일반적으로 vision에서 `d_k, d_v << n`이므로 차이가 크다.

## Scaling normalization: 정확한 등가

논문이 scaling normalization이라 부르는 경우를 단순화하면 score를 `n`으로 나눈다.

```math
D(Q,K,V)=\left(\frac{QK^{\top}}{n}\right)V
```

query와 key를 각각 `sqrt(n)`으로 나누면 다음이 성립한다.

```math
\begin{aligned}
\rho_q(Q)&=\frac{Q}{\sqrt n},
&\rho_k(K)&=\frac{K}{\sqrt n},\\
\rho_q(Q)\left(\rho_k(K)^{\top}V\right)
&=\frac{Q}{\sqrt n}\left(\left(\frac{K}{\sqrt n}\right)^{\top}V\right)\\
&=\left(\frac{QK^{\top}}{n}\right)V.
\end{aligned}
```

이 경우는 근사가 아니다. 연산 순서만 바뀌고 출력은 수치 오차 범위에서 동일하다. 논문의 “equivalent”라는 표현은 우선 이 설정에 가장 명확하게 적용된다.

## Softmax normalization: 분리 가능한 근사

표준 attention의 softmax는 query 하나마다 모든 key score를 함께 정규화한다.

```math
P_{ij}=\frac{\exp\!\left(q_i^{\top}k_j\right)}{\sum_t\exp\!\left(q_i^{\top}k_t\right)}
```

이 연산은 일반적으로 `softmax(QK^T) = softmax(Q) softmax(K)^T`로 분해되지 않는다. 저자들은 대신 다음 normalization을 사용한다.

```text
rho_q(Q) : Q의 각 row에 softmax      # feature/channel 방향
rho_k(K) : K의 각 column에 softmax   # position 방향
```

그 결과

```math
Y=\mathrm{softmax}_{\mathrm{row}}(Q)
\left[\mathrm{softmax}_{\mathrm{col}}(K)^{\top}V\right]
```

가 된다. `softmax_col(K)`의 각 channel은 전체 위치에서 합이 1이므로 `K^T V` 단계는 위치 전체의 weighted context를 만든다. `softmax_row(Q)`는 각 위치가 그 context channel들을 어떻게 섞을지 정한다.

이 구조의 장점은 모든 가중치가 음이 아니고 정규화된다는 점이다. 그러나 query-key 내적의 row-wise softmax와 동일한 확률분포는 아니다. query마다 완전히 다른 sharp alignment를 직접 만드는 능력은 제한될 수 있다.

## 계산량과 메모리

논문의 일반적인 설정인 `d_v=d`, `d_k=d/2`를 대입하면 다음과 같이 정리된다.

| 방법 | 추가 메모리 | 주요 계산량 |
| --- | ---: | ---: |
| Non-local/full attention | `4dn + n²` | `(4d²+d)n + 3dn²` |
| Efficient Attention | `4dn + d²/2` | `(6d²+d)n` |

즉 `d`가 고정되어 있고 `n`이 커질 때 Efficient Attention은 메모리 `O(dn+d²)`, 계산 `O(d²n)`이다. 반면 full attention은 `n²` score의 저장과 연산이 지배한다.

여기서 “linear”는 `n`에 대한 복잡도다. channel dimension `d`에 대해서는 `d²` 항이 남는다. 작은 feature map이나 매우 큰 channel에서는 projection 비용이 우세할 수 있다.

## 알고리즘 흐름

```python
# Q: [B, N, Dk], K: [B, N, Dk], V: [B, N, Dv]

q = softmax(Q, dim=-1)       # channel-wise for every position
k = softmax(K, dim=-2)       # position-wise for every channel

context = transpose(k, -2, -1) @ V   # [B, Dk, Dv]
output = q @ context                  # [B, N, Dv]
```

multi-head에서는 head마다 같은 계산을 수행한 뒤 concatenate하고 output projection을 거친다. 구현상 핵심은 `[N,N]` tensor가 graph 안에 한 번도 생성되지 않게 하는 것이다.

## 직관: attention map 대신 context basis를 만든다

표준 attention은 각 query가 `n`개의 value를 직접 선택한다. Efficient Attention은 먼저 key channel별로 value를 모아 `d_k`개의 전역 context basis를 만든다.

```text
standard:
query_i -> n개의 위치 가중치 -> n개 value 조합

efficient:
all keys -> d_k개의 global context
query_i  -> d_k개의 context 가중치 -> output
```

따라서 정보 경로가 `query → key position → value`에서 `query → latent context channel → value`로 바뀐다. 이 관점은 뒤의 Linformer, Nyströmformer, kernel attention과도 연결되지만, 여기서는 latent 축이 projection rank가 아니라 key channel이다.

## 실험 결과

### MS COCO object detection / instance segmentation

Mask R-CNN 계열에서 Efficient Attention을 여러 feature level에 넣은 결과는 다음과 같다.

| 구성 | Box AP | Mask AP | 메모리 | 계산량 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 39.4 | 35.1 | - | - |
| Efficient Attention, res3-4 + FPN1-5 | **41.2** | **36.7** | 354 MB | 8.85 G |
| 대응하는 full non-local | - | - | 약 22.4 GB | 약 712 G |

full non-local을 같은 위치에 배치하는 것은 당시 GPU 메모리에서 사실상 불가능했고, Efficient Attention은 넓은 spatial feature에 전역 상호작용을 추가하면서 AP를 개선했다.

Normalization ablation에서는 scaling과 softmax 변형의 차이가 매우 작았다.

| 정규화 | Box AP | Mask AP |
| --- | ---: | ---: |
| Scaling | 40.2 | 35.9 |
| Softmax | 40.2 | 36.0 |

`d_k`를 32, 64, 128로 바꿔도 성능이 거의 유지되어, 이 실험 범위에서는 작은 context basis로도 충분했음을 보여준다.

### Stereo matching

Scene Flow의 PSMNet에 적용했을 때 end-point error는 baseline `0.51`에서 `0.48`로 감소했다. Efficient Attention block의 메모리는 약 `796 MB`였지만, 동일 위치의 non-local block을 직접 구성하면 추정치가 약 `9.68 TB`에 이른다. 고해상도 cost volume에서 quadratic map을 피하는 효과가 특히 크다.

## 실험을 해석할 때 주의할 점

실험은 “표준 Transformer softmax를 완전히 같은 출력으로 더 빠르게 계산했다”는 증거가 아니다. vision backbone에 넣는 global context module로서 품질과 비용이 좋다는 증거에 가깝다. 특히 COCO 개선은 attention 수학뿐 아니라 어느 stage에 module을 배치했는지, 당시 non-local baseline의 구현 비용과도 연결된다.

또한 논문의 GPU 수치와 FLOPs는 현대 fused kernel 환경의 절대 속도를 예측하지 않는다. 연산량은 작더라도 두 번의 matrix multiplication과 normalization, data layout 변환이 실제 latency를 좌우한다.

## 장점과 기여

- 결합 법칙이라는 단순한 관찰로 `[n,n]` 중간 행렬을 제거했다.
- scaling normalization에서는 정확한 등가성을 명확히 제시했다.
- softmax를 separable normalization으로 바꿔 non-negative, normalized weight를 유지했다.
- 고해상도 vision feature에 전역 context를 저비용으로 적용할 수 있음을 detection과 stereo에서 보였다.
- 이후 linear attention을 이해할 때 필요한 “먼저 `K^T V`를 계산한다”는 기본 패턴을 정립했다.

## 한계와 비판적 관점

### 1. 표준 softmax attention과는 다르다

row-wise `softmax(QK^T)`를 정확히 보존하지 않는다. 따라서 sharp retrieval이나 query별 pairwise alignment가 중요한 문제에서 품질 차이가 날 수 있다.

### 2. Causal mask가 자연스럽게 결합되지 않는다

전체 sequence의 `K^T V`를 먼저 계산하면 미래 token 정보까지 섞인다. causal attention에는 위치별 prefix accumulation이 필요하며, 논문이 주로 다루는 bidirectional vision setting과 별도의 구현이 된다.

### 3. 작은 latent context가 병목이 될 수 있다

모든 위치 간 상호작용이 `d_k`개의 context channel을 통과한다. `d_k`가 작으면 효율적이지만, 동시에 표현할 수 있는 global interaction의 rank를 제한한다.

### 4. 이론 복잡도와 하드웨어 효율은 다르다

full attention은 quadratic이지만 GPU에서 매우 잘 최적화된 dense kernel을 사용할 수 있다. 짧은 sequence에서는 Efficient Attention의 normalization과 작은 GEMM이 오히려 불리할 수 있다.

## 후속 연구와의 연결

- **Linear Transformer / Performer**: `K^T V`를 먼저 만드는 구조를 kernel feature map으로 일반화한다.
- **Linformer**: sequence 축을 학습된 low-rank projection으로 줄인다.
- **Nyströmformer**: landmark를 이용해 attention matrix를 세 factor로 근사한다.
- **FlashAttention**: attention 분포를 근사하지 않고, exact softmax attention의 IO와 tiling을 최적화한다.

이 논문은 “attention을 선형화하는 방법”과 “exact attention을 메모리 효율적으로 계산하는 방법”이 서로 다른 연구 축임을 구분하는 좋은 출발점이다.

## 구현 체크리스트

- `softmax(Q, dim=-1)`와 `softmax(K, dim=-2)`의 축이 정확한가?
- 중간 graph에 `[N,N]` tensor가 숨어서 생성되지 않는가?
- mixed precision에서 softmax와 accumulation을 FP32로 처리할 필요가 있는가?
- causal task라면 prefix sum 또는 recurrent state가 필요한가?
- `d_k` 변화에 따른 품질과 latency를 함께 측정했는가?
- baseline과 같은 normalization, projection 수, residual 구조로 비교했는가?

## 온디바이스·비전 관점

온디바이스 vision에서는 `N=H×W`가 크고 memory bandwidth가 제한되므로 `[N,N]` materialization을 피하는 장점이 직접적이다. 특히 segmentation, depth, stereo처럼 spatial resolution을 늦게까지 유지하는 네트워크에서 유리하다. 반면 mobile accelerator가 작은 batched GEMM이나 transpose를 잘 지원하지 않으면 이론 FLOPs 감소가 latency로 이어지지 않을 수 있다.

실무에서는 다음을 함께 측정해야 한다.

```text
peak activation memory
operator fusion 가능성
NCHW/NHWC transpose 비용
softmax 지원 여부
quantization 시 exp와 accumulation 안정성
```

## 최종 평가

Efficient Attention의 가장 큰 가치는 복잡한 근사 이론보다 **attention 계산 그래프를 다시 괄호 치는 것만으로 quadratic intermediate를 제거할 수 있다**는 명료한 통찰이다. scaling normalization에서는 정확하고, separable softmax에서는 효율적인 global context module이 된다. 다만 표준 Transformer의 softmax를 그대로 보존한다고 이해하면 안 된다. 이 정확성의 경계를 분명히 인식할 때, 논문은 linear attention 계열의 수학적 출발점이자 고해상도 vision attention의 실용적 설계로 평가할 수 있다.

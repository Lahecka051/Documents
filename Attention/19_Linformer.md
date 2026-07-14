# 19. Linformer: Self-Attention with Linear Complexity

## 논문 정보

- 제목: **Linformer: Self-Attention with Linear Complexity**
- 저자: Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma
- 발표: 2020
- 핵심 키워드: low-rank attention, sequence projection, linear complexity, projection sharing

## 한눈에 보는 요약

Linformer는 self-attention matrix가 실제로는 낮은 rank로 근사될 수 있다는 관찰에서 출발한다. key와 value의 **sequence length 축** `n`을 학습된 projection `E,F`로 `k`까지 줄인 뒤 attention을 계산한다.

```math
\begin{aligned}
K\in\mathbb{R}^{n\times d}&\longmapsto EK\in\mathbb{R}^{k\times d},\\
V\in\mathbb{R}^{n\times d}&\longmapsto FV\in\mathbb{R}^{k\times d}.
\end{aligned}
```

query는 `n`개를 유지하지만 key/value slot은 `k`개만 남으므로 score가 `[n,n]` 대신 `[n,k]`가 된다. `k << n`이고 `k`를 길이와 무관한 상수로 두면 time과 memory가 `O(nk)`, 즉 `n`에 대해 선형이다.

이 방법은 sparse attention처럼 일부 원본 token을 선택하는 것이 아니다. projection의 각 row가 sequence 전체를 섞어 `k`개의 latent key/value를 만든다. 따라서 dense global information을 낮은 rank bottleneck으로 압축하는 방법이다.

<p align="center"><img src="https://github.com/user-attachments/assets/05740013-ccb8-4cc9-b593-477f77c1e7b7" alt="Linformer projection and inference time" width="760"></p>
<p align="center"><sub>Figure 2 — sequence-axis projection과 길이에 따른 Linformer 추론 시간</sub></p>

## 문제의식

한 head의 표준 self-attention은 다음과 같다.

```math
\begin{aligned}
P&=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt d}\right)
&&\in\mathbb{R}^{n\times n},\\
Y&=PV&&\in\mathbb{R}^{n\times d}.
\end{aligned}
```

`P`를 직접 만들면 `O(n²)` memory와 compute가 필요하다. 저자들은 pretrained Transformer의 attention map singular value를 조사해 몇 개의 큰 singular value가 대부분의 energy를 차지하는 경향을 관찰한다. 즉 `P`의 effective rank가 `n`보다 훨씬 작을 수 있다는 가설을 세운다.

그렇다면 `n`개의 key/value를 그대로 유지할 필요 없이, attention이 사용하는 중요한 subspace만 `k`개 latent slot으로 표현할 수 있다.

## 핵심 수식

입력 `X ∈ R^{n×d_model}`에서 head별 query/key/value를 만든다.

```math
\begin{aligned}
Q&=XW_i^Q&&\in\mathbb{R}^{n\times d_h},\\
K&=XW_i^K&&\in\mathbb{R}^{n\times d_h},\\
V&=XW_i^V&&\in\mathbb{R}^{n\times d_h}.
\end{aligned}
```

projection matrix는 sequence 축에 작용한다.

```math
\begin{aligned}
E_i&\in\mathbb{R}^{k\times n},
&F_i&\in\mathbb{R}^{k\times n},\\
\bar K&=E_iK&&\in\mathbb{R}^{k\times d_h},\\
\bar V&=F_iV&&\in\mathbb{R}^{k\times d_h}.
\end{aligned}
```

Linformer head의 출력은 다음과 같다.

```math
\begin{aligned}
\mathrm{head}_i
&=\mathrm{softmax}\!\left(\frac{Q\bar K^{\top}}{\sqrt{d_h}}\right)\bar V\\
&=\mathrm{softmax}\!\left(\frac{Q(E_iK)^{\top}}{\sqrt{d_h}}\right)(F_iV).
\end{aligned}
```

score shape은 `[n,k]`, output은 `[n,d_h]`다. query마다 `k`개의 latent key를 선택하고, 같은 projection으로 압축된 value를 조합한다.

### 왜 K와 V 둘 다 projection하는가

key만 줄이면 score는 `[n,k]`지만 그것을 원래 `[n,d]` value에 곱할 수 없다. value도 같은 수의 latent slot으로 줄여야 attention 확률의 마지막 축과 맞는다.

```text
attention probability : [n,k]
projected value       : [k,d]
output                : [n,d]
```

`E`와 `F`는 동일할 수도 있고 별도로 학습할 수도 있다.

## Low-rank 관점

표준 attention output을 `P V`라고 하자. 논문의 이론은 임의의 value vector `v`에 대해 `Pv`의 작용을 낮은 rank projection으로 근사할 수 있다는 방향으로 전개된다. Johnson–Lindenstrauss 유형의 결과를 이용해 적절한 `k`가 `n`에 직접 비례하지 않아도 근사 가능하다는 bound를 제시한다.

여기서 주의할 점은 중요하다.

- 이론은 모든 입력에서 attention matrix 자체가 정확히 rank `k`라는 뜻이 아니다.
- 특정 `P`가 value에 작용한 결과를 확률적으로 근사하는 결과에 가깝다.
- 실제 모델의 `E,F`는 random projection이 아니라 task와 함께 학습된다.

따라서 논문 제목의 linear complexity는 `k`를 작은 상수로 고정한다는 실용적 가정과 결합되어 성립한다.

## Projection 공유 방식

head와 layer마다 `E,F ∈ R^{k×n}`를 별도로 두면 parameter가 커진다. 논문은 여러 sharing scheme을 비교한다.

### Headwise projection

각 head가 별도 `E_i,F_i`를 갖는다. 표현력은 가장 크지만 projection parameter도 `H×k×n`으로 늘어난다.

### Key-value sharing

한 head 안에서 `E_i = F_i`로 둔다. key와 value를 동일한 sequence mixture로 압축한다.

### Head sharing

한 layer의 모든 head가 같은 `E,F`를 사용한다. head별 Q/K/V channel projection은 다르지만 sequence 축 압축은 공유한다.

### Layer sharing

모든 layer가 하나의 `E,F`를 공유한다. 가장 parameter-efficient하며 논문 실험에서 성능 손실이 작았다.

Projection을 average pooling이나 1D convolution으로 대체하는 변형도 실험한다. 학습된 dense projection이 가장 일반적이지만, pooling/convolution은 길이 변화와 local inductive bias 측면에서 장점이 있다.

## 계산 복잡도

표준 attention과 Linformer의 주요 항을 비교하면 다음과 같다.

| 연산 | 표준 | Linformer |
| --- | ---: | ---: |
| Score | `O(n² d_h)` | `O(n k d_h)` |
| Value aggregation | `O(n² d_h)` | `O(n k d_h)` |
| Attention memory | `O(n²)` | `O(nk)` |
| Sequence projection | 없음 | `O(n k d_h)` |

`k`가 128 또는 256으로 고정되고 `n`이 수천 이상이면 큰 절감이 난다. 반대로 `n`이 512이고 `k=256`이면 압축 비율이 작아 projection overhead까지 고려해야 한다.

## 길이와 projection matrix

Dense `E,F ∈ R^{k×n}`는 학습 시 최대 sequence length `n`에 묶인다. 길이가 달라질 때는 다음 선택이 필요하다.

```text
shorter input : E,F의 앞 column만 사용하거나 padding
longer input  : 새 column이 없으므로 직접 적용 불가
```

이 점은 positional interpolation처럼 임의 길이로 자연스럽게 extrapolate하는 방법과 다르다. convolution/pooling projection은 길이에 더 유연하지만 논문의 주된 학습된 matrix와 표현이 달라진다.

## Causal attention에서의 문제

`E K`와 `F V`는 sequence 전체 위치를 섞는다. autoregressive token `i`의 query가 latent slot을 볼 때, 그 slot 안에 미래 위치 `j>i`의 key/value가 포함되어 있으면 정보 누출이 생긴다.

표준 causal mask를 `[n,k]` score에 단순 적용해도 projection 전에 미래 정보가 섞인 문제를 해결하지 못한다. causal Linformer에는 position-dependent projection 또는 lower-triangular 구조, prefix-specific latent state가 필요하다. 원 논문의 중심 실험이 bidirectional encoder인 이유와 연결된다.

## 작은 예시

길이 8을 `k=2`로 줄인다고 하자.

```text
E row 1 = 위치 1~4의 가중합
E row 2 = 위치 5~8의 가중합

K_bar = [앞부분 latent key,
         뒷부분 latent key]
```

실제 dense projection은 모든 위치에 weight를 가질 수 있으므로 단순 pooling보다 유연하다. query는 두 latent slot만 고르지만, 각 slot이 문서 전체의 mixture이므로 global information은 유지된다. 동시에 token 3과 4를 별도로 구분해야 하는 관계는 projection에서 손실될 수 있다.

## 실험 결과

### Pretraining 길이와 projection rank

논문은 길이 512에서 `k=128`, 길이 1024에서 `k=256` 같은 설정을 사용한다. 이 범위에서 full Transformer와 비슷한 pretraining loss를 보이며, attention spectrum의 effective rank가 sequence length보다 작다는 가설을 뒷받침한다.

### GLUE downstream

논문 표의 평균 점수는 다음과 같다.

| 모델 | GLUE 평균 |
| --- | ---: |
| RoBERTa baseline | 92.25 |
| Linformer, `k=256`, layer-shared projection | **92.30** |

개별 task에서 작은 등락은 있지만, 강하게 projection을 공유해도 평균 성능을 유지했다. 이는 `E,F` parameter를 head/layer마다 복제하지 않아도 encoder representation에 충분한 low-rank structure가 있음을 시사한다.

### 속도와 메모리

V100 기반 논문 측정에서 `k=128` 설정의 예시는 다음과 같다.

| Sequence length | Training speedup | Memory reduction |
| ---: | ---: | ---: |
| 8,192 | 약 5.5× | 약 28× |
| 65,536 | 약 20× | 약 60× |

길이가 커질수록 quadratic baseline과의 격차가 커진다. 다만 이는 당시 dense attention 구현과 비교한 수치이며, 최신 fused exact attention과 같은 hardware 조건의 재측정은 별도로 필요하다.

## Attention spectrum 실험의 해석

저자들은 pretrained model의 attention matrix singular value가 빠르게 감소하는 시각화를 제시한다. 이것은 low-rank approximation이 가능할 이유를 제공하지만 다음을 조심해야 한다.

- 평균 spectrum이 낮은 rank라는 것과 모든 head·입력에서 rank가 낮다는 것은 다르다.
- softmax matrix는 input-dependent인데 `E,F`는 보통 input-independent parameter다.
- rare retrieval pattern은 평균 energy가 작아도 task에 결정적일 수 있다.

따라서 spectrum은 설계 동기이지, 작은 `k`가 모든 장문 task를 보존한다는 보장은 아니다.

## 장점과 기여

- sequence 축을 직접 projection해 `[n,n]` score를 `[n,k]`로 줄이는 단순한 구조를 제시했다.
- sparse edge가 아니라 dense global mixture를 유지한다.
- projection을 head/layer 사이에서 공유해 parameter overhead를 줄였다.
- encoder pretraining과 GLUE에서 full attention 수준 품질을 보여줬다.
- 긴 길이에서 time과 memory 이득이 커지는 실험을 제시했다.

## 한계와 비판적 관점

### 1. Fixed maximum length

학습된 `E,F`의 column 수가 최대 길이에 묶인다. 더 긴 context로 바로 extrapolate하기 어렵고, 길이별 parameter 처리도 필요하다.

### 2. Causal masking의 비호환성

projection 단계에서 미래 token을 섞을 수 있으므로 decoder self-attention에 단순 적용할 수 없다. 논문의 강한 결과는 주로 bidirectional encoder에 해당한다.

### 3. Input-independent compression

어떤 위치를 합칠지 `E,F`가 모든 입력에서 공유한다. 문서마다 다른 중요한 token을 동적으로 보존하는 능력은 content-based routing보다 약할 수 있다.

### 4. Rank bottleneck

copy, associative recall, needle retrieval처럼 개별 위치를 정밀하게 구분해야 하는 task는 작은 `k`에서 손실될 수 있다. 평균 benchmark 점수가 rare long-range dependency를 충분히 검증하지 않을 수도 있다.

### 5. 현대 exact kernel과 비교 필요

FlashAttention은 full attention 분포를 유지하면서 score materialization을 피한다. 중간 길이에서는 근사 방법의 kernel overhead보다 exact tiled attention이 빠를 수 있다.

## 관련 방법과 비교

| 방법 | 줄이는 축 | 핵심 중간 크기 | Exact full attention? |
| --- | --- | ---: | --- |
| Linformer | sequence rank | `[n,k]` | 아니오 |
| Performer | kernel feature | `[n,r]` | random-feature 근사 |
| Nyströmformer | landmark | `[n,m]` | landmark 근사 |
| Longformer | edge 집합 | `[n,w+g]` | 선택 edge 안에서 exact |
| FlashAttention | IO/materialization | tile | 예 |

## 구현 체크리스트

- `E,F`가 channel 축이 아니라 sequence 축에 곱해지는가?
- shape convention이 `E:[k,n]`인지 framework transpose와 일치하는가?
- shorter sequence에서 projection을 crop/pad하는 규칙이 일관적인가?
- head/layer sharing 설정이 parameter 수 계산과 맞는가?
- causal task에서 미래 정보가 projection에 들어가지 않는가?
- rank `k`별 품질·latency·memory curve를 측정했는가?
- projection GEMM과 transpose 비용을 profiler에 포함했는가?

## 온디바이스 관점

Linformer는 irregular sparsity나 sort가 없고 dense GEMM만 사용하므로 accelerator 친화적일 가능성이 높다. `E K`, `F V`, `[n,k]×[k,d]`가 모두 정적 shape라 compiler가 최적화하기 쉽다. 그러나 `E,F` parameter가 `n`에 비례하고 긴 최대 길이에 묶이는 점은 storage와 다양한 input length 지원에 부담이 된다.

Vision에서는 patch 수가 고정된 image encoder에 잘 맞는다. 반대로 해상도가 자주 바뀌면 sequence projection을 interpolation하거나 resolution별 weight를 준비해야 한다. quantization 시에는 projection weight와 softmax activation의 dynamic range도 확인해야 한다.

## 최종 평가

Linformer는 full attention의 sequence dimension을 학습된 low-rank bottleneck으로 압축한다는 매우 직접적인 해법이다. `k`가 작을 때 memory와 compute가 선형이고, encoder benchmark에서는 projection을 강하게 공유해도 품질을 유지했다. 다만 input-independent projection, 최대 길이에 종속된 parameter, causal masking의 어려움 때문에 범용 decoder attention 대체로 보기는 어렵다. “attention map은 얼마나 낮은 rank인가?”라는 질문을 실용적인 architecture로 연결한 대표 연구로서 가치가 크다.

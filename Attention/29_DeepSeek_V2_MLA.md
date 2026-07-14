# 29. DeepSeek-V2의 Multi-head Latent Attention (MLA)

## 논문 정보

- 원 논문: **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**
- 저자: DeepSeek-AI
- 발표: 2024
- 리뷰 범위: DeepSeek-V2 전체 중 **Multi-head Latent Attention(MLA)** 중심
- 핵심 키워드: KV cache compression, low-rank joint compression, projection absorption, decoupled RoPE, latent attention

## 한눈에 보는 요약

Multi-head Latent Attention(MLA)은 각 token의 full multi-head K와 V를 cache하는 대신, K와 V를 공동으로 생성할 수 있는 작은 latent vector `c_KV`만 cache한다. 별도의 low-rank down-projection으로 hidden state를 압축하고, key와 value는 각각 up-projection으로 복원한다.

```math
\begin{aligned}
c_{KV}&=W_{DKV}h,\\
k_C&=W_{UK}c_{KV},\\
v_C&=W_{UV}c_{KV}.
\end{aligned}
```

Inference에서는 행렬 곱의 결합 법칙으로 `W_UK`를 query projection 쪽에, `W_UV`를 output projection 쪽에 흡수할 수 있어 매 step 과거의 full K/V를 복원할 필요가 없다.

문제는 RoPE다. Position-dependent rotation이 `W_UK`와 query 사이에 들어가면 projection absorption이 깨진다. DeepSeek-V2는 content key/query에는 low-rank latent 경로를 사용하고, 작은 별도 query/key에만 RoPE를 적용하는 **decoupled RoPE**로 해결한다. Cache에는 `c_KV`와 작은 shared RoPE key만 저장한다.

DeepSeek-V2 설정에서 KV cache는 MHA보다 93.3% 줄고, 이론상 GQA 2.25 group 정도의 cache 크기로 MHA보다 강한 결과를 보고한다.

<p align="center"><img src="https://github.com/user-attachments/assets/151eda74-9bef-4c43-85b2-5892101206d5" alt="DeepSeek-V2 MLA compressed latent KV" width="680"></p>
<p align="center"><sub>Figure 3 핵심 패널 — compressed latent KV와 추론 시 cache 구조</sub></p>

## MHA의 KV cache 병목

Hidden dimension `d`, head 수 `n_h`, head dimension `d_h`인 MHA는 token `t`에서

```math
\begin{aligned}
q_t&=W^Qh_t,\\
k_t&=W^Kh_t,\\
v_t&=W^Vh_t.
\end{aligned}
```

를 만들고 `n_h`개의 head로 나눈다. Autoregressive inference에서는 모든 과거 `k_t,v_t`를 layer마다 저장한다.

```math
\text{cache elements per token across }l\text{ layers}=2n_hd_hl
```

DeepSeek-V2처럼 head가 128개이고 context가 128K이면 KV cache가 batch size와 throughput을 제한한다. MQA/GQA는 KV head를 공유해 줄이지만 논문의 ablation에서는 hard benchmark 품질이 MHA보다 낮았다.

## MLA의 핵심: K/V 공동 low-rank compression

Token hidden state `h_t ∈ R^d`를 작은 latent로 down-project한다.

```math
\begin{aligned}
c_t^{KV}&=W^{DKV}h_t,\\
c_t^{KV}&\in\mathbb{R}^{d_c},\\
W^{DKV}&\in\mathbb{R}^{d_c\times d}.
\end{aligned}
```

이 latent에서 content key와 value를 up-project한다.

```math
\begin{aligned}
k_t^C&=W^{UK}c_t^{KV},\\
v_t^C&=W^{UV}c_t^{KV},\\
W^{UK},W^{UV}&\in\mathbb{R}^{(n_hd_h)\times d_c}.
\end{aligned}
```

`d_c << n_h d_h`이므로 full K/V 대신 `c_t^KV`만 cache하면 layer당 token당 `d_c` element만 필요하다. K와 V에 별도 latent를 두지 않고 **joint latent 하나**를 공유하는 것이 추가 절감이다.

## Query low-rank compression

Training activation memory를 줄이기 위해 query도 별도 latent를 거친다.

```math
\begin{aligned}
c_t^Q&=W^{DQ}h_t,\\
q_t^C&=W^{UQ}c_t^Q.
\end{aligned}
```

Query는 현재 token에서 한 번 계산하고 cache하지 않으므로 decode KV cache 크기를 직접 줄이지는 않는다. 하지만 큰 `n_h d_h` query activation과 projection 구조를 low-rank로 제한해 training memory/parameterization에 영향을 준다.

## Projection absorption

Naive inference라면 과거 latent마다 `W^UK c_j^KV`, `W^UV c_j^KV`를 복원해야 해 cache는 작아도 compute가 커진다. MLA의 핵심 최적화는 결합 법칙이다.

### Key up-projection 흡수

Content score를 쓰면

```math
\begin{aligned}
(q_t^C)^{\top}k_j^C
&=(W^{UQ}c_t^Q)^{\top}(W^{UK}c_j^{KV})\\
&=(c_t^Q)^{\top}\left[(W^{UQ})^{\top}W^{UK}\right]c_j^{KV}.
\end{aligned}
```

두 projection을 미리 결합한 matrix로 query를 latent key space에 보내면 cached `c_j^KV`와 직접 dot product할 수 있다. 논문 표현으로 `W^UK`를 query projection에 absorb한다.

### Value up-projection 흡수

Attention weight `a_j`에 대해

```math
\begin{aligned}
\sum_ja_jv_j^C
&=\sum_ja_jW^{UV}c_j^{KV}\\
&=W^{UV}\left(\sum_ja_jc_j^{KV}\right).
\end{aligned}
```

Head concatenate 뒤 output projection `W^O`와 `W^UV`를 결합할 수 있다. 먼저 latent value weighted sum을 만든 뒤 absorbed output projection을 적용한다.

따라서 decode kernel은 full K/V를 매번 복원하지 않고 latent cache를 직접 읽는다.

## RoPE가 absorption을 깨뜨리는 이유

RoPE는 위치 `j`에 따라 key에 rotation matrix `R_j`를 적용한다.

```math
k_j=R_jW^{UK}c_j^{KV}
```

Score는

```math
q_t^{\top}R_jW^{UK}c_j^{KV}
```

가 된다. `R_j`가 token position마다 다르므로 `W^UK`를 query projection에 하나의 고정 matrix로 미리 흡수할 수 없다. 행렬 곱은 결합 가능하지만 교환 가능하지 않아 `R_j`를 지나서 projection 순서를 바꿀 수 없다.

Full key를 cache하지 않으면 매 decode step 모든 과거 position의 rotated key를 다시 만들어야 하므로 MLA의 compute 이득이 사라진다.

## Decoupled RoPE

DeepSeek-V2는 attention query/key를 두 부분으로 나눈다.

```text
content part: low-rank MLA, RoPE 없음
position part: small separate q_R, shared k_R, RoPE 적용
```

Query head `i`와 key는

```math
\begin{aligned}
q_{t,i}&=\left[q_{t,i}^C;q_{t,i}^R\right],\\
k_{j,i}&=\left[k_{j,i}^C;k_j^R\right].
\end{aligned}
```

이다. RoPE 부분은

```math
\begin{aligned}
q_t^R&=\mathrm{RoPE}(W^{QR}c_t^Q),\\
k_t^R&=\mathrm{RoPE}(W^{KR}h_t).
\end{aligned}
```

로 계산한다. `q^R`은 multi-head지만 `k^R`은 head 사이에서 공유한다. Score는 content dot product와 position dot product의 합이다.

```math
\mathrm{score}_{t,j,i}
=\frac{(q^C)^{\top}k^C+(q^R)^{\top}k^R}{\sqrt{d_h+d_h^R}}
```

RoPE는 작은 position subspace에만 존재하므로 content `W^UK` absorption은 유지된다. Cache에는 latent `c_j^KV`와 shared rotated key `k_j^R`만 저장한다.

## 최종 KV cache 크기

MLA의 layer당 token cache는

```math
d_c+d_h^R
```

이고 전체 `l` layer에서는

```math
(d_c+d_h^R)l
```

이다. 비교하면 다음과 같다.

| Attention | Token당 cache element, l layers |
| --- | ---: |
| MHA | `2 n_h d_h l` |
| GQA | `2 n_g d_h l` |
| MQA | `2 d_h l` |
| MLA | `(d_c + d_h^R) l` |

DeepSeek-V2는 `d_c=4d_h`, `d_h^R=d_h/2`를 사용하므로 MLA cache는 `4.5d_h l`, 즉 MQA의 `2.25×`, GQA 2.25 group과 같은 규모다. MHA head 128개와 비교하면 매우 작다.

## DeepSeek-V2 구체 설정

논문 설정은 다음과 같다.

```math
\begin{aligned}
\text{attention heads }n_h&=128,\\
\text{per-head dim }d_h&=128,\\
\text{KV latent dim }d_c&=512,\\
\text{query latent dim }d'_c&=1536,\\
\text{RoPE head dim }d_h^R&=64,\\
\text{context length}&=128\mathrm{K}.
\end{aligned}
```

Layer당 token cache는 latent 512와 RoPE key 64, 총 576 element다. MHA의 `2×128×128=32768` element와 비교하면 약 98.2% 작은 이론 element 수다. 논문의 실제 배포 전체 비교에서는 precision·다른 최적화를 포함해 KV cache 93.3% 감소를 보고한다.

## Attention 계산 흐름

```text
current token h_t
  ├─ W_DQ -> c_Q -> content query
  ├─ c_Q -> RoPE query per head
  └─ W_DKV -> c_KV(current), cache
              └─ W_KR h_t -> shared RoPE key, cache

decode against prefix:
  content score: absorbed query · cached c_KV
  position score: RoPE query · cached RoPE key
  softmax over prefix
  weighted sum of cached c_KV
  absorbed value/output projection
```

MLA 전용 kernel은 두 cache component와 absorbed weight layout을 직접 지원해야 한다.

## MQA/GQA/MHA ablation

논문은 약 7B dense model을 같은 1.33T token으로 학습해 비교한다.

| Benchmark | MQA | GQA-8 | MHA |
| --- | ---: | ---: | ---: |
| BBH | 33.2 | 35.6 | **37.0** |
| MMLU | 37.9 | 41.2 | **45.2** |
| C-Eval | 30.0 | 37.7 | **42.9** |
| CMMLU | 34.6 | 38.4 | **43.5** |

이 결과는 극단적 KV head sharing이 hard benchmark capacity를 낮출 수 있다는 MLA의 동기를 강화한다.

## MLA와 MHA 직접 비교

Architecture를 attention 외에는 맞춘 small/large MoE model 결과는 다음과 같다.

| 항목 | Small MHA | Small MLA | Large MHA | Large MLA |
| --- | ---: | ---: | ---: | ---: |
| KV cache/token | 110.6K | **15.6K** | 860.2K | **34.6K** |
| BBH | 37.9 | **39.0** | 46.6 | **50.7** |
| MMLU | 48.7 | **50.0** | 57.5 | **59.0** |
| C-Eval | **51.6** | 50.9 | 57.9 | **59.2** |
| CMMLU | 52.3 | **53.4** | 60.7 | **62.5** |

MLA cache는 small에서 MHA의 약 14%, large에서 약 4%이고 대부분 benchmark에서 MHA보다 높다. 다만 model layer/activated parameter가 완전히 동일하지 않은 항목이 있어 attention 하나의 순수 효과로 과도하게 일반화하면 안 된다.

## 전체 DeepSeek-V2 효율 결과

DeepSeek-V2는 236B total parameter 중 token당 21B를 activate하는 MoE model이고 8.1T token으로 학습되었다. 이전 DeepSeek 67B 대비 논문은 다음을 보고한다.

- Training cost 42.5% 절감
- KV cache 93.3% 절감
- Maximum generation throughput 5.76×
- 배포 generation throughput 50K tokens/s 이상

이 수치는 MLA뿐 아니라 DeepSeekMoE, quantization, serving optimization의 결합 결과다. MLA의 독립 기여로 동일시해서는 안 된다.

## 장점과 기여

- K와 V를 별도 head 공유가 아니라 하나의 low-rank latent로 공동 압축했다.
- Projection absorption으로 cache 압축이 과거 K/V 재계산으로 이어지지 않게 했다.
- RoPE와 absorption의 충돌을 정확히 지적하고 decoupled RoPE로 해결했다.
- MQA급 cache 크기에서 MHA 이상의 benchmark 품질 가능성을 보였다.
- 128K context 대형 MoE model에서 실제 throughput 개선으로 연결했다.

## 한계와 비판적 관점

### 1. 구현과 kernel이 복잡하다

일반 MHA/GQA kernel에 단순 shape 변경만으로 넣기 어렵다. Latent content score, shared RoPE key, absorbed value output을 지원해야 한다.

### 2. Low-rank bottleneck

모든 K/V 정보가 `d_c` latent를 통과한다. DeepSeek 규모에서 잘 작동했지만 작은 model, multimodal high-rank feature, exact retrieval에서 optimal rank가 다를 수 있다.

### 3. Decoupled RoPE의 표현 제약

Position interaction이 작은 별도 subspace에 제한된다. Full key dimension 전체에 RoPE를 적용하는 MHA와 inductive bias가 다르다.

### 4. 전체 모델 결과와 MLA 기여 분리

5.76× throughput과 93.3% cache 절감에는 quantization, MoE, system optimization이 함께 들어간다. MLA-only matched deployment benchmark가 더 필요하다.

### 5. Training parameter/compute

Down/up projection과 query compression이 추가되고 training forward에는 full multi-head representation이 필요할 수 있다. Decode cache 절감과 training FLOPs 절감은 같은 문제가 아니다.

## GQA와 MLA 비교

| 축 | GQA | MLA |
| --- | --- | --- |
| 압축 방식 | KV head 공유 | K/V 공동 low-rank latent |
| Cached state | G개의 full KV head | `c_KV` + small RoPE key |
| 위치 처리 | full head RoPE | decoupled RoPE |
| 기존 MHA 변환 | mean pooling+uptraining 가능 | architecture 변경·재학습 필요 |
| Kernel | 비교적 단순 broadcast | absorbed projection 전용 kernel |

## 구현 체크리스트

- `c_KV` 하나가 K와 V up-projection에 공동 사용되는가?
- Inference에서 과거 full `k_C,v_C`를 복원하지 않는가?
- `W_UK`와 `W_UV` absorption의 transpose/ head layout이 정확한가?
- RoPE가 content key가 아니라 decoupled q_R/k_R에만 적용되는가?
- Shared RoPE key를 head 수만큼 물리적으로 복제하지 않는가?
- Cache allocator가 `(d_c+d_h^R)` 기준으로 실제 줄었는가?
- Prefill, decode, cache memory, quality를 분리해 측정했는가?

## 온디바이스 관점

MLA는 긴 context의 RAM과 DRAM bandwidth를 크게 줄일 잠재력이 있어 온디바이스 LLM에 매력적이다. Cache가 latent 512+position 64처럼 고정된 작은 vector면 context capacity와 energy가 개선된다. 그러나 absorbed projection을 효율적으로 실행하는 custom kernel과 weight conversion이 필요하며 기존 GQA model을 바로 MLA로 바꿀 수 없다.

작은 NPU에서는 low-rank GEMM이 잘 맞지만 RoPE branch와 latent attention fusion이 지원되지 않으면 kernel launch/transpose overhead가 커질 수 있다. 따라서 architecture 선택은 학습 단계부터 device compiler와 co-design해야 한다.

## 최종 평가

MLA는 MQA/GQA처럼 KV head 수만 줄이는 대신 **각 token의 K와 V를 생성하는 공통 latent state 자체를 cache**한다. Projection absorption으로 복원 compute를 없애고, decoupled RoPE로 position encoding과 결합 법칙의 충돌을 해결한 점이 가장 독창적이다. 구현 난도와 low-rank 가정은 크지만, 매우 작은 cache로 MHA급 이상의 품질을 보인 결과는 KV-cache 설계의 새로운 축을 열었다.

# 88. Kimi K3: Open Frontier Intelligence

## 논문 정보

- 원문 파일: k3_tech_report.pdf
- 제목: **Kimi K3: Open Frontier Intelligence**
- 저자: Kimi Team
- 공개 시점: 2026년 7월 기술 보고서
- 모델 유형: Native multimodal Mixture-of-Experts language model
- 전체 파라미터: 2.78T
- 토큰당 활성 파라미터: 104.2B
- 최대 문맥 길이: 1M tokens
- 공식 모델 가중치: [MoonshotAI/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)
- 주요 주제: Kimi Delta Attention, Gated MLA, Attention Residuals, Stable LatentMoE, native vision, long-context RL, distributed training, inference serving

> 이 문서는 47쪽 기술 보고서의 구조, 수식, 학습 및 시스템 설계, 평가 결과를 함께 해설한다. 논문이 주장하는 성능 수치뿐 아니라 무엇이 실제로 공개되었고 무엇이 재현에 부족한지, 전체 파라미터와 활성 파라미터가 각각 무엇을 의미하는지, 알고리즘상의 복잡도와 실제 하드웨어 속도가 왜 다를 수 있는지도 구분한다.

## 한눈에 보는 결론

Kimi K3는 단순히 Kimi K2를 크게 만든 모델이 아니다. 이 보고서의 핵심은 다음 네 축을 동시에 바꾼 데 있다.

1. **시퀀스 방향:** 69개의 Kimi Delta Attention(KDA) 층과 24개의 Gated MLA 층을 3:1 비율로 섞어 1M-token 문맥을 처리한다.
2. **깊이 방향:** Attention Residuals가 현재 층에 직전 층 하나만 넘기는 대신, 임베딩과 이전 블록 표현을 선택적으로 읽게 한다.
3. **채널 방향:** Stable LatentMoE가 896개 routed expert 중 토큰마다 16개를 활성화하고, 2개의 shared expert를 항상 사용한다.
4. **학습 및 시스템 방향:** Native vision, 9개 RL expert의 통합, KDA 전용 커널, 완전 균형 expert parallelism, 1M-context용 캐시와 sandbox 인프라를 모델 구조와 함께 설계한다.

가장 중요한 성과는 **2.78T 파라미터의 open-weight MoE를 실제 학습·후처리·서빙 가능한 시스템으로 묶었다는 것**이다. 단일 모듈의 새로움만 보면 KDA, MLA, MoE, speculative decoding, quantization은 각각 선행 연구가 있다. 그러나 3T급 모델에서 이들을 함께 작동시키기 위해 수치 안정성, 라우팅 균형, activation memory, context parallelism, prefix cache, long-horizon RL까지 연결한 전체 시스템 설계는 강한 기여다.

반대로 논문의 가장 큰 약점도 이 지점에서 나온다. 보고서는 많은 기술을 한꺼번에 도입했지만 다음 근거는 충분히 공개하지 않는다.

- 2.5배 scaling efficiency 개선에서 각 구성요소가 차지하는 기여
- 총 pre-training token 수, 전체 compute, cluster 규모와 학습 기간
- KDA, MoonEP, Block AttnRes의 end-to-end throughput 및 latency 표
- 1M 문맥의 위치별 검색·추론 정확도를 보여 주는 정규 long-context benchmark
- 대부분의 내부 benchmark 데이터와 평가기
- 2.78T 모델을 재현할 수 있는 완전한 optimizer, batch, parallelism configuration

따라서 Kimi K3를 읽을 때에는 다음 두 판단을 동시에 유지해야 한다.

- **연구·시스템적 중요성:** 매우 높다. 특히 long-context hybrid attention과 초대형 sparse MoE를 실제 시스템으로 연결한 보고서로 가치가 크다.
- **개별 주장에 대한 재현 가능성:** 제한적이다. 가중치는 공개되지만 데이터, 전체 학습 recipe, 대규모 infra 및 여러 내부 평가는 닫혀 있다.

## 먼저 구분해야 할 세 가지

### 전체 파라미터와 활성 파라미터

Kimi K3의 “2.8T 모델”이라는 표현은 저장해야 하는 전체 파라미터 수를 말한다. 한 토큰을 처리할 때 모든 2.78T 파라미터가 계산되는 것은 아니다. MoE router가 896개 routed expert 중 16개를 고르기 때문에 토큰당 활성 파라미터는 104.2B이다.

전체 파라미터 대비 토큰당 활성 파라미터 비율은

```math
\frac{104.2\mathrm{B}}{2.78\mathrm{T}}\approx3.75\%
```

이다. 이는 routed expert만 놓고 계산한 선택 비율

```math
\frac{16}{896}\approx1.79\%
```

와 다르다. 104.2B에는 선택된 routed expert뿐 아니라 두 shared expert, KDA와 MLA, embedding 및 기타 항상 활성화되는 weight도 포함되기 때문이다.

이를 구분하면 다음과 같다.

| 항목 | 의미 | Kimi K3에서의 영향 |
|---|---|---|
| Total parameters | 체크포인트에 존재하는 모든 weight | 저장 공간, 전체 모델 배치, expert sharding 비용 |
| Active parameters | 한 토큰의 forward에 실제 참여하는 weight | 토큰당 연산량과 메모리 대역폭의 주요 근사 |
| Trainable parameters | 학습 시 gradient와 optimizer state를 유지하는 weight | pre-training memory 및 communication |
| Resident parameters | 특정 GPU 또는 노드에 실제 올라간 weight | serving topology와 expert parallelism |

MoE는 dense 2.78T 모델보다 토큰당 계산을 크게 줄이지만, 전체 weight를 어딘가에는 저장하고 필요한 expert를 배치해야 한다. 그러므로 “active 104B이므로 104B dense 모델과 완전히 같다”도 틀리다. Router, expert dispatch, all-to-all 통신, 16개 expert의 weight 접근, shared expert, KDA/MLA 층이 추가로 필요하기 때문이다.

### 학습 scaling efficiency와 추론 속도

보고서의 “Kimi K2 대비 약 2.5배 scaling efficiency”는 같은 validation loss에 도달하기 위한 학습 FLOPs가 줄었다는 scaling-law 주장이다. 이는 다음 주장과 같지 않다.

- 추론 latency가 2.5배 빨라졌다.
- 토큰 처리량이 2.5배 증가했다.
- 전력 소비가 2.5배 감소했다.
- KDA 하나만으로 2.5배 개선되었다.

Kimi K3는 Kimi K2보다 active parameters가 32.6B에서 104.2B로 약 3.2배 증가했다. 품질당 학습 효율이 좋아졌더라도 한 토큰의 절대 추론 비용은 더 클 수 있다. 이 구분은 모델 선택과 배포 판단에서 매우 중요하다.

### 알고리즘 복잡도와 실제 latency

KDA의 recurrent state는 문맥 길이에 따라 커지는 full KV cache를 대체하므로 긴 문맥에서 메모리상 유리하다. 하지만 recurrence는 GPU가 선호하는 대규모 병렬 GEMM과 충돌한다. 그래서 논문은 FlashKDA, context parallelism, state-aware prefix cache, speculative replay kernel까지 별도로 만든다.

즉,

    선형 복잡도
      != 자동으로 빠른 GPU 실행
      != 모든 NPU에서 빠른 실행
      != 짧은 문맥에서도 full attention보다 빠름

Kimi K3의 중요한 교훈은 좋은 점근 복잡도보다 **그 복잡도를 실제 커널과 분산 시스템에서 실현하는 방법**이 더 어렵다는 것이다.

## 연구 배경과 문제의식

기존 LLM의 scaling은 주로 배포 전에 더 큰 모델과 더 많은 데이터에 compute를 쓰는 pre-training scaling이었다. 이후 reasoning model은 추론 시 더 긴 사고, 더 많은 sample, tool use에 compute를 쓰는 test-time scaling을 두 번째 축으로 만들었다.

Kimi K3는 open-weight 생태계가 test-time reasoning 기법은 빠르게 따라가지만, pre-trained foundation 자체는 대체로 1T급 부근에 머물러 있다고 진단한다. 비슷한 크기의 기반 모델에 점점 정교한 RL만 반복하면 가장 강한 proprietary model과의 차이가 줄어들지 않을 수 있다는 문제의식이다.

따라서 Kimi K3의 연구 질문은 다음과 같이 정리할 수 있다.

1. 3T급 sparse MoE를 안정적으로 학습할 수 있는가?
2. 1M-token 문맥에서 attention memory와 계산을 감당할 수 있는가?
3. 깊어진 93개 층에서 정보가 직전 residual state에만 압축되는 병목을 줄일 수 있는가?
4. 896개 expert와 Top-16 routing을 균형 있게 분산 실행할 수 있는가?
5. Vision encoder를 사전 contrastive 학습 없이 처음부터 language objective와 공동 학습할 수 있는가?
6. 수천 번의 tool call과 수백만 누적 token을 갖는 trajectory를 RL로 학습할 수 있는가?
7. 이 모델을 연구용 체크포인트가 아니라 실제 online serving system으로 운영할 수 있는가?

## Kimi K3 핵심 사양

| 구성 | Kimi K2 | Kimi K3 | 변화 |
|---|---:|---:|---:|
| Architecture | MoE | MoE | 동일 계열 |
| Layers | 61 | 93 | +52% |
| Total parameters | 1.04T | 2.78T | +167% |
| Activated parameters | 32.6B | 104.2B | +220% |
| Hidden dimension | 7,168 | 7,168 | 동일 |
| Latent MoE dimension | 없음 | 3,584 | model width의 0.5배 |
| Expert hidden dimension | 2,048 | 3,072 | +50% |
| Routed experts | 384 | 896 | +133% |
| Active experts/token | 8 | 16 | +100% |
| Shared experts | 1 | 2 | +100% |
| Attention heads | 64 | 96 | +50% |
| Dense layers | 1 | 1 | 동일 |
| Vocabulary | 160K | 160K | 동일 |
| 최대 training context | 128K | 1M | 8배 |
| Attention | MLA | 69 KDA + 24 MLA | Hybrid |
| FFN activation | SwiGLU | SiTU-GLU | bounded activation |
| MTP layer | 1 | 1 | 동일 |
| Vision encoder | 없음 | MoonViT-V2, 401M | native vision 추가 |
| ViT depth | 없음 | 27 layers | - |
| ViT patch size | 없음 | 14 | - |
| ViT attention heads | 없음 | 12 | - |

이 표에서 주목할 점은 hidden dimension이 7,168로 유지됐다는 것이다. Kimi K3는 폭 자체를 키우기보다 층 수, attention head 수, expert 수, expert당 hidden dimension과 활성 expert 수를 늘렸다. 특히 routed expert는 896개지만 latent width를 3,584로 줄여 통신 및 expert 계산을 억제한다.

## 전체 아키텍처

Kimi K3는 정보 혼합을 세 방향으로 나눈다.

| 방향 | 담당 모듈 | 질문 |
|---|---|---|
| Token/sequence mixing | KDA + Gated MLA | 멀리 떨어진 token 정보를 어떻게 섞는가? |
| Layer/depth mixing | Attention Residuals | 이전 어느 깊이의 표현을 다시 읽을 것인가? |
| Channel mixing | Stable LatentMoE | 어떤 expert 변환을 적용할 것인가? |

전체 데이터 흐름은 다음과 같다.

    image / video
        |
        v
    MoonViT-V2
        |
        v
    2 x 2 pixel shuffle -> MLP projector
        |
        +----------------------+
                               v
    text tokens ----------> shared token stream
                               |
                               v
                    [KDA + Stable LatentMoE] x 3
                               |
                               v
                  [Gated MLA + Stable LatentMoE] x 1
                               |
                         block 반복
                               |
                               v
                    final global Gated MLA
                               |
                               v
                            output

Layer 수를 전개하면 hybrid block은 23개이다.

```math
23\times(3\,\mathrm{KDA}+1\,\mathrm{MLA})
+1\,\mathrm{final\ MLA}
=69\,\mathrm{KDA}+24\,\mathrm{MLA}=93
```

즉 23개의 3:1 hybrid block 뒤에 별도의 final MLA를 두어, backbone의 마지막 layer는 항상 global attention을 수행한다.

각 attention layer 뒤에는 Stable LatentMoE가 온다. 3개의 KDA와 1개의 MLA가 한 주기를 이루고, backbone 끝에는 global interaction을 보장하기 위한 Gated MLA가 한 층 더 배치된다. 총합은 69 KDA와 24 MLA, 즉 93개 attention layer이다.

## Hybrid Attention

### 왜 KDA와 MLA를 섞는가

Full softmax attention은 모든 token pair를 직접 비교하므로 특정 과거 token을 선택적으로 다시 읽는 능력이 강하다. 반면 길이 $T$가 증가하면 attention matrix 계산이 대략 $O(T^2)$, KV cache가 $O(T)$로 증가한다.

Linear/recurrent attention은 과거를 고정 크기 state에 압축한다. Decode에서 과거 전체 KV를 다시 읽지 않아도 되지만, 압축된 state가 원래 token을 완전히 보존하지 못할 수 있다.

Kimi K3의 3:1 hybrid는 다음 절충이다.

- KDA 3개 층: 긴 문맥을 고정 크기 state로 효율적으로 누적
- Gated MLA 1개 층: 원래 token-to-token global interaction 보완
- 마지막 Gated MLA: 최종 표현이 반드시 global attention을 거치도록 보장

이는 “linear attention이 full attention을 완전히 대체한다”는 설계가 아니다. 오히려 recurrent compression의 약점을 periodic global attention으로 보정한다.

### Kimi Delta Attention의 상태 갱신

한 attention head에서 다음 shape를 생각한다.

| 기호 | Shape | 의미 |
|---|---|---|
| $x_t$ | $\mathbb{R}^{d}$ | 시점 $t$의 hidden state |
| $q_t,k_t$ | $\mathbb{R}^{d_k}$ | query와 key |
| $v_t$ | $\mathbb{R}^{d_v}$ | value |
| $S_t$ | $\mathbb{R}^{d_k \times d_v}$ | 과거를 압축한 recurrent state |
| $\alpha_t$ | $(0,1)^{d_k}$ | key channel별 retention |
| $\beta_t$ | $(0,1)$ | 현재 token의 write strength |

KDA의 핵심 recurrence는 다음과 같다.

```math
S_t =
\left(I-\beta_t k_t k_t^\top\right)
\mathrm{Diag}(\alpha_t)S_{t-1}
+\beta_t k_t v_t^\top
```

```math
\widetilde{o}_t=S_t^\top q_t
```

이를 세 단계로 읽으면 이해하기 쉽다.

1. $\mathrm{Diag}(\alpha_t)S_{t-1}$: channel마다 과거 기억을 서로 다른 비율로 감쇠한다.
2. $I-\beta_t k_tk_t^\top$: 현재 key 방향에 이미 기록된 성분을 delta rule 방식으로 수정한다.
3. $\beta_tk_tv_t^\top$: 현재 key-value association을 state에 쓴다.

단순 누적식 $S_t=S_{t-1}+k_tv_t^\top$은 같은 key가 반복될 때 오래된 값을 계속 더한다. Delta rule은 현재 key 방향의 기존 예측과 새 value 사이의 오차를 수정하는 형태이므로 연관 정보를 덮어쓸 수 있다. 여기에 $\alpha_t$의 channel-wise forgetting을 결합한 것이 KDA이다.

Query, key, value와 gate는 다음 흐름으로 만들어진다.

    x_t
      +-> linear -> short convolution -> Swish -> L2Norm -> q_t
      +-> linear -> short convolution -> Swish -> L2Norm -> k_t
      +-> linear -> short convolution -> Swish          -> v_t
      +-> linear -> Sigmoid                              -> beta_t
      +-> low-rank projection + head bias                -> z_t -> alpha_t

Short convolution은 인접 token의 국소 문맥을 projection 단계에 섞고, query/key L2 normalization은 delta update의 scale을 안정화한다.

### Chunkwise parallel form

Token을 하나씩 recurrence로 처리하면 학습 시 GPU parallelism이 너무 작다. KDA는 sequence를 길이 $C$의 chunk로 나누어 다음 두 계산으로 분리한다.

- **Inter-chunk:** 이전 chunk에서 넘어온 state가 현재 chunk의 query에 주는 영향
- **Intra-chunk:** 현재 chunk 내부 token끼리 만드는 causal interaction

현재 chunk를 $[t]$로 표시하고, 위치 $i$에서 $j$까지의 누적 retention을 다음과 같이 정의한다.

```math
\gamma_{i\rightarrow j}^{[t]}
=
\prod_{r=i}^{j}\alpha_r^{[t]}
```

$\Gamma_{1\rightarrow C}^{[t]}$는 첫 위치부터 각 위치까지의 누적 retention을 행별로 쌓은 행렬이다. 논문의 UT transform이 만든 $U^{[t]}$와 $W^{[t]}$를 이용해 pseudo-value를

```math
\widetilde{V}^{[t]}
=U^{[t]}-W^{[t]}S^{[t]}
```

로 두면 chunk 출력은 개념적으로 다음처럼 분해된다.

```math
A^{[t]}=
\mathrm{Tril}
\left[
\left(Q^{[t]}\odot\Gamma_{1\rightarrow C}^{[t]}\right)
\left(K^{[t]}/\Gamma_{1\rightarrow C}^{[t]}\right)^\top
\right]
```

```math
O^{[t]}
=
\underbrace{
\left(\Gamma_{1\rightarrow C}^{[t]}\odot Q^{[t]}\right)S^{[t]}
}_{\text{previous chunks}}
+
\underbrace{
A^{[t]}\widetilde{V}^{[t]}
}_{\text{current chunk}}
```

$\mathrm{Tril}$은 upper triangle을 제거해 미래 token을 보지 못하게 한다. 이 형태의 장점은 chunk 내부의 많은 연산을 dense matrix multiplication으로 바꿀 수 있다는 것이다. 하지만 chunk와 chunk 사이에는 여전히 recurrent state 전달 순서가 존재한다.

### 누적 decay에서 발생하는 수치 문제

$\Gamma$는 $0\lt\alpha\lt1$인 값들의 곱이다. 긴 구간에서는 매우 작아지고, $K/\Gamma$의 reciprocal은 매우 커질 수 있다.

예를 들어 모든 retention이 0.1이고 16개 token을 곱하면

```math
\Gamma=0.1^{16}=10^{-16}
```

이므로 reciprocal은 $10^{16}$이다. 실제 $\alpha$는 가변적이지만 문제의 방향은 같다. Kimi Linear는 log space 계산과 16-token secondary tile을 이용했으나 diagonal tile은 position-pair 전용 계산을 필요로 했다. 이는 Tensor Core에 잘 맞는 dense GEMM 경로를 깨뜨리는 병목이다.

### Lower-bounded decay

Kimi Linear의 log-decay는 대략 다음 범위를 갖는다.

```math
g_t=-\exp(A)\,\mathrm{Softplus}(z_t)\in(-\infty,0)
```

Kimi K3는 이를 하한이 있는 scaled sigmoid로 바꾼다.

```math
g_t=g_{\min}\,\mathrm{Sigmoid}\!\left(\exp(A)z_t\right)
\in(g_{\min},0)
```

```math
\alpha_t=\exp(g_t)
\in\left(\exp(g_{\min}),1\right)
```

논문에서는 $g_{\min}=-5$를 사용한다. 따라서 한 step의 retention은

```math
\alpha_{t,j}\gt\exp(-5)\approx 6.7\times10^{-3}
```

이고, 16-token tile의 누적 log-decay는 $(-80,0)$ 범위에 있다. Reciprocal rescaling은 $\exp(80)$보다 작아 BF16의 동적 범위 안에 남는다. 이 제한 덕분에 diagonal tile도 position-pair 특수 경로가 아니라 dense Tensor Core matrix multiplication으로 계산할 수 있다.

중요한 점은 이 변경이 단순한 numerical trick이 아니라 **수학적 parameterization이 GPU kernel 형태를 바꾼 algorithm-system co-design**이라는 것이다.

다만 하한을 두는 데에는 trade-off가 있다.

- 장점: overflow 범위를 통제하고 Tensor Core 경로를 통일한다.
- 잠재적 비용: 모델이 한 step에서 특정 channel을 거의 완전히 지우는 극단적 forget gate를 표현할 수 없다.
- 논문에서 부족한 점: 하한 $-5$와 다른 값 사이의 품질·속도 ablation이 제시되지 않는다.

### Full-rank output gate

KDA는 recurrent output에 head-wise RMSNorm을 적용한 뒤 full-rank gate로 channel을 선택한다.

```math
y_t
=W_o
\left[
\mathrm{Sigmoid}(W_gx_t)
\odot
\mathrm{RMSNorm}(\widetilde{o}_t)
\right]
```

이전 Kimi Linear의 low-rank gate보다 parameter와 계산은 늘지만, 각 token이 모든 output channel을 독립적으로 조절할 수 있다. 논문의 설명대로라면 head마다 gradient scale이 다른 상황과 매우 큰 모델에서 표현력을 확보하려는 선택이다. 그러나 low-rank와 full-rank gate의 직접적인 ablation 수치는 보고서에 없다.

### KDA의 memory 및 계산 복잡도

Full attention과 KDA를 간단히 비교하면 다음과 같다.

| 항목 | Full attention | KDA |
|---|---|---|
| Training attention 연산 | 대략 $O(T^2d)$ | chunkwise 기준 대략 $O(TCd)$와 state update |
| Decode history memory | token 수에 비례하는 KV cache | head별 고정 크기 $d_k\times d_v$ state |
| 과거 token의 직접 재접근 | 가능 | state에 압축된 정보만 접근 |
| 병렬성 | 큰 GEMM에 유리 | recurrence 처리와 전용 kernel 필요 |
| Prefix reuse | KV block 복사 | state checkpoint와 MLA cache를 함께 관리 |

Head 하나의 chunk 길이를 $C$라고 두면, KDA의 대략적인 연산량은

```math
O(Td_kd_v)+O\!\left(TC(d_k+d_v)\right)
```

로 정리할 수 있다. 첫 항은 recurrent state 갱신, 두 번째 항은 chunk 내부 interaction에 해당한다. $C$를 고정하면 sequence length $T$에 선형이지만, 상수항은 $d_k$, $d_v$, $C$와 kernel 효율에 좌우된다. 보고서는 실제 K3의 $d_k$, $d_v$, chunk 크기를 공개하지 않으므로 hidden width 7,168이나 head 수 96만으로 state 크기와 FLOPs를 확정해서는 안 된다.

KDA state는 문맥 길이 $T$와 무관하지만 “작다”는 말과 동일하지 않다. State 크기는 KDA layer 수, head 수, $d_kd_v$, precision에 비례한다.

```math
M_{\mathrm{KDA}}
\propto
L_{\mathrm{KDA}}\,H\,d_kd_v
```

Kimi K3에는 KDA layer가 69개이므로 state traffic을 제대로 fuse하지 않으면 decode 병목이 될 수 있다. 논문이 state snapshot 대신 projected draft input을 저장하고 on-chip replay하는 이유가 여기에 있다.

## Gated MLA

### MLA의 역할

Multi-head Latent Attention은 token마다 head별 full K/V를 그대로 cache하는 대신 저차원 latent vector를 저장한다.

```math
c_t=W_cx_t
```

Attention 계산 시 $c_t$에서 content key와 value를 up-project한다. 이 방식은 global softmax attention을 유지하면서 KV-cache footprint를 줄인다.

여기서 “latent”는 KV-cache의 channel 차원을 압축한다는 뜻이지 token-pair 계산을 없앤다는 뜻이 아니다. MLA 24개 층은 여전히 전역 token pair를 비교하므로 attention 계산은 sequence length에 대해 quadratic이다. 보고서는 MLA latent dimension과 실제 KV-cache 절감 배수를 공개하지 않는다. 따라서 Kimi K3 전체를 완전한 $O(T)$ 모델이라고 부르면 부정확하다.

Kimi K3에서 MLA는 다음 역할을 담당한다.

- KDA state 압축 때문에 사라질 수 있는 token-level global interaction 보완
- 특정 과거 token을 content similarity로 다시 찾는 경로 제공
- Backbone 마지막 층에서 최종 global attention 보장

### NoPE MLA

Kimi K2와 K2.5의 MLA와 달리 Kimi K3의 MLA에는 RoPE를 포함한 explicit positional encoding이 없다. 위치 정보는 주로 KDA의 recurrence, decay, short convolution에서 들어온다.

이 설계의 장점은 context extension 시 RoPE base 변경, positional interpolation, YaRN 같은 별도 조정이 필요 없다는 것이다. 반면 MLA 단독으로는 query와 key에 절대적·상대적 위치 신호가 명시적으로 들어가지 않는다. 따라서 hybrid 전체가 제대로 학습되어 KDA 표현 속 위치 정보를 MLA가 활용해야 한다.

이는 모듈 독립성이 낮다는 뜻이기도 하다.

- Gated MLA만 떼어내면 위치에 둔감할 수 있다.
- KDA와 MLA의 layer ratio를 바꾸면 위치 정보 전달 특성도 달라질 수 있다.
- 3:1 비율이 최적인지 보여 주는 공개 ablation은 없다.

### MLA output gate와 FP32 output

MLA에도 KDA와 같은 full-rank channel gate가 사용된다.

```math
y_t=W_o
\left[
\mathrm{Sigmoid}(W_gx_t)\odot\widetilde{o}_t
\right]
```

또한 flash attention의 biased rounding error를 줄이기 위해 학습 중 attention output을 FP32로 유지한다. 이는 output tile의 on-chip footprint를 두 배로 만들기 때문에, 저자들은 output tile을 query tile 대신 KV staging buffer와 overlap하도록 kernel을 재설계했다.

이 사례 역시 “precision을 높이면 메모리가 증가한다”에서 끝나지 않는다.

    FP32 output 선택
        |
        v
    shared-memory footprint 증가
        |
        v
    기존 tile 배치 불가능
        |
        v
    KV staging과 수명 구간을 겹치도록 kernel 재설계

모델의 수치 정확도 결정이 kernel pipeline 구조까지 전파된다.

## Attention Residuals

### 기존 residual connection의 깊이 병목

표준 Transformer의 residual 흐름은 대략 다음과 같다.

```math
h_{l+1}=h_l+f_l(h_l)
```

여기서 $h_l$에는 이전 모든 층의 정보가 누적되어 있지만, 현재 층은 과거의 특정 층을 직접 선택할 수 없다. 모든 과거가 하나의 vector에 압축된다는 점에서 깊이 방향의 recurrence처럼 볼 수 있다.

Attention Residuals는 sequence 방향에서 attention이 과거 token을 선택하듯, depth 방향에서 현재 layer가 이전 layer 또는 block 표현을 선택하게 한다.

### Full Attention Residuals

Layer $l$은 학습 가능한 layer-specific pseudo-query $q_l=w_l$를 갖는다. Token embedding과 이전 layer output이 key와 value가 된다.

```math
k_i=v_i=
\begin{cases}
h_1, & i=0\\
f_i(h_i), & 1\le i\le l-1
\end{cases}
```

Score kernel은 다음과 같다.

```math
\phi(q,k)
=
\exp\left(q^\top\mathrm{RMSNorm}(k)\right)
```

```math
\alpha_{i\rightarrow l}
=
\frac{\phi(q_l,k_i)}
{\sum_{j=0}^{l-1}\phi(q_l,k_j)}
```

```math
h_l=\sum_{i=0}^{l-1}\alpha_{i\rightarrow l}v_i
```

RMSNorm은 magnitude가 큰 특정 layer가 score를 독점하는 것을 막는다. Pseudo-query가 token별 입력에서 만들어지는 것이 아니라 layer마다 학습되는 vector라는 점도 중요하다. 즉, 같은 layer의 모든 token은 동일한 learned query parameter를 사용한다. 다만 key $k_i$는 각 token의 representation에서 나오므로 score와 depth-attention weight 자체는 token마다 달라진다. 따라서 “완전히 정적인 layer mixing”도 아니고, “입력에서 매번 새 query를 만드는 full token-dependent attention”도 아니다.

Full form의 arithmetic은 대략 $O(L^2d)$이다. $L\lt100$이면 sequence attention의 $T^2$보다 작아 계산 자체는 감당할 수 있다. 문제는 모든 과거 layer output을 살아 있게 유지해야 하는 $O(Ld)$ activation memory와 pipeline-stage communication이다.

### Block Attention Residuals

Kimi K3는 layer를 block으로 묶는다. Block $n$ 안의 layer output 합을

```math
b_n=\sum_{j\in B_n}f_j(h_j)
```

로 만들고, 현재 block의 첫 layer는 이전 block 표현만 읽는다.

```math
V=
[b_0,b_1,\ldots,b_{n-1}]^\top
```

현재 block의 두 번째 이후 layer는 현재 block의 partial sum도 후보에 넣는다.

```math
V=
[b_0,b_1,\ldots,b_{n-1},b_n^{i-1}]^\top
```

여기서 $b_0=h_1$로 token embedding을 항상 source에 포함한다. 이렇게 하면 모든 layer output 대신 block 수 $N$개의 표현만 유지하므로 memory 및 cross-stage communication이 $O(Ld)$에서 $O(Nd)$로 감소한다.

Kimi K3는 12개 layer 단위로 나누고, embedding source까지 세면 총 9개의 block-level source를 갖는다. 논문은 약 8개 block이면 full Attention Residuals의 이득 대부분을 회복한다고 설명한다.

### 직관

표준 residual은 모든 과거 정보를 동일한 통로에 계속 더한다.

    embedding -> layer 1 -> layer 2 -> ... -> layer 93

Block AttnRes는 현재 layer가 과거 block 중 필요한 표현을 가중합한다.

    embedding -----------+
    block 1 output ------+
    block 2 output ------+--> depth attention --> current block
    ... -----------------+

이는 다음 상황에 유용할 수 있다.

- 얕은 층의 lexical 또는 local feature를 깊은 층에서 다시 사용
- 중간 추론 표현이 반복 residual addition으로 희석되는 현상 완화
- 93-layer 모델의 gradient 및 information flow 개선

### 해석상의 주의점

AttnRes를 “각 token이 모든 이전 layer를 자유롭게 attention한다”고 이해하면 과장이다.

- 실제 Kimi K3는 full layer-level이 아니라 block-level 근사를 사용한다.
- Pseudo-query는 layer-specific learned vector라 입력별 동적 query보다 제한적이다.
- Block 내부는 partial sum으로 합쳐지므로 개별 layer의 정체성이 일부 사라진다.
- 보고서에는 표준 residual, Full AttnRes, Block AttnRes의 Kimi K3 규모 end-to-end ablation 표가 없다.

그럼에도 depth를 단순 누적 경로가 아닌 선택적 retrieval 문제로 다시 정의했다는 점은 중요한 설계 관점이다.

## Stable LatentMoE

### 왜 일반 MoE가 아니라 LatentMoE인가

일반 MoE에서는 routed expert가 model hidden width $d$ 전체를 입력으로 받는다. 토큰당 활성 expert 수 $k$를 늘리면 다음 비용이 함께 증가한다.

- Expert로 보내는 activation communication
- Expert weight memory traffic
- Grouped GEMM scheduling 복잡도
- Expert별 token imbalance

Kimi K3는 shared expert와 routed expert의 폭을 분리한다.

- Shared experts: full width $d=7{,}168$에서 공통 변환 수행
- Routed experts: latent width $\ell=3{,}584=d/2$에서 전문 변환 수행

입력 $x\in\mathbb{R}^{d}$에 대해 routed path는 먼저 down-project한다.

```math
z=W_{\downarrow}x
\in\mathbb{R}^{\ell}
```

Top-k로 선택된 expert 집합을 $T_k(x)$라 하면 routed output은

```math
u=
\sum_{i\in T_k(x)}
p_iE_i^{\mathrm{routed}}(W_{\downarrow}x)
\in\mathbb{R}^{\ell}
```

이고 전체 layer output은

```math
y=
\sum_{j=1}^{N_s}E_j^{\mathrm{shared}}(x)
+
W_{\uparrow}\mathrm{RMSNorm}(u)
```

이다. Kimi K3에서는

```math
N_s=2,\qquad n=896,\qquad k=16
```

이다. 즉 routed expert 활성 비율은

```math
\frac{16}{896}=\frac{1}{56}\approx1.79\%
```

이다. 그러나 shared expert 두 개는 모든 token에 활성화된다.

### Tensor 흐름

    x: [B, T, 7168]
      |
      +-----------------------> shared expert 1: full width
      |                         shared expert 2: full width
      |
      v
    W_down
      |
      v
    z: [B, T, 3584]
      |
      v
    router -> Top-16 of 896
      |
      v
    selected latent experts
      |
      v
    weighted sum u: [B, T, 3584]
      |
      v
    RMSNorm -> W_up
      |
      v
    routed output: [B, T, 7168]
      |
      +---- add shared outputs
      |
      v
    y: [B, T, 7168]

Latent width를 절반으로 줄였기 때문에 16개 expert를 활성화하면서도 full-width expert 16개를 사용하는 것보다 activation dispatch와 expert 내부 연산을 줄일 수 있다. 다만 $W_{\downarrow}$와 $W_{\uparrow}$가 추가되고, routed path가 여러 matrix multiplication의 긴 사슬이 된다.

### Stable이라는 이름이 붙은 이유

2.8T 규모와 896-expert routing에서는 두 문제가 커진다.

1. Down projection, gated expert FFN, up projection이 연속되어 내부 activation이 폭발할 수 있다.
2. 거의 1,000개 expert를 fixed-step bias update로 균형화하면 반응이 느리거나 oscillation이 생길 수 있다.

Stable LatentMoE는 다음 세 요소를 추가한다.

- Up projection 전 RMSNorm
- Bounded activation인 SiTU-GLU
- Quantile Balancing

### Normalized LatentMoE

선택된 expert와 routing probability에 따라 $u$의 scale이 달라진다. 이를 곧바로 $W_{\uparrow}$에 넣으면 큰 activation이 full-width 출력으로 증폭될 수 있다.

```math
y_{\mathrm{routed}}
=
W_{\uparrow}\mathrm{RMSNorm}(u)
```

RMSNorm은 expert 조합에 따른 scale 변동을 줄인다. 특히 low-precision 학습에서는 outlier와 overflow 감소에 도움이 된다. 논문은 validation loss와 downstream benchmark가 일관되게 좋아졌다고 서술하지만, 개선 폭을 보여 주는 표는 제공하지 않는다.

## SiTU-GLU

### SwiGLU의 outlier 문제

SwiGLU의 단순 scalar 형태는 다음처럼 볼 수 있다.

```math
f_{\mathrm{SwiGLU}}(x)
=
\left[x\,\mathrm{Sigmoid}(x)\right]x
```

Gate 쪽의 $x\mathrm{Sigmoid}(x)$와 value 쪽의 $x$가 모두 양의 큰 값에서 제한 없이 증가한다. 두 branch에 동시에 outlier가 생기면 곱은 더 빠르게 커지고 FP8/BF16 학습의 overflow 위험이 증가한다.

### SiTU-GLU 정의

Kimi K3는 두 linear branch에 각각 smooth cap을 적용한다.

```math
\mathrm{softcap}(x,\beta)
=
\beta\tanh\left(\frac{x}{\beta}\right)
```

```math
\mathrm{SiTU\text{-}GLU}(x)
=
\left[
\beta_1\tanh\left(\frac{W_gx}{\beta_1}\right)
\odot
\mathrm{Sigmoid}(W_gx)
\right]
\odot
\left[
\beta_2\tanh\left(\frac{W_ux}{\beta_2}\right)
\right]
```

사용한 값은

```math
\beta_1=4,\qquad \beta_2=25
```

이다.

### 수학적 성질

원점 근처에서

```math
\beta\tanh\left(\frac{x}{\beta}\right)
=x+O\left(\frac{x^3}{\beta^2}\right)
```

이므로 작은 activation에서는 SwiGLU와 유사하다. $\beta_1,\beta_2\rightarrow\infty$이면 SwiGLU를 회복한다.

반면 모든 coordinate에서

```math
\left|\mathrm{SiTU\text{-}GLU}(x)_i\right|
\leq
\beta_1\beta_2
=100
```

이라는 상한이 있다. Hard clamp와 달리 포화 구간으로 부드럽게 접근하므로 경계에서 gradient가 갑자기 0으로 끊기지 않는다.

이 activation의 목적은 표현력을 무조건 키우는 것이 아니라 **SwiGLU의 원점 부근 거동을 유지하면서 대규모 저정밀 학습의 tail activation을 통제하는 것**이다.

### 비판적으로 볼 점

- $\beta_1=4,\beta_2=25$의 선택 근거와 민감도 표가 없다.
- Bounded output이 extreme-value 정보를 손실하는지 분석하지 않는다.
- 전체 안정성 향상에서 RMSNorm, SiTU-GLU, QB 각각의 독립 효과가 수치로 분리되지 않는다.
- FP8 activation에서 실제 overflow rate 또는 scale histogram을 제공하지 않는다.

## Quantile Balancing

### 기존 auxiliary-loss-free routing

Token $x_i$의 raw router score를

```math
s_i=\mathrm{Sigmoid}(W_rx_i)
```

라 한다. Expert-specific bias $b$는 선택에만 사용된다.

```math
T_i=\mathrm{TopK}(s_i+b)
```

선택된 expert의 실제 mixture weight는 raw score로 정규화한다.

```math
p_{i,j}
=
\frac{s_{i,j}}
{\sum_{r\in T_i}s_{i,r}},
\qquad j\in T_i
```

Bias가 mixture weight에 들어가지 않으므로 load-balancing 조정이 model output의 gradient 경로를 직접 왜곡하지 않는다. 기존 fixed-step 방식은 expert load가 평균보다 적으면 bias를 일정량 올리고 많으면 내린다.

```math
b_j^{(t+1)}
=
b_j^{(t)}
+\gamma\,\mathrm{sign}
\left(\bar{\ell}-\ell_j^{(t)}\right)
```

하지만 $\gamma$가 작으면 balance 회복이 느리고, 크면 load가 진동한다.

### Quantile로 목표 load를 직접 맞추기

Batch에 token이 $m$개, expert가 $n$개, token당 선택 수가 $k$개이면 expert당 목표 load는

```math
q=\frac{mk}{n}
```

이다.

각 token에서 biased score의 Top-(k+1)을 구한다. 상위 $k$개는 실제 route이고, $(k+1)$번째 score $\alpha_i^{(t)}$는 어떤 expert가 Top-k에 들어가기 위해 넘어야 하는 cutoff이다.

Expert $j$의 margin은

```math
r_{i,j}=s_{i,j}-\alpha_i^{(t)}
```

이다. 정확히 $q$개의 token margin이 threshold를 넘도록 bias를 고르면 해당 expert의 목표 load를 맞출 수 있다. 논문의 update는 다음 형태다.

```math
\widehat{b}_j^{(t+1)}
=
-\mathrm{Quantile}_{1-k/n}
\left(s_{:,j}-\alpha^{(t)}\right)
```

공통 offset은 Top-k 결과를 바꾸지 않으므로 평균을 제거한다.

```math
b^{(t+1)}
=
\widehat{b}^{(t+1)}
-\mathrm{mean}\!\left(\widehat{b}^{(t+1)}\right)\mathbf{1}
```

새 bias는 다음 step부터 적용한다. 현재 batch에서 얻은 bias로 같은 batch를 다시 routing하지 않으므로 training causality를 지킨다. Inference에서는 최종 bias를 고정한다.

### 작은 예시

Token 8개, expert 4개, Top-1이라면

```math
m=8,\quad n=4,\quad k=1,\quad q=2
```

이다. 기존 routing load가 $(4,3,1,0)$이어도 각 expert의 margin distribution에서 두 번째 token까지 선택되도록 threshold를 정하면 목표 $(2,2,2,2)$에 가까운 bias를 한 번에 계산할 수 있다.

### Global histogram estimator

실제 학습에서는 global batch의 margin이 수백만 개이고 여러 rank와 gradient accumulation step에 분산된다. 모든 margin을 gather해 exact quantile을 계산하는 것은 비싸다.

Kimi K3는 expert마다 margin histogram을 만든다.

1. 각 rank가 local margin을 $B$개 bin에 scatter-add한다.
2. Micro-batch 동안 count를 누적한다.
3. Step 끝에 integer count matrix 하나를 all-reduce한다.
4. 누적 count가 목표 rank $q$에 도달하는 bin에서 선형 보간한다.
5. Quantile의 exponential moving average로 batch noise를 줄일 수 있다.

Router score가 $(0,1)$이고 현재 bias 범위를 알기 때문에 binning interval도 유한하게 잡을 수 있다. 논문은 $B=1000$일 때 오차가 bin width 이내, 보통 수 $10^{-3}$ 이하이며 raw margin 교환 대비 통신 비용이 1% 미만이라고 설명한다.

### Quantile Balancing의 의미

QB는 load balance를 loss penalty로 유도하는 대신 dispatch decision boundary를 직접 조절한다. 이는 다음 이점이 있다.

- Expert 사용량 목표를 더 빠르게 추적
- Auxiliary loss가 language objective와 경쟁하지 않음
- 거의 1,000개 expert에서도 dying expert 위험 감소
- MoonEP의 완전 균형 실행 계획에 더 좋은 입력 제공

하지만 QB와 MoonEP는 같은 문제를 다른 층위에서 다룬다.

| 기술 | 조절 대상 | 목표 |
|---|---|---|
| Quantile Balancing | Router가 token을 어떤 expert에 보낼지 | Expert별 장기적 load 균형 |
| MoonEP | 이미 결정된 routing을 어느 rank에서 실행할지 | Rank별 즉시 compute 균형 |

QB가 완벽해도 micro-batch마다 rank load가 다를 수 있고, MoonEP가 rank를 균형화해도 특정 expert 자체가 충분히 학습되지 않을 수 있다. 두 기술은 대체 관계가 아니라 보완 관계다.

## Native Vision

### 여기서 native가 의미하는 것

Kimi K3의 native multimodal은 “이미지 patch를 text tokenizer가 직접 처리한다”는 뜻이 아니다. 별도의 vision encoder와 MLP projector는 여전히 존재한다.

Native라는 표현은 주로 다음 학습 방식을 뜻한다.

- Pre-trained LLM에 vision tower를 나중에 붙이지 않는다.
- Vision encoder를 SigLIP 같은 contrastive checkpoint로 초기화하지 않는다.
- Language와 vision token을 학습 시작부터 한 sequence에 interleave한다.
- 하나의 next-token prediction objective로 vision encoder, projector, LLM backbone을 공동 최적화한다.

따라서 **구조적으로는 encoder-plus-projector VLM이고, 학습 일정상 native**라고 해석하는 것이 정확하다.

### MoonViT-V2

MoonViT-V2 사양은 다음과 같다.

| 항목 | 값 |
|---|---:|
| Parameters | 약 401M |
| Layers | 27 |
| Attention heads | 12 |
| Patch size | 14 |
| Normalization | RMSNorm |
| Linear/attention bias | 제거 |
| 최대 보고 이미지 크기 | $3584\times3584$ |

이미지와 비디오는 parameter를 공유한다. Video에서는 intra-frame spatial attention과 inter-frame temporal attention을 factorize하고, temporal pooling으로 시간 방향 token을 줄인다.

Vision encoder 뒤에서 2 x 2 pixel shuffle downsampling을 적용해 visual token 수를 4분의 1로 줄인 후 MLP projector를 거쳐 LLM hidden space로 보낸다.

최대 정사각형 입력이 정확히 나누어진다고 가정하면 patch grid는

```math
3584/14=256,\qquad 256\times256=65{,}536
```

개 위치다. 2 x 2 pixel shuffle 뒤에는

```math
128\times128=16{,}384
```

개의 spatial token이 LLM 쪽으로 전달된다. 이 계산은 special token, padding, 비정사각형 입력의 처리를 제외한 기본 patch-grid 계산이며, 16,384개도 일반적인 VLM 입력과 비교하면 매우 큰 visual context다.

    image/video
        |
        v
    patchify + MoonViT-V2
        |
        v
    visual grid [H_p, W_p, d_v]
        |
        v
    2 x 2 pixel shuffle
        |
        v
    token count N_v -> N_v / 4
        |
        v
    MLP projector
        |
        v
    [N_v / 4, 7168]
        |
        v
    text token과 interleave

주의할 점은 pixel shuffle이 **LLM에 들어가는 visual token 수**를 줄인다는 것이다. MoonViT-V2는 그 전에 고해상도 patch를 이미 처리하므로 vision encoder 자체의 attention FLOPs가 4분의 1이 되는 것은 아니다.

### From-scratch vision encoder를 택한 이유

저자들은 SigLIP-initialized MoonViT-3D를 LLM과 공동 학습하면 vision-tower gradient norm이 높고 spike가 자주 나타났다고 보고한다. MoonViT-V2를 처음부터 next-token prediction으로 학습했을 때 gradient가 더 안정적이었다.

저자들의 해석은 다음과 같다.

- Contrastive pre-training은 global semantics에 유리하지만 OCR, 구조, 정밀 위치 같은 language-generation용 feature와 objective가 다를 수 있다.
- 큰 LLM에 이미 pre-trained vision tower를 연결하면 두 subsystem의 scale과 optimization state가 달라 불안정할 수 있다.
- 충분한 multimodal data와 compute가 있다면 vision representation을 처음부터 language objective에 맞출 수 있다.

보고서는 from-scratch MoonViT-V2가 SigLIP 초기화 baseline과 vision evaluation에서 동등하다고 말한다. 그러나 상세 benchmark 표나 여러 seed는 제시하지 않고 gradient plot도 단일 trajectory 수준이므로, “contrastive pre-training이 일반적으로 불필요하다”로 확대하면 안 된다. 이는 Kimi K3 규모와 데이터 조건에서의 결과다.

## Per-Head Muon

Kimi K3는 matrix parameter optimizer로 Muon을 사용한다. Muon은 momentum matrix에 Newton-Schulz orthogonalization을 적용해 update 방향을 정규화한다.

기존 방식이 full Q, K, V projection matrix 전체를 하나의 block으로 orthogonalize했다면, Kimi K3는 momentum matrix를 attention head 축으로 나눈다.

    full projection momentum
        |
        v
    [head 1 block | head 2 block | ... | head H block]
        |
        v
    head별 Newton-Schulz orthogonalization
        |
        v
    다시 결합하여 update

Full-matrix 방식에서는 gradient scale이 큰 head가 전체 update 방향을 지배하고 작은 head는 충분히 정규화되지 않을 수 있다. Per-head 방식은 head별 update scale을 균등하게 하고, tall한 작은 block에 Newton-Schulz iteration을 적용하므로 optimizer overhead도 약간 줄어든다고 설명한다.

하지만 보고서에는 다음 정보가 부족하다.

- Newton-Schulz iteration 수와 numerical precision
- Muon이 적용되는 정확한 parameter 목록
- AdamW-only baseline과의 최종 품질 및 안정성 차이
- Per-head와 full-matrix Muon의 정량 ablation

## Pre-Training

### 데이터 구성

Text corpus는 네 개의 큰 domain으로 나뉜다.

- Web text
- Code
- Mathematics
- Knowledge

각 domain은 rule-based heuristic, classifier quality score, deduplication을 함께 사용하고, 작은 모델의 ablation으로 domain sampling rate를 정한다. Knowledge와 mathematics corpus에는 Kimi K2에서 사용한 rephrasing pipeline을 적용한다.

Rephrasing은 단순 paraphrase가 아니다.

1. 다양한 style과 관점의 prompt로 source document를 다시 표현한다.
2. 긴 문서는 chunk-wise autoregressive generation으로 처리한다.
3. 원문과의 fidelity를 검증한다.

Vision corpus는 다음 범주를 포함한다.

- Captioned image
- Interleaved image-text document
- OCR
- Perception 및 localization
- Video
- Visual coding
- SVG, 3D asset, webpage, game, CAD schematic처럼 code와 rendering이 짝을 이루는 programmatic data

좌표 supervision은 absolute coordinate와 $[0,1]$ normalized coordinate를 함께 사용한다. 이는 해상도가 달라져도 localization 표현을 유지하기 위한 선택이다.

보고서는 data taxonomy와 filtering 방향은 설명하지만 다음 핵심 수치를 공개하지 않는다.

- 전체 token 수와 modality별 token 비중
- Raw 및 deduplicated document 수
- 언어별·domain별 구성 비율
- Synthetic/rephrased data 비율
- Vision image/video 수와 해상도 분포
- Train/test contamination audit
- Copyright 및 licensing 분포

따라서 가중치를 이용한 추론은 가능해도 동일한 pre-training을 재현하기는 어렵다.

### Scaling law

Kimi K3 team은 architecture뿐 아니라 batch size, learning rate, tokens-per-parameter, model shape를 별도로 탐색했다. Held-out out-of-distribution validation loss에 fitted scaling curve를 만들었고, 같은 loss에 도달하는 FLOPs 기준으로 Kimi K2보다 약 2.5배 효율적이라고 주장한다.

개념적으로 loss를 compute $C$의 함수로

```math
L(C)=L_\infty+aC^{-b}
```

처럼 fitting했다면, Kimi K3 curve가 같은 $L$에서 요구하는 $C$가 Kimi K2의 약 $1/2.5$이라는 의미다. 다만 보고서는 실제 fit coefficient, confidence interval, model scale별 raw point와 독립 holdout 수치를 제공하지 않는다.

또한 2.5배는 다음 변화가 합쳐진 결과다.

- KDA와 Gated MLA의 hybrid
- Attention Residuals
- Stable LatentMoE 및 더 많은 active experts
- SiTU-GLU와 QB
- Per-Head Muon
- 데이터 filtering 및 mixture 변화
- Batch, learning rate, tokens-per-parameter 재탐색

그러므로 이를 특정 모듈 하나의 기여로 돌릴 수 없다.

### Cosine decay와 WSD 비교

저자들은 Warmup-Stable-Decay(WSD)보다 cosine decay가 더 좋았다고 보고한다. 중요한 방법론적 지적은 schedule을 비교할 때 peak learning rate와 batch size를 동일하게 고정하면 공정하지 않을 수 있다는 것이다.

    schedule A -- 최적 LR / batch search
    schedule B -- 최적 LR / batch search
                    |
                    v
             각자의 optimum끼리 비교

같은 hyperparameter를 강제로 쓰면 우연히 그 값에 더 잘 맞는 schedule이 이긴다. Kimi K3 team은 schedule마다 scaling-law search를 독립적으로 수행했고, 각 optimum에서 cosine의 final loss가 낮았다고 한다.

최종 공개 recipe 중 명시된 값은 다음과 같다.

- Cosine learning-rate schedule
- 전체 schedule의 1% linear warmup
- Weight decay 0.1
- Matrix parameter에 Per-Head Muon
- Kimi K2의 weight clipping
- MoE load balance에 Quantile Balancing

그러나 peak learning rate, global batch token 수, gradient clipping 값, Muon 세부값, training token 수는 공개하지 않는다.

### Native multimodal next-token prediction

Text와 visual token은 처음부터 하나의 sequence에 interleave되고 같은 next-token prediction loss로 학습된다.

```math
\mathcal{L}_{\mathrm{NTP}}
=
-\sum_t
\log p_\theta(z_t\mid z_{\lt t})
```

여기서 $z_t$는 text token일 수도 있고, visual context를 조건으로 한 text output일 수도 있다. Vision encoder의 representation도 최종 language loss로부터 gradient를 받는다.

이 방식의 장점은 post-hoc alignment stage가 필요 없다는 것이다. 반대로 초기 학습부터 401M vision tower와 2.78T MoE backbone을 동시에 안정화해야 하므로 데이터 mixture와 optimizer에 매우 민감하다.

## 1M-token Long-Context Extension

### 네 단계 curriculum

Context length는 한 번에 1M으로 올리지 않는다.

| 구간 | Context length | 학습 단계 |
|---|---:|---|
| Stage 1 | 8K | 초기 pre-training |
| Stage 2 | 64K | 후속 pre-training |
| Stage 3 | 256K | cooldown |
| Stage 4 | 1M | cooldown |

긴 sequence는 attention뿐 아니라 activation, communication, batch 구성 비용이 매우 크다. 전체 학습의 작은 후반 구간에만 비싼 long-context training을 집중해 비용을 통제한다.

### NoPE와 context extrapolation

Kimi K3에는 explicit positional embedding이 없다. KDA의 recurrent order, short convolution, decay가 위치와 순서를 암묵적으로 표현한다. 따라서 1M으로 확장할 때 RoPE frequency base를 바꾸거나 positional interpolation을 적용하지 않는다.

하지만 “position parameter를 수정하지 않는다”와 “학습하지 않은 길이에 완벽히 일반화한다”는 다른 주장이다. 실제로 저자들도 256K와 1M cooldown data를 사용한다. 즉 구조적 extrapolation 가능성은 있지만, 장거리 dependency를 실제로 사용하는 능력은 long-context data와 curriculum으로 학습한다.

### Long-context data

자연에서 얻은 긴 문서와 비디오는 길다고 해서 유용하지 않다. 다음 오염이 많다.

- Near duplicate
- Binary blob
- Truncated file
- 잘못 생성된 log
- 중복 video clip
- 구조가 깨진 document

저자들은 exact/fuzzy deduplication, frame perceptual hash, heuristic 및 classifier quality filter, structural validation을 적용한다. 진짜 길고 coherent한 sample이 부족하므로 cooldown에서 upsample한다.

또한 단지 긴 document를 넣는 것만으로 모델이 먼 정보를 사용하지 않을 수 있다. 이를 방지하기 위해 multimodal document와 subtask를 조심스럽게 permutation 및 concatenation해, context 전체에 흩어진 정보를 읽어야만 풀 수 있는 synthetic long-context task를 만든다.

### 1M context 주장의 범위

보고서는 1M-context training과 serving infrastructure를 자세히 설명하지만, 다음 종류의 정규 평가가 부족하다.

- Needle 위치를 0%부터 100%까지 바꾼 retrieval curve
- 여러 needle과 distractor가 있을 때 precision/recall
- 1M 위치에서의 exact copying 또는 code dependency resolution
- 64K, 256K, 512K, 1M별 quality degradation
- Context length별 TTFT, prefill throughput, cache memory

BrowseComp에서 context management 없이 full 1M을 사용하면 90.4%, 300K에서 compaction을 적용한 설정은 91.2%라고 보고한다. 이는 1M window가 유효한 기능임을 보여 주지만, 더 긴 raw context가 항상 더 좋은 결과를 보장하지는 않는다는 사례이기도 하다.

## Post-Training 전체 구조

Post-training은 세 단계다.

    Supervised Fine-Tuning
        |
        v
    domain x reasoning-effort별 RL expert
        |
        v
    Multi-Teacher On-Policy Distillation
        |
        v
    하나의 통합 Kimi K3 policy

### Supervised Fine-Tuning

SFT는 RL의 cold-start policy를 만든다. 이전 Kimi 계열의 domain-specialized model로 복잡한 agent trajectory를 합성하고, multi-stage verification과 human-in-the-loop annotation을 적용한다.

모든 agent trajectory는 XTML 기반 chat template로 직렬화한다. SFT가 학습시키려는 것은 단순 instruction following보다 넓다.

- Adaptive reasoning
- Precise tool calling
- Long-horizon state 유지
- 오류 후 recovery
- Thinking, response, tools channel 분리
- Dynamic tool declaration

SFT 단계부터 routed expert weight에는 MXFP4, expert input activation에는 MXFP8을 사용하는 quantization-aware training을 시작한다.

### RL domain과 reasoning effort

RL은 세 broad domain으로 나뉜다.

| Domain | 포함 작업 |
|---|---|
| General | Reasoning, knowledge, vision, faithfulness, search, professional work |
| General agents | Long-horizon assistant, deep research, long-form writing |
| Coding agents | SWE, coding experience, GPU kernel, web development |

각 domain마다 reasoning effort를 low, high, max 세 수준으로 학습하므로

```math
3\ \mathrm{domains}
\times
3\ \mathrm{effort\ levels}
=9\ \mathrm{expert\ policies}
```

가 만들어진다. 여기서 expert는 MoE 내부 expert가 아니라 **post-training된 policy teacher**를 뜻한다. 두 종류의 expert를 혼동하면 안 된다.

### Partial rollout

한 iteration에 prompt $N$개, prompt당 completion $K$개를 생성하면 활성 trajectory는 $N\times K$개다. 모든 trajectory가 끝날 때까지 기다리면 긴 task 하나가 전체 iteration을 막는다.

Partial rollout은 완료 비율 $\lambda$에 도달하면 generation을 멈춘다.

```math
\mathrm{completed}
\geq
\lambda NK
```

완료된 response는 optimization으로 보내고, 미완료 trajectory는 queue에 넣어 다음 iteration에서 우선 resume한다.

    iteration t
      |- completed trajectories -> policy update
      |- paused trajectories ----+
                                  |
    iteration t+1 <---------------+

이 방식은 straggler latency를 줄이지만, 한 trajectory가 여러 policy version에 걸쳐 생성되므로 매우 stale하고 off-policy가 된다. 논문은 per-token regularization으로 update를 local neighborhood에 제한해 안정성을 유지한다고 설명한다. 그러나 regularizer의 정확한 식과 coefficient는 이 보고서에 제시되지 않는다.

### Reasoning-effort budget control

Problem $x$마다 cold-start model에서 초기 token budget $b_0(x)$를 추정한다. Trajectory $y$의 사용량 $T(y)$가

```math
T(y)\gt\tau b_0(x)
```

이면 task reward를 $-1$로 덮어쓴다.

- General reasoning: $T(y)$는 thinking token 수
- Agentic task: thinking과 tool-call argument를 포함한 누적 output token 수

먼저 큰 $\tau$로 max-effort model을 만들고, $\tau$를 줄여 high와 low expert를 얻는다. 문제 난이도에 따라 기준 budget을 다르게 잡으므로 단일 absolute token limit보다 합리적이다.

다만 cold-start가 비효율적으로 길거나 너무 짧으면 $b_0(x)$ 자체가 편향된다. $\tau$를 domain별 human guidance로 조정하므로 완전 자동화된 effort calibration도 아니다.

### Agentic Generative Reward Model

정답을 자동 검증할 수 없는 general task에는 tournament-style binary comparison을 수행하는 Agentic GRM을 사용한다. Judge는 반드시 다음 순서를 따른다.

1. 결과물 또는 응답을 읽는다.
2. 평가 rubric을 만든다.
3. 후보별로 rubric score를 계산한다.
4. Scorepad에 근거와 점수를 기록한다.

긴 답변이 유리해지는 reward hacking을 줄이기 위해 cold-start verbosity $\ell_0$와 multiplier $\sigma$를 사용한다.

```math
\mathrm{length}(y)\gt\sigma\ell_0
```

인 후보는 binary comparison에서 자동 패배한다. 이 제약은 verbosity를 통제하지만, 긴 설명이 실제로 필요한 task에서 quality를 손상할 수 있으므로 task별 calibration이 중요하다.

## Multi-Teacher On-Policy Distillation

9개의 domain-effort teacher를 하나의 student로 합친다. Domain $d$, effort $e$, 입력 $x$, prefix $y_{\lt t}$에서 teacher와 student의 token probability 비율을 reward로 사용한다.

```math
r_{\mathrm{OPD}}^d
\left(y_t\mid e,x,y_{\lt t}\right)
=
\mathrm{clip}
\left[
\mathrm{sg}
\left(
\log
\frac{
\pi_{\mathrm{teacher}}^{(d,e)}
\left(y_t\mid x,y_{\lt t}\right)
}{
\pi_\theta
\left(y_t\mid e,x,y_{\lt t}\right)
}
\right),
-R_{\max},
R_{\max}
\right]
```

$\mathrm{sg}$는 stop-gradient이고 clipping은 teacher/student ratio가 매우 클 때 advantage가 폭발하는 것을 막는다. Token마다 dense reward를 얻기 때문에 sparse task reward만 쓰는 것보다 teacher behavior를 세밀하게 전달할 수 있고, 기존 partial-rollout RL infrastructure에도 통합할 수 있다.

저자들은 top-k logit을 더 자세히 맞추는 distillation도 시험했지만 convergence와 final performance에서 명확한 이득을 보지 못했다고 보고한다.

이 단계가 필요한 이유는 domain expert를 그대로 ensemble하면 다음 문제가 있기 때문이다.

- Request마다 teacher를 선택해야 함
- 9개 체크포인트를 별도로 serving해야 함
- Domain이 섞인 task에서 routing 기준이 모호함
- 하나의 대화에서 effort를 바꾸기 어려움

MOPD 후에는 chat option으로 effort를 지정하는 하나의 policy가 된다.

## Deployment-Aware Post-Training

### MXFP4 expert weight와 MXFP8 activation

전체 parameter memory 대부분을 차지하는 MoE expert weight를 MXFP4로 quantize하고, expert activation은 MXFP8로 계산한다. 다음 구성은 더 높은 precision에 남긴다.

- Attention projection
- Latent MoE down/up projection
- Shared experts
- Router
- 기타 non-expert component

SFT부터 RL 전체에 QAT를 적용하며 rollout과 training이 같은 quantization을 사용한다. 이는 high-precision learner와 low-precision rollout 사이의 policy mismatch를 줄인다.

저장량을 거칠게 하한 추정하면, 모든 2.78T parameter가 4-bit라고 가정해도

```math
2.78\times10^{12}\times0.5\ \mathrm{byte}
\approx1.39\ \mathrm{TB}
```

이다. 실제로는 non-expert component와 scale metadata가 더 높은 precision이므로 실 checkpoint 및 resident memory는 이보다 크다. 따라서 MXFP4를 쓴다고 해도 Kimi K3가 consumer GPU나 mobile device에 들어가는 것은 아니다.

Active parameter 104.2B를 모두 4-bit라고 단순 가정한 token당 active weight byte 하한도 약 52.1GB이다. 실제 inference에서는 batching, expert weight residency와 cache reuse가 있으므로 매 token마다 정확히 52.1GB를 새로 전송한다고 볼 수는 없지만, 여전히 다중 accelerator serving이 전제된다.

### EAGLE-3 draft model

Kimi K3는 backbone block과 유사한 1개 MTP layer를 pre-training한다. 이를 EAGLE-3 style draft model로 fine-tune한다.

- Target Kimi K3는 frozen
- Draft layer와 feature-fusion projection만 update
- Training에서는 draft를 7 step unroll
- 첫 step 이후에는 newest target feature가 없으므로 draft 자신의 과거 output을 사용

Draft input은 첫 번째, 네 번째, 마지막 AttnRes block의 low/mid/high-level feature를 concat해 만든다.

```math
f_{\mathrm{draft}}
=
W_{\mathrm{E3}}
[h_{\mathrm{low}};h_{\mathrm{mid}};h_{\mathrm{high}}]
```

$W_{\mathrm{E3}}$는 처음에 $[0\ 0\ I]$로 초기화해 기존 MTP가 학습했던 high-level feature와 동일하게 시작하고, 이후 low/mid feature를 점진적으로 사용한다.

### Acceptance-rate 직접 최적화

Target distribution을 $p$, draft distribution을 $q$라 하면 lossless speculative sampling의 token acceptance probability는

```math
A(p,q)
=
\sum_{x\in\mathcal{V}}
\min\left(p(x),q(x)\right)
```

이다. 일반 KL divergence를 줄여도 작은 draft model에서 이 acceptance가 최대로 된다는 보장은 없다. Kimi K3는 이를 직접 최대화한다.

```math
\mathcal{L}_{\mathrm{LK}}
=
-\log
\sum_{x\in\mathcal{V}}
\min\left(p(x),q(x)\right)
```

Temperature 1에서 계산하며 auxiliary ground-truth cross-entropy를 사용하지 않는다. 이는 deploy metric을 surrogate가 아니라 loss에 직접 넣은 사례다.

## RL Task Synthesis와 Agentic Environment

Kimi K3의 post-training에서 중요한 변화는 정적인 question-answer dataset보다 **행동 결과를 검증할 수 있는 environment**를 대규모로 만든 것이다.

### Unified White-Box RL Environment

하나의 agent harness에만 RL하면 model이 특정 tool schema, system prompt, context compaction 또는 message format에 과적합할 수 있다. Kimi K3의 white-box environment는 harness를 구성요소로 분해한다.

- Tool interface
- System prompt
- Context management
- Skill
- Memory
- Subagent
- Interaction protocol

이 모듈을 조합해 Kimi Code, Claude Code, Codex, OpenClaw, Hermes와 유사한 harness 또는 새로운 harness를 동적으로 만든다.

여기서 목표는 특정 product UI를 흉내 내는 것이 아니라 다음 invariance를 학습하는 것이다.

    같은 목표
      + 다른 tool 이름
      + 다른 system prompt
      + 다른 context policy
      + 다른 subagent 구조
      -> 여전히 task를 완수

다만 평가에서도 model마다 다른 harness가 사용되는 경우가 있어, training의 다양성은 강점이지만 benchmark 간 model 본체와 harness 효과를 분리하기 어렵게 만든다.

### Knowledge-Graph-Guided Task Synthesis

Post-training material의 다양성을 확보하기 위해 coarse domain에서 atomic concept까지 이어지는 directed acyclic graph를 agent가 확장한다.

    seed domain
        |
        v
    agent web exploration
        |
        +-> equivalent node 검사와 중복 제거
        |
        v
    coarse -> fine concept edge 추가
        |
        v
    여러 수준의 node 또는 related node 묶음 sampling
        |
        v
    web material retrieval
        |
        v
    knowledge / coding / vision task synthesis

Fine-grained node는 드문 전문 지식을 찾게 하고, broad node sampling은 전체 coverage를 유지한다. Ancestor context를 query에 포함해 keyword만으로 의미가 모호해지는 것을 줄인다.

위험도 있다.

- Web retrieval bias가 graph에 누적될 수 있다.
- Agent가 atomic하다고 판단하는 기준이 불명확하다.
- Synthetic task 품질이 teacher model과 verifier에 종속된다.
- Benchmark와 같은 공개 material을 검색하면 contamination 가능성이 있다.

### Verifiable multimodal reasoning

Visual STEM 문제, chart, puzzle에서는 Python interpreter를 가진 isolated sandbox를 제공한다. Model은 한 번에 이미지를 보는 데 그치지 않고 다음 loop를 반복한다.

    observe image
        |
        v
    crop / zoom / transform code 작성
        |
        v
    sandbox 실행
        |
        v
    새로운 image와 수치 결과 관찰
        |
        v
    reasoning 수정 및 검증

이는 native vision과 tool-augmented vision을 결합한다. 평가에서도 Math-Vision, CharXiv, ZeroBench 등에 Python tool을 붙이면 성능이 크게 상승한다. 따라서 vision score를 볼 때 pure perception과 agentic visual computation을 분리해야 한다.

### GPU kernel optimization task

Task 범위는 단일 operator부터 fused mega-kernel까지 포함한다.

- CUDA
- Triton
- CuTe DSL
- Gluon
- ThunderKittens
- TileLang
- BF16, FP8, FP4
- 여러 GPU architecture

각 task에는 PyTorch reference가 있고 numerical threshold를 넘으면 reward가 0이다. Expert implementation과 같은 성능이면 0.5, hardware roofline에 가까워질수록 1에 접근한다.

저자들은 다음 reward hacking을 탐지한다고 설명한다.

- CUDA graph replay로 실제 일반 kernel을 구현하지 않음
- 입력 caching
- 허용되지 않은 precision 축소
- 특정 test shape만 암기

Correctness gate를 먼저 통과하고 그다음 performance reward를 주는 구조는 compiler/kernel RL에 적합하다. 다만 hidden test coverage가 좁으면 여전히 shape overfitting이 가능하다.

### Personal Assistant Task

Gmail, Notion, Slack, Canvas와 유사한 mock application을 만들어 외부 API rate limit 없이 장기간 interaction을 재현한다. 한 rollout은 여러 simulated day, 수십 개의 상호 의존 event, 최대 수천 tool call과 수백만 context token을 포함할 수 있다.

Mock app은 reproducibility와 안전성에는 좋지만 실제 product의 예외, permission, latency, UI 변화까지 동일하지는 않다. Sim-to-real agent transfer를 별도로 평가해야 한다.

### Autonomous Execution Task

AET는 정답 trajectory를 제공하지 않는다. 환경은 다음만 제공한다.

- Initial state
- Constrained goal
- Tool-based action space
- Execution budget
- Independent verifier

Agent가 decomposition, tool selection, planning, recovery, termination을 직접 결정한다. Reward는 “완료했다”는 model의 말이 아니라 최종 environment state에 대한 verifier 결과로 계산한다.

Reward hacking을 줄이기 위해

- Agent와 verifier를 격리
- Diagnostic public verifier와 held-out hidden verifier를 함께 사용
- Submission 횟수 제한
- 실패 및 잘못된 접근에 penalty

를 적용한다.

### Web development task

한 줄 설명부터 장문의 specification까지 입력으로 사용하고 website, game, 3D/WebGL, visualization, SVG, full-stack app을 생성한다. Build failure, runtime error, reference를 속이는 fake 구현에는 reward 0을 준다.

평가는 deterministic functional check, structure/pixel similarity, source inspection, artifact interaction을 조합한다. Pixel similarity만 최적화하면 기능이 깨지고, 기능 test만 쓰면 시각 품질이 낮아질 수 있기 때문에 여러 signal을 함께 사용한다.

## Infrastructure 개요

Kimi K3 시스템은 세 가지 어려움을 동시에 처리한다.

1. Recurrent KDA와 global MLA가 섞인 hybrid attention
2. 3T-class sparse multimodal pre-training
3. 1M-token long-horizon RL과 online serving

보고서의 시스템 설계는 lifecycle별로 연결된다.

| 단계 | 핵심 문제 | 주요 해결책 |
|---|---|---|
| Training/prefill | KDA recurrence와 GPU 병렬성 충돌 | FlashKDA, intra-device CP, KCP |
| 3T pre-training | Expert imbalance, activation/optimizer memory | MoonEP, unified activation manager, ZeRO, offload |
| Multimodal training | 큰 image/video의 device imbalance | Dynamic CP, ViT를 pipeline bubble에 배치 |
| 1M RL | KV cache와 training memory 경쟁 | External cache pool, auto-throttling, partial rollout |
| Sandbox | 장기 state와 안전한 실행 | AgentENV microVM, pause/resume/fork/snapshot |
| Online serving | KDA state와 MLA KV의 cache lifetime 차이 | Unified paged cache, fine hash block, state checkpoint |
| Fleet | 1M request burst와 cache locality | Affinity routing, budget-based admission control |

## KDA 전용 kernel

### Training과 prefill용 FlashKDA

Chunkwise KDA는 chunk 내부는 병렬이지만 chunk 사이 state propagation은 직렬이다. 단순 구현은 다음처럼 GPU를 번갈아 놀게 한다.

    intra-chunk parallel compute
        -> serial state propagation
        -> intra-chunk parallel compute
        -> serial state propagation

FlashKDA는 token-parallel stage와 head-parallel recurrence를 별도로 schedule하고 overlap한다. CUTLASS 기반으로 구현되며 training과 inference prefill에 모두 사용되고 flash-linear-attention backend로 자동 dispatch된다.

논문은 Triton reference보다 크게 빠르다고 말하지만 shape별 latency, throughput, speedup table은 본문에 없다. 재현하려면 공개 implementation에서 정확한 GPU, dtype, head dimension, chunk size를 고정해 측정해야 한다.

### Intra-device context parallelism

Tensor parallelism은 head를 나누지만 sequence recurrence 길이는 줄이지 않는다. Ultra-long prefill에서 rank당 head 수가 적으면 각 head의 긴 recurrence 때문에 SM utilization이 낮아질 수 있다.

KDA segment의 state transition은 입력 state에 대한 affine transform으로 표현할 수 있다.

```math
S_{\mathrm{out}}
=
M_{\mathrm{segment}}S_{\mathrm{in}}
+\widetilde{S}_{\mathrm{segment}}
```

여기서 $M_{\mathrm{segment}}$와 zero-state에서 생성한 $\widetilde{S}$는 incoming state가 오기 전에 계산할 수 있다. 한 GPU 안에서 sequence segment를 여러 SM에 나누어 이 두 값을 병렬 계산하고, 이후 exact initial state를 합성한다.

## KDA Context Parallelism

### 왜 일반 linear-attention CP를 그대로 쓸 수 없는가

단순 additive linear attention이라면 각 rank가 $S=0$에서 local state를 계산하고 이전 rank들의 state를 더하면 된다.

```math
S_{\mathrm{total}}=\sum_i\widetilde{S}_i
```

그러나 KDA update에는 token-dependent transition matrix가 있다.

```math
S_t=M_tS_{t-1}+\beta_tk_tv_t^\top
```

```math
M_t=
\left(I-\beta_tk_tk_t^\top\right)
\mathrm{Diag}(\alpha_t)
```

따라서 앞 rank의 state는 뒤 rank local token들의 $M_t$를 모두 거쳐야 한다. Zero-state output만 더하면 틀린다.

### Segment를 두 조각으로 표현하기

Rank $i$는 local segment에 대해 다음 두 값을 계산한다.

- $M_i$: segment 전체의 cumulative transition
- $\widetilde{S}_i$: incoming state가 0일 때 segment가 만드는 state

두 segment의 합성은

```math
(M_b,\widetilde{S}_b)
\circ
(M_a,\widetilde{S}_a)
=
\left(
M_bM_a,\;
\widetilde{S}_b+M_b\widetilde{S}_a
\right)
```

이다. 이 연산은 associative하므로 prefix scan이 가능하다.

각 rank는 local fragment를 계산하고 한 번의 all-gather로 교환한 뒤, 앞선 document fragment를 순서대로 합성해 자신의 incoming state를 복구한다. 통신 payload는 sequence length에 비례하는 full KV block이 아니라 fixed-size transition/state fragment다.

| Context parallelism | 통신 대상 |
|---|---|
| Softmax attention CP | 길이에 따라 증가하는 K/V block |
| 단순 linear attention CP | Additive local state |
| KDA KCP | Cumulative transition $M$ + zero-state output $\widetilde{S}$ |

KCP는 compute scaling을 선형화하지만 $M\in\mathbb{R}^{d_k\times d_k}$, $S\in\mathbb{R}^{d_k\times d_v}$의 all-gather와 local composition 비용은 남는다. 실제 이득은 head dimension, CP degree, interconnect에 좌우된다.

## 3T-class Pre-Training Parallelism

Kimi K3는 여러 parallelism을 결합한다.

- Pipeline Parallelism(PP)과 virtual stage(VP)
- Expert Parallelism(EP)
- ZeRO-1 Data Parallelism
- Pipeline ZeRO-2 gradient sharding
- Context Parallelism

MoE all-to-all dispatch/combine는 expert computation과 overlap하고, shared expert도 별도 stream에서 다른 kernel과 겹친다.

### MoonEP: 완전 균형 expert parallelism

일반 EP에서는 rank마다 받은 token 수가 다르다.

    rank 0: 10,000 token-expert pairs
    rank 1:  4,000
    rank 2:  7,000
    rank 3:  2,000

전체 step은 가장 느린 rank를 기다리고, variable shape 때문에 allocator fragmentation과 host synchronization도 생긴다.

MoonEP는 router output을 본 뒤 일부 expert weight를 다른 rank에 동적으로 중복 배치해 모든 rank가 정확히 같은 $S\times K$ token-expert pair를 처리하도록 한다.

### 중복 expert 상한

Expert 수를 $E$, EP rank 수를 $R$이라 하면 논문은 rank당 최대

```math
\frac{E}{R}
```

개의 redundant expert slot이면 어떤 routing output에서도 완전 균형 plan이 존재함을 증명한다. 직관은 overloaded rank의 token을 underloaded rank가 정확히 차도록 이동시키고, 각 underloaded rank가 한 번만 채워지도록 구성하는 것이다.

또한 특정 worst-case routing에서는

```math
\left\lceil
\frac{E(R-1)}{R^2}
\right\rceil
\approx
\frac{E}{R}
```

개가 필요해 상한이 본질적으로 tight하다고 보인다.

이론적 상한을 예약하면 feasible plan이 없어서 학습이 중단되는 일을 피할 수 있다. Exact ILP는 대표 case의 reference로만 쓰고 실제 step에서는 near-optimal GPU planner를 사용한다.

### 완전 균형이 주는 시스템 효과

1. **Static shape:** 모든 rank가 $S\times K$를 처리하므로 layer마다 device count를 host로 가져올 필요가 없다.
2. **Zero-copy dispatch:** Planner가 token의 최종 expert-grouped destination을 미리 계산해 communication buffer view를 GEMM에 직접 넘긴다.
3. **고정 buffer:** Worst case에서 conventional path가 $S\times K\times R$ buffer를 요구할 수 있는 반면 MoonEP는 rank당 고정 $S\times K$ buffer를 사용한다.
4. **Predictable makespan:** Rank-level imbalance를 제거해 pipeline stall을 줄인다.

Rank aggregate가 균형이어도 한 rank 안에서 expert별 token 수는 다르다. MoonEP는 analytical cost model과 offline autotuning coefficient를 사용해 grouped GEMM worker schedule을 current distribution에 맞춘다.

### MoonEP와 QB의 연결

QB가 좋은 statistical balance를 만들고 MoonEP가 남은 per-step physical imbalance를 해결한다.

    router raw scores
        |
        v
    Quantile Balancing
        |
        v
    expert assignment
        |
        v
    MoonEP online planner
        |
        v
    redundant expert migration
        |
        v
    rank별 exact S x K execution

이 조합은 품질 수준의 routing과 시스템 수준의 load placement를 분리한다는 점에서 설계가 깔끔하다.

## Memory-Efficient Training

### Unified activation manager

Backward에 필요한 tensor마다 storage backend를 추상화한다.

- GPU에 유지
- Recomputation
- Quantization
- CPU offload
- 다른 PP rank memory로 remote offload

정책은 tensor annotation으로 지정하고 model code와 분리한다. Function granularity recomputation을 지원해 cross-layer 영역도 다시 계산할 수 있다.

모든 GPU allocation을 main compute stream의 단일 memory pool에서 관리해 multi-stream fragmentation과 host overhead를 줄인다. Prefetch는 layer granularity로 수행하고 compute와 overlap한다.

Kimi K3의 주요 activation 정책은 다음과 같다.

- 대부분 activation: block-wise FP8 quantization + offload 또는 remote offload
- Element-wise operator: recomputation
- AttnRes: block representation 공유 + checkpointing

### MoE activation 절약

Grouped GEMM backward에서 forward output 전체를 저장하는 대신 gradient 식을 intermediate activation과 upstream gradient만으로 다시 표현한다. Dispatch input만 보존하고 backward에서 grouped GEMM input을 다시 dispatch해 복구한다.

이 recomputation은 추가 communication을 만들지만 grouped-GEMM backward와 overlap해 activation 저장을 줄인다.

### Attention Residual memory

Block representation은 boundary layer에서 한 번 만들고 뒤 layer가 공유한다. AttnRes 전체를 checkpointing으로 감싸 각 layer가 backward를 위해 저장하는 activation 양을 standard residual 수준으로 맞춘다.

Pipeline parallelism에서는 새로 생성된 block만 다음 stage로 incremental transfer하고 micro-batch가 끝나면 즉시 해제한다.

### PP rank 사이 activation 재배치

Interleaved 1F1B schedule에서는 pipeline warmup 때문에 초기 PP rank에 더 많은 activation이 살아 있고 뒤 rank에는 적다. 가장 붐비는 rank 기준으로 전체 batch를 줄이면 GPU memory가 낭비된다.

Kimi K3는 여유가 있는 다른 PP rank의 memory로 activation을 remote offload해 rank별 memory를 균형화한다. 이는 CPU offload보다 빠른 device-to-device 경로를 활용하면서 cluster 내부 여유 memory를 pooling하는 방식이다.

### Gradient와 optimizer state

Pipeline ZeRO-2는 gradient를 DP rank에 shard하고, shard는 CPU memory에 저장한다. GPU에는 double gradient buffer를 두어 reduce와 CPU accumulation을 overlap한다.

Muon orthogonalization은 full parameter matrix가 필요하지만 distributed optimizer는 parameter를 DP rank에 shard한다. 모든 rank가 전체 parameter buffer를 all-gather하는 대신, 각 rank가 자신이 담당한 parameter에 필요한 shard만 owner rank로부터 P2P로 가져온다. Model-chunk 단위로 communication과 computation을 pipeline해 full-buffer memory를 제거한다.

### 평가

이 절의 강점은 “FP8과 offload를 쓴다”는 단편적 설명보다 activation, gradient, optimizer, pipeline imbalance를 하나의 schedule로 묶었다는 것이다.

그러나 다음 수치가 빠져 있다.

- GPU당 peak memory
- Recomputation으로 늘어난 FLOPs
- CPU/NVMe/remote offload bandwidth
- Offload 없는 baseline 대비 step time
- Parallelism degree별 model FLOPs utilization
- Cluster failure 및 straggler 통계

따라서 아이디어는 상세하지만 Kimi K3 학습의 실제 효율을 외부에서 검증하기는 어렵다.

## Multimodal Encoder Optimization

### Dynamic context parallelism

긴 video 또는 큰 image 하나가 특정 rank에 몰리면 vision encoder 시간이 text backbone보다 길어져 pipeline critical path가 된다.

Kimi K3는 큰 image를 patch 축으로 여러 device에 나누고 attention에서 CP rank 사이 K/V를 gather한다. CP group을 더 작은 sub-CP group으로 나눠 여러 큰 image를 병렬 배치한다.

목표는 두 가지다.

- 한 visual sample의 latency 감소
- Sample별 해상도 및 frame 수 차이로 생기는 device imbalance 감소

### Vision compute를 pipeline bubble에 숨기기

ViT와 text training을 완전히 같은 stage schedule로 실행하면 vision forward/backward가 critical path에 노출된다. 저자들은 ViT computation을 분해한다.

1. 첫 PP micro-batch들의 ViT forward는 upfront 실행
2. 나머지 ViT forward는 text pipeline bubble에 삽입
3. ViT backward도 반대쪽 bubble에 배치

이를 통해 vision encoder compute 대부분을 pipeline의 빈 구간에 숨긴다고 주장한다.

이 결과는 wall-clock 효율에는 중요하지만, “vision compute가 사라진다”는 뜻은 아니다. Cluster가 이미 다른 작업으로 bubble을 채우고 있거나 visual sample 분포가 달라지면 overlap 이득도 달라진다.

## 1M-context Agentic RL Infrastructure

### Co-located training과 rollout

1M-context Kimi K3 RL experiment를 수백 GPU 안에 유지하기 위해 training과 rollout resource를 co-locate한다. 이는 resource utilization을 높이지만 rollout KV cache와 training weight/optimizer state가 같은 memory를 경쟁한다.

### External KV cache pool

Partial rollout에서 미완료 trajectory는 다음 iteration에 재개된다. 1M prefix를 매번 prefill하면 비용이 매우 크다.

Kimi K3는 write-back cache를 사용한다.

- Active decoding block: GPU에 유지
- GPU에서 eviction되는 reusable idle prefix: CPU DRAM external pool에 write-back
- 재사용 직전: GPU로 prefetch
- KDA state와 해당 MLA KV block: 같은 lifecycle로 이동

Write-through처럼 모든 active block을 CPU에도 복제하지 않으므로 DRAM과 PCIe traffic을 줄인다.

Training iteration이 끝나면 model weight와 optimizer state를 NVMe로 offload해 CPU DRAM을 prefix pool에 내준다. Rollout이 끝나면 pool을 해제해 다음 training과 충돌하지 않게 한다.

이 설계는 GPU, CPU DRAM, NVMe를 단계별로 재할당하는 hierarchical memory system이다.

### Rollout auto-throttling

Long-horizon trajectory는 진행될수록 context가 길어진다. 평균 길이로 고정 concurrency를 잡으면 초반에는 GPU가 놀고 후반에는 KV cache가 넘칠 수 있다.

Scheduler는 다음 runtime signal로 새 request 투입량을 조절한다.

- Active request count
- Queued request count
- KV-cache utilization

초반에는 concurrency를 높이고 cache pressure가 커질수록 낮춘다. 이는 static batch size보다 안정적이지만 scheduler control law와 SLO 결과는 공개하지 않는다.

### Non-policy model용 gradient buffer reuse

RL loss에는 reference model처럼 forward-only model이 필요할 수 있다. 전체 weight를 GPU에 상주시킬 수 없으므로 CPU에 두었다가 필요한 chunk만 가져온다.

Kimi K3는 policy model의 FP32 gradient buffer storage를 reference weight materialization slot으로 재사용한다. 실제 gradient 계산 전에는 이 buffer가 비어 있으므로 추가 GPU allocation 없이 사용할 수 있다.

ZeRO-2로 GPU당 두 VPP chunk의 gradient buffer만 남는 설정에서는 한 slot으로 current forward를 하고 다른 slot으로 next chunk를 prefetch한다.

## AgentENV Sandbox

Agentic RL에서는 model이 code, shell, container, disk, 때로는 VM까지 조작한다. 단순 container isolation에서 kernel panic과 deadlock이 관찰되었기 때문에 Firecracker microVM 기반 AgentENV를 사용한다.

### 세 가지 설계 목표

1. **High-fidelity isolation:** Agent가 disk mount, container, VM 같은 현실적 작업을 시도하면서도 host를 격리
2. **Flexible lifecycle:** Pause, resume, fork, snapshot으로 수일짜리 trajectory state 유지
3. **High density:** 수만 sandbox를 빠르게 만들고 memory를 공유

### 보고된 수치

| 항목 | 보고 값 |
|---|---:|
| Incremental checkpoint latency | 최저 133 ms |
| Resume latency | 최저 49 ms |
| Agent가 inference를 기다리는 비율 | sandbox lifetime의 최대 98% |
| Memory overcommit | 최대 6.5배 |
| 생성된 sandbox 수 | 51,219,741 |
| 사용한 image 수 | 1,505,678 |

Pause된 sandbox가 CPU와 memory를 사용하지 않게 하고, fork는 원본을 계속 실행한 채 동일 state에서 judge용 복사본을 만든다. Snapshot은 오류 복구 지점을 제공한다.

OverlayBD, custom ublk driver, layer sharing, P2P transport로 대규모 launch를 sub-second 수준으로 줄이고, copy-on-write memory와 page cache 최적화로 density를 높인다.

이 수치들은 인상적이지만 “최저 latency”와 “실제 대규모 p50/p95”는 다르다. Sandbox 5천만 개도 독립 task 수가 아니라 생성 event 수다. 평균 lifetime, failure rate, concurrent peak가 없으므로 capacity를 완전히 해석할 수 없다.

## Inference와 Online Serving

### Hybrid cache의 두 종류

Kimi K3는 한 request에서 두 cache를 함께 유지한다.

| Cache | 특성 |
|---|---|
| MLA KV cache | Token 수에 따라 증가, token/page 단위 |
| KDA recurrent state | Sequence당 고정 크기, sparse checkpoint |

Prefix는 두 cache가 같은 token boundary에서 복구되어야만 재사용할 수 있다.

### Unified paged layout

KDA state를 MLA KV와 같은 paged block pool에 넣고 page byte 크기를 맞춘다. Allocation, reference count, eviction, cross-node transfer를 하나의 manager로 처리한다.

KDA state 안에서는 head별 byte stream을 contiguous하게 저장한다. Prefill과 decode node의 tensor-parallel degree가 다르면 transfer path에서 layout을 바꾸고 GPU-side reshuffle은 피한다.

### Physical block과 hash block 분리

KDA state checkpoint가 크기 때문에 자주 저장할 수 없다. 하나의 block size를 cache storage와 hash에 함께 쓰면 physical block이 1,024에서 6,144 token처럼 커지고, 짧은 prefix는 전혀 hit하지 못한다.

Kimi K3는 다음 두 granularity를 분리한다.

- Physical allocation block: 예시에서 6,144 token
- Prefix hash block: 예시에서 512 token

MLA page 안에는 여러 hash endpoint가 있고 KDA state는 일부 endpoint, 주로 conversation-turn boundary에 sparse checkpoint된다.

논문 예시에서는 request의 첫 2,800 token이 일치할 때

```math
B=2560=5\times512
```

에서 MLA hash block 5개와 KDA checkpoint를 복구하고 $[0,B)$를 다시 prefill하지 않는다.

### Cache consistency

부분적으로 찬 block을 여러 request가 공유하면 race가 발생한다. 보고서는 다음 invariants를 둔다.

- 모든 cache group이 같은 free list를 사용
- Hit한 block은 새 private allocation 전에 모든 group에서 pin
- 현재 scheduling step에서 copy가 끝나지 않은 block은 match 대상에서 제외
- 한 KDA group checkpoint가 evict되면 sibling group도 atomic invalidation
- Shared checkpoint는 read-only이고 request private state로 copy 후 갱신

가장 긴 MLA hash match가 있어도 모든 KDA group에 같은 boundary checkpoint가 없으면 hit를 더 짧은 지점으로 내린다.

### Speculative decoding에서 KDA rollback

Draft token 여러 개로 KDA state를 미리 update한 뒤 일부가 reject되면 state가 마지막 accepted token보다 앞서 가 있다. Draft 위치마다 full state snapshot을 저장하면 state traffic이 너무 크다.

Kimi K3는 full state 대신 draft token의 projected input만 cache한다. Accepted prefix가 결정되면 작은 projection을 이용해 state를 on-chip에서 replay한다.

하나의 fused recurrent loop가 다음을 처리한다.

- Short convolution
- Input normalization
- Gate
- KDA recurrence
- Output normalization
- Accepted token replay
- Bonus token
- 다음 draft window

이는 recurrent model에서 speculative decoding을 적용할 때 상태 rollback을 “복사”가 아니라 “재계산”으로 푸는 방법이다.

### Block AttnRes kernel

Block AttnRes는

1. 과거 block representation을 한 번 읽는 inter-block pass
2. 현재 block partial sum과 online-softmax로 합치는 intra-block pass

로 나뉜다.

Prefill에서는 tensor-parallel all-reduce를 reduce-scatter와 all-gather로 분해하고 그 사이 sequence-sharded activation에서 intra-block kernel을 실행한다. 각 token의 block representation이 한 rank에만 materialize되어 중복 memory를 줄인다.

Decode에서는 inter-block kernel을 side stream에 실행해 main computation과 overlap하고, intra-block merge와 뒤 RMSNorm을 앞선 TP all-reduce에 fuse한다.

### Stable LatentMoE kernel

Latent down projection과 router를 하나의 GEMM으로 fuse하고, latent weight를 rank에 shard한다. Output all-gather는 multimem store를 이용해 GEMM epilogue에 fuse하며 shared-expert computation과 overlap한다.

Small-batch decode에서 routed expert GEMM은 compute-bound가 아니라 weight streaming memory-bound가 된다. 일반 tile-centric GEMM 대신 token-centric WarpDecode 계열을 사용한다.

- 한 warp가 한 output neuron 담당
- Warp를 더 작은 lane team으로 나누어 여러 expert 처리
- Partial result를 warp-wide reduction
- Weight layout을 offline permutation해 runtime dequantization 감소

### Fleet-level scheduling

1M context request와 2K 이하 request의 cost는 약 세 자릿수 차이가 난다. 평균 request 기준 capacity planning은 long-context burst에서 무너진다.

Cache-aware affinity scheduling은 session을 prefix cache가 있는 primary cluster로 보낸다. Consistent hashing으로 secondary cluster도 미리 정하지만 cache는 복제하지 않는다. Primary failure 시 secondary가 re-prefill하고, secondary assignment가 fleet에 고르게 분산되어 failover burst를 나눈다.

논문이 든 전형적 coding request 예시는 이미 약 400K-token prefix를 갖고 새로 추가할 부분은 4K뿐인 경우다. Cache hit라면 4K increment만 처리하지만 miss이면 400K prefix를 다시 prefill해야 하므로 affinity가 단순한 소폭 최적화가 아니라 비용 차수 자체를 바꾼다. 다만 보고서는 이 예시에 대한 실제 TTFT 분포나 hit/miss별 wall-clock 표는 제공하지 않는다.

Budget-based admission control은 short, medium, ultra-long request class에 별도 resource budget을 준다. 1M request가 자신의 budget을 모두 써도 short request용 capacity는 남게 해 전체 TTFT SLO 붕괴를 막는다.

이 방식은 평균 latency 최적화보다 tail isolation과 predictability를 우선한다.

## Evaluation Protocol

### 평가 축

공개 benchmark는 네 capability axis로 구성된다.

| 축 | 대표 benchmark |
|---|---|
| Reasoning & Knowledge | GPQA Diamond, CritPt, AA-LCR, HLE-Full |
| Coding | DeepSWE, ProgramBench, Terminal-Bench 2.1, FrontierSWE, SWE-Marathon, PostTrainBench, MLS-Bench-Lite, SciCode |
| Agentic | BrowseComp, DeepSearchQA, MCPMark, OSWorld, OfficeQA, SpreadsheetBench, 금융·법률 task 등 |
| Vision | WorldVQA, OmniDocBench, PerceptionBench, Video-MME, MMVU, MMMU-Pro, CharXiv, Math-Vision, ZeroBench |

비교 대상은 Claude Fable 5, GPT-5.6 Sol, Claude Opus 4.8, GPT-5.5, open-weight GLM-5.2다.

### 공통 inference 설정

Kimi K3는

- Reasoning effort: max
- Temperature: 1.0
- Single-step reasoning 및 vision without tools: top-p 0.95
- Coding 및 agentic: top-p 1.0

을 사용한다. 비교 모델도 대체로 max지만 GPT-5.5는 xhigh다.

여기서 각 provider의 max, xhigh는 같은 token budget 또는 같은 test-time compute를 의미하지 않는다. 따라서 Table 2는 엄밀한 compute-matched model comparison보다 **각 model의 강한 reasoning mode와 지원 agent stack을 함께 비교한 system-level evaluation**에 가깝다.

### Tool score 표기

다음 benchmark는 한 cell에 두 점수를 쓴다.

- HLE-Full: tool 없음 / general tool 사용
- MMMU-Pro, CharXiv, Math-Vision, ZeroBench: Python 없음 / Python 사용

Vision은 보통 3회 평균이고 ZeroBench-main은 공식 protocol에 따라 5회 실행한다.

### Harness 차이

Coding model은 Kimi Code, Claude Code, Codex 중 하나에서 평가된다.

- Terminal-Bench 2.1: model별 여러 harness 중 최고 점수
- Agents' Last Exam: Kimi K3는 Kimi Code, GPT 계열은 Codex, Claude와 GLM은 Claude Code
- SWE-Marathon: 공식 final v1.1 이전 2026-07-09 branch를 H20에 맞게 재보정
- PostTrainBench: 공식 H100 대신 H20에서 3회 평균

Agent benchmark도 tool schema, context management와 judge가 다르다. 결과는 실제 product stack의 유용성을 보여 주지만 model weight만의 능력을 분리하지는 못한다.

### Fallback, refusal, cyberguard

- Claude Fable 5 결과에는 fallback behavior가 포함된다.
- GPT-5.6 Sol 결과에는 potential cyberguard 개입이 포함된다.
- Agents' Last Exam의 Fable 5 항목은 xhigh이며 task 40%가 downgraded로 표시된 leaderboard entry다.
- SWE-Marathon에서 Fable 5는 task 35%에 fallback이 발생했다.

안전 정책과 fallback을 포함하는 것은 제품 수준 비교에는 타당하지만, 순수 capability 비교에는 confounder다.

## Main Results: Reasoning과 Knowledge

| Benchmark | Kimi K3 | Claude Fable 5 | GPT-5.6 Sol | Claude Opus 4.8 | GPT-5.5 | GLM-5.2 |
|---|---:|---:|---:|---:|---:|---:|
| GPQA Diamond | 93.5 | 92.6 | **94.1** | 91.0 | 93.5 | 91.2 |
| CritPt | 23.4 | 28.6 | **32.3** | 20.9 | 27.1 | 20.9 |
| AA-LCR | **74.7** | 70.0 | 73.7 | 67.7 | 74.3 | 71.3 |
| HLE-Full, no tool/tool | 43.5/56.0 | **53.3/63.0** | 44.5/58.0 | 49.8/57.9 | 41.4/52.2 | - |

### 해석

GPQA에서는 frontier model과 거의 같은 수준이다. AA-LCR에서는 74.7로 가장 높지만 GPT-5.5의 74.3, GPT-5.6 Sol의 73.7과 차이는 작다.

반면 research-level task인 CritPt와 HLE-Full에서는 차이가 남는다.

- CritPt: K3 23.4, Sol 32.3
- HLE no tool: K3 43.5, Fable 53.3
- HLE tool: K3 56.0, Fable 63.0

HLE에서 tool이 K3 점수를 12.5점 높이지만 가장 강한 proprietary model을 넘지는 못한다. 따라서 K3의 강점은 일반 graduate-level knowledge와 agentic tool use이고, frontier research reasoning은 여전히 개선 영역이라는 저자들의 결론이 타당하다.

### 1M context와 연결해 읽기

AA-LCR 및 장기 agent task가 long-context 능력을 일부 보여 주지만, 1M window 자체를 체계적으로 분해한 실험은 아니다. BrowseComp에서는

- 300K에서 context compaction 적용: 91.2
- Context management 없이 full 1M: 90.4

이다. 긴 raw window를 제공하는 것보다 어떤 정보를 유지·압축하는지가 더 중요할 수 있음을 보여 준다.

## Main Results: Coding

| Benchmark | Kimi K3 | Fable 5 | GPT-5.6 Sol | 최고 결과 해석 |
|---|---:|---:|---:|---|
| DeepSWE | 67.5 | 70.0 | **73.0** | Sol 우세 |
| ProgramBench | **77.8** | 76.8 | 77.6 | K3 근소 1위 |
| Terminal-Bench 2.1 | 88.3 | 88.0 | **88.8** | 사실상 근접 |
| FrontierSWE | 81.2 | **86.6** | 71.3 | K3 2위 |
| SWE-Marathon | **42.0** | 35.0 | 39.0 | K3 강점 |
| PostTrainBench | 36.6 | **41.4** | 34.6 | Fable 우세 |
| MLS-Bench-Lite | 48.3 | **49.9** | 46.2 | Fable 근소 우세 |
| SciCode | 58.7 | **60.2** | 56.1 | Fable 우세 |

Kimi K3는 ProgramBench, SWE-Marathon에서 가장 높고 Terminal-Bench에서는 Sol과 0.5점 차이다. Long-horizon FrontierSWE에서도 81.2로 강하다.

그러나 DeepSWE에서는 Sol, PostTrainBench·MLS-Bench-Lite·SciCode에서는 Fable 5가 앞선다. 따라서 “모든 coding에서 최고”가 아니라 다음 영역에서 특히 강하다고 보는 편이 정확하다.

- 긴 tool-use trajectory
- GPU kernel 최적화
- Production-oriented software task
- Web/visual artifact 생성

Harness와 hardware caveat가 크므로 1~2점 차이는 model architecture의 순수 우위로 해석하지 않아야 한다.

## Main Results: Agentic

### Kimi K3가 가장 높은 대표 항목

| Benchmark | Kimi K3 |
|---|---:|
| BrowseComp | **91.2** |
| DeepSearchQA | **95.0 F1** |
| ResearchRubrics | **76.2** |
| MCPMark-Verified | **94.5** |
| AutomationBench | **30.8** |
| SpreadsheetBench 2 | **34.8** |
| $\tau^3$-Banking | **33.4** |
| Harvey Lab-AA | **94.6** criterion pass |

### 경쟁 모델이 앞선 주요 항목

| Benchmark | Kimi K3 | 최고 모델 |
|---|---:|---|
| GDPval-AA v2 | 1,686 Elo | Fable 5, 1,747 |
| AA-Briefcase | 1,548 Elo | Fable 5, 1,583 |
| Agents' Last Exam | 28.3 | GPT-5.6 Sol, 29.6 |
| APEX-Agents | 41.0 | Fable 5, 43.3 |
| OfficeQA Pro | 63.3 | Fable 5, 69.9 |
| OSWorld-Verified | 84.8 | Fable 5, 85.0 |
| OSWorld 2.0 | 58.3 | Fable 5, 66.1 |
| SaaS-Bench | 60.1 | GPT-5.6 Sol, 61.4 |
| CorpFin v2 | 71.6 | Fable 5, 71.8 |
| Finance Agent v2 | 54.4 | Fable 5, 56.3 |
| Legal Research Bench | 44.2 | Fable 5, 49.5 |

K3의 상대적 강점은 search, research, MCP tool use, spreadsheet, banking처럼 여러 도구와 긴 state를 조율하는 작업이다. 어려운 computer use, 일부 전문 지식 업무와 enterprise evaluation에서는 Fable 5의 우위가 남는다.

평가별 특수 조건도 중요하다.

- OfficeQA Pro: 전체 PDF corpus를 image로 제공하고 machine-readable text는 없음
- MCP-Atlas: Public 500-task subset, 100-turn limit, Gemini 3.1 Pro judge
- AutomationBench: Public 600-task subset
- 산업 benchmark 일부: 논문팀 재실행이 아니라 Artificial Analysis 또는 Vals AI 결과 인용

## Main Results: Vision

| Benchmark | Kimi K3 | 가장 높은 비교 결과 |
|---|---:|---:|
| WorldVQA | 51.0 | Fable 5, 56.7 |
| OmniDocBench | **91.1** | Kimi K3 |
| PerceptionBench | 58.5 | GPT-5.6 Sol, 59.7 |
| Video-MME | **90.0** | Kimi K3 |
| MMVU | **82.1** | Kimi K3 |
| BabyVision + Python | 85.7 | Fable 5, 90.5 |
| MMMU-Pro no tool/Python | 81.6/83.4 | Sol 83.0 no tool, Fable 86.5 tool |
| CharXiv no tool/Python | 84.8/91.3 | Fable 88.9/93.5 |
| Math-Vision no tool/Python | 94.3/97.8 | Sol 95.8 no tool, Fable 98.6 tool |
| ZeroBench pass@5 no tool/Python | 23.0/41.0 | Fable 23.0/46.0 |

Kimi K3는 document parsing, video, multimodal understanding에서 강하다. 특히 OmniDocBench, Video-MME, MMVU는 가장 높다.

Python tool의 증가 폭은 다음과 같다.

| Benchmark | 증가 |
|---|---:|
| MMMU-Pro | +1.8 |
| CharXiv | +6.5 |
| Math-Vision | +3.5 |
| ZeroBench | +18.0 |

ZeroBench의 18점 상승은 native visual representation만의 결과가 아니다. Model이 이미지를 보고 Python으로 crop, 계산, 검증하는 agentic visual reasoning system의 성능이다.

Pure vision 성능을 보려면 no-tool score를 우선 봐야 하고, 실제 문제 해결 시스템을 보려면 tool score가 더 중요하다. 둘 중 하나만 선택해 모델을 홍보하면 다른 능력을 가리게 된다.

또한 main table에서 GLM-5.2의 vision 항목은 비어 있다. “Open-weight multimodal frontier”라는 주장에 대해 같은 공개 조건의 open-weight 직접 비교가 충분하지 않다.

## Internal Evaluation

내부 benchmark는 public suite가 놓치는 failure mode를 빠르게 반영할 수 있지만 dataset, judge prompt, 전체 sample과 raw trajectory가 공개되지 않아 재현성이 낮다.

### 주요 내부 결과

| Benchmark | Kimi K3 | 가장 높은 비교 결과 |
|---|---:|---:|
| Kimi Code Bench 2.0, Claude Code | 73.7 | Fable 5, 76.9 |
| Coding Experience, Claude Code | **59.9** | K3 |
| 24/7 ClawBench | 48.3 | Sol, 52.0 |
| MIRA | 64.1 | Fable 5, 72.9 |
| KAET | 83.5 | Sol, 85.4 |
| CLIF | **52.4** | K3 |
| Agentic Vision | 78.3 | Sol, 82.9 |
| Swarm | **76.3** | K3 |
| Online Experience | 77.9 | Sol, 84.0 |
| Deep Research | **90.0** | K3 |
| Finance | 62.6 | Sol, 62.7 |
| KWV | 64.7 | Sol, 66.9 |
| DECK | 73.5 | Sol, 74.7 |
| Agent Behavior | 65.0 | Sol, 76.4 |
| Faithfulness | 85.5 | GPT-5.5, 86.5 |
| Chat All-in-One | 85.2 | Fable 5, 88.0 |

K3의 orchestration과 deep research 강점이 공개 평가와 일치한다. 반면 Agent Behavior, MIRA, 장기간 always-on assistant, agentic vision은 여전히 약하다.

### Kimi Webdev Bench

K3와 Claude Opus 4.8을 모두 Claude Code harness에서 실행하고 blind expert가 code quality, feature completeness, visual fidelity, interaction experience를 평가한다.

| 분야 | K3 승 | 동률 | 패 | 승-패 |
|---|---:|---:|---:|---:|
| Games | 55.6% | 3.7% | 40.7% | +14.9%p |
| 3D/WebGL/Shader | 72.7% | 13.7% | 13.6% | +59.1%p |
| Website/UI Clone | 52.6% | 21.1% | 26.3% | +26.3%p |
| 전체 | 58.6% | 13.8% | 27.6% | +31.0%p |

같은 harness와 blind judging을 사용했다는 점에서 내부 결과 중 비교적 강한 증거다. 하지만 prompt 수, annotator 수, inter-rater agreement와 confidence interval이 없다.

### Refusal과 fallback

내부 표에도 product safety behavior가 섞여 있다.

- KCB 2.0 Fable 5: 80개 중 fallback 13, refusal 1
- KCB 2.0 GPT-5.6 Sol: refusal 10/80
- KCB 2.0 GPT-5.5: refusal 3/80
- 24/7 ClawBench Fable 5: refusal task 2개 포함
- Online Experience Fable 5: refusal 14개 포함
- Agent Behavior Fable 5: refusal 6/95

“도움을 실제로 받을 수 있는가”를 평가한다면 refusal도 실패로 세는 것이 합리적이다. “안전 정책이 없을 때의 latent capability”를 비교하려면 별도 protocol이 필요하다.

## Cybersecurity Evaluation

Cyber capability를 두 단계로 나눈다.

- Tier 1: 새로운 vulnerability discovery와 proof of concept
- Tier 2: End-to-end exploit development

### Tier 1

OS kernel, database, AI service, web framework, blockchain, VPN을 포함한 여러 system에서 수백 개 candidate vulnerability를 찾았다. Human review를 받은 결과의 약 70%가 genuine으로 확인되었고, 6개 project의 previously unknown vulnerability 16개가 포함되었다고 보고한다.

Linux kernel 사례로

- Remote denial-of-service가 가능한 heap out-of-bounds write
- RDMA subsystem의 read-only page에 kernel write가 가능한 Dirty-COW 계열 local privilege escalation

을 제시한다.

이 결과는 방어적 code audit 능력의 강한 사례지만 전체 candidate 수, review된 subset의 선택 기준, 공개 disclosure 상태가 없으므로 70%를 전체 탐지 precision으로 그대로 해석할 수는 없다.

### Tier 2

총 36개 exploit task를 사용한다.

| Track | Task 수 | 내용 |
|---|---:|---|
| User-space | 16 | PostgreSQL, XWiki, Apache HTTP Server 등 real CVE |
| Linux kernel | 20 | QEMU 환경에서 unprivileged user에서 root로 escalation |

Kimi K3는

```math
\frac{14}{36}=38.9\%
```

를 해결했고 GLM-5.2는

```math
\frac{8}{36}=22.2\%
```

를 해결했다. K3 성공 14개 중 10개는 user-space다. Kernel track에서는 두 모델 모두 task의 75% 이상을 해결하지 못했다.

실패 원인은 다음처럼 분석된다.

1. 이미 exploit primitive를 얻고도 chain의 마지막 단계를 완성하지 못함
2. Mitigation에 맞지 않는 전략을 계속 고수
3. 장시간 비생산적 debugging loop
4. 제출 전에 최종 exploit을 충분히 검증하지 않음

Proprietary frontier model은 cyber request를 거부해 비교에서 제외되었다. 따라서 이 suite로 Kimi K3가 proprietary model보다 강하다고 결론 내릴 수 없다.

UK AISI와 CAISI의 외부 평가는 ExploitBench에서 K3 32%, GLM 24%, 32-step simulated enterprise network에서 17 대 11 step을 보고한다. 그러나 arbitrary code execution은 41개 task 중 0개였다. 발견·부분 진전 능력과 완전 exploit 성공을 분리해야 한다.

## Third-Party Evaluation

2026년 7월 23일 시점의 독립 평가 snapshot은 다음과 같다.

| Benchmark | Kimi K3 | Fable 5 | GPT-5.6 Sol | Opus 4.8 | GPT-5.5 | GLM-5.2 |
|---|---:|---:|---:|---:|---:|---:|
| Artificial Analysis Intelligence Index v4.1 | 57.1, #4/580 | **59.9** | 58.9 | 55.7 | 55.0 | 51.1 |
| Vals Index | 74.7, #2/39 | **75.1** | 73.1 | 70.4 | 68.0 | 65.0 |
| WebDev Arena Elo | **1,678, #1/99** | 1,634 | 1,630 | 1,565 | 1,507 | 1,592 |
| Text Arena Elo | 1,486, #8/200 | **1,507** | 1,485 | 1,484 | 1,482 | 1,469 |
| Agent Arena | 9.1, #4/37 | **12.7** | 10.1 | 9.8 | 8.8 | 6.5 |

외부 결과에서도 WebDev는 강하고 text 및 agent overall에서는 최상위 proprietary model 뒤에 있다는 패턴이 반복된다.

해석 시 주의점은 다음과 같다.

- 각 기관의 자체 protocol 결과를 모은 표이지 동일 환경 재평가가 아니다.
- Elo는 match가 누적될수록 바뀐다.
- Agent Arena는 voting을 시작한 지 얼마 되지 않은 7월 23일 snapshot이다.
- Effort variant가 model마다 다를 수 있다.
- 1~2점 또는 수십 Elo의 작은 차이에 통계적 확정을 부여하면 안 된다.

## Cost Efficiency

Figure 13은 Kimi Code Bench 2.0, BrowseComp, GDPval-AA v2, AA-Briefcase 네 suite에서 task당 API 비용과 score를 비교한다.

### 논문의 주요 주장

- KCB 2.0: K3 max는 Fable 5보다 4.0점 낮지만 비용은 38%
- K3 high: Opus 4.8 max와 비슷한 score를 약 3분의 1 비용에 달성
- BrowseComp: K3 max 91.2, task당 2.03달러
- BrowseComp의 GPT-5.6 Sol 90.4보다 점수는 높고 비용은 약 절반
- GDPval-AA v2: Sol보다 약 50 Elo 낮지만 13% 저렴, Fable 5보다 2.6배 저렴
- AA-Briefcase: Fable 5 다음 score를 Fable 비용의 약 절반에 제공

### 비용 비교의 한계

1. KCB에서 K3는 Kimi Code, 다른 model은 Claude Code를 사용해 token 수와 retry behavior가 달라질 수 있다.
2. K3 비용은 내부 측정이고 경쟁 모델은 외부 chart 또는 공개 API 가격이다.
3. Provider별 max/high/xhigh의 compute budget이 같지 않다.
4. API price만 보며 TTFT, latency, energy, failure retry, 2.78T self-hosting cost는 제외한다.
5. 선택된 네 workload의 frontier를 전체 task로 일반화할 수 없다.
6. API 가격이 바뀌면 결론도 바뀐다.

특히 두 “효율”을 혼동하지 않아야 한다.

| 주장 | 측정 대상 |
|---|---|
| K2 대비 2.5배 scaling efficiency | Pre-training FLOPs 대비 validation loss |
| Figure 13 cost efficiency | 2026년 7월 API 가격 기준 task score/달러 |

첫 번째가 좋다고 두 번째가 자동으로 좋지 않고, API 가격이 저렴하다고 architecture FLOPs가 적다는 뜻도 아니다.

## Case Studies

### GPU kernel optimization

동일 sandbox에서 model마다 task당 최대 24시간을 주고 AttnRes, DSA, KDA, MLA kernel을 최적화한다.

| Kernel/항목 | Kimi K3 결과 |
|---|---:|
| AttnRes | 283.6 ms -> 114.4 ms, 59.7% 개선 |
| DSA | Runtime 55.1% 감소 |
| KDA | Runtime 73.6% 감소 |
| MLA | Peak TFLOPS의 절반 이상 |

AttnRes trajectory의 최종 speedup은 K3 59.7%, Fable 5 57.1%, GPT-5.5 30.8%, GPT-5.6 Sol 17.3%다.

강한 실무 coding 사례지만 다음 한계가 있다.

- Kernel별 전체 비교 표가 없다.
- 여러 독립 run의 분산이 없다.
- 초기 K3 checkpoint가 자체 kernel development에 사용되어 held-out 독립성이 불분명하다.
- Hardware와 shape별 generalization을 충분히 보여 주지 않는다.

### MiniTriton compiler

Kimi K3가 만든 MiniTriton은 다음 범위를 포함한다.

- Tile-level Python frontend와 layout system
- Warp-level MLIR annotation 및 optimization
- PTX code generation
- PyTorch-like eager/compiled tensor library
- Reverse-mode autograd
- Neural-network module
- NCCL distributed primitive
- Sparse 및 visualization primitive

NVIDIA L20에서 core suite geometric mean 기준 PyTorch eager와 torch.compile보다 빠르고, 큰 tensor-core matrix multiplication은 측정 machine roof의 약 90%에 도달한다. DSL KDA prefill kernel은 matched Triton reference보다 빠르다.

Character-level GPT의 loss curve는 PyTorch reference를 따라가며 full-model gradient 차이는 FP64 reference 기준 PyTorch 자체 FP32 rounding 수준인 약 $10^{-4}$ 이내라고 보고한다. 자체 NCCL primitive로 2-GPU DDP도 single-GPU와 유사하게 수렴한다.

공개 code가 있다는 점은 큰 장점이다. 그러나 이는 다른 model과 같은 protocol의 benchmark가 아니며, benchmark shape 선택, 실패한 시도와 총 개발 비용을 포함한 정량 비교는 없다.

### Chip design

48시간 autonomous run에서 hybrid KDA/NoPE-MLA, Block AttnRes, shared expert와 INT4 MoE routing을 가진 nano model용 inference chip prototype을 설계했다.

| 항목 | 결과 |
|---|---:|
| Standard-cell library | Nangate45 |
| Analytical area budget | $4\ \mathrm{mm}^2$ |
| Clock | 100 MHz timing closure |
| Decode throughput | RTL simulation에서 8,700 token/s 이상 |
| Standard cells | 1.46M |
| SRAM | 0.277 MiB |

이 사례는 RTL generation과 open-source EDA 자동화의 proof of concept이다. 실제 tape-out 또는 silicon measurement가 아니며 전력, thermal, memory traffic도 없다. 8,700 token/s는 작은 nano model의 RTL simulation 값이지 Kimi K3 throughput이 아니다.

### Coding for research

I-Love-Q universal relation 재현에서

- 20편 이상 논문 검토
- 300개 이상 equation of state 평가
- 3,000줄 이상 Python
- Interactive HTML dashboard
- 약 2시간 소요

를 보고한다. 숙련 연구자의 1~2주와 비교하지만 독립적인 correctness audit와 인간 비교 protocol이 없다.

### Knowledge work

AI ASIC 산업 조사에서는

- 42년 범위
- 120회 이상 iterative refinement
- 87개 quarterly report와 99개 PDF
- 11,000페이지 이상
- Web search 2,800회 이상
- Terminal query 1,100회 이상

을 사용했다.

GWTC-5 분석은 gravitational-wave event 391개, concurrent subagent 20개 이상, visualization 7개, summary table 2개, 10편 이상 literature synthesis를 포함한다.

이 수치는 long-horizon orchestration 규모를 보여 주지만 정보량 자체가 정확성을 보장하지 않는다. Final claim의 citation correctness, numerical reproduction과 expert review가 필요하다.

### Video editing과 motion design

자기 architecture를 설명하는 3Blue1Brown-style animation과 source clip 56개로 teaser video를 제작했다. Clip selection, motion-matched cut, beat synchronization, audio processing과 반복 수정이 포함된다.

Native vision이 단순 VQA뿐 아니라 생성한 visual artifact를 다시 보고 수정하는 loop에 사용된 사례다. 그러나 human editor의 1~2일과 비교한 시간 주장은 품질을 blind test한 결과가 아니라 정성적 추정이다.

## Appendix에서 보강되는 내용

### SiTU-GLU

원점 근처에서 scaled tanh가 linear term을 보존하고 $\beta_1=4,\beta_2=25$일 때 coordinate output의 absolute bound가 100임을 보인다.

### Quantile Balancing derivation

Balanced assignment의 linear-program dual에서 token threshold와 expert threshold를 도출한다. 각 threshold를 고정하고 다른 쪽을 quantile로 갱신하는 과정은 coordinate-wise exact minimization으로 해석된다.

Inference에서는 최종 Top-k bias를 frozen하므로 histogram 또는 quantile 연산이 필요 없다.

### Histogram quantile

Expert별 margin을 1,000개 bin에 누적한다. Error는 bin width 이하이고 raw margin을 모든 rank에서 모으는 방식 대비 communication이 1% 미만이라고 주장한다.

### MoonEP bound

Rank당 $E/R$개의 redundant expert가 항상 충분하고, worst case에는 $\lceil E(R-1)/R^2\rceil$이 필요해 asymptotically 같은 크기임을 증명한다.

### XTML chat template

XTML은 XML의 angle bracket 대신 reserved special token을 사용한다.

- open
- sep
- close
- end_of_msg

Assistant message는 think, response, tools channel로 나뉜다. Tool call에는 parallel call을 구분하는 index와 typed argument가 있고, result가 같은 tool/index를 반복해 대응 관계를 명확히 한다.

Global option인 tool declaration과 reasoning effort는 history 앞에 두고, 매 request마다 바뀌는 tool choice와 response format은 history 뒤에 둔다. 후자의 변경이 기존 history KV cache를 무효화하지 않게 하기 위한 설계다.

K3는 preserved thinking을 사용한다. Thinking mode에서는 think channel이 비어 있어도 history에 구조가 유지된다. Format consistency에는 유리하지만 매우 긴 agent session에서 reasoning trace 저장량과 privacy 관리가 중요해진다.

## 논문의 증거 수준 점검

### 비교적 강하게 뒷받침되는 주장

| 주장 | 근거 |
|---|---|
| Kimi K3의 정확한 architecture 규모 | Table 1에 K2와 K3 사양 공개 |
| Lower-bounded decay가 BF16 범위를 제한 | $g_{\min}=-5$, 16-token tile에서 reciprocal $\lt e^{80}$ 수학적 설명 |
| SiTU-GLU output이 bounded | Appendix B의 $\beta_1\beta_2=100$ 상한 |
| QB의 quantile update | Appendix C의 balanced assignment/dual derivation |
| Histogram estimator의 오차 구조 | Appendix D의 bin-width bound와 $B=1000$ |
| MoonEP redundant expert 상한 | Appendix E의 $E/R$ upper bound와 near-tight construction |
| 여러 공개 benchmark에서 강한 agent/coding/vision 결과 | Table 2와 외부 leaderboard snapshot |
| Web development의 외부 선호도 | WebDev Arena 1위라는 제3자 결과 |

### 방향은 설득력 있지만 정량 근거가 약한 주장

| 주장 | 부족한 근거 |
|---|---|
| MoonViT-V2가 SigLIP 초기화와 동등한 vision 성능 | Task별 표와 seed 분산 없음 |
| RMSNorm LatentMoE가 loss와 downstream을 개선 | 개선 폭 미공개 |
| Per-Head Muon이 안정성과 overhead 개선 | Full-Muon/AdamW 비교 표 없음 |
| 약 8개 AttnRes block이면 대부분의 이득 회복 | K3 자체 block-count ablation 없음 |
| KDA full-rank gate의 필요성 | Low-rank와의 controlled ablation 없음 |
| 3:1 KDA/MLA가 최적 | Ratio sweep 없음 |
| Infrastructure가 높은 utilization을 달성 | End-to-end MFU, throughput, latency 표 없음 |

### 강하게 제한해서 읽어야 하는 주장

#### 2.5배 scaling efficiency

Architecture, parameter scale, active compute, data, optimizer와 training recipe가 동시에 바뀌었다. Raw scaling point, fit coefficient, confidence interval도 없다. Kimi K3 family 전체가 K2 family보다 같은 loss에 필요한 FLOPs가 적다는 aggregate claim으로만 받아들여야 한다.

#### 1M direct extrapolation

NoPE 수정이 필요 없다는 뜻이지 64K model을 그대로 1M에서 zero-shot 사용했다는 뜻이 아니다. 실제 training curriculum에 256K와 1M cooldown이 있다.

#### Open frontier within everyone's reach

Weight 공개는 중요하지만 2.78T model의 4-bit theoretical lower bound만 약 1.39TB다. 일반 개인 연구자가 쉽게 self-host할 수 있는 크기는 아니다. Training data, internal evaluation과 production infrastructure 전체도 공개되지 않는다.

#### Cost-efficiency frontier

선택된 네 benchmark와 당시 API price에서의 결과다. Hardware cost, latency, energy, retry, harness 차이를 포함한 일반적 inference efficiency claim은 아니다.

## 주요 기여

### 1. Sequence-depth-width를 분리한 architecture

KDA, AttnRes, LatentMoE가 서로 같은 attention 변형이 아니라 각각 다른 정보 흐름 병목을 담당한다.

- KDA/MLA: token axis
- AttnRes: layer axis
- LatentMoE: channel/expert axis

모듈의 역할 분리가 명확해 architecture를 분석하고 후속 ablation을 설계하기 좋다.

### 2. 수학적 parameterization과 kernel의 공동 설계

Lower-bounded decay는 overflow 방지뿐 아니라 diagonal tile까지 Tensor Core GEMM으로 바꾼다. FP32 MLA output은 shared-memory layout을 다시 설계하게 한다. KDA affine transition은 context parallel prefix scan으로 이어진다.

즉 수식, precision, kernel, distributed execution이 한 단계씩 연결되어 있다.

### 3. 초대형 sparse MoE의 실제 운영 문제 해결

896 expert를 선언하는 것보다 중요한 것은 다음 문제를 함께 다룬 점이다.

- Router-level balance: QB
- Rank-level balance: MoonEP
- Variable-shape 제거
- Expert weight migration
- Zero-copy dispatch
- Small-batch weight-streaming decode kernel

### 4. 1M-context RL을 environment state까지 포함해 처리

긴 trajectory의 상태는 LLM KV cache만이 아니다.

- Sandbox memory와 disk
- Tool state
- External application state
- Paused rollout
- Reference model weight

AgentENV와 external KV pool은 model state와 world state를 함께 보존한다.

### 5. Deployment-aware post-training

QAT를 SFT부터 RL까지 유지하고 speculative draft의 acceptance rate를 직접 최적화한다. Training objective와 실제 serving constraint의 간격을 줄이는 접근이다.

## 논문의 한계

### 1. 개별 architecture ablation 부족

KDA, MLA ratio, AttnRes, LatentMoE normalization, SiTU, QB, Per-Head Muon의 독립 기여를 분리할 수 없다. 2.5배 개선이 어느 조합에서 오는지 알 수 없다.

### 2. Pre-training 재현 정보 부족

총 token, modality 비율, data cutoff, optimizer 세부값, global batch, peak LR, parallelism degree, cluster와 training duration이 빠져 있다.

### 3. Infrastructure 성능표 부족

설계 설명은 매우 상세하지만

- FlashKDA vs Triton throughput
- KCP degree별 scaling
- MoonEP vs 기존 EP step time
- Memory optimization별 peak GB
- Prefix hit/miss TTFT
- 1M serving throughput

을 한눈에 검증할 표가 없다.

### 4. 1M 문맥 quality 검증 부족

Max context support와 effective context utilization은 다르다. 위치별 retrieval, multi-hop distance, distractor density와 long-context generation consistency가 필요하다.

### 5. 평가 조건 혼합

Reasoning effort, harness, tool availability, fallback와 hardware가 model마다 다르다. 시스템 전체 비교로는 의미가 있지만 architecture 또는 base model의 순수 우위를 말하기 어렵다.

### 6. 내부 평가 의존

Kimi Code Bench, MIRA, KAET, Agent Behavior, Webdev 등 중요한 결론이 비공개 task와 judge에 의존한다. Sample 수와 confidence interval도 대부분 없다.

### 7. Open-weight와 open science의 차이

가중치는 공개되지만 training data, full recipe, internal benchmark, production scheduler와 cluster configuration은 완전히 공개되지 않는다. Open-weight는 큰 기여지만 완전 재현 가능한 open-source training project와 동일하지 않다.

### 8. 매우 높은 배포 장벽

Active parameter가 104.2B이고 전체 weight가 2.78T다. MXFP4를 사용해도 expert sharding과 고속 network가 필요하다. Small lab이나 edge deployment에는 직접 사용할 수 없다.

### 9. Native vision 주장의 범위

Vision tower와 projector는 별도로 존재한다. Native는 joint-from-scratch training을 뜻한다. 더 작은 data/compute에서 contrastive initialization 없이 같은 결과가 나올지는 검증되지 않았다.

### 10. Safety와 capability가 섞인 평가

Cybersecurity에서는 proprietary model의 refusal 때문에 비교가 불가능하다. 일반 benchmark에서도 refusal/fallback을 실패로 포함한다. 실제 제품 유용성과 latent technical capability를 별도 표로 보고할 필요가 있다.

## 온디바이스와 Edge AI 관점

### Kimi K3 자체를 온디바이스에 배포할 수 있는가

현실적으로 불가능하다.

- 전체 weight 2.78T
- 모든 weight를 4-bit라고 가정한 하한 약 1.39TB
- Active parameters 104.2B
- 93개 backbone layer
- Token당 routed expert 16개와 shared expert 2개
- 401M vision encoder
- KDA와 MLA의 두 cache 체계

이는 mobile NPU뿐 아니라 단일 datacenter GPU에도 들어가지 않는다.

### 그래도 온디바이스 연구에 중요한 이유

Kimi K3 전체가 아니라 일부 설계 원리는 작은 모델에 옮길 수 있다.

#### 1. Hybrid KDA + sparse global attention

긴 context에서 모든 layer에 full attention을 쓰지 않고 일부 layer만 global attention으로 남기는 전략은 memory가 제한된 device에 유용할 수 있다.

단, mobile NPU가 KDA의

- Short convolution
- Dynamic gate
- Rank-one state update
- In-place recurrent state
- Head-wise normalization

을 fuse하지 못하면 CPU fallback과 tensor copy 때문에 full attention보다 느릴 수 있다.

#### 2. Lower-bounded decay

BF16뿐 아니라 INT8/FP16 recurrence에서도 cumulative scaling 범위를 제한하는 아이디어는 유용하다. Hardware dynamic range에 맞춰 $g_{\min}$과 tile size를 공동 설계할 수 있다.

#### 3. Visual token compression

2 x 2 pixel shuffle로 LLM visual token을 4분의 1로 줄이는 것은 prefill과 language KV cache를 줄인다.

하지만 vision encoder 뒤에 적용되므로 ViT compute는 그대로다. On-device에서는 early pooling, patch merge 또는 hierarchical encoder와 비교해야 한다.

#### 4. Stable bounded activation

SiTU-GLU의 bounded output은 low-precision activation calibration과 overflow 통제에 도움이 될 수 있다. 그러나 tanh operator가 NPU에서 비싸거나 근사 구현만 지원될 수 있으므로 다음을 측정해야 한다.

- Native tanh latency
- LUT/polynomial approximation error
- Saturation 비율
- INT8 scale stability
- Accuracy 변화

#### 5. Latent expert

Full hidden state가 아니라 latent width에서 expert를 실행하면 dispatch byte를 줄일 수 있다. 그러나 mobile에서는 896개 중 16개처럼 큰 expert pool보다 작은 Top-1/Top-2와 static expert placement가 현실적이다.

Dynamic Top-16은 irregular weight access가 커서 batch 1에서 flash storage 또는 DRAM bandwidth 병목을 만들 수 있다.

#### 6. KDA fixed state

Full attention KV는 context와 함께 증가하지만 KDA state는 고정이다. Always-on assistant에는 매력적이다.

그러나 Kimi K3도 24개 MLA layer가 있어 전체 cache가 완전 고정은 아니다. 소형 edge model에서는

- KDA-only
- 대부분 KDA + 매우 드문 global layer
- Sliding-window attention + KDA

를 비교할 필요가 있다.

#### 7. Per-Head Muon

이는 training 안정화 방법이며 inference kernel을 빠르게 하지 않는다. On-device deployment benefit과 training benefit을 혼동하지 않아야 한다.

### On-device에서 먼저 측정할 항목

| 범주 | 필수 측정 |
|---|---|
| Accuracy | Perplexity, long-context retrieval, downstream task |
| Prefill | 1K/4K/16K context latency와 peak memory |
| Decode | Batch 1 token/s, recurrent state update latency |
| Kernel | NPU support, CPU fallback, fusion count |
| Memory | Weight, KDA state, remaining KV cache, activation |
| Power | Joule/token, sustained temperature, throttling |
| Robustness | 10분 이상 generation, state drift, overflow/NaN |

Theoretical FLOPs가 줄어도 kernel fallback과 memory layout이 나쁘면 실제 latency는 악화될 수 있다.

## 소규모 재현을 위한 구현 순서

Kimi K3 전체를 재현하려 하지 말고 구성요소를 단계별로 검증하는 편이 좋다.

### Phase 1: KDA recurrence correctness

1. FP32 sequential recurrence 구현
2. Chunkwise form과 token별 output 비교
3. Random length와 padding에서 max absolute error 측정
4. Gradient check
5. Lower-bounded decay와 기존 Softplus decay 비교

핵심 invariant는 다음이다.

```math
\max_t
\left\|
O_t^{\mathrm{sequential}}
-
O_t^{\mathrm{chunkwise}}
\right\|
\lt\epsilon
```

### Phase 2: Hybrid ratio

같은 parameter와 training token budget에서 다음을 비교한다.

| Arm | 구성 |
|---|---|
| A | Full attention only |
| B | KDA only |
| C | KDA:MLA = 1:1 |
| D | KDA:MLA = 3:1 |
| E | KDA:MLA = 7:1 |

Short-context perplexity, long-context retrieval, prefill/decode latency와 memory를 모두 보고해야 한다.

### Phase 3: AttnRes

- Standard residual
- Full AttnRes
- Block size 4/8/12/16

를 비교한다. Final loss뿐 아니라 activation memory, pipeline communication, kernel 수를 측정한다.

### Phase 4: Stable LatentMoE factorial

다음 2 x 2 x 2 factorial이 각 요소의 main effect와 interaction을 분리한다.

| 축 | Off | On |
|---|---|---|
| Normalization | No RMSNorm | RMSNorm before $W_{\uparrow}$ |
| Activation | SwiGLU | SiTU-GLU |
| Balance | Fixed-step bias | Quantile Balancing |

측정값은 loss, activation percentile, overflow/NaN, expert load CV, router entropy, token drop, step time이다.

### Phase 5: Vision initialization

같은 MoonViT architecture에서

- Random initialization + joint NTP
- SigLIP initialization + joint NTP
- SigLIP freeze 후 unfreeze

를 동일 data와 seed로 비교해야 한다. Gradient norm plot뿐 아니라 OCR, localization, document, video와 general VQA 성능을 함께 본다.

### Phase 6: Deployment

Training checkpoint가 만들어져도 끝이 아니다.

    PyTorch reference
        |
        v
    optimized eager/fused kernel
        |
        v
    exported graph
        |
        v
    target runtime
        |
        v
    target NPU/GPU executable

각 경계에서 tensor output, perplexity와 downstream metric을 비교한다. Conversion 성공만으로 배포 성공을 선언하면 안 된다.

## 권장 추가 실험

1. 동일 harness, 동일 token/tool budget, 동일 hardware에서 model 재평가
2. Low/high/max별 accuracy, output token, tool call, latency, cost 공개
3. 8K부터 1M까지 context length sweep
4. Needle 위치, needle 수, distractor density, multi-hop distance별 평가
5. KDA/MLA ratio와 final MLA 제거 ablation
6. $g_{\min}$ 및 chunk/tile size sweep
7. Full-rank와 low-rank output gate 비교
8. AttnRes block 수와 block sum 이외의 compression 비교
9. SiTU $\beta_1,\beta_2$ sensitivity와 activation histogram
10. QB bin 수, EMA, tie 및 non-integer target 처리 분석
11. MoonEP end-to-end throughput, buffer와 planner overhead
12. 1M prefix-cache hit/miss TTFT와 state checkpoint density
13. Native vision과 Python-assisted vision을 분리한 평가
14. Internal benchmark의 fixed held-out test와 외부 judge
15. Cost에 latency, energy, retry와 self-hosting amortization 포함
16. Case study를 여러 seed와 blind expert panel로 반복

## 강점

- Architecture의 각 모듈이 해결하는 축이 명확하다.
- KDA 수식에서 kernel과 context parallelism까지 연결이 일관된다.
- 896-expert MoE의 statistical balance와 physical execution balance를 별도로 해결한다.
- Long-horizon RL에서 model state와 environment state를 함께 관리한다.
- Native vision, agent tool use, kernel coding이 실제 capability 결과로 연결된다.
- 가중치와 MoonEP, AgentENV, MiniTriton, nano-KPU 등 여러 artifact를 공개한다.
- 저자들이 최상위 proprietary model보다 여전히 약한 영역을 main table에 함께 보고한다.

## 한계 요약

- 개별 구성요소의 controlled ablation이 부족하다.
- Pre-training data와 compute recipe가 충분히 공개되지 않는다.
- 1M context의 정규 quality/latency curve가 없다.
- Infrastructure의 end-to-end 성능표가 부족하다.
- 평가의 effort, harness, tool과 hardware가 완전히 통제되지 않았다.
- Internal benchmark와 judge 의존도가 높다.
- API 비용 효율과 architecture 효율이 혼재될 수 있다.
- Open-weight이지만 일반 연구자가 self-host하기 어려운 규모다.
- Native vision의 이득을 동일 architecture/seed로 엄밀히 분리하지 않는다.
- 최상위 research reasoning과 어려운 computer use에서는 proprietary model의 우위가 남는다.

## 최종 평가

Kimi K3의 가장 큰 의미는 benchmark 하나의 1위가 아니다. **2.78T 전체·104.2B 활성 sparse multimodal model을 1M-context pre-training, long-horizon RL, sandbox state, distributed expert execution, prefix caching과 production serving까지 하나의 일관된 설계로 구현했다는 것**이다.

Architecture 차원에서는

- KDA가 sequence memory를 압축하고
- Gated MLA가 global retrieval을 보완하며
- Attention Residuals가 depth bottleneck을 줄이고
- Stable LatentMoE가 width를 확장한다.

System 차원에서는

- Lower-bounded decay가 Tensor Core 경로를 만들고
- KCP가 recurrence를 device 사이에 분할하며
- MoonEP가 expert rank를 균형화하고
- AgentENV와 external cache가 수백만-token trajectory를 유지하며
- Hybrid prefix cache와 speculative replay가 online serving을 가능하게 한다.

성능 측면에서는 open-weight model의 기준을 크게 높였고 coding, search, tool orchestration, document/video understanding에서 매우 강하다. 그러나 Fable 5와 GPT-5.6 Sol은 research reasoning, 어려운 computer use와 일부 vision/coding에서 여전히 앞선다.

따라서 가장 공정한 결론은 다음과 같다.

> Kimi K3는 “모든 면에서 가장 강한 모델”이라기보다, open-weight 모델이 frontier급 agentic capability와 초장문 multimodal execution에 얼마나 가까이 갈 수 있는지를 보여 주는 대규모 architecture-system co-design 보고서다. 그 핵심 기여는 2.8T라는 숫자 자체보다 KDA, AttnRes, sparse MoE, RL과 serving infrastructure를 실제로 이어 붙인 데 있다. 다만 개별 모듈의 기여와 1M context의 유효성, 비용 효율의 일반성은 추가 통제 실험이 필요하다.

## 읽은 뒤 확인할 체크리스트

- [ ] 2.78T total과 104.2B active를 구분했는가?
- [ ] Active parameter와 routed expert 선택 비율 16/896을 구분했는가?
- [ ] 2.5배 scaling efficiency를 inference speedup으로 해석하지 않았는가?
- [ ] KDA 69개와 MLA 24개이므로 전체가 완전한 linear attention은 아님을 이해했는가?
- [ ] NoPE extrapolation과 1M cooldown training을 구분했는가?
- [ ] Native vision이 별도 vision encoder가 없다는 뜻이 아님을 이해했는가?
- [ ] Pixel shuffle이 LLM token을 줄이지만 ViT 앞단 compute를 제거하지 않음을 이해했는가?
- [ ] QB와 MoonEP가 서로 다른 수준의 load balance임을 구분했는가?
- [ ] SFT/RL의 policy expert 9개와 MoE 내부 expert 896개를 구분했는가?
- [ ] Python-assisted vision score와 no-tool score를 구분했는가?
- [ ] Model과 harness를 합친 평가임을 확인했는가?
- [ ] API price efficiency와 hardware efficiency를 구분했는가?
- [ ] RTL simulation과 실제 silicon을 구분했는가?
- [ ] Open-weight와 full training reproducibility를 구분했는가?

## 참고 링크

- [Kimi K3 model weights](https://huggingface.co/moonshotai/Kimi-K3)
- [MoonEP](https://github.com/MoonshotAI/MoonEP)
- [AgentENV](https://github.com/kvcache-ai/AgentENV)
- [MiniTriton](https://github.com/MoonshotAI/minitriton)
- [nano-KPU](https://github.com/MoonshotAI/nano-kpu)

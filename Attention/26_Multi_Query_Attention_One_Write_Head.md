# 26. Fast Transformer Decoding: One Write-Head is All You Need

## 논문 정보

- 제목: **Fast Transformer Decoding: One Write-Head is All You Need**
- 저자: Noam Shazeer
- 발표: 2019
- 핵심 키워드: multi-query attention, incremental decoding, KV cache, memory bandwidth, shared key-value heads

## 한눈에 보는 요약

Multi-Query Attention(MQA)은 query head는 여러 개 유지하되 모든 head가 **하나의 key head와 value head를 공유**한다. Multi-Head Attention(MHA)의 KV cache는 head 수 `H`에 비례하지만 MQA는 head 축이 사라져 cache와 decode memory traffic이 약 `H`배 줄어든다.

```text
MHA: H query heads, H key heads, H value heads
MQA: H query heads, 1 key head, 1 value head
```

Autoregressive decode에서는 매 step query가 하나뿐이라 계산 parallelism이 작고, 각 layer가 긴 과거 KV cache를 HBM에서 반복 읽는다. 이 과정은 compute보다 memory bandwidth에 제한되기 쉽다. MQA는 score 개수 자체를 줄이지 않지만 읽어야 할 key/value byte를 크게 줄여 latency와 batch throughput을 개선한다.

논문 실험에서 WMT14 translation의 품질 손실은 작았고, TPUv2 greedy decoder 부분 비용은 token당 `46 μs → 3.8 μs`로 크게 감소했다. 대신 여러 value subspace를 head별로 유지하던 MHA의 capacity가 줄어드는 것이 핵심 trade-off다.

<p align="center"><img src="https://github.com/user-attachments/assets/d682061f-5660-4583-a818-0aab6d1c27f5" alt="Multi-Head와 Multi-Query Attention 비교" width="820"></p>
<p align="center"><sub>보조 도식 — MHA와 MQA의 query/KV head 구성 및 KV cache 비교</sub></p>

## Training과 incremental decoding의 차이

Training에서는 전체 sequence의 query를 한 번에 계산한다.

```text
Q,K,V : [B,H,N,D_h]
score : [B,H,N,N]
```

큰 GEMM으로 병렬화할 수 있어 accelerator utilization이 높다. 반면 decoding step `t`에서는 새 query가 head마다 하나다.

```text
q_t      : [B,H,1,D_h]
K_cache  : [B,H,t,D_h]
V_cache  : [B,H,t,D_h]
```

각 step마다 모든 layer의 `K_cache,V_cache`를 읽어 score와 weighted sum을 계산한다. Arithmetic intensity가 낮아지고 batch가 작을수록 memory bandwidth가 지배한다.

## Multi-Head Attention의 KV cache

MHA는 head별 projection을 사용한다.

```math
\begin{aligned}
q_h&=W_h^Qx_t,
&k_h&=W_h^Kx_t,
&v_h&=W_h^Vx_t,\\
o_h&=\mathrm{softmax}\!\left(\frac{q_hK_h^{\top}}{\sqrt{D_h}}\right)V_h.
\end{aligned}
```

Layer당 sequence 길이 `N`의 KV cache element 수는

```math
2B H N D_h
```

이다. 총 hidden dimension `D=H D_h`라면 token당 `2D` element를 layer마다 저장한다.

## Multi-Query Attention 수식

MQA에서는 query projection은 head별로 유지하지만 key/value projection은 하나다.

```math
\begin{aligned}
q_h&=W_h^Qx_t,&&h=1,\ldots,H,\\
k&=W^Kx_t,&&\text{shared},\\
v&=W^Vx_t,&&\text{shared},\\
o_h&=\mathrm{softmax}\!\left(\frac{q_hK^{\top}}{\sqrt{D_h}}\right)V.
\end{aligned}
```

여기서 모든 query head가 같은 cached `K,V`를 보지만 score는 query가 다르므로 head마다 다르다. 즉 attention pattern의 다양성은 남고, key/value representation의 다양성만 공유된다.

Cache element 수는

```math
2B N D_h
```

로 줄어 MHA 대비 정확히 `H`분의 1이다.

## Tensor shape

```text
MHA
Q : [B,N,H,D_h]
K : [B,N,H,D_h]
V : [B,N,H,D_h]

MQA
Q : [B,N,H,D_h]
K : [B,N,1,D_h]
V : [B,N,1,D_h]
```

Attention kernel은 shared K/V를 head 축으로 논리적으로 broadcast한다. 실제 tensor를 `H`번 복제하면 cache 절감은 유지돼도 read traffic과 temporary memory가 다시 늘 수 있으므로, stride-0 broadcast나 MQA 전용 kernel이 필요하다.

## 왜 head 수를 그냥 줄이는 것보다 낫나

KV cache를 줄이는 단순한 방법은 전체 attention head 수를 1로 줄이거나 head dimension을 작게 하는 것이다. 그러나 그러면 query와 output subspace의 수도 함께 줄어든다.

MQA는 다음을 분리한다.

```text
query/output diversity : H개 유지
stored memory vectors  : 1개로 축소
```

논문 실험에서도 1/2/4-head MHA를 parameter-matched하게 만든 baseline보다 MQA가 품질을 더 잘 유지했다. Memory traffic에 직접 관련된 KV head만 줄이는 것이 효율적이라는 증거다.

## Decode memory bandwidth 분석

한 token을 생성할 때 layer마다 `K,V` 전체 prefix를 읽는다. MHA의 대략적인 byte는

```math
\mathrm{bytes}_{\mathrm{MHA}}\approx 2B H N D_h\,\mathrm{bytes\_per\_element}
```

이다. MQA는

```math
\mathrm{bytes}_{\mathrm{MQA}}\approx 2B N D_h\,\mathrm{bytes\_per\_element}
```

이므로 ideal bandwidth-bound 구간에서는 attention decode가 `H`배까지 빨라질 여지가 있다. 실제 latency에는 model weight read, Q projection, softmax, output projection, framework overhead가 있어 전체 속도는 그보다 작다.

MQA가 training FLOPs를 같은 비율로 줄이지 않는 이유도 여기 있다. Training에서는 Q head가 모두 있고 score `[H,N,N]`를 계산하므로 attention score 연산은 여전히 head 수에 비례한다. 큰 이득은 cache write/read가 지배하는 incremental inference에 집중된다.

## Encoder self-attention과 cross-attention

원 논문은 encoder self-attention, decoder self-attention, encoder-decoder attention을 모두 MQA로 바꾼 모델을 실험한다. 하지만 효율 동기는 주로 decoder의 반복 read다.

- Encoder self-attention은 한 번 병렬 계산되므로 bandwidth 이득이 제한적이다.
- Decoder self-attention은 매 생성 step prefix KV를 읽으므로 가장 큰 이득이다.
- Cross-attention은 encoder K/V를 모든 decoder step에서 반복 읽으므로 역시 MQA가 유리하다.

후속 GQA 논문은 encoder에는 적용하지 않고 decoder self/cross attention에 집중한다.

## 실험 설정

WMT14 English→German encoder-decoder Transformer를 사용한다. Baseline은 6 layer, `d_model=1024`, head 8개, head dimension 128이다. Parameter 수를 비슷하게 맞추기 위해 MQA 모델은 FFN dimension을 조정한다.

Billion Word language modeling에서도 decoder-only 모델을 비교한다. 품질과 속도를 같은 parameter budget 안에서 평가한다.

## 실험 결과: WMT14 품질

| Attention | dev ln(PPL) | dev BLEU | test BLEU beam 1 / 4 |
| --- | ---: | ---: | ---: |
| Multi-head, 8 heads | **1.424** | **26.7** | 27.7 / 28.4 |
| Multi-query, 8 query heads | 1.439 | 26.5 | 27.5 / **28.5** |
| Multi-head local | 1.427 | 26.6 | 27.5 / 28.3 |
| Multi-query local | 1.437 | 26.5 | 27.6 / 28.2 |

MQA의 dev perplexity와 BLEU는 약간 나쁘지만 차이는 작고, beam-4 test BLEU는 오히려 28.5로 가장 높았다. Local attention과 MQA를 함께 써도 동작해 두 최적화가 직교함을 보였다.

Head 수 또는 dimension만 줄인 MHA baseline은 dev BLEU가 약 25.8~26.2로 더 크게 하락했다.

## 실험 결과: Decode 속도

Sequence length 128, TPUv2에서 token당 amortized 비용은 다음과 같다.

| Attention | Training | Greedy inference: encoder + decoder | Beam-4: encoder + decoder |
| --- | ---: | ---: | ---: |
| Multi-head | 13.2 μs | 1.7 + 46 μs | 2.0 + 203 μs |
| Multi-query | 13.0 μs | 1.5 + **3.8 μs** | 1.6 + **32 μs** |
| Multi-head local | 13.2 μs | 1.7 + 23 μs | 1.9 + 47 μs |
| Multi-query local | 13.0 μs | 1.5 + **3.3 μs** | 1.6 + **16 μs** |

Training time은 거의 같지만 decoder incremental step은 greedy에서 약 `12×`, beam-4에서 약 `6×` 빨라졌다. 논문의 bandwidth 분석과 일치한다.

## 실험 결과: Billion Word LM

| Attention | dev perplexity |
| --- | ---: |
| MHA, 8 heads | **29.9** |
| MQA, 8 query heads | 30.2 |
| Reduced-head MHA variants | 30.9~31.2 |

MQA는 baseline보다 0.3 PPL 나쁘지만 cache 절감을 위해 전체 head 수를 줄이는 방법보다 훨씬 낫다.

## 장점과 기여

- Decode 병목을 FLOPs가 아니라 KV cache memory bandwidth로 정확히 진단했다.
- Query head와 KV head 수를 분리하는 간단한 architecture를 제안했다.
- KV cache 크기와 read traffic을 head 수 `H`배 줄였다.
- Training 비용은 거의 유지하면서 translation decode를 크게 가속했다.
- 이후 PaLM, GQA, 현대 LLM inference 설계의 출발점이 되었다.

## 한계와 비판적 관점

### 1. KV 표현력 감소

모든 query head가 같은 key/value basis를 공유한다. 특히 value head 하나는 head별로 다른 정보를 저장하는 MHA보다 capacity가 낮다.

### 2. 품질 손실과 training instability

원 실험에서는 작지만 일관된 perplexity 하락이 있다. 모델·task가 커지면 MQA가 MHA에 비해 hard benchmark에서 더 크게 떨어질 수 있으며 GQA가 이를 보완한다.

### 3. 전용 kernel 필요

Framework가 K/V를 head 수만큼 물리적으로 expand하면 bandwidth 이득이 사라진다. Shared KV broadcast를 직접 지원해야 한다.

### 4. Prefill 가속과 decode 가속은 다르다

긴 prompt prefill은 여전히 여러 query head의 quadratic score를 계산한다. MQA의 주 이득은 token-by-token decode와 cache capacity다.

### 5. Cache가 유일한 병목은 아니다

Batch가 작으면 model weight read도 지배한다. MQA만으로 end-to-end latency가 ideal `H×` 개선되지는 않는다.

## MHA·MQA·GQA 관계

```math
\begin{aligned}
\mathrm{MHA}:&\quad \#\mathrm{KV\ heads}=H,\\
\mathrm{GQA}:&\quad \#\mathrm{KV\ heads}=G,\quad 1<G<H,\\
\mathrm{MQA}:&\quad \#\mathrm{KV\ heads}=1.
\end{aligned}
```

MQA는 가장 작은 cache, MHA는 가장 큰 capacity, GQA는 그 사이의 조절 가능한 절충이다.

## 구현 체크리스트

- K/V projection output에 head 축이 실제로 1인가?
- Cache allocator도 `[B,N,1,D_h]` 기준으로 줄었는가?
- Attention kernel이 K/V를 물리적으로 H번 복제하지 않는가?
- Query head `h`가 올바른 shared K/V에 broadcast되는가?
- Tensor parallel에서 shared KV head가 중복 저장/통신되지 않는가?
- Prefill과 decode latency를 분리해 측정했는가?
- Quality 비교에서 parameter 수와 training compute를 맞췄는가?

## 온디바이스 관점

온디바이스 LLM은 memory capacity와 DRAM bandwidth가 모두 작으므로 MQA의 이득이 직접적이다. KV cache가 `H`배 줄면 더 긴 context나 더 큰 batch를 같은 RAM에 넣을 수 있고, 매 token의 DRAM read energy도 감소한다. 구현도 irregular sparsity 없이 dense dot product와 broadcast로 구성할 수 있다.

반면 quality budget이 엄격한 assistant나 vision-language model에서는 KV head 하나가 병목이 될 수 있다. 소수 GQA group 또는 MLA와 latency·quality를 함께 비교하는 것이 실용적이다.

## 최종 평가

Multi-Query Attention의 공헌은 간단하지만 영향력이 크다. Autoregressive inference가 compute-bound가 아니라 **과거 K/V를 다시 읽는 memory-bandwidth-bound workload**라는 점을 짚고, query diversity는 유지하면서 cache의 head dimension만 제거했다. 약간의 품질 손실로 매우 큰 decode speedup을 얻으며, 이후 GQA와 MLA가 발전시킨 KV-cache 압축 연구의 기준점이 되었다.

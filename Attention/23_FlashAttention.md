# 23. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

## 논문 정보

- 제목: **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**
- 저자: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
- 발표: NeurIPS 2022
- 핵심 키워드: exact attention, IO awareness, tiling, online softmax, recomputation, HBM, SRAM

## 한눈에 보는 요약

FlashAttention은 attention의 `O(n²)` FLOPs를 줄이지 않는다. 대신 GPU의 느리고 큰 HBM과 빠르고 작은 on-chip SRAM 사이에서 오가는 byte 수를 줄여 **표준 softmax attention을 정확하게** 더 빠르고 적은 memory로 계산한다.

표준 구현은 `S=QK^T`, `P=softmax(S)`, `O=PV`를 별도 kernel로 실행하면서 `[n,n]` 행렬 `S,P`를 HBM에 쓰고 다시 읽는다. FlashAttention은 Q/K/V를 tile로 SRAM에 올리고, score·softmax·value aggregation을 한 fused kernel 안에서 처리한다. `[n,n]` attention matrix는 HBM에 저장하지 않는다.

Softmax는 한 row 전체가 있어야 normalization할 수 있지만, **online softmax**의 running maximum과 running sum을 유지하면 key block을 차례로 보면서도 정확한 결과를 얻을 수 있다. backward에서는 저장하지 않은 attention probability를 Q/K/V와 softmax 통계로 다시 계산한다.

결과적으로 FLOPs는 여전히 `O(n²d)`지만 추가 activation memory는 `O(n)` 수준이고, HBM IO가 크게 줄어 실제 wall-clock이 빨라진다.

<p align="center"><img src="https://github.com/user-attachments/assets/43b88503-21cb-4369-9b4c-05c62a31c442" alt="FlashAttention IO-aware tiling" width="760"></p>
<p align="center"><sub>원 논문 Figure 1 핵심 패널 — HBM-SRAM 데이터 이동을 줄이는 FlashAttention tiling</sub></p>

## 문제의식: FLOPs만 줄여서는 충분하지 않다

GPU memory hierarchy는 비대칭이다. 논문이 예로 든 A100은 HBM이 수십 GB이고 bandwidth가 약 1.5 TB/s인 반면, SRAM은 SM 전체 합계가 수십 MB로 작지만 약 19 TB/s 수준으로 훨씬 빠르다.

표준 attention은 다음 과정을 거친다.

```text
kernel 1: Q,K를 HBM에서 읽음 -> S=QK^T -> S를 HBM에 씀
kernel 2: S를 읽음 -> P=softmax(S) -> P를 HBM에 씀
kernel 3: P,V를 읽음 -> O=PV -> O를 HBM에 씀
```

연산량이 많더라도 Tensor Core GEMM은 매우 빠르다. 오히려 거대한 `S,P`를 HBM에 쓰고 읽는 시간이 병목이 된다. 일부 approximate attention은 FLOPs를 줄여도 irregular memory access와 kernel overhead 때문에 실제로 빠르지 않았다는 것이 논문의 출발점이다.

## 표준 attention과 정확성 목표

```math
\begin{aligned}
S&=\frac{QK^{\top}}{\sqrt d}&&\in\mathbb{R}^{N\times N},\\
P&=\operatorname{softmax}_{\mathrm{row}}(S)&&\in\mathbb{R}^{N\times N},\\
O&=PV&&\in\mathbb{R}^{N\times d}.
\end{aligned}
```

FlashAttention이 출력해야 하는 `O`는 위 식과 동일하다. sparsity, low-rank, random feature를 사용하지 않는다. 차이는 `[N,N]` 중간 결과를 materialize하지 않는 계산 순서뿐이다.

## Tiling 구조

Q를 row block `Q_i`, K/V를 column block `K_j,V_j`로 나눈다.

```math
\begin{aligned}
Q_i&\in\mathbb{R}^{B_r\times d},
&K_j&\in\mathbb{R}^{B_c\times d},\\
V_j&\in\mathbb{R}^{B_c\times d},
&S_{ij}=Q_iK_j^{\top}&\in\mathbb{R}^{B_r\times B_c}.
\end{aligned}
```

한 key/value block을 SRAM에 올리고 여러 Q block을 순회한다. 각 Q block에 대해 작은 `S_ij`만 SRAM에서 만들고 softmax state와 output accumulator를 update한 뒤 버린다.

```text
for K_j, V_j block:
    load K_j, V_j HBM -> SRAM
    for Q_i block:
        load Q_i and row statistics
        compute S_ij in SRAM
        update exact softmax output
        write only O_i and statistics
```

Block size는 `Q_i,K_j,V_j,S_ij,O_i`가 SRAM에 맞도록 선택한다. SRAM이 클수록 더 큰 tile을 사용해 HBM 왕복을 줄일 수 있다.

## Online softmax

한 query row의 score를 key block별로 `x^(1), x^(2), ...`라 하자. 지금까지 본 block의 running maximum `m`, exponential sum `l`, unnormalized output accumulator를 유지한다.

새 block score `x`가 들어오면

```math
\begin{aligned}
m_{\mathrm{new}}&=\max\!\left(m_{\mathrm{old}},\max(x)\right),\\
l_{\mathrm{new}}&=\exp\!\left(m_{\mathrm{old}}-m_{\mathrm{new}}\right)l_{\mathrm{old}}
+\sum\exp\!\left(x-m_{\mathrm{new}}\right).
\end{aligned}
```

이다. 기존 exponential은 기준 maximum이 `m_old`에서 `m_new`로 바뀌므로 `exp(m_old-m_new)`로 rescale한다.

Normalized output을 직접 유지한다면 새 block의 value contribution과 기존 output을 동일한 scale로 합친다.

```math
\begin{aligned}
\tilde P&=\exp\!\left(x-m_{\mathrm{new}}\right),\\
O_{\mathrm{new}}
&=\frac{l_{\mathrm{old}}\exp\!\left(m_{\mathrm{old}}-m_{\mathrm{new}}\right)O_{\mathrm{old}}
+\tilde P V_{\mathrm{block}}}{l_{\mathrm{new}}}.
\end{aligned}
```

모든 key block을 처리한 뒤 `m,l,O`는 전체 row softmax와 정확히 같다. 순서를 바꿔도 max-shifted exponential 합을 algebraically 재조합했기 때문이다.

## Tensor state

HBM에 유지하는 핵심 state는 다음뿐이다.

```text
Q,K,V,O : [N,d]
m       : [N]      # row maximum
l       : [N]      # row exponential sum
```

실제 구현은 backward를 위해 `logsumexp = m + log(l)` 형태를 저장할 수 있다. `[N,N]` `S`나 `P`는 저장하지 않는다.

## Causal mask와 dropout

### Causal attention

Query block과 key block의 원래 position을 비교한다.

- 완전히 미래인 block은 계산 자체를 건너뛴다.
- diagonal boundary block만 element-wise triangular mask를 적용한다.
- 완전히 과거인 block은 mask 없이 계산한다.

이는 dense `[N,N]` mask를 만들지 않고 causal semantics를 보존한다.

### Dropout

Attention dropout mask를 저장하면 quadratic memory가 다시 생긴다. FlashAttention은 random seed/state를 저장하고 backward에서 동일한 pseudo-random mask를 재생성한다. forward와 backward RNG mapping이 정확히 같아야 한다.

## Backward: 저장 대신 재계산

일반 attention backward는 `P`를 저장해 `dV=P^T dO`, `dP=dO V^T`, `dS=dsoftmax(dP)`를 계산한다. FlashAttention은 Q/K/V/O와 row logsumexp를 tile로 읽어 `S_ij,P_ij`를 다시 계산한다.

Softmax gradient의 row reduction은 다음 identity를 이용한다.

```math
\begin{aligned}
dS_{ij}&=P_{ij}\left(dP_{ij}-D_i\right),\\
D_i&=\sum_jP_{ij}dP_{ij}
=\operatorname{dot}(dO_i,O_i).
\end{aligned}
```

`D_i`를 `dO_i`와 forward output `O_i`에서 구할 수 있으므로 전체 `P`를 저장하지 않아도 된다. FLOPs는 재계산 때문에 늘지만 HBM traffic이 크게 줄어 backward도 빨라질 수 있다. 이 논문이 보여주는 핵심은 compute를 조금 더 써서 memory IO를 줄이는 trade-off다.

## 복잡도

### FLOPs와 memory

```math
\begin{aligned}
\mathrm{FLOPs}&:\ O(N^2d)\quad\text{(표준 attention과 동일한 차수)},\\
\text{추가 HBM memory}&:\ O(N)\quad\text{(}N^2\text{ attention matrix 없음)}.
\end{aligned}
```

### IO complexity

Head dimension `d`, SRAM 크기 `M`일 때 논문은 다음을 보인다.

```math
\begin{aligned}
\text{standard attention HBM access}&:\ \Theta(Nd+N^2),\\
\text{FlashAttention HBM access}&:\ \Theta\!\left(\frac{N^2d^2}{M}\right).
\end{aligned}
```

일반적인 `d <= M <= Nd` 범위에서 FlashAttention이 훨씬 적은 HBM access를 사용한다. 또한 모든 SRAM 크기 범위에서 동시에 이를 asymptotically 개선하는 exact attention algorithm은 없다는 lower-bound 성질을 제시한다.

이 결과는 FlashAttention이 FLOP-optimal이라는 뜻이 아니라, 특정 memory hierarchy에서 IO-aware하다는 뜻이다.

## Block-sparse FlashAttention

미리 정한 block mask가 있다면 nonzero tile만 같은 fused algorithm으로 계산한다. sparsity ratio를 `s`라 할 때 HBM IO가 대략 그 비율만큼 더 줄어든다. 이 변형은 full attention의 exact output이 아니라 지정된 sparse attention graph 안에서 exact다.

FlashAttention의 tiling은 sparsity와 경쟁하는 아이디어가 아니라 sparse attention도 hardware-efficient하게 실행하는 하부 kernel 원리다.

## 실험 결과: Attention kernel

GPT-2 medium의 길이 1,024 설정에서 표준 attention은 약 `40.3 GB`, FlashAttention은 약 `4.4 GB`의 HBM read/write를 보였고 attention 계산은 최대 `7.6×` 빨라졌다. Block size를 키워 HBM access가 줄어드는 구간에서 runtime도 함께 감소해 IO가 병목이라는 가설을 검증한다.

GPU와 shape에 따라 A100에서 대체로 `2~4×`, RTX 3090에서 약 `2.5~4.5×` attention speedup을 보고한다. Head dimension 128에서는 SRAM에 맞추기 위해 tile이 작아져 이득이 줄 수 있다.

## 실험 결과: End-to-end training

### BERT-large

Sequence length 512의 BERT-large 학습에서 MLPerf 1.1 기록 대비 약 `15%` end-to-end speedup을 보였다. 짧은 길이에서도 fused IO 절감이 전체 training time에 영향을 준다는 결과다.

### GPT-2

| 모델 | 구현 | PPL | 학습 시간 |
| --- | --- | ---: | ---: |
| GPT-2 small | HuggingFace | 18.2 | 9.5일 |
| GPT-2 small | Megatron-LM | 18.2 | 4.7일 |
| GPT-2 small | FlashAttention | 18.2 | **2.7일** |
| GPT-2 medium | HuggingFace | 14.2 | 21.0일 |
| GPT-2 medium | Megatron-LM | 14.3 | 11.5일 |
| GPT-2 medium | FlashAttention | 14.3 | **6.9일** |

Approximation 없이 perplexity를 유지하면서 HuggingFace 대비 최대 약 `3×`, Megatron 대비 약 `1.7~1.8×` 빨라졌다.

### 더 긴 context

GPT-2 small에서 context를 1K에서 4K로 늘리면 perplexity가 `18.2 → 17.5`로 개선되었다. FlashAttention 4K 학습은 3.6일로, Megatron 1K의 4.7일보다도 약 30% 빨랐다. 빠른 kernel이 단순 비용 절감뿐 아니라 더 긴 context라는 model capability를 가능하게 한 사례다.

### Long Range Arena

길이 1K~4K의 LRA에서 약 `2.4×` end-to-end speedup을 보고한다. Path-X 16K에서 `61.4%`, block-sparse 변형으로 Path-256 64K에서 `63.1%`를 기록해 당시 Transformer가 chance를 넘기 어려웠던 긴 문제를 처리했다.

## 장점과 기여

- attention을 FLOP 수가 아니라 memory hierarchy와 IO 관점에서 재정의했다.
- online softmax와 tiling으로 full softmax attention을 정확히 계산한다.
- `[N,N]` activation 저장을 제거해 memory를 sequence length에 선형으로 줄였다.
- backward recomputation이 IO-bound GPU에서는 실제로 더 빠를 수 있음을 보였다.
- 이론 IO bound, CUDA kernel, end-to-end 품질을 함께 검증했다.

## 한계와 비판적 관점

### 1. FLOPs는 여전히 quadratic

Memory cap은 크게 완화하지만 매우 긴 context에서 `N²d` compute 자체가 병목이 된다. 1M-token dense attention의 근본 연산량을 없애지는 않는다.

### 2. Hardware·shape 의존

Head dimension, sequence length, GPU SRAM과 bandwidth에 따라 optimal tile과 speedup이 달라진다. 작은 shape에서는 kernel launch와 setup overhead가 우세할 수 있다.

### 3. 구현 복잡도

Online softmax rescaling, dropout RNG, causal boundary, backward gradient를 fused CUDA kernel에서 정확히 구현해야 한다. framework-level reference보다 검증 부담이 크다.

### 4. Training과 decode는 다른 병목

논문의 중심은 parallel full-sequence attention이다. Autoregressive decode에서는 query가 한 개이고 KV cache read bandwidth가 주요 병목이므로 MQA/GQA/MLA 같은 별도 기법이 중요하다.

## 구현 체크리스트

- row maximum과 sum을 새 block maximum 기준으로 모두 rescale하는가?
- old output accumulator도 같은 scale로 보정하는가?
- causal mask에서 완전 미래 tile을 skip하고 diagonal tile만 mask하는가?
- backward가 forward와 동일한 dropout mask를 재생성하는가?
- softmax 통계와 accumulation을 FP32로 유지하는가?
- `[N,N]` tensor가 autograd graph에 숨어서 생성되지 않는가?
- kernel benchmark와 end-to-end speedup을 구분해 측정했는가?

## 온디바이스 관점

FlashAttention의 원리는 GPU에 한정되지 않는다. 작은 on-chip SRAM/scratchpad와 느린 외부 DRAM을 가진 NPU에서도 tile fusion으로 external-memory traffic을 줄일 수 있다. 다만 device가 dynamic loop, online reduction, fused exp를 지원해야 한다.

Vision의 고해상도 patch attention에서 activation memory를 크게 줄일 수 있지만 compute는 여전히 quadratic이다. 따라서 작은/중간 resolution에는 exact tiled attention, 더 큰 resolution에는 window·sparse attention을 결합하는 것이 현실적이다.

## 최종 평가

FlashAttention은 “attention을 근사해야만 빠르게 만들 수 있다”는 전제를 뒤집었다. 수학적 output은 그대로 두고, tile·online softmax·recomputation으로 HBM IO와 activation storage를 줄여 실제 GPU 시간을 개선했다. 이후 attention 연구에서 FLOPs와 wall-clock, algorithm과 memory hierarchy를 함께 보게 만든 전환점이다. 단, compute complexity는 여전히 quadratic이며 decoding KV cache 문제는 별도의 축이다.

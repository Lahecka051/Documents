# 16. Generating Long Sequences with Sparse Transformers

## 논문 정보

- 제목: **Generating Long Sequences with Sparse Transformers**
- 저자: Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever
- 발표: 2019, OpenAI technical report
- 핵심 키워드: sparse attention, strided attention, fixed attention, block-sparse kernel, autoregressive modeling

## 한눈에 보는 요약

Sparse Transformer는 모든 과거 token을 보는 dense causal attention을 구조화된 sparse edge로 대체한다. 길이 `n`에서 dense attention은 `O(n²)`이지만, 논문은 각 token이 약 `O(sqrt(n))`개의 위치만 보도록 설계해 attention 연산을 `O(n sqrt(n))`으로 줄인다.

핵심은 무작위로 edge를 지우는 것이 아니다. sparse graph를 여러 layer에 쌓았을 때 모든 과거 위치의 정보가 짧은 경로로 현재 위치에 도달하도록 **factorized attention pattern**을 설계한다. 이미지와 audio에는 주기적 구조를 이용하는 strided pattern, text에는 block의 끝 요약 위치를 참조하는 fixed pattern을 사용한다.

논문은 sparse pattern 외에도 100층이 넘는 Transformer를 학습하기 위한 pre-activation residual, residual branch 초기화, activation recomputation, mixed precision, block-sparse CUDA kernel을 함께 제시한다. 즉 알고리즘과 시스템 설계가 결합된 초기 장문 sequence 연구다.

<p align="center"><img src="https://github.com/user-attachments/assets/1b741c59-5e19-4520-8bd9-0ab23024cfab" alt="Sparse Transformer attention patterns" width="760"></p>
<p align="center"><sub>Figure 3 — full attention과 strided·fixed sparse pattern 비교</sub></p>

## 문제 설정

Autoregressive Transformer는 위치 `i`에서 과거 집합 `{0, …, i}`를 모두 본다.

```math
A_i=\left\{j\mid j\le i\right\}
```

각 layer의 score 수는 약 `n²/2`이고, activation memory도 quadratic이다. 긴 이미지 raster sequence, raw audio, byte-level text에서는 sequence가 수만에서 백만까지 늘어 dense attention을 적용하기 어렵다.

논문의 목표는 다음 두 조건을 함께 만족하는 attention graph를 만드는 것이다.

1. 각 위치가 직접 보는 key 수는 `n`보다 훨씬 작다.
2. 여러 layer를 거치면 임의의 과거 위치 정보가 현재 위치까지 전달된다.

## Factorized self-attention

저자들은 full attention 집합을 `p`개의 sparse pattern으로 분해한다.

```math
A_i=A_i^{(1)}\cup A_i^{(2)}\cup\cdots\cup A_i^{(p)}
```

각 pattern에서 위치당 edge 수를 대략 `O(n^(1/p))`로 만들고, `p`번의 attention step으로 모든 위치가 연결되도록 한다. `p=2`일 때 각 위치가 약 `sqrt(n)`개를 보고 전체 복잡도는 `O(n sqrt(n))`이다.

논문이 요구하는 직관적 연결 조건은 다음과 같다.

```text
모든 j <= i에 대해,
j의 정보가 p+1개 이하의 attention step으로 i에 도달해야 한다.
```

한 layer가 local edge를, 다음 layer가 long-range edge를 사용하면 직접 연결되지 않은 token도 중간 token을 통해 정보를 교환한다.

## Strided attention

Strided pattern은 이미지나 audio처럼 regular structure가 있는 sequence를 위한 패턴이다. stride `l ≈ sqrt(n)`을 잡고 각 위치가 다음 두 종류의 과거 위치를 본다.

```text
1. 최근 l개 위치: local context
2. 현재 위치와 l 간격으로 떨어진 위치: strided context
```

개념적으로는

```math
\begin{aligned}
A_i^{(\mathrm{local})}
&=\left\{j:i-l<j\le i\right\},\\
A_i^{(\mathrm{stride})}
&=\left\{j:j\le i\ \text{and}\ (i-j)\bmod l=0\right\}.
\end{aligned}
```

이다. 2D 이미지를 row-major로 펼치고 `l`을 한 row 길이와 맞추면 local edge가 같은 row의 가까운 pixel을, stride edge가 같은 column의 pixel을 연결한다. 두 pattern을 한두 layer 거치면 전체 이미지의 정보가 전달된다.

장점은 규칙적이라 block-sparse kernel로 구현하기 쉽다는 점이다. 단점은 sequence flatten 순서와 데이터 구조가 맞지 않으면 중요한 관계가 sparse graph에서 멀어질 수 있다는 점이다.

## Fixed attention

Text에서는 `l` 간격의 동일 residue class가 특별한 의미를 갖지 않는다. 저자들은 sequence를 길이 `l`의 block으로 나누고 다음 pattern을 사용한다.

```text
local head:
  현재 block의 이전 token을 본다.

fixed head:
  이전 block들에서 미리 정한 c개의 summary position을 본다.
```

논문의 text 실험에서는 대체로 `l=128` 또는 `256`, block당 summary 수 `c=8,16,32`를 사용한다. summary 위치는 각 block 끝부분에 고정된다. 이 token들이 block 내부 정보를 먼저 모으고, 뒤의 block이 그 summary를 참조하면서 장거리 정보가 이동한다.

“fixed”는 attention weight가 고정된다는 뜻이 아니다. **참조할 수 있는 위치 집합이 고정**되고, 그 안의 score는 학습된다.

## Pattern을 layer와 head에 배치하는 방법

논문은 세 가지 구성을 논의한다.

- layer마다 한 pattern만 사용하고 pattern을 교대로 배치한다.
- 한 layer에서 여러 pattern의 key를 합쳐 하나의 attention을 계산한다.
- multi-head의 head마다 서로 다른 pattern을 할당한다.

실제 표현력은 edge 수뿐 아니라 이 배치에 좌우된다. local head만 연속되면 receptive field가 느리게 커지고, long-range head가 너무 적으면 summary 위치가 병목이 된다.

## Tensor와 mask 관점

Dense causal attention은 논리적으로 다음 tensor를 만든다.

```text
Q,K,V : [B, H, N, D]
score : [B, H, N, N]
mask  : lower triangular [N,N]
```

Sparse Transformer는 각 query block에 허용된 key block만 계산한다.

```text
query block               : [B, H, Bq, D]
selected key/value blocks : [B, H, S, Bk, D]
local score blocks        : [B, H, Bq, S*Bk]
```

중요한 점은 dense `[N,N]`을 만든 뒤 mask로 0을 만드는 것이 아니라, 허용된 block만 kernel에서 계산해야 실제 메모리와 시간 이득이 생긴다는 것이다.

## 깊은 네트워크 학습 기법

Sparse attention만으로 100층 이상 모델이 안정적으로 학습되는 것은 아니다. 논문은 다음 변경을 사용한다.

### Pre-activation residual

LayerNorm과 activation을 residual branch 전에 배치한다.

```math
\begin{aligned}
x&\leftarrow x+\mathrm{Attention}(\mathrm{LN}(x)),\\
x&\leftarrow x+\mathrm{FFN}(\mathrm{LN}(x)).
\end{aligned}
```

이는 gradient가 identity path를 통해 흐르도록 돕는다. 이후 pre-LN Transformer의 일반적인 형태와 연결된다.

### Residual branch 초기화

깊이가 `N`인 모델에서 residual branch의 output projection을 대략 `1/sqrt(2N)`으로 축소 초기화한다. 논문에서는 attention의 `W_p`와 feed-forward의 두 번째 projection `W_2`에 적용한다. 깊이가 커져 residual 합의 분산이 폭증하는 것을 완화한다.

### Activation recomputation과 mixed precision

backward에 필요한 attention/FFN activation을 모두 저장하지 않고 다시 계산한다. 계산은 늘지만 peak memory가 줄어 더 긴 sequence와 큰 모델을 학습할 수 있다. FP16 연산과 FP32 master weight/accumulation도 사용한다.

### 학습된 다차원 positional embedding

이미지를 flatten하더라도 row와 column 위치를 별도 embedding으로 표현한다. 이는 strided graph의 구조와 원래 데이터 차원의 대응을 모델에 제공한다.

## Block-sparse kernel

일반 framework에서 irregular gather/scatter로 sparse attention을 만들면 FLOPs는 줄어도 GPU utilization이 낮다. 저자들은 attention pattern을 고정 크기 block으로 묶은 custom CUDA kernel을 구현했다.

```text
dense: 모든 query-key tile 계산
sparse: mask가 허용한 tile만 GEMM
```

따라서 논문의 성능은 이론 `O(n sqrt(n))`뿐 아니라 pattern의 block regularity에 의존한다. 현대 하드웨어에서도 동일한 교훈이 유지된다. sparse edge가 많지 않다는 사실과 실제 kernel이 빠르다는 사실은 별개다.

## 실험 결과

### CIFAR-10 density modeling

Pixel을 autoregressive sequence로 생성하는 실험에서 Sparse Transformer는 `2.80 bits/dim`을 기록해 당시 PixelSNAIL의 `2.85`보다 개선했다. 단순히 sequence를 짧게 자르지 않고 이미지 전체의 장거리 구조를 sparse path로 연결할 수 있음을 보였다.

### ImageNet 64×64

ImageNet 64×64 density modeling에서 `3.44 bits/dim`을 기록해 이전 최고치 약 `3.52`를 개선했다. 이미지 한 장을 수천 길이 sequence로 다루는 상황에서 strided pattern이 유효했다.

### Enwik8

| 모델 | BPC | iteration time |
| --- | ---: | ---: |
| Dense Transformer | 1.00 | 1.31 |
| Sparse fixed | **0.99** | 0.55 |
| Sparse strided | 1.13 | **0.35** |
| Transformer-XL | 0.99/1.03 | - |

Fixed pattern은 text에 맞는 summary routing 덕분에 dense와 비슷하거나 나은 품질을 더 짧은 iteration time으로 달성했다. 반대로 strided pattern은 빠르지만 text 구조와 맞지 않아 BPC가 악화되었다. 이는 sparse topology가 단순한 계산 옵션이 아니라 inductive bias라는 직접적인 증거다.

### Raw audio

| Sequence length | Model size | BPC |
| ---: | ---: | ---: |
| 65,536 | 152M | 1.97 |
| 262,144 | 25M | 2.17 |
| 1,048,576 | 3M | 2.99 |

백만 길이 sequence도 계산 가능함을 보였지만, 길이가 커질수록 메모리 제약 때문에 모델 크기를 줄였고 품질도 나빠졌다. 이 표는 “백만 token을 처리할 수 있다”와 “동일한 용량·품질로 처리한다”가 다르다는 점을 보여준다.

## 실험의 의미

가장 설득력 있는 부분은 task마다 pattern의 적합성이 다르다는 결과다. text에서 fixed가 strided보다 훨씬 낫고, 이미지에서는 2D 구조와 맞춘 stride가 효과적이다. 따라서 보편적인 sparse mask 하나가 dense attention을 대체한다고 주장하기보다, 데이터의 topology를 attention graph에 넣는 방식으로 읽는 것이 정확하다.

반면 당시 baseline과 training recipe가 현대 long-context 모델과 다르고, custom kernel의 절대 속도도 최신 FlashAttention 계열과 직접 비교하기 어렵다.

## 장점과 기여

- 구조화된 sparse attention으로 `O(n²)`을 `O(n sqrt(n))`으로 낮췄다.
- 여러 layer에서 전체 정보가 연결되도록 graph path 관점의 설계 원칙을 제시했다.
- 이미지·text에 다른 sparse topology가 필요함을 실험으로 보였다.
- 100층 이상의 Transformer를 위한 pre-activation과 초기화 기법을 함께 정리했다.
- block-sparse kernel, recomputation, mixed precision까지 포함해 장문 학습을 실제로 구현했다.

## 한계와 비판적 관점

### 1. Pattern이 데이터와 맞아야 한다

고정된 stride나 summary 위치가 실제 dependency와 어긋나면 필요한 token 사이 경로가 길어진다. Enwik8 strided 결과가 이를 보여준다.

### 2. 전역 정보는 여러 layer를 거친다

Dense attention은 한 layer에서 임의의 두 token을 직접 연결하지만, sparse graph는 중간 token을 거친다. 같은 global dependency를 표현하려면 더 깊은 network가 필요할 수 있다.

### 3. Custom kernel 의존성이 크다

이론적 sparsity를 실제 속도로 바꾸려면 특정 block size와 GPU에 최적화된 kernel이 필요하다. 작은 batch나 mobile accelerator에서는 지원이 없을 수 있다.

### 4. Autoregressive cache가 단순하지 않다

추론 시 각 새 token이 참조하는 sparse key 위치를 관리해야 하고, fixed summary가 충분한 정보를 보유하는지도 학습에 의존한다. dense KV cache처럼 단순한 contiguous append만으로 끝나지 않을 수 있다.

### 5. `O(n sqrt(n))`은 여전히 super-linear다

백만 길이에서는 `sqrt(n)`도 약 1,000이다. 처리 가능성은 늘지만 compute가 가벼워지는 것은 아니다.

## 후속 연구와의 연결

- **Longformer**: local window에 소수 global token과 dilation을 결합해 `O(nw)` 구조를 만든다.
- **BigBird**: local, random, global edge를 결합하고 graph 이론과 표현력 분석을 강화한다.
- **Routing/LSH attention**: content에 따라 sparse neighbor를 선택한다.
- **FlashAttention**: sparsity 없이 exact dense attention의 IO를 줄이는 다른 축이다.

Sparse Transformer는 이후 long-context 모델의 “local + long-range route”라는 설계 문법을 만든 초기 논문으로 볼 수 있다.

## 구현 체크리스트

- dense score를 계산한 후 mask하는 가짜 sparsity가 아닌가?
- sparse block의 크기가 accelerator tile과 맞는가?
- causal boundary에서 현재·미래 token이 섞이지 않는가?
- pattern을 바꿨을 때 graph diameter와 receptive field가 어떻게 변하는가?
- positional embedding이 원래 2D/다차원 구조를 보존하는가?
- recomputation으로 늘어난 학습 시간이 memory 절감과 균형을 이루는가?
- quality, FLOPs, wall-clock, peak memory를 모두 측정했는가?

## 온디바이스·비전 관점

고정 sparse pattern은 입력과 무관하므로 동적 routing보다 배포가 쉽다. 이미지 patch grid의 row/column 구조를 이용하면 segmentation이나 generation에서 장거리 context를 줄인 비용으로 추가할 수 있다. 다만 mobile NPU가 arbitrary block-sparse GEMM을 지원하지 않으면 dense tile 여러 개로 변환하면서 이득이 사라질 수 있다.

온디바이스에서는 완전히 irregular한 pattern보다 window와 stride가 정적으로 정해진 pattern, 또는 local convolution과 결합한 구조가 유리하다. 실제 선택은 이론 edge 수보다 compiler가 어떤 sparse layout을 native로 실행하는지에 달려 있다.

## 최종 평가

Sparse Transformer의 핵심 공헌은 “attention을 줄인다”가 아니라 **어떤 sparse graph가 긴 sequence의 정보를 잃지 않고 전달하는가**를 계산 구조와 시스템 구현으로 함께 보여준 데 있다. 결과는 pattern이 task 구조와 맞을 때 dense attention에 가까운 품질을 훨씬 낮은 비용으로 낼 수 있음을 보여준다. 동시에 고정 topology, multi-hop 지연, custom kernel 의존이라는 한계도 분명하다. 이후 Longformer와 BigBird를 읽기 위한 가장 중요한 선행 연구다.

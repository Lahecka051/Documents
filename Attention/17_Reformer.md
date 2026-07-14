# 17. Reformer: The Efficient Transformer

## 논문 정보

- 제목: **Reformer: The Efficient Transformer**
- 저자: Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya
- 발표: ICLR 2020
- 핵심 키워드: locality-sensitive hashing, LSH attention, reversible residual layer, activation recomputation, long sequence

## 한눈에 보는 요약

Reformer는 Transformer의 두 가지 메모리 병목을 서로 다른 방법으로 해결한다.

1. attention score의 `O(n²)` 문제는 **LSH attention**으로 줄인다.
2. layer activation을 모두 저장하는 `O(nL)` 문제는 **reversible residual layer**로 줄인다.

LSH attention은 query와 key를 content 기반 hash bucket으로 묶고 같은 bucket의 token끼리만 attention한다. 유사한 vector가 같은 bucket에 들어갈 확률이 높다는 locality-sensitive hashing을 사용해 approximate nearest-neighbor search를 attention에 적용한다. 정렬 비용을 포함한 복잡도는 대략 `O(n log n)`이다.

Reversible layer는 출력으로부터 입력을 복원할 수 있게 residual block을 설계한다. backward에서 이전 activation을 저장하는 대신 다시 계산하므로 memory를 크게 줄인다. 여기에 feed-forward를 position chunk로 나누는 기법까지 결합해 수만 길이 sequence를 처리한다.

<p align="center"><img src="https://github.com/user-attachments/assets/d3bf37f5-0053-400b-8af3-2b49297dcf32" alt="Reformer LSH attention process" width="760"></p>
<p align="center"><sub>원 논문 Figure 2 — LSH bucket·sort·chunk로 attention 후보를 좁히는 Reformer</sub></p>

## 표준 Transformer의 세 메모리 항

길이 `n`, hidden dimension `d`, layer 수 `L`인 Transformer의 학습 memory를 단순화하면 다음 세 항으로 볼 수 있다.

```math
\begin{aligned}
\text{attention logits/weights} &:\ O(Ln^2),\\
\text{layer activations} &:\ O(Lnd),\\
\text{feed-forward activations} &:\ O(Lnd_{ff}).
\end{aligned}
```

긴 sequence에서는 `n²`이 먼저 병목이지만, attention을 선형화한 뒤에도 깊은 layer의 activation 저장이 남는다. Reformer는 이 항들을 각각 LSH, reversible layer, chunking으로 대응한다.

## LSH attention의 출발점

표준 self-attention은 query `q_i`와 모든 key `k_j`의 내적을 계산한다. softmax는 큰 내적을 가진 key에 질량을 집중시키므로, 출력에 가장 큰 영향을 주는 것은 query와 가까운 key다. Reformer는 모든 pair를 검사하는 대신 approximate nearest neighbor를 먼저 찾는다.

논문은 query와 key projection을 공유한다.

```math
\begin{aligned}
&Q\text{와 }K\text{가 같은 projection을 공유},\\
k_j&=\frac{q_j}{\lVert q_j\rVert}.
\end{aligned}
```

이렇게 하면 query와 key가 같은 공간에 있고 angular similarity 기반 LSH를 사용할 수 있다. 자기 자신은 항상 가장 가까운 vector이므로, 다른 token을 찾도록 self-attention score를 보통 mask한다.

## Angular locality-sensitive hashing

무작위 rotation matrix `R`을 만들고 다음 hash를 사용한다.

```math
h(x)=\operatorname*{arg\,max}\left([xR;-xR]\right)
```

`xR`의 각 coordinate와 부호 반전 값을 이어 붙인 뒤 가장 큰 항의 index를 bucket으로 삼는다. 방향이 비슷한 vector는 같은 최대 coordinate를 가질 가능성이 높다.

한 번의 hash는 충돌 운에 민감하므로 `n_hashes`개의 독립 rotation을 사용한다. 여러 hash에서 같은 bucket에 들어간 key들의 합집합이 query의 candidate set이 된다.

## Hash → sort → chunk

각 token을 hash bucket id와 원래 sequence index로 정렬한다. 같은 bucket의 token이 memory에서 연속되므로 고정 길이 chunk 단위 dense attention으로 계산할 수 있다.

```text
1. 모든 q/k vector를 bucket id로 hash
2. (bucket id, sequence position) 기준 sort
3. 정렬된 sequence를 길이 m의 chunk로 분할
4. 각 chunk가 자기 chunk와 이전 chunk를 attention
5. 결과를 원래 순서로 unsort
```

왜 이전 chunk도 보는가? 하나의 bucket이 chunk 경계에서 잘릴 수 있기 때문이다. 같은 bucket token이 인접 두 chunk에 나뉘어도 서로 만날 수 있게 한다.

논문은 bucket 수를 `n_buckets`, 평균 bucket 크기를 `n/n_buckets`라 할 때 chunk 길이 `m ≈ 2n/n_buckets`를 사용한다. 너무 큰 bucket은 연산을 늘리고, 너무 작은 bucket은 중요한 neighbor가 boundary 밖으로 빠질 위험을 높인다.

## Causal LSH attention

Autoregressive modeling에서는 정렬 후에도 원래 sequence의 미래 token을 보지 않도록 해야 한다. bucket 정렬은 시간 순서를 섞으므로 causal mask는 **정렬된 index가 아니라 원래 position index**를 비교해 적용한다.

```math
\operatorname{allow}(i,j)=
\begin{cases}
1,&\text{same candidate bucket and }\operatorname{pos}(j)\le\operatorname{pos}(i),\\
0,&\text{otherwise}.
\end{cases}
```

동일 pair가 여러 hash round에서 선택되면 softmax normalization과 output을 중복 보정해야 한다. 구현의 정확성이 단순 local attention보다 훨씬 까다로운 이유다.

## LSH attention의 계산 복잡도

대략적인 비용은 다음과 같다.

```math
\begin{aligned}
\text{hash projection} &:\ O(ndn_{\mathrm{hashes}}),\\
\text{bucket sorting} &:\ O(n\log n),\\
\text{chunk attention} &:\ O(nmdn_{\mathrm{hashes}}).
\end{aligned}
```

bucket/chunk 크기를 상수 또는 `O(log n)` 수준으로 관리하면 전체를 `O(n log n)`으로 볼 수 있다. 그러나 실제 wall-clock은 sort, permutation, gather/scatter, 여러 hash round의 중복에 좌우된다. dense attention보다 FLOPs가 적어도 sequence가 짧으면 정렬 overhead가 더 클 수 있다.

## Approximation 품질을 좌우하는 요소

### Hash round 수

round가 많을수록 유사 pair가 최소 한 번 같은 bucket에 들어갈 확률이 커진다. 품질은 좋아지지만 계산과 memory가 비례해 늘어난다.

### Bucket 크기

큰 bucket은 candidate recall이 높지만 chunk attention이 비싸다. 작은 bucket은 빠르지만 충돌 누락과 경계 효과가 커진다.

### Shared Q/K

hashing을 쉽게 하지만 표준 attention처럼 query와 key가 독립 projection을 갖는 표현력은 포기한다.

### Randomness와 redraw

random rotation에 따라 candidate가 달라진다. seed, 학습/평가 시 hash 처리, distributed replica 간 일관성을 관리해야 한다.

## Reversible residual layer

일반 residual layer는

```math
y=x+F(x)
```

이며 `F`의 입력을 backward에 사용하려고 `x`를 저장한다. Reformer는 hidden state를 두 부분 `x1, x2`로 나누고 다음 coupling을 사용한다.

```math
\begin{aligned}
y_1&=x_1+F(x_2),\\
y_2&=x_2+G(y_1).
\end{aligned}
```

`F`는 attention, `G`는 feed-forward network다. 출력 `y1,y2`가 있으면 입력을 정확히 재구성할 수 있다.

```math
\begin{aligned}
x_2&=y_2-G(y_1),\\
x_1&=y_1-F(x_2).
\end{aligned}
```

따라서 각 layer의 activation을 저장하지 않고 최종 activation만 유지한 채, backward에서 `G`와 `F`를 다시 실행해 입력과 gradient를 복원한다.

### Memory와 compute trade-off

activation storage는 layer 수에 거의 선형으로 늘지 않지만, backward에서 forward 연산을 다시 하므로 계산량이 증가한다. dropout mask나 random hash처럼 stochastic 연산은 재계산 시 같은 결과를 재현해야 하므로 RNG state 관리가 필요하다. low precision에서는 반복적인 덧셈·뺄셈의 수치 오차도 점검해야 한다.

## Feed-forward chunking

Transformer의 FFN은 각 position에 독립적으로 적용된다.

```math
\operatorname{FFN}(X)[i]=W_2\,\operatorname{activation}\!\left(W_1X[i]\right)
```

따라서 모든 `n` position의 `d_ff` activation을 동시에 만들 필요가 없다. sequence를 여러 chunk로 나누어 FFN을 실행하고 결과를 concatenate한다.

```python
ys = []
for x_chunk in split(x, chunk_size, dim=sequence):
    ys.append(ffn(x_chunk))
y = concat(ys, dim=sequence)
```

출력은 동일하고 peak memory만 줄어든다. 대신 kernel launch 수가 늘고 작은 GEMM의 효율이 낮아질 수 있다.

## 전체 계산 흐름

```text
input embedding + position encoding
  ↓ split channels
LSH attention F(x2)
  ↓ reversible update y1
chunked FFN G(y1)
  ↓ reversible update y2
repeat without storing every layer input
  ↓
final normalization and output
```

Reformer의 “효율성”은 하나의 magic attention이 아니라 세 memory 기법의 결합이다.

## 실험 결과

### Synthetic duplication task

Sequence 앞부분의 symbol을 뒤에서 복사하는 synthetic task로 LSH recall을 분석한다. full attention으로 학습한 모델을 평가 시 LSH로 바꾸면 손실이 생기지만, LSH로 학습한 모델은 여러 hash round를 사용했을 때 거의 완벽하게 문제를 해결한다. 특히 4회 hash에서 성능이 크게 회복되고, 평가 때 hash 수를 늘리면 추가 개선되는 경향을 보인다.

이는 approximation을 학습 중부터 노출하면 모델이 hash 구조에 적응할 수 있음을 뜻한다. 반대로 pretrained full-attention weight를 무조건 LSH로 치환하면 같은 품질을 기대하기 어렵다.

### WMT machine translation

문장 길이가 대체로 128보다 짧아 translation 실험에서는 LSH를 사용하지 않고 reversible Transformer 자체를 비교한다. Base/Big 설정에서 일반 Transformer와 비슷한 BLEU를 유지하면서 activation memory를 줄인다. 이 실험은 LSH 품질보다 reversible architecture가 표준 task에서도 큰 손실 없이 작동하는지 확인하는 역할을 한다.

### Enwik8

최대 약 64K character context를 사용하는 모델에서 12-layer Reformer가 약 `1.05 bits/dim`을 기록한다. 당시 매우 긴 context를 단일 example로 처리하면서 경쟁력 있는 성능을 보였다는 의미가 크다.

### ImageNet 64×64

RGB channel을 sequence로 펼치면 길이가 약 12K가 된다. 6-layer Reformer가 12-layer baseline 계열과 경쟁적인 density modeling 결과를 보이며, 긴 image sequence에서도 LSH attention을 적용할 수 있음을 보였다.

## 실험을 읽는 관점

Reformer의 가장 강한 증거는 “표준 attention과 완전히 동일한 품질”이 아니라, 기존 GPU memory로 불가능하던 긴 input을 처리하면서 합리적인 품질을 얻었다는 점이다. 실험마다 LSH, reversibility, chunking이 모두 같은 정도로 사용된 것은 아니므로 기여를 분리해 읽어야 한다.

또한 이후 exact attention kernel과 long-context benchmark가 크게 발전했다. 당시의 속도 비교를 현대 GPU에 그대로 적용하기보다, sorting 기반 sparse attention의 구조적 장단점을 보는 것이 적절하다.

## 장점과 기여

- content-based LSH로 dense pair search를 approximate nearest-neighbor 문제로 바꿨다.
- hash bucket을 sort/chunk해 GPU에서 처리 가능한 계산 형태로 만들었다.
- reversible residual로 layer activation 저장을 제거했다.
- FFN chunking까지 결합해 attention 외 메모리 병목도 다뤘다.
- 수만 길이 text와 image sequence를 실제 학습으로 시연했다.

## 한계와 비판적 관점

### 1. Exact softmax attention이 아니다

다른 bucket의 key는 score가 아무리 높아도 후보가 되지 않는다. 여러 hash round가 recall을 높이지만 비용도 함께 증가한다.

### 2. Sorting과 permutation overhead

이론 복잡도는 낮지만 GPU에서 radix sort, gather, unsort가 필요하다. 짧거나 중간 길이에서는 잘 최적화된 dense attention보다 느릴 수 있다.

### 3. Randomness와 재현성

hash rotation과 bucket collision이 결과에 영향을 준다. reversible recomputation에서는 동일 random state를 복원해야 한다.

### 4. Shared Q/K의 제약

query와 key projection을 공유하는 것은 LSH를 단순화하지만 표준 attention의 자유도를 줄인다. self-score를 별도로 mask해야 하는 부작용도 있다.

### 5. Reversibility는 compute를 memory로 교환한다

activation을 저장하지 않는 대신 backward에서 attention과 FFN을 재실행한다. memory가 충분한 환경에서는 학습 시간이 더 중요할 수 있다.

### 6. Autoregressive inference 이점이 제한적일 수 있다

한 token씩 생성할 때 KV cache의 모든 key를 다시 sort/hash하는 비용과 bucket 관리를 고려해야 한다. training의 긴 parallel sequence 효율이 decode latency 개선과 동일하지 않다.

## 후속 연구와의 연결

- **Routing Transformer**는 clustering/routing으로 content-based sparse neighbor를 찾는다.
- **Reversible networks**는 이후 memory-efficient deep model에 널리 사용된다.
- **Performer/Linformer**는 sorting 대신 kernel 또는 low-rank factorization으로 선형화를 시도한다.
- **FlashAttention**은 approximate neighbor selection 없이 exact attention을 tile 단위로 계산한다.

현대 모델에서 Reformer 전체 구조가 표준이 되지는 않았지만, content-aware sparsity와 activation recomputation이라는 두 아이디어는 계속 재사용된다.

## 구현 체크리스트

- hash bucket 계산이 query/key에 동일하게 적용되는가?
- bucket boundary를 위해 이전 chunk까지 포함하는가?
- causal mask가 원래 position 기준으로 적용되는가?
- 여러 hash에서 중복된 pair의 확률을 보정하는가?
- reversible recomputation에서 dropout/RNG가 동일하게 재현되는가?
- FP16 재구성 오차가 layer 깊이에 따라 누적되지 않는가?
- sort/gather 비용을 포함한 wall-clock을 측정했는가?

## 온디바이스 관점

LSH attention은 theoretical memory는 좋지만 sort와 dynamic gather가 많은 점이 mobile NPU에 불리하다. 고정 local window보다 compiler 최적화가 어렵고, 입력마다 bucket 분포가 달라 execution shape가 불규칙해질 수 있다. 반면 reversible block과 FFN chunking은 memory가 제한된 training 또는 edge adaptation에서 여전히 유용하다.

실제 온디바이스 추론이라면 다음을 비교해야 한다.

```text
LSH의 sort/gather latency
local-window attention의 정적 kernel
FlashAttention류 exact tiled kernel 지원
recompute로 절약되는 RAM과 늘어나는 전력
```

## 최종 평가

Reformer는 attention 하나만 바꾼 논문이라기보다, 긴 sequence를 막는 memory 항을 해부하고 각각에 다른 해법을 붙인 시스템적 연구다. LSH attention은 nearest-neighbor 후보만 계산해 quadratic score를 피하고, reversible layer는 깊이에 따른 activation 저장을 없앤다. approximation과 sorting overhead 때문에 오늘날 모든 모델의 기본 구조가 되지는 않았지만, 긴 context 문제를 알고리즘·architecture·memory scheduling의 결합으로 풀어야 한다는 점을 매우 선명하게 보여준다.

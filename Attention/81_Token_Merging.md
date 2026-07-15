# 81. Token Merging: Your ViT But Faster

## 논문 정보

- 제목: Token Merging: Your ViT But Faster
- 저자: Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, Judy Hoffman
- 소속: Georgia Tech, Meta AI
- 발표: ICLR 2023
- arXiv: [2210.09461](https://arxiv.org/abs/2210.09461)
- 코드: [facebookresearch/ToMe](https://github.com/facebookresearch/ToMe)
- 원본 파일: `81_Token_Merging.pdf`

## 한눈에 보는 요약

Token Merging, 줄여서 ToMe는 Vision Transformer의 token을 버리는 대신 비슷한 token끼리 합쳐서 이후 block의 연산량을 줄이는 방법이다. 핵심은 이미 학습된 ViT의 attention key를 token 유사도 표현으로 재사용하고, 반복 clustering 대신 한 번의 bipartite matching으로 매 block에서 정확히 $r$개의 token을 줄이는 것이다. 고정된 수만큼 줄이므로 batch 안의 sample shape가 같고, 별도 학습 없이 기존 checkpoint에 삽입할 수도 있다.

논문의 가장 중요한 결과는 다음과 같다.

- off-the-shelf ViT-L/ViT-H에 적용해 image throughput을 약 2배 높이면서 큰 모델에서는 정확도 하락을 0.2-0.3% 수준으로 억제했다.
- video ViT-L에서는 2.2배 inference throughput과 약 절반의 fine-tuning 시간을 보고했다.
- audio ViT-B에서는 throughput을 거의 2배 높이면서 mAP 하락을 0.4%로 줄였다.
- ViT-L/16 MAE, ImageNet-1k, $r=8$ 조건에서 bipartite matching은 84.25% top-1, 182.9 images/s를 기록했다. baseline은 85.96%, 93.3 images/s다.
- merging은 parameter 수나 checkpoint 크기를 줄이지 않는다. 이 방법이 줄이는 것은 각 block 뒤에 전달되는 token 수와 그에 따른 attention, MLP, activation 비용이다.

ToMe를 한 문장으로 정리하면 다음과 같다.

> attention이 이미 만든 key 공간에서 서로 비슷한 patch token을 빠르게 찾아 크기 가중 평균으로 합치고, 합쳐진 token이 대표하는 patch 수를 다음 attention에 반영한다.

## 문제 설정: 왜 pruning이 아니라 merging인가

ViT의 입력 token 수를 $N$, hidden dimension을 $D$라고 하자. 한 block의 self-attention score 행렬은 head마다 $N\times N$이고, QKV projection과 MLP는 대체로 $N$에 선형으로 비례한다. 따라서 앞 block에서 token 수를 줄이면 뒤의 여러 block이 동시에 싸진다.

기존 token pruning은 중요도가 낮은 token을 제거한다. 그러나 제거된 token에 들어 있던 정보는 사라진다. reduction 비율이 커질수록 이 손실이 누적되며, 동적으로 sample마다 다른 token 수를 남기면 일반적인 dense batch로 묶기도 어렵다. training 중 실제 token을 제거하지 않고 mask나 padding을 쓰면 연산량 절감도 제한된다.

ToMe의 관찰은 vision token 사이에 중복이 많다는 것이다. 하늘, 잔디, 털처럼 유사한 영역은 여러 patch가 거의 같은 정보를 전달한다. 이 token들을 하나의 대표 token으로 평균하면 개별 위치의 세부 정보는 줄지만 전체 내용 자체를 버리는 것보다는 손실이 작다. 논문은 image뿐 아니라 인접 frame과 spectrogram에서도 이러한 중복이 존재함을 실험한다.

## Transformer block 안에서의 위치

표준 pre-norm ViT block을 단순화하면 다음과 같다.

$$
X' = X + \mathrm{MSA}(\mathrm{LN}(X)),
$$

$$
Y = X' + \mathrm{MLP}(\mathrm{LN}(X')).
$$

ToMe는 attention branch가 끝난 뒤, MLP branch가 시작되기 전에 들어간다.

$$
X' \longrightarrow \mathrm{Merge}(X', K, s) \longrightarrow \mathrm{MLP}.
$$

이 위치에는 두 가지 의미가 있다.

1. 사라질 token도 현재 block의 attention에는 참여하므로 다른 token에 정보를 전달할 기회가 있다.
2. 현재 attention에서 계산한 key $K$를 추가 projection 없이 유사도 feature로 재사용할 수 있다.

반대로 block 입구에서 먼저 줄이면 현재 block의 attention까지 싸지지만, merge될 token이 정보를 전파하지 못하고 이전 block의 feature만으로 pair를 정해야 한다. Table 1의 $X_{pre}$와 $X$ 비교는 attention 뒤에서 merge하는 설계가 더 정확하다는 근거다.

## Bipartite soft matching

### 입력과 shape

batch를 $B$, token 수를 $N$, hidden dimension을 $D$, head 수를 $H_a$, head dimension을 $d_h=D/H_a$라고 하자.

- block 입력 $X$: $[B,N,D]$
- attention key $K$: $[B,H_a,N,d_h]$
- head 평균 key: $[B,N,d_h]$
- token size $s$: $[B,N,1]$
- merge 후 token: $[B,N-r,D]$
- merge 후 size: $[B,N-r,1]$

CLS token처럼 보호할 token은 matching source에서 제외한다. 원문 구현은 첫 token의 matching score를 $-\infty$로 설정해 CLS token이 다른 token에 흡수되지 않도록 한다.

### 다섯 단계

각 block에서 token을 대략 같은 크기의 두 집합 $A$와 $B$로 나눈다. 기본 구현은 짝수 index와 홀수 index를 번갈아 배정한다.

1. $A$와 $B$로 token을 분할한다.
2. 각 $a_i\in A$에 대해 가장 cosine similarity가 높은 $b_j\in B$ 하나를 찾는다.
3. 모든 source edge 중 similarity가 가장 큰 $r$개만 남긴다.
4. 선택된 source token을 destination token에 size-weighted average로 합친다.
5. merge되지 않은 $A$ token과 갱신된 $B$ token을 concatenate한다.

head 평균 key를 $\bar K\in\mathbb{R}^{B\times N\times d_h}$라 하고 L2 normalize한 뒤,

$$
M = \mathrm{normalize}(K_A)\mathrm{normalize}(K_B)^\top
$$

를 계산한다. $M$의 shape는 대략 $[B,\lceil N/2\rceil,\lfloor N/2\rfloor]$다. 각 row에서 max를 구하고, 그 max score들의 top-$r$ source만 실제로 merge한다.

이 방식은 global one-to-one matching이 아니다. 여러 $A$ token이 같은 $B$ token을 목적지로 선택할 수 있다. 그래서 저자들은 hard matching이 아니라 soft matching이라고 부른다. 다만 한 번에 source의 일부만 이동시키므로 k-means처럼 한 cluster가 갑자기 지나치게 커지는 현상은 줄어든다.

### 왜 alternating partition인가

집합을 앞 절반과 뒤 절반으로 순차 분할하면 spatial ordering과 강하게 결합될 수 있다. alternating 분할은 이웃한 token들이 서로 다른 집합에 놓이므로 가까운 patch도 비교할 수 있고, concatenate 후 다음 layer에서는 다시 다양한 pair가 비교된다.

Table 1e의 결과는 이를 뒷받침한다.

| partition | top-1 | images/s |
|---|---:|---:|
| sequential | 81.07 | 183.0 |
| alternating | 84.25 | 182.9 |
| random | 83.80 | 181.7 |

속도 차이는 작지만 sequential은 정확도가 크게 낮다.

## Token size와 proportional attention

두 token $x_i,x_j$가 각각 $s_i,s_j$개의 원래 patch를 대표한다고 하자. merge 결과는 단순 평균이 아니라 다음 가중 평균이다.

$$
x_{ij}=\frac{s_i x_i+s_j x_j}{s_i+s_j},\qquad s_{ij}=s_i+s_j.
$$

초기에는 모든 patch token의 $s=1$이다. merge가 반복되면 token마다 대표하는 patch 수가 달라진다. 이 크기를 무시하면 원래 동일한 key가 여러 번 softmax에 등장하던 효과가 사라진다.

원래 동일한 key $k$가 $s$번 있었다면 query $q$에 대한 softmax 분모 기여는

$$
s\exp(q^\top k/\sqrt d)
=\exp(q^\top k/\sqrt d + \log s)
$$

이다. 따라서 ToMe는 attention logit에 $\log s$를 더한다.

$$
A=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt d}+\log s\right).
$$

$\log s$는 key axis로 broadcast된다. query $i$와 key $j$를 명시하면

$$
A_{ij}=\frac{\exp(q_i^\top k_j/\sqrt d+\log s_j)}
{\sum_t\exp(q_i^\top k_t/\sqrt d+\log s_t)}.
$$

이 보정은 동일한 key를 정확히 복제한 상황과 동등하지만, 서로 다른 token을 평균한 경우까지 완전히 복원하는 것은 아니다. 즉, proportional attention은 multiplicity 보정이지 lossless merge 증명이 아니다.

Table 1f에서는 pretraining recipe에 따라 효과가 달랐다.

| source | proportional attention | top-1 | images/s |
|---|---:|---:|---:|
| MAE | 없음 | 84.25 | 182.9 |
| MAE | 사용 | 83.84 | 180.9 |
| AugReg | 없음 | 82.15 | 182.8 |
| AugReg | 사용 | 83.51 | 180.8 |

저자들은 MAE가 pretraining 중 token을 이미 제거하기 때문에 off-the-shelf MAE 모델에는 보정이 불필요할 수 있다고 해석한다. 학습을 다시 하면 이 차이는 줄어든다. 따라서 구현에서는 checkpoint 계열별 ablation이 필요하다.

## 구현 pseudocode

아래는 논문 Appendix D의 핵심을 shape가 드러나도록 다시 쓴 pseudocode다.

```python
def make_merge(key, r, protect_cls=True):
    # key: [B, N, Ck], usually K averaged over heads
    key = key / norm(key, dim=-1, keepdim=True)

    a = key[:, 0::2, :]                 # [B, Na, Ck]
    b = key[:, 1::2, :]                 # [B, Nb, Ck]
    sim = a @ transpose(b, -1, -2)      # [B, Na, Nb]

    if protect_cls:
        sim[:, 0, :] = -inf

    best_score, best_dst = max(sim, dim=-1)  # [B, Na]
    order = argsort(best_score, descending=True)

    src_merge = order[:, :r]            # source indices in A
    src_keep = sort(order[:, r:])
    dst_merge = gather(best_dst, src_merge)

    def merge_tensor(x, reduce="sum"):
        # x can be feature [B,N,D] or size [B,N,1]
        src = x[:, 0::2, :]
        dst = x[:, 1::2, :]
        kept = gather_rows(src, src_keep)
        moved = gather_rows(src, src_merge)
        dst = scatter_reduce(dst, dst_merge, moved, reduce=reduce)
        return concat([kept, dst], dim=1)

    return merge_tensor
```

실제 weighted average는 feature 자체를 바로 평균하는 대신 $x\cdot s$와 $s$를 각각 sum-reduce한 뒤 나누면 안정적으로 구현할 수 있다.

```python
merge = make_merge(key, r)
new_size = merge(size, reduce="sum")
new_token = merge(token * size, reduce="sum") / new_size
```

top-$r$, gather, scatter-add가 모두 batch 병렬로 실행되며 $r$번 반복하는 Python loop가 없다. 다만 이 연산들이 모바일 NPU에서 native kernel로 지원된다는 뜻은 아니다.

## Merge schedule

### Constant schedule

기본 schedule은 모든 block에서 $r$개씩 줄인다.

$$
N_{\ell+1}=N_\ell-r.
$$

$L$개 block 전체에서 최대 $rL$개를 줄인다. sample 내용과 관계없이 같은 수를 줄이므로 batch shape가 고정된다. 논문은 15,000개의 random schedule을 비교해 constant schedule이 특히 reduction이 큰 구간에서 Pareto frontier에 가깝다고 보고한다.

### Decreasing schedule

속도를 더 높이기 위한 schedule은 첫 layer에서 $2r$, 마지막 layer에서 0을 줄이도록 선형 감소한다. 총 reduction은 constant와 같은 $rL$이지만 token을 일찍 줄여 뒤 block을 더 많이 절약한다. 그 대가로 초기 feature가 덜 성숙한 상태에서 더 공격적으로 합치므로 정확도 손실이 커질 수 있다.

## 복잡도와 activation memory 손계산

이 절의 숫자는 논문 표의 측정값이 아니라, 구조를 이해하기 위한 별도 계산이다.

### ViT-L/16, 224 입력 예시

$224\times224$ 이미지를 $16\times16$ patch로 나누면 patch는 $14\times14=196$개다. CLS token을 포함하면 초기 token 수는 $N_0=197$이다. ViT-L은 $L=24$, $D=1024$다.

$r=8$ constant schedule에서 각 block attention이 받는 token 수를 merge 전 기준으로 쓰면

$$
197,189,181,\ldots,21,13
$$

이고 마지막 merge 뒤에는 보호 token을 포함해 약 5개가 남는다.

baseline의 block 전체 token 합은

$$
24\times197=4728,
$$

ToMe는

$$
\sum_{\ell=0}^{23}(197-8\ell)=2520
$$

이다. 따라서 QKV projection과 MLP처럼 $N$에 선형인 항의 이상적 누적 비율은 약

$$
2520/4728=0.533
$$

이다.

attention score element 수는 baseline에서

$$
24\times197^2=931{,}416,
$$

ToMe에서

$$
\sum_{\ell=0}^{23}(197-8\ell)^2=338{,}200
$$

이므로 약 0.363배다. score만 FP16으로 저장한다고 단순화하면 24개 block 누적 원소를 한 번에 저장하는 값이 아니라 비교용 총량 기준으로 1.777 MiB 대 0.645 MiB다.

한 layer의 token activation $[N,D]$만 보면 초기 FP16 크기는

$$
197\times1024\times2\text{ bytes}=0.385\text{ MiB}
$$

이고 마지막 merge 뒤 $[5,1024]$는 약 0.0098 MiB다. 실제 peak memory에는 QKV, attention probability, MLP expansion, residual, allocator workspace가 더해지며 training에서는 backward용 tensor도 보존된다.

### Matching overhead

첫 block의 bipartite similarity는 대략 $99\times98=9702$개 원소다. full attention score의 $197^2=38{,}809$개보다 작지만 공짜는 아니다. 또한 cosine normalization, top-$r$, gather, scatter가 추가된다. GPU처럼 이러한 operation을 효율적으로 실행하는 장치에서는 논문과 같은 speedup이 가능하지만, gather/scatter kernel launch와 memory indirection이 비싼 장치에서는 이론적 FLOPs 감소만큼 latency가 줄지 않을 수 있다.

### 무엇이 줄고 무엇이 줄지 않는가

| 항목 | ToMe 영향 |
|---|---|
| patch embedding | 감소하지 않음 |
| 첫 block attention | merge가 attention 뒤라서 감소하지 않음 |
| 이후 block QKV/MLP | token 수에 비례해 감소 |
| 이후 attention score | token 수 제곱에 비례해 감소 |
| parameter/checkpoint | 감소하지 않음 |
| intermediate activation | 감소 가능 |
| matching/gather/scatter | 새로 추가 |

## 핵심 ablation

모든 수치는 ViT-L/16 MAE, ImageNet-1k, off-the-shelf, $r=8$, FP32 V100 조건이다. baseline은 85.96%, 93.3 images/s다.

### 유사도 feature와 distance

| feature | top-1 | images/s |
|---|---:|---:|
| attention 전 $X_{pre}$ | 83.02 | 186.8 |
| attention 후 $X$ | 83.70 | 182.8 |
| $K$ | 84.25 | 182.9 |
| $Q$ | 84.04 | 182.8 |
| $V$ | 83.80 | 182.9 |

| distance | top-1 | images/s |
|---|---:|---:|
| Euclidean | 84.26 | 182.5 |
| cosine | 84.25 | 182.9 |
| raw dot product | 82.78 | 183.0 |
| softmax score | 82.00 | 183.0 |

Euclidean이 0.01% 높지만 cosine이 조금 빠르다. raw dot product는 vector norm에 영향을 받아 정확도가 낮다. 논문의 기본값은 head 평균 $K$와 cosine이다.

### aggregation 방식

| 방식 | top-1 | images/s |
|---|---:|---:|
| destination 하나만 유지 | 81.01 | 185.4 |
| max pool | 83.50 | 184.6 |
| unweighted average | 83.57 | 183.8 |
| size-weighted average | 84.25 | 182.9 |

가중 평균이 약간 느리지만 정보 보존이 가장 좋다.

### reduction algorithm 비교

| 방식 | top-1 | images/s |
|---|---:|---:|
| random pruning | 79.22 | 184.4 |
| attention pruning | 79.48 | 183.8 |
| k-means 2회 | 80.19 | 169.7 |
| k-means 5회 | 80.29 | 147.5 |
| greedy matching | 84.36 | 179.4 |
| bipartite matching | 84.25 | 182.9 |

greedy가 0.11% 높지만 순차성이 있어 느리다. bipartite 방식은 random pruning과 거의 같은 throughput에서 정확도는 크게 높다.

## ImageNet 결과

### 큰 모델일수록 손실이 작은 경향

off-the-shelf constant schedule로 약 2배 throughput을 만들었을 때 AugReg ViT-B/S/Ti는 4-5%가량 떨어지지만 ViT-L은 224 입력에서 약 2%, 384 입력에서 0.7% 하락했다. SWAG ViT-L 512와 ViT-H 518은 0.3% 하락에 그쳤다. 저자들은 큰 모델이 더 깊어 merge가 더 점진적으로 일어날 수 있기 때문일 가능성을 제시한다.

### MAE fine-tuning 비교

| 모델 | schedule | top-1 | GFLOPs | images/s |
|---|---|---:|---:|---:|
| ToMe ViT-L MAE | decreasing $r=8$ | 84.2 | 22.3 | 241 |
| ToMe ViT-L MAE | constant $r=8$ | 85.1 | 31.0 | 183 |
| ViT-L MAE baseline | 없음 | 85.7 | 61.6 | 93 |
| ToMe ViT-H MAE | decreasing $r=7$ | 86.1 | 72.6 | 81 |
| ToMe ViT-H MAE | constant $r=7$ | 86.5 | 92.9 | 63 |
| ViT-H MAE baseline | 없음 | 86.9 | 167.4 | 35 |

constant schedule은 ViT-L에서 0.6% 하락과 약 1.97배 throughput, ViT-H에서 0.4% 하락과 1.8배 throughput을 보인다. decreasing은 더 빠르지만 정확도 손실도 커진다.

### pruning 계열 비교

| 방법 | top-1 | GFLOPs | images/s | training speed |
|---|---:|---:|---:|---:|
| DeiT-S baseline | 79.8 | 4.6 | 930 | 1.0배 |
| DynamicViT | 79.3 | 2.9 | 1505 | 1.0배 |
| ToMe DeiT, $r=13$ | 79.4 | 2.7 | 1552 | 1.5배 |
| ToMe AugReg off-the-shelf, $r=13$ | 79.3 | 2.7 | 1550 | 해당 없음 |

ToMe는 training에서도 실제 token tensor를 줄일 수 있어 1.5배 speedup을 보고했다. 또한 AugReg checkpoint에는 재학습 없이 비슷한 결과를 얻었다.

## Video와 audio 결과

### Kinetics-400 video

Spatiotemporal MAE ViT-L에 같은 코드를 적용했다. constant $r=65$ 조건에서 baseline 7.3 clips/s가 16.3 clips/s로 2.2배가 되었고, accuracy는 84.7%에서 84.5%로 0.2% 하락했다. 8 GPU fine-tuning 시간은 추정 263시간에서 136시간으로 줄었다.

| 모델 | clips/s | 상대 속도 | fine-tuning 시간 |
|---|---:|---:|---:|
| ViT-L MAE | 7.3 | 1.0배 | 263시간, 추정 |
| ToMe ViT-L, constant $r=65$ | 16.3 | 2.2배 | 136시간 |
| ToMe ViT-L, decreasing $r=65$ | 24.9 | 3.4배 | 미보고 |

논문의 video token은 $2\times16\times16$ patch라 한 token이 두 frame을 포함한다. visualization에서는 같은 공이나 물체 부분이 여러 frame에 걸쳐 하나의 token으로 합쳐지는 현상을 보였다. 이는 명시적 tracker가 아니라 key similarity와 positional encoding에서 emergent하게 나온 결과다.

### AudioSet-2M audio

Audio MAE의 spectrogram token에 적용했다.

| 설정 | mAP | GFLOPs | samples/s |
|---|---:|---:|---:|
| 공개 ViT-B MAE baseline | 47.3 | 48.6 | 103 |
| off-the-shelf $r=20$ | 46.2 | 36.3 | 136 |
| off-the-shelf $r=40$ | 43.1 | 24.7 | 200 |
| 재현 baseline | 46.4 | 48.6 | 103 |
| training with ToMe, $r=20$ | 46.3 | 36.3 | 136 |
| training with ToMe, $r=40$ | 46.0 | 24.7 | 200 |

마지막 비교에서는 약 1.94배 throughput에서 mAP가 0.4% 하락했다. 공개 baseline과 저자 재현 baseline이 다르므로 반드시 같은 training implementation 안에서 비교해야 한다.

## 논문 증거와 해석의 경계

### 논문이 직접 보여 준 것

- V100 GPU의 최적 batch size, 주로 FP32 조건에서 image throughput을 측정했다.
- video fine-tuning은 8 GPU 시간으로 비교했다.
- key cosine, weighted average, alternating partition이 기본 조합에서 좋은 trade-off임을 ablation했다.
- 큰 ViT, video, audio에서 token 중복을 활용할 수 있음을 보였다.
- fixed reduction count라 dynamic pruning보다 dense batching이 쉽다.

### 이 리뷰에서 별도로 추론한 것

- $N$과 $N^2$ 합으로 계산한 activation/attention 절감률은 이상적인 shape 기반 계산이다.
- 모바일 NPU에서 gather/scatter가 병목일 수 있다는 평가는 논문의 V100 결과를 모바일에 그대로 옮길 수 없다는 시스템 관점의 추론이다.
- segmentation/tracking에 쓸 가능성은 visualization이 주는 힌트이지 논문이 downstream 성능으로 검증한 결과가 아니다.

## 온디바이스 관점

### 장점

- checkpoint와 weight를 바꾸지 않고 runtime patch로 적용할 수 있다.
- token 수가 layer별로 결정적이므로 static memory plan과 batch 처리가 dynamic pruning보다 쉽다.
- parameter size는 그대로지만 activation과 DRAM traffic을 줄일 가능성이 있다.
- 카메라 frame, video clip, spectrogram처럼 중복이 큰 입력에서 효과가 커질 수 있다.
- $r$을 조절해 battery, thermal state, latency budget에 따라 품질과 속도를 바꿀 수 있다.

### 위험 요소

- `topk`, `argsort`, `gather`, `scatter_add`가 target delegate에 없으면 CPU fallback이 발생할 수 있다.
- token 수가 layer마다 달라지는 graph는 일부 NPU compiler의 static shape 요구와 충돌할 수 있다. 고정 schedule이어도 layer별 shape가 서로 다른 것은 그대로다.
- 첫 block과 patch embedding은 절약되지 않는다. 작은 ViT에서는 matching overhead 비중이 커질 수 있다.
- detection과 segmentation은 작은 객체와 경계 정보를 보존해야 하므로 classification보다 merge tolerance가 낮을 수 있다.
- proportional attention의 $\log s$ broadcast가 fused attention kernel과 호환되지 않을 수 있다.
- 모델 파일 크기와 weight bandwidth는 그대로여서 weight-bound 모델에는 이득이 제한될 수 있다.

### 실제 디바이스 실험 설계

1. baseline과 ToMe를 같은 precision, input, batch=1로 export한다.
2. `r`을 0, 2, 4, 8처럼 sweep하고 layer별 token shape를 기록한다.
3. end-to-end p50/p95 latency와 함께 merge kernel 시간, attention, MLP 시간을 분리한다.
4. peak RSS/allocator memory와 DRAM bandwidth counter를 가능하면 측정한다.
5. CPU fallback 여부와 delegate partition을 확인한다.
6. classification이면 top-1, detection이면 small-object AP, segmentation이면 boundary IoU를 같이 본다.
7. 5분 이상 지속 실행해 온도와 throttling 후 latency를 다시 측정한다.

## 장점과 기여

1. token reduction을 pruning이 아닌 information aggregation 문제로 바꿨다.
2. 이미 계산한 key를 재사용해 별도 scoring network와 parameter가 없다.
3. iterative clustering 대신 병렬 matching을 사용해 실제 throughput을 확보했다.
4. sample마다 동일한 $r$을 사용해 batching 가능성을 유지했다.
5. off-the-shelf 적용과 training-time 적용을 모두 지원한다.
6. image, video, audio에서 같은 기본 원리를 검증했다.
7. token size와 proportional attention으로 merge multiplicity를 명시적으로 추적했다.

## 한계와 비판적 관점

1. 주요 latency 결과가 V100 중심이라 모바일 SoC에서의 operator 비용은 검증되지 않았다.
2. 정확도와 throughput의 최적점은 pretraining recipe, model depth, 입력 해상도에 의존한다.
3. classification 위주이며 dense prediction에서 boundary와 작은 객체 보존을 정량 검증하지 않았다.
4. matching은 local adjacency를 강제하지 않아 멀리 떨어진 동일 패턴을 합칠 수 있다. classification에는 유리할 수 있지만 위치 정밀도가 필요한 task에는 위험하다.
5. parameter와 checkpoint 크기는 줄지 않는다.
6. token을 일단 합치면 원래 개별 feature를 정확히 복원할 수 없다.
7. size-weighted average는 first moment만 보존하며 multimodal한 cluster 내부 분산은 잃는다.
8. decreasing schedule의 속도 이점은 크지만 초기 layer의 정보 손실 위험도 높다.

## 재현 및 구현 체크리스트

- [ ] CLS/distillation token을 merge source에서 보호했는가
- [ ] $K$를 head 평균한 뒤 L2 normalize했는가
- [ ] alternating partition을 사용했는가
- [ ] 매 block의 $r$이 현재 source 수를 넘지 않도록 clamp했는가
- [ ] feature와 token size에 같은 source/destination mapping을 적용했는가
- [ ] weighted average에서 분모가 0이 되지 않는가
- [ ] proportional attention의 $\log s$가 key dimension에 broadcast되는가
- [ ] MAE off-the-shelf에서는 proportional attention 사용 여부를 따로 ablation했는가
- [ ] merge를 attention 뒤, MLP 앞에 넣었는가
- [ ] training 시 backward가 scatter/gather를 통과하는지 확인했는가
- [ ] throughput 측정에서 batch size, dtype, warm-up, synchronization을 고정했는가
- [ ] export graph에 unsupported operation이나 CPU fallback이 없는가
- [ ] FLOPs와 실제 p50/p95 latency를 별도로 기록했는가
- [ ] dense prediction task에서 작은 객체와 경계 성능을 따로 평가했는가

## 후속 연구와의 연결

ToMe는 DynamicViT 같은 token pruning과 다른 축이다. pruning이 중요도를 기준으로 token 존재 여부를 결정한다면 ToMe는 표현 중복을 기준으로 token 집합을 압축한다. Token Merging 이후의 연구에서는 spatial constraint, task-aware merging, reversible token map, hierarchical backbone과의 결합을 탐색할 수 있다.

온디바이스 비전에서는 특히 다음 세 방향이 유망하다.

- detector의 ROI 밖 token을 더 공격적으로 merge하고 ROI 안은 보존하는 task-aware schedule
- frame 간 motion consistency를 사용해 video token merge map을 cache하는 방식
- hardware가 지원하는 fixed pooling/gather pattern으로 matching을 근사해 NPU friendly graph를 만드는 방식

## 최종 평가

ToMe의 가장 큰 가치는 복잡한 새 backbone을 설계하지 않고도 기존 ViT의 중복 계산을 직접 줄인다는 데 있다. 수학적으로는 cosine matching과 weighted pooling이라는 단순한 조합이지만, attention key 재사용, fixed reduction, proportional attention, block 내부 배치 위치가 함께 맞물려 실제 GPU throughput을 만든다.

다만 온디바이스 적용에서는 FLOPs보다 `topk/gather/scatter`, layer별 variable token shape, attention kernel 수정 지원 여부가 더 중요하다. 따라서 이 논문을 재현할 때는 ImageNet 정확도 표만 따라가기보다 operator trace, activation memory, p95 latency, fallback 여부를 함께 측정해야 한다. 그 검증까지 통과한다면 ToMe는 큰 ViT를 모바일이나 edge 장치에 맞추는 매우 실용적인 runtime token compression 기법이 될 수 있다.

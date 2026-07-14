# 05. Self-Attention with Relative Position Representations

## 논문 정보

- 원본 파일: `05_Self_Attention_with_Relative_Position_Representations.pdf`
- 제목: Self-Attention with Relative Position Representations
- 저자: Peter Shaw, Jakob Uszkoreit, Ashish Vaswani
- 발표: NAACL 2018
- 주제: self-attention의 key와 value에 query-key 상대 거리 embedding을 결합하는 방법
- 핵심 키워드: relative position representation, relation-aware self-attention, clipped distance, relative key, relative value

## 한눈에 보는 요약

기본 Transformer는 input embedding에 absolute position encoding을 더한다.

```math
x_i=\operatorname{token\_embedding}_i+\operatorname{absolute\_position}_i
```

이 논문은 위치 정보를 input에 한 번 더하는 대신, self-attention에서 position $i$가 position $j$를 볼 때 두 위치의 상대 거리 $j-i$를 직접 사용한다.

```math
\operatorname{relative\ distance}=j-i
```

Query $q_i$와 key $k_j$의 compatibility를 계산할 때 거리별 vector $a^K_{i,j}$를 key에 더한다.

```math
e_{i,j}
=\frac{q_i(k_j+a^K_{i,j})^\top}{\sqrt{d_z}}
```

Value를 합산할 때도 별도의 거리 vector $a^V_{i,j}$를 더할 수 있다.

```math
z_i=\sum_j\alpha_{i,j}(v_j+a^V_{i,j})
```

거리 종류가 sequence length만큼 무한히 늘어나지 않도록 $[-k,k]$ 범위로 clipping한다.

```math
r_{i,j}=\operatorname{clip}(j-i,k)
```

따라서 학습할 vector는 key용 $2k+1$개, value용 $2k+1$개다. 방향이 있으므로 $-3$과 $+3$은 다른 relation이다.

WMT 2014 translation에서 absolute sinusoidal encoding을 relative representation으로 완전히 교체했을 때 BLEU가 향상됐다. Absolute와 relative를 같이 사용해도 추가 이득은 없었다. Ablation에서는 value-side relative vector를 제거해도 성능이 유지되고, score에 들어가는 relative key가 더 중요했다.

이 논문은 sequence를 labeled directed complete graph로 보고 relative position을 edge label로 해석한다. 이 관점은 이후 graph relation attention과 다양한 relative bias 연구로 이어진다.

## 연구 배경과 문제의식

### Transformer는 순서에 불변이다

Position 정보가 없는 self-attention은 token 순서를 바꾸면 출력도 같은 방식으로 순열되는 permutation-equivariant 연산이다.

```math
\begin{aligned}
Q &= XW_Q,\\
K &= XW_K,\\
V &= XW_V,\\
\operatorname{Attention}(X)
&=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt d}\right)V.
\end{aligned}
```

RNN은 recurrence 순서로, CNN은 kernel의 local offset으로 상대 위치를 구조 안에 포함한다. Transformer는 recurrence와 convolution이 없으므로 별도 position signal이 필요하다.

### Absolute position의 한계

원 Transformer는 sinusoidal position vector를 token embedding에 더한다.

```math
x_i=\operatorname{word}_i+p_i
```

Self-attention이 상대 거리 패턴을 사용하려면 $p_i$와 $p_j$에서 $j-i$ 관계를 간접적으로 학습해야 한다.

하지만 sequence 관계에서 중요한 것은 종종 절대 index보다 상대 offset이다.

```text
"previous word": -1
"next word":     +1
"two tokens ago": -2
```

같은 syntactic pattern이 sentence의 어느 absolute 위치에 나타나든 같은 relative relation으로 처리하는 편이 자연스럽다.

### 논문의 목표

Self-attention의 pairwise interaction 자체가 position relation을 받게 만든다.

```text
node feature: token representation x_i
edge feature: relative position relation between i and j
```

이를 sequence 전용 trick이 아니라 arbitrary relation-aware self-attention의 한 사례로 제시한다.

## 기본 self-attention

Input sequence를 다음과 같이 두자.

```math
\begin{aligned}
X&=(x_1,\ldots,x_n),\\
x_i&\in\mathbb{R}^{d_x}.
\end{aligned}
```

한 attention head의 query, key, value는 다음과 같다.

```math
\begin{aligned}
q_i &= x_iW_Q,\\
k_j &= x_jW_K,\\
v_j &= x_jW_V.
\end{aligned}
```

Shape는 다음과 같다.

```text
W_Q, W_K, W_V: [d_x, d_z]
q_i, k_j, v_j: [d_z]
```

기본 score와 output은 다음이다.

```math
\begin{aligned}
e_{i,j} &= \frac{q_ik_j^\top}{\sqrt{d_z}},\\
\alpha_{i,j} &= \operatorname{softmax}_j(e_{i,j}),\\
z_i &= \sum_j\alpha_{i,j}v_j.
\end{aligned}
```

## Relation-aware self-attention

### Directed edge representation

Position $i$에서 $j$로 향하는 edge에 두 vector를 둔다.

```text
a^K_(i,j): score/key에 사용하는 relation vector
a^V_(i,j): output/value에 사용하는 relation vector
```

두 vector를 별도로 두는 이유는 key와 value의 역할이 다르기 때문이다.

- $a^K$: 어떤 node를 얼마나 볼지 바꾼다.
- $a^V$: 선택한 relation 자체의 정보를 output에 전달한다.

### Relative key를 포함한 score

```math
e_{i,j}
=\frac{q_i(k_j+a^K_{i,j})^\top}{\sqrt{d_z}}
```

전개하면 두 항이다.

```math
e_{i,j}
=\frac{q_ik_j^\top+q_i(a^K_{i,j})^\top}{\sqrt{d_z}}
```

첫 항은 content-to-content compatibility이고 둘째 항은 query-to-relation compatibility다.

```math
\begin{aligned}
\text{content term:}\quad &q_ik_j^\top,\\
\text{position term:}\quad &q_i(a^K_{i,j})^\top.
\end{aligned}
```

같은 거리라도 query content가 다르면 position term이 달라진다. 단순 scalar distance bias보다 content-dependent하다.

### Relative value를 포함한 output

```math
z_i=\sum_j\alpha_{i,j}(v_j+a^V_{i,j})
```

Output에는 두 종류의 정보가 합쳐진다.

```math
\sum_j\alpha_{i,j}v_j
+
\sum_j\alpha_{i,j}a^V_{i,j}
```

첫 항은 선택한 token content, 둘째 항은 선택한 edge/relation의 weighted summary다.

## Relative distance embedding

### Clipping

Sequence에서 edge label은 signed relative distance다.

```math
\operatorname{distance}(i,j)=j-i
```

큰 거리는 최대 절댓값 $k$로 clipping한다.

```math
\begin{aligned}
\operatorname{clip}(x,k)&=\max(-k,\min(k,x)),\\
r_{i,j}&=\operatorname{clip}(j-i,k).
\end{aligned}
```

Distance embedding은 다음 table에서 조회한다.

```math
\begin{aligned}
a^K_{i,j}&=w^K_{r_{i,j}},\\
a^V_{i,j}&=w^V_{r_{i,j}}.
\end{aligned}
```

Learned label 수는 각각 $2k+1$개다.

```math
-k,\ldots,-2,-1,0,+1,+2,\ldots,+k
```

<p align="center"><img src="https://github.com/user-attachments/assets/01376754-622e-42eb-8623-0ceb2dafabe8" alt="상대 거리를 directed edge로 표현하는 self-attention" width="680"></p>
<p align="center"><sub>원 논문 Figure 1 — 상대 거리를 directed edge representation으로 표현하는 방식</sub></p>

### 방향성이 중요하다

$j-i$를 사용하므로 왼쪽과 오른쪽 relation이 다르다.

```text
j = i-1 -> -1 -> one token to the left
j = i+1 -> +1 -> one token to the right
```

Implementation에서 $i-j$를 사용해도 convention 전체가 일관되면 학습은 가능하지만, checkpoint와 수식 convention은 호환되지 않는다.

### 왜 먼 거리를 하나로 묶는가

논문은 일정 거리 이상에서는 정확한 offset이 덜 중요하다고 가정한다.

```text
distance +20, +30, +100
-> all mapped to +k
```

장점은 다음과 같다.

- Parameter 수가 sequence length와 무관하다.
- 학습보다 긴 sequence에도 같은 edge label을 적용할 수 있다.
- 가까운 local order는 정밀하게 표현한다.

단점은 $k$ 밖의 모든 거리가 구분되지 않는다는 점이다.

## Tensor shape와 구현

Batch $B$, head $H$, length $N$, head dimension $d_z$를 사용하자.

```text
Q, K, V: [B, H, N, d_z]
```

Relative table은 다음 shape다.

```text
W_rel_K: [2k+1, d_z]
W_rel_V: [2k+1, d_z]
```

Position pair별 index matrix를 만든다.

```math
\begin{aligned}
I[i,j]&=\operatorname{clip}(j-i,k)+k,\\
\operatorname{shape}(I)&=[N,N].
\end{aligned}
```

Gather하면 pairwise relation tensor가 된다.

```math
\begin{aligned}
A_K&=W_{\mathrm{rel},K}[I],\\
A_V&=W_{\mathrm{rel},V}[I],\\
\operatorname{shape}(A_K)
=\operatorname{shape}(A_V)&=[N,N,d_z].
\end{aligned}
```

Naive하게 batch와 head로 복제하면 메모리가 커진다.

```text
naive broadcast:
[B,H,N,N,d_z]
```

논문은 relation tensor를 batch와 head 사이에 공유하고 reshape/matmul로 계산한다.

### Score 분해

```math
\begin{aligned}
\operatorname{score}_{\mathrm{content}}&=QK^\top,\\
\operatorname{shape}(\operatorname{score}_{\mathrm{content}})&=[B,H,N,N].
\end{aligned}
```

Relative term은 query position별로 다음을 계산한다.

```math
\operatorname{score}_{\mathrm{relative}}[b,h,i,j]
=\operatorname{dot}(Q[b,h,i,:],A_K[i,j,:])
```

두 score를 더하고 scaling한다.

```math
E=\frac{\operatorname{score}_{\mathrm{content}}
+\operatorname{score}_{\mathrm{relative}}}{\sqrt{d_z}}
```

Value aggregation도 두 항으로 나눌 수 있다.

```math
\begin{aligned}
Z_{\mathrm{content}} &= \mathrm{Alpha}\,V,\\
Z_{\mathrm{relative}}[b,h,i,:]
&=\sum_j\mathrm{Alpha}[b,h,i,j]A_V[i,j,:].
\end{aligned}
```

현대 framework에서는 `einsum`이나 skew/relative-shift 계열 trick으로 구현할 수 있다.

## 메모리와 성능 비용

논문은 relation을 head 사이에 공유하면 edge representation 저장을 다음처럼 줄일 수 있다고 설명한다.

```math
O(HN^2d_z)\ \longrightarrow\ O(N^2d_z)
```

전체 self-attention activation 관점에서는 다음 항이 추가된다.

```math
\begin{aligned}
\text{base:}\quad &O(BHNd_z),\\
\text{relative:}\quad &+O(N^2d_a).
\end{aligned}
```

실험 구현에서는 P100 기준 training steps/sec가 약 7% 감소했지만 같은 model/batch size를 유지했다.

논문은 공유가 가능하다고 설명하지만 실제 실험 설정은 model size에 따라 다르다.

- Base: layer와 head별 unique relation representation
- Big: layer별 unique representation, head 사이 공유

따라서 parameter/memory 수를 재현할 때 어느 축으로 share하는지 확인해야 한다.

## 작은 예시

Clipping distance $k=2$, query position $i=3$이라고 하자.

| Key position $j$ | $j-i$ | clipped label |
| ---: | ---: | ---: |
| 0 | -3 | -2 |
| 1 | -2 | -2 |
| 2 | -1 | -1 |
| 3 | 0 | 0 |
| 4 | +1 | +1 |
| 5 | +2 | +2 |
| 6 | +3 | +2 |

Position 0과 1은 정확한 거리는 다르지만 모두 far-left relation $-2$를 사용한다. Position 5와 6은 far-right $+2$를 사용한다.

Score는 content와 relation을 함께 본다.

```math
\operatorname{score}(i=3,j=4)
=q_3k_4^\top+q_3(w^K_{+1})^\top
```

같은 token content가 왼쪽에 있을 때와 오른쪽에 있을 때 다른 score가 나온다.

## 실험 설정

### 데이터

- WMT 2014 English-German: 약 4.5M sentence pair
- WMT 2014 English-French: 약 36M sentence pair
- WordPiece vocabulary: 32,768
- Beam size: 4
- Length penalty: 0.6
- Adam $\beta_1=0.9$, $\beta_2=0.98$, $\epsilon=10^{-9}$
- Warmup: 4,000 step
- Label smoothing: 0.1

Baseline도 같은 code/library 설정으로 다시 학습해 position method의 영향을 분리한다.

### Model 설정

Base model:

```text
layers: 6 encoder + 6 decoder
d_model: 512
heads: 8
head dimension: 64
FFN inner dimension: 1024
k: 16
training: 100k steps on 8 K40
```

Big model:

```text
d_model: 1024
heads: 16
head dimension: 64
FFN inner dimension: 4096
k: 8
training: 300k steps on 8 P100
last 20 checkpoints averaged
```

## 실험 결과

| 모델 | Position 정보 | EN-DE BLEU | EN-FR BLEU |
| --- | --- | ---: | ---: |
| Transformer base | Absolute | 26.5 | 38.2 |
| Transformer base | Relative | 26.8 | 38.7 |
| Transformer big | Absolute | 27.9 | 41.2 |
| Transformer big | Relative | 29.2 | 41.5 |

Relative-only model의 향상은 다음과 같다.

```text
base EN-DE: +0.3
base EN-FR: +0.5
big EN-DE:  +1.3
big EN-FR:  +0.3
```

Absolute sinusoidal encoding을 relative representation과 함께 사용해도 추가 이득은 없었다. 적어도 이 translation 설정에서는 relative relation만으로 필요한 order signal을 제공할 수 있었다.

## Clipping distance ablation

Base model, EN-DE development set 결과다.

| $k$ | BLEU |
| ---: | ---: |
| 0 | 12.5 |
| 1 | 25.5 |
| 2 | 25.8 |
| 4 | 25.9 |
| 16 | 25.8 |
| 64 | 25.9 |
| 256 | 25.8 |

$k=0$은 모든 pair가 같은 relation label을 사용하므로 position 구분이 사라져 성능이 무너진다. $k\ge2$에서는 차이가 작다.

이 결과를 "거리 2 이상은 항상 필요 없다"로 해석하면 안 된다.

- 여러 layer를 쌓으면 local relation이 전파될 수 있다.
- Translation sentence length와 task 특성에 의존한다.
- 긴 context language modeling에서는 다른 결과가 나올 수 있다.

## Relative key/value ablation

| $a^V$ | $a^K$ | EN-DE BLEU |
| --- | --- | ---: |
| Yes | Yes | 25.8 |
| No | Yes | 25.8 |
| Yes | No | 25.3 |
| No | No | 12.5 |

핵심 관찰은 다음이다.

```text
relative key only = full model
```

Value-side relation을 제거해도 BLEU가 유지된다. 반면 key-side relation이 없으면 성능이 떨어진다. Position 정보는 무엇을 볼지 정하는 compatibility에 들어가는 것이 더 중요했다.

이 결과는 이후 T5 relative bias, Transformer-XL, RoPE처럼 주로 attention score에 position을 넣는 방법과 연결된다.

## 장점과 핵심 기여

### 1. Relative position을 attention pair에 직접 넣었다

Input embedding이 아니라 query-key relation 자체가 거리 정보를 갖는다.

### 2. 방향 있는 거리 표현이다

왼쪽 $-r$과 오른쪽 $+r$을 다른 label로 처리한다.

### 3. 길이에 독립적인 parameterization이다

$2k+1$개 table만 사용하므로 learned absolute table처럼 최대 index별 parameter가 필요하지 않다.

### 4. Relation-aware graph view를 제시했다

Relative position을 edge label로 보고 arbitrary directed labeled graph로 확장 가능한 framework를 제안했다.

### 5. Key/value 역할을 ablation했다

Score-side relative key가 핵심이고 relative value는 translation에서 필수가 아님을 보였다.

## 한계와 비판적 관점

### 1. Pairwise relation tensor가 크다

Parameter table은 작지만 length $N$의 pair index와 relation contribution은 $N^2$에 해당한다. Long sequence memory 문제를 해결하지 않는다.

### 2. Clipping 밖 거리를 구분하지 못한다

$k$보다 먼 모든 왼쪽/오른쪽 거리가 각각 하나의 label로 합쳐진다. Fine-grained long-distance pattern을 표현하기 어렵다.

### 3. Translation 중심의 증거다

$k\ge2$ 안정성이나 relative value 불필요성은 다른 task에 그대로 일반화되지 않을 수 있다.

### 4. Cross-attention relation은 자연스럽지 않다

같은 sequence 안의 self-attention은 $j-i$가 명확하지만 source와 target의 index 차이는 직접적인 언어학적 거리가 아니다. 논문의 핵심 적용 범위는 self-attention이다.

### 5. 길이 extrapolation을 보장하지 않는다

새 길이에서 edge label을 계산할 수는 있지만 clipping된 relation과 model content computation이 긴 sequence에서 잘 일반화한다는 정리는 아니다.

### 6. Head/layer sharing 설정이 복잡하다

Relation table을 head 또는 layer마다 둘지 공유할지에 따라 parameter와 inductive bias가 달라진다.

## 후속 논문과의 연결

이 논문의 score를 다시 쓰면 다음과 같다.

```math
\operatorname{score}_{i,j}
=q_ik_j^\top+q_i(a^K_{j-i})^\top
```

후속 연구는 이 position term을 여러 방식으로 바꾼다.

- Transformer-XL: content/position term을 분해하고 relative shift를 사용
- T5: query-dependent vector 대신 distance bucket scalar bias 사용
- DeBERTa: content-to-position과 position-to-content를 분리
- RoPE: Q/K를 회전해 dot product에 상대 phase를 내장
- ALiBi: distance에 비례하는 head별 linear scalar bias

공통 질문은 같다.

```text
relative position should enter Q, K, V, or the score bias?
```

## 구현 체크리스트

```text
1. Relative distance가 j-i convention인지 확인했는가?
2. Clipping이 [-k,k] 양쪽에서 대칭적인가?
3. Table index offset +k를 적용했는가?
4. Causal mask와 relative distance sign을 혼동하지 않는가?
5. Padding pair의 score를 mask하는가?
6. Relative table이 head/layer 중 어느 축에서 공유되는가?
7. a^K term에도 sqrt(d_z) scaling이 함께 적용되는가?
8. a^V가 attention weight 적용 후 합산되는가?
9. [N,N,d] relation tensor를 batch/head로 불필요하게 복제하지 않는가?
10. Sequence length가 바뀔 때 index matrix를 재생성하는가?
```

## 개인 학습/연구 메모

기억해야 할 핵심은 self-attention을 node-only 연산에서 edge-aware 연산으로 확장했다는 점이다.

```text
before:
score depends on x_i and x_j

after:
score depends on x_i, x_j, and relation(i,j)
```

가장 중요한 식은 다음이다.

```math
e_{i,j}=\frac{q_i(k_j+a^K_{j-i})^\top}{\sqrt d}
```

이 한 줄이 content와 directed relative distance를 attention compatibility 안에서 결합한다.

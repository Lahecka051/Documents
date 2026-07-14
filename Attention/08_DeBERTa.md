# 08. DeBERTa: Decoding-enhanced BERT with Disentangled Attention

## 논문 정보

- 원본 파일: `Attention/08_DeBERTa.pdf`
- 제목: DeBERTa: Decoding-enhanced BERT with Disentangled Attention
- 저자: Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen
- 발표: ICLR 2021
- 주제: pre-trained language model, disentangled attention, relative position, enhanced mask decoder
- 핵심 키워드: content-to-content, content-to-position, position-to-content, EMD, SiFT

## 한눈에 보는 요약

DeBERTa는 BERT/RoBERTa 계열 encoder model을 두 가지 구조적 아이디어로 개선한다.

1. **Disentangled attention**
   - token content와 position을 처음부터 더하지 않고 별도 vector로 유지한다.
   - attention score를 content-content, content-position, position-content 항으로 계산한다.

2. **Enhanced Mask Decoder, EMD**
   - Transformer stack에서는 relative position을 사용한다.
   - masked token을 decode하기 직전에 absolute position을 보완 정보로 넣는다.

논문의 문제의식은 다음과 같다.

```text
BERT input:
token representation = content embedding + absolute position embedding

DeBERTa:
content representation and relative position representation remain separate
```

content와 position을 일찍 더하면 이후 attention이 두 정보의 역할을 명시적으로 구분하기 어렵다. DeBERTa는 attention score의 각 상호작용을 분리해 학습한다.

대표 score는 다음과 같다.

```math
\tilde A_{i,j}
=
Q_i^cK_j^{c\top}
+Q_i^cK_{\delta(i,j)}^{r\top}
+K_j^cQ_{\delta(j,i)}^{r\top}
```

첫 항은 content-to-content, 둘째는 content-to-position, 셋째는 position-to-content다. position-to-position 항은 relative position 설정에서 추가 정보가 작다고 판단해 제거한다.

![DeBERTa의 disentangled attention과 enhanced mask decoder](https://github.com/user-attachments/assets/242f8251-7067-4311-9d0c-b760a6cd6bb0)

## 연구 배경과 문제의식

### BERT의 embedding 합

BERT류 model은 입력에서 보통 다음을 더한다.

```math
x_i=e_i^{token}+e_i^{position}+e_i^{segment}
```

이 방식은 간단하지만 content와 position이 하나의 hidden vector에 섞인다. attention의 query와 key는 혼합 vector에서 생성된다.

```text
mixed input
-> Q/K projection
-> one dot product
```

모델이 content 관계와 positional 관계를 학습할 수는 있지만 두 요소의 상호작용이 구조적으로 분리되어 있지 않다.

### 단어 관계는 content와 distance에 함께 의존한다

논문은 `deep learning`을 예로 든다. 두 단어가 바로 붙어 있을 때와 서로 다른 문장에 있을 때 content는 같아도 관계 강도는 다르다.

```text
deep learning        -> strong phrase relation
deep ... sentence ... learning -> weaker relation
```

따라서 attention score는 다음을 구분할 필요가 있다.

- query content와 key content의 적합성
- query content가 특정 상대 위치를 선호하는 정도
- 특정 query 위치가 key content를 선택하는 방식

### Relative position만으로 충분하지 않은 decoding

MLM에서 같은 local pattern이 여러 번 나타날 수 있다.

```text
A new [MASK] opened beside the new [MASK].
```

두 mask 모두 `new` 뒤에 있고 주변 상대 위치가 유사하다. 하지만 첫 mask는 subject `store`, 두 번째는 object `mall`이다. 논문은 문장 내 absolute position이 이런 syntactic role을 구분하는 보완 정보가 될 수 있다고 본다.

DeBERTa는 absolute position을 input에 섞지 않고 MLM decoder 직전에 넣는다.

## Disentangled representation

token `i`를 두 종류의 vector로 생각한다.

```text
H_i: token i의 content representation
P_(i-j): token i와 j 사이의 relative position representation
```

두 token `i`, `j` 사이의 완전한 상호작용을 전개하면 네 항을 생각할 수 있다.

```math
A_{i,j}
=H_iH_j^\top
+H_iP_{j|i}^\top
+P_{i|j}H_j^\top
+P_{i|j}P_{j|i}^\top
```

각 항은 다음과 같다.

| 항 | 이름 | 의미 |
|---|---|---|
| $H_iH_j^\top$ | content-to-content | 두 token content의 관계 |
| $H_iP_{j|i}^\top$ | content-to-position | query content가 key의 상대 위치를 선호 |
| $P_{i|j}H_j^\top$ | position-to-content | query 위치 관점에서 key content를 선택 |
| $P_{i|j}P_{j|i}^\top$ | position-to-position | 위치끼리의 상호작용 |

DeBERTa 구현은 마지막 position-to-position 항을 제외한다. relative position끼리의 항이 추가하는 정보가 제한적이고 계산 비용을 늘린다고 판단했기 때문이다.

## Relative distance index

### Clipping

최대 상대 거리 `k`를 두고 실제 offset `i-j`를 `0`에서 `2k-1` 사이 index로 바꾼다.

```math
\delta(i,j)=
\begin{cases}
0, & i-j\le -k \\
2k-1, & i-j\ge k \\
i-j+k, & \text{otherwise}
\end{cases}
```

따라서 relative embedding table은 다음 shape를 가진다.

```math
P\in\mathbb{R}^{2k\times d}
```

논문 pre-training에서는 `k=512`를 사용한다.

```text
offset <= -512 -> first clipped index
-511 ... 511   -> distinct indices
offset >= 512  -> last clipped index
```

clipping은 먼 token을 attention에서 제거하는 sparse mask가 아니다. 먼 거리의 **정확한 위치 차이만 같은 embedding으로 묶는다**. content-content score는 여전히 모든 token pair에 대해 계산된다.

### 방향성이 중요하다

content-to-position은 query `i`에서 key `j`를 보는 방향이므로 `\delta(i,j)`를 쓴다.

position-to-content는 key content `j`가 query position `i`와 어떤 관계인지 보므로 반대 방향 `\delta(j,i)`를 쓴다.

```text
C2P: delta(i, j)
P2C: delta(j, i)
```

두 index를 같게 쓰면 relative direction 정보가 잘못된다. 구현에서 가장 흔한 오류 중 하나다.

## Disentangled attention 수식

### Projection

content hidden state를 `H`, relative table을 `P`라고 하자.

```math
Q_c=HW_{q,c},
\quad
K_c=HW_{k,c},
\quad
V_c=HW_{v,c}
```

```math
Q_r=PW_{q,r},
\quad
K_r=PW_{k,r}
```

shape는 다음과 같다.

```text
H:   [N, d]
P:   [2k, d]

Qc:  [N, d]
Kc:  [N, d]
Vc:  [N, d]
Qr:  [2k, d]
Kr:  [2k, d]
```

### 세 score 항

single head 기준으로 다음과 같다.

```math
A_{i,j}^{c2c}
=Q_{c,i}K_{c,j}^\top
```

```math
A_{i,j}^{c2p}
=Q_{c,i}K_{r,\delta(i,j)}^\top
```

```math
A_{i,j}^{p2c}
=K_{c,j}Q_{r,\delta(j,i)}^\top
```

최종 logit은 세 항의 합이다.

```math
\tilde A_{i,j}
=A_{i,j}^{c2c}
+A_{i,j}^{c2p}
+A_{i,j}^{p2c}
```

### Scaling factor

표준 attention은 dot product 하나의 variance를 고려해 `sqrt(d)`로 나눈다. DeBERTa logit은 세 dot product의 합이므로 논문은 다음 scaling을 사용한다.

```math
A_{i,j}
=
\frac{\tilde A_{i,j}}{\sqrt{3d}}
```

그 다음 softmax와 content value aggregation을 수행한다.

```math
H_o=\operatorname{softmax}(A)V_c
```

multi-head 구현에서는 `d`가 보통 head dimension `d_h`에 대응한다.

## Tensor shape

batch와 head를 포함한 대표 shape는 다음과 같다.

```text
B: batch size
Hh: attention heads
N: sequence length
D: hidden size
Dh: head dimension
k: maximum relative distance

content Q/K/V: [B, Hh, N, Dh]
relative Q/K:  [Hh or 1, 2k, Dh]

C2C score: [B, Hh, N, N]
raw C2P:   [B, Hh, N, 2k]
raw P2C:   [B, Hh, N, 2k]
gathered C2P/P2C: [B, Hh, N, N]

output: [B, Hh, N, Dh]
```

relative embedding table `P`는 논문에서 모든 layer에 공유된다. 각 layer의 content/position projection은 기본적으로 별도 parameter다.

## 효율적인 구현

### Naive relative tensor

모든 `(i,j)` pair마다 relative vector를 복제하면 다음 크기의 tensor가 필요하다.

```math
O(N^2d)
```

DeBERTa는 가능한 relative position이 `2k`개뿐이라는 점을 사용한다.

### C2P 계산

먼저 모든 query와 relative key table을 곱한다.

```math
S^{c2p}=Q_cK_r^\top
```

shape는 `[N, 2k]`다. 이후 pair별 `\delta(i,j)` index로 gather해 `[N,N]` score를 만든다.

```python
raw_c2p = q_content @ k_relative.transpose(-1, -2)
# [B, heads, N, 2k]

c2p = gather_by_relative_index(raw_c2p, delta_ij)
# [B, heads, N, N]
```

### P2C 계산

key content와 relative query table을 곱고 반대 방향 index를 gather한다.

```python
raw_p2c = k_content @ q_relative.transpose(-1, -2)
# [B, heads, N, 2k]

p2c = gather_by_relative_index(raw_p2c, delta_ji)
# transpose/broadcast convention은 구현에 따라 다름
```

relative table 저장은 `O(kd)`로 줄지만 최종 dense score matrix는 여전히 `O(N^2)`다. DeBERTa는 linear attention 방법이 아니다.

### 추가 비용

논문은 C2P/P2C 계산의 추가 complexity를 `O(Nkd)`로 설명한다. base/large 설정에서 일반 BERT/RoBERTa보다 계산량이 약 30% 증가한다고 보고한다.

projection을 content와 position 사이에 공유하면 parameter 수를 줄일 수 있다.

```math
W_{q,r}=W_{q,c},
\qquad
W_{k,r}=W_{k,c}.
```

base model ablation에서는 성능을 거의 유지하면서 parameter를 RoBERTa 수준으로 줄였다.

## Enhanced Mask Decoder

### Absolute position을 늦게 넣는다

BERT는 input embedding 단계에서 absolute position을 더한다.

```text
BERT:
content + absolute position
-> every Transformer layer
-> MLM head
```

DeBERTa는 Transformer stack에서 disentangled relative attention을 사용한 뒤 MLM decoding 직전에 absolute position을 넣는다.

```text
DeBERTa:
content
-> relative-position Transformer stack
-> hidden H + absolute position I
-> Enhanced Mask Decoder
-> MLM softmax
```

논문의 해석은 absolute position을 너무 일찍 섞으면 relative relation을 충분히 학습하는 데 방해가 될 수 있다는 것이다.

### EMD 입력

EMD는 두 입력을 받을 수 있다.

```text
H: previous Transformer hidden states
I: decoding에 필요한 보완 정보
```

논문에서는 첫 EMD layer의 `I`로 absolute position embedding을 사용한다. EMD를 두 layer 쌓고 두 layer가 weight를 공유한다.

### Masked token에만 적용

MLM은 15% 정도의 token만 예측한다. EMD는 masked position decoding에만 적용할 수 있으므로 전체 sequence에 두 decoder layer를 추가하는 것보다 저렴하다.

논문 계산은 다음 추가 비용을 제시한다.

```text
base, 12 layers:  약 3%
large, 24 layers: 약 2%
```

### DeBERTa-AP 비교

appendix의 language modeling 실험은 absolute position을 input에 넣은 DeBERTa-AP보다 EMD 방식 DeBERTa가 더 낮은 perplexity를 보였다.

| Model | WikiText-103 test PPL |
|---|---:|
| RoBERTa | 21.6 |
| DeBERTa-AP | 20.0 |
| DeBERTa | 19.9 |
| DeBERTa-MT | 19.5 |

차이는 작지만 absolute position의 주입 시점에 대한 논문 주장과 일치한다.

## MLM과 pre-training

기본 objective는 BERT와 같은 masked language modeling이다.

```math
\max_\theta
\sum_{i\in C}
\log P_\theta(x_i\mid \tilde X)
```

`C`는 masked token index 집합이다. BERT convention에 따라 선택된 15% 중 일부는 `[MASK]`, 일부는 random token, 일부는 원 token로 유지된다.

DeBERTa는 추가로 최대 길이 3의 span masking을 사용한다.

### Pre-training data

base/large model은 deduplication 이후 약 78GB를 사용한다.

| Source | 크기 |
|---|---:|
| English Wikipedia | 12GB |
| BookCorpus | 6GB |
| OpenWebText | 38GB |
| Stories | 31GB |

합계는 raw 합보다 작으며 dedup 이후 약 78GB다.

DeBERTa 1.5B는 약 160GB pre-training data와 128K vocabulary를 사용했다.

## Model size

| Model | Layers | Hidden | FFN | Heads | Parameters |
|---|---:|---:|---:|---:|---:|
| DeBERTa-base | 12 | 768 | 3072 | 12 | 약 134M |
| DeBERTa-large | 24 | 1024 | 4096 | 16 | 수억 parameter |
| DeBERTa-900M | - | - | - | - | 900M |
| DeBERTa-1.5B | 48 | 1536 | 6144 | 24 | 1.5B |

모든 설정의 head dimension은 64다.

### 1.5B에서 추가된 변경

대형 모델은 단순히 layer만 늘린 것이 아니다.

- content/position projection matrix 공유
- 첫 Transformer layer 옆에 convolution layer 추가
- 128K vocabulary
- 더 큰 160GB dataset
- SuperGLUE fine-tuning에서 SiFT 사용

따라서 1.5B 결과 전체를 disentangled attention과 EMD만의 효과로 해석하면 안 된다.

## Scale-Invariant Fine-Tuning

### Virtual adversarial training의 문제

virtual adversarial training은 input embedding에 작은 perturbation을 추가하고 원본과 perturbation 입력의 output distribution이 비슷하도록 regularize한다.

그러나 word embedding norm은 token과 model size에 따라 다르다. 같은 perturbation 크기도 상대적 영향이 달라질 수 있다.

### SiFT

SiFT는 embedding을 normalize한 뒤 perturbation을 적용한다.

```text
word embedding
-> normalize
-> adversarial perturbation
-> consistency regularization
```

scale에 덜 민감해 큰 모델의 fine-tuning 안정성을 높이려는 방법이다.

중요한 범위 제한이 있다. 논문에서 SiFT는 **DeBERTa-1.5B의 SuperGLUE task에만 적용**되었다. 모든 DeBERTa 결과에 사용된 기본 구성 요소는 아니다.

## 실험 결과

### GLUE large model

| Model | GLUE average |
|---|---:|
| BERT-large | 84.05 |
| RoBERTa-large | 88.82 |
| XLNet-large | 89.15 |
| ELECTRA-large | 89.46 |
| DeBERTa-large | 90.00 |

DeBERTa-large는 MNLI matched/mismatched `91.1/91.1`, RTE `88.3`, MRPC `91.9`를 기록했다.

논문은 RoBERTa/XLNet이 약 160GB와 더 많은 training sample을 사용한 반면 DeBERTa-large는 약 78GB, 1M step, batch 2K로 학습했다고 강조한다.

### Reading comprehension와 NLU

| Benchmark | RoBERTa-large | DeBERTa-large |
|---|---:|---:|
| MNLI-m | 90.2 | 91.1 |
| SQuAD v1.1 F1 | 94.6 | 95.5 |
| SQuAD v2.0 F1 | 89.4 | 90.7 |
| RACE accuracy | 83.2 | 86.8 |
| ReCoRD F1 | 90.6 | 91.4 |
| SWAG accuracy | 89.9 | 90.8 |

### Base model

| Model | MNLI-m | SQuAD v1.1 F1/EM | SQuAD v2.0 F1/EM |
|---|---:|---:|---:|
| RoBERTa-base | 87.6 | 91.5/84.6 | 83.7/80.5 |
| DeBERTa-base | 88.8 | 93.1/87.2 | 86.2/83.1 |

base에서도 일관된 개선을 보여 architecture 차이가 scale에만 의존하지 않는다는 근거를 제공한다.

## Ablation study

분석용 base model은 Wikipedia+BookCorpus, 1M steps, batch 256으로 학습되었다. main base 결과와 data/batch가 다르므로 표 사이 숫자를 직접 혼합하면 안 된다.

| Model | MNLI-m | SQuAD v1.1 F1 | SQuAD v2.0 F1 | RACE |
|---|---:|---:|---:|---:|
| RoBERTa reimplementation | 84.9 | 91.1 | 79.5 | 66.8 |
| DeBERTa-base | 86.3 | 92.1 | 82.5 | 71.7 |
| without EMD | 86.1 | 91.8 | 81.3 | 70.3 |
| without C2P | 85.9 | 91.6 | 81.3 | 69.3 |
| without P2C | 86.0 | 91.7 | 80.8 | 69.6 |
| without EMD and C2P | 85.8 | 91.5 | 80.3 | 68.1 |
| without EMD and P2C | 85.8 | 91.3 | 80.2 | 68.5 |

세 구성 요소 중 하나를 제거해도 모든 benchmark에서 성능이 하락했다.

RACE에서는 full model `71.7`에서

- EMD 제거: `70.3`
- C2P 제거: `69.3`
- P2C 제거: `69.6`

로 떨어졌다. positional interaction이 긴 reading comprehension에서 특히 중요한 것으로 해석할 수 있다.

## SuperGLUE와 human baseline

DeBERTa-1.5B+SiFT는 SuperGLUE test macro average `89.9`, ensemble은 `90.3`을 기록했다. 당시 표의 human baseline은 `89.8`이었다.

논문도 이 수치가 human-level language understanding을 의미하지 않는다고 명시한다. benchmark average는 제한된 task와 metric의 집합이며 인간의 compositional generalization과는 거리가 있다.

## Long sequence에 대한 해석

appendix는 relative distance clipping과 layer stacking을 근거로 더 긴 sequence를 처리할 수 있다고 설명하고, RACE에서 length `512 -> 768`로 늘렸을 때 accuracy가 `86.3 -> 86.8`로 개선됨을 보였다.

하지만 다음을 구분해야 한다.

```text
positional representation range
!=
attention compute complexity
```

relative offset이 `k`에서 clip되어도 content-content attention은 dense다. sequence length `N`에 대한 score memory와 compute는 여전히 `O(N^2)`다. DeBERTa의 position scheme이 긴 입력에 정의된다는 것과 효율적인 long-context architecture라는 것은 다른 주장이다.

## 구현 관점의 전체 흐름

```text
1. token content embedding 생성
2. relative position index matrix delta(i,j) 생성
3. content Q/K/V projection
4. shared relative table에서 Qr/Kr projection
5. C2C score 계산
6. Qc @ Kr^T 후 delta(i,j) gather로 C2P 계산
7. Kc @ Qr^T 후 delta(j,i) gather로 P2C 계산
8. 세 score를 합하고 sqrt(3Dh)로 scaling
9. attention mask와 softmax 적용
10. Vc weighted sum
11. Transformer layer 반복
12. masked positions에서 absolute position과 EMD 결합
13. MLM softmax로 token 복원
```

### 축약 pseudocode

```python
def disentangled_attention(h, relative_table, delta_ij, mask):
    qc = project_q_content(h)
    kc = project_k_content(h)
    vc = project_v_content(h)

    qr = project_q_relative(relative_table)
    kr = project_k_relative(relative_table)

    c2c = qc @ kc.transpose(-1, -2)

    raw_c2p = qc @ kr.transpose(-1, -2)
    c2p = gather_relative(raw_c2p, delta_ij)

    raw_p2c = kc @ qr.transpose(-1, -2)
    p2c = gather_relative_reverse(raw_p2c, delta_ij)

    logits = (c2c + c2p + p2c) / math.sqrt(3 * head_dim)
    logits = logits + mask
    prob = logits.softmax(dim=-1)
    return prob @ vc
```

실제 구현은 batch/head dimension, transpose, broadcasting, gather index layout을 세심하게 맞춰야 한다.

## 구현 체크리스트

```text
1. relative distance가 i-j인지 j-i인지 정의했는가?
2. delta(i,j) clipping이 [0, 2k) 범위인가?
3. C2P는 delta(i,j), P2C는 delta(j,i)를 쓰는가?
4. scaling이 sqrt(3 * head_dim)인가?
5. P2P 항을 실수로 추가하지 않았는가?
6. relative embedding table이 layer 간 공유되는 checkpoint인가?
7. position projection을 content projection과 공유하는 variant인지 확인했는가?
8. gather index dtype과 device가 score tensor와 맞는가?
9. padding/causal mask가 disentangled score 합 이후 적용되는가?
10. EMD가 masked position에만 적용되는가?
11. absolute position이 encoder input에 중복 추가되지 않는가?
12. EMD layer 수와 weight sharing이 checkpoint와 맞는가?
13. span masking과 standard MLM 비율이 data pipeline과 일치하는가?
14. long sequence에서 O(N^2) memory를 별도로 확인했는가?
```

## 자주 헷갈리는 지점

### “Disentangled representation은 hidden state 두 개를 layer마다 따로 전달하는가?”

content hidden state와 relative position table을 분리해 score를 계산한다. token마다 독립적인 absolute position hidden sequence 두 개를 병렬로 계속 전달하는 구조로 단순화하면 부정확하다.

### “Position-to-position 항도 사용하는가?”

개념적 네 항에는 존재하지만 실제 DeBERTa attention에서는 제거한다.

### “C2P와 P2C는 같은 score를 transpose한 것인가?”

아니다. projection과 relative direction이 다르다. 하나를 단순 transpose해 대체하면 일반적으로 같지 않다.

### “EMD는 downstream inference에도 항상 필요한가?”

EMD는 pre-training MLM의 masked token decoding을 강화하는 구성이다. downstream task에서 사용하는 head와 구조는 task에 따라 다르다.

### “SiFT가 DeBERTa architecture의 필수 부분인가?”

아니다. fine-tuning regularization이며 논문에서 제한적으로 1.5B SuperGLUE에 적용되었다.

### “Relative clipping으로 attention이 local sparse해지는가?”

아니다. 먼 offset이 같은 relative embedding을 쓸 뿐 content attention은 dense하다.

## 장점과 핵심 기여

### 1. Content와 position 상호작용을 명시적으로 분리했다

하나의 혼합 vector dot product 대신 C2C, C2P, P2C를 각각 모델링한다.

### 2. Relative direction의 양쪽 관계를 포함했다

기존 relative attention에서 자주 생략되던 position-to-content 항을 추가했다.

### 3. Absolute position의 주입 시점을 재설계했다

relative relation을 encoder 전체에서 학습한 뒤 MLM decoding 시점에 absolute position을 보완한다.

### 4. Pair별 relative vector 복제를 피했다

`2k` table과 gather를 사용해 relative representation 저장을 `O(kd)`로 줄였다.

### 5. 다양한 scale에서 실험했다

base, large, 900M, 1.5B와 여러 benchmark에서 효과를 보이고 ablation으로 각 component의 기여를 확인했다.

## 한계와 비판적 관점

### 계산량 증가

C2P와 P2C 두 score가 추가되어 standard attention보다 compute가 증가한다. 논문은 BERT/RoBERTa 대비 약 30%를 보고한다.

### Gather 구현의 복잡성

relative direction, clipping, head/batch broadcasting을 잘못 구현하기 쉽다. custom fused kernel이 없으면 메모리 이동도 병목이 될 수 있다.

### EMD의 개선 폭이 task마다 다르다

RACE와 SQuAD v2.0에서는 유의미하지만 MNLI에서는 차이가 작다. absolute position 보완의 가치가 task 특성에 의존한다.

### 대형 모델 비교의 confounder

1.5B에서는 dataset, vocabulary, convolution, projection sharing, SiFT가 함께 바뀐다. 단일 architecture component의 순수 scale effect를 분리하기 어렵다.

### Benchmark saturation

SuperGLUE human baseline 초과는 중요한 결과지만 일반 언어 지능의 증거는 아니다. 논문도 compositional generalization이 남아 있음을 인정한다.

### Long-context 효율을 해결하지 않는다

relative representation은 길이에 비교적 유연하지만 dense attention의 quadratic cost는 그대로다.

## 다른 relative position 방법과 비교

| 방법 | 핵심 score 구성 | 특징 |
|---|---|---|
| Shaw | content-content + content-position | edge embedding을 key/value에 추가 |
| Transformer-XL | C2C, C2P, global content/position bias | recurrence와 함께 사용 |
| T5 | bucketed scalar bias | 매우 단순하고 layer 공유 |
| DeBERTa | C2C + C2P + P2C | content/position 양방향 상호작용 |
| RoPE | Q/K rotation | dot product에 relative phase 유도 |
| ALiBi | distance-proportional scalar bias | 별도 position embedding 없음 |

DeBERTa의 특징은 단순히 relative embedding을 쓴 것이 아니라 **position-to-content 항까지 구조적으로 분리했다**는 점이다.

## 후속 연구와의 연결

DeBERTa-v2/v3 계열은 tokenizer, convolution, projection sharing, pre-training objective를 발전시켰다. 특히 DeBERTa-v3는 ELECTRA-style replaced token detection을 활용해 pre-training efficiency를 높였다.

논문 appendix도 ELECTRA의 RTD와 DeBERTa architecture를 결합한 실험을 다룬다. 이는 architecture와 objective가 독립적으로 결합될 수 있음을 보여준다.

disentangled attention의 관점은 이후 position/content factorization, relation-aware attention, 2D relative bias 설계와도 연결된다.

## 온디바이스 및 비전 관점의 메모

### 온디바이스 비용

DeBERTa는 같은 hidden size의 BERT보다 relative projection과 두 score 항이 추가된다. latency가 중요한 장치에서는 다음을 검토해야 한다.

- projection sharing
- relative table quantization
- gather kernel fusion
- maximum sequence length 제한
- head pruning과 distillation

parameter 수만 같아도 runtime은 같지 않다. C2P/P2C의 memory access pattern을 실제 hardware에서 측정해야 한다.

### 2D vision으로의 확장

image patch `i`, `j`의 relative position은 하나의 scalar `i-j`보다 `(\Delta r,\Delta c)`가 자연스럽다.

```math
A_{i,j}^{c2p}
=Q_{c,i}K_{r,\Delta r}^\top
+Q_{c,i}K_{c,\Delta c}^\top
```

P2C도 row/column 방향을 반대로 적용할 수 있다. 다만 score 항 수가 늘어 compute와 parameter가 커진다.

### EMD와 masked image modeling

mask된 patch를 복원할 때 encoder는 relative spatial relation을 학습하고 decoder에서 absolute 2D coordinate를 보완하는 구조를 생각할 수 있다.

```text
relative geometry in encoder
+ absolute coordinate at reconstruction head
```

이는 DeBERTa의 EMD 철학을 vision masked modeling에 옮긴 형태다.

## 개인 학습/연구 메모

DeBERTa를 기억할 때 가장 중요한 식은 다음이다.

```math
\tilde A_{i,j}
=
Q_i^cK_j^{c\top}
+Q_i^cK_{\delta(i,j)}^{r\top}
+K_j^cQ_{\delta(j,i)}^{r\top}
```

각 항을 한 문장으로 읽으면 다음과 같다.

```text
C2C: 이 content끼리 관련 있는가?
C2P: 이 query content는 어느 상대 위치를 선호하는가?
P2C: 이 query 위치에서 어떤 key content가 적절한가?
```

그리고 EMD의 핵심은 다음이다.

```text
relative position throughout representation learning
absolute position only when masked-token decoding needs it
```

논문의 전체 기여는 다음 결합으로 정리할 수 있다.

```text
separate content and position
+ bidirectional content-position interactions
+ efficient relative table gathering
+ late absolute-position decoding
= stronger BERT-style encoder pre-training
```

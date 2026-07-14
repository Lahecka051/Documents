# 06. Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

## 논문 정보

- 원본 파일: `Attention/06_Transformer_XL.pdf`
- 제목: Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
- 저자: Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov
- 발표: ACL 2019
- 주제: long-context language modeling, segment recurrence, state reuse, relative positional encoding
- 핵심 키워드: memory, stop-gradient, context fragmentation, relative shift, effective context length

## 한눈에 보는 요약

Transformer-XL은 Transformer language model이 고정 길이 segment 단위로 문장을 처리할 때 생기는 두 문제를 해결한다.

1. **최대 문맥 길이가 segment 길이에 묶이는 문제**
2. **segment 경계에서 앞 문맥이 사라지는 context fragmentation 문제**

해결책은 이전 segment의 hidden state를 현재 segment의 key와 value에 재사용하는 **segment-level recurrence**다. 이전 hidden state는 memory로 저장되지만 gradient는 끊는다. 따라서 학습 그래프를 무한히 길게 유지하지 않으면서도 현재 segment가 과거 표현을 읽을 수 있다.

그러나 단순히 hidden state만 재사용하면 absolute positional encoding이 충돌한다. 이전 segment의 첫 token과 현재 segment의 첫 token이 같은 절대 위치 embedding을 갖기 때문이다. 논문은 attention score에 상대 거리 `i-j`를 직접 넣는 새로운 relative positional encoding을 도입한다.

핵심 구조는 다음과 같다.

```text
previous segment hidden states
        |
        | cache + stop-gradient
        v
memory [M, d] ----+
                  +--> key/value source [M + L, d]
current [L, d] ---+
        |
        +--------------> query source [L, d]
```

이 구조는 다음 효과를 만든다.

- 학습 segment보다 긴 문맥을 추론 시 사용할 수 있다.
- segment 시작 token도 과거 문맥을 본다.
- 이전 계산 결과를 재사용하므로 autoregressive 평가가 훨씬 빨라진다.
- absolute position 대신 relative distance를 사용해 memory 길이가 달라져도 일반화한다.

논문의 가장 중요한 메시지는 단순히 “KV cache를 쓴다”가 아니다. **숨은 상태 재사용과 상대 위치 표현이 함께 있어야 시간적 일관성이 유지된다**는 점이다.

## 연구 배경과 문제의식

### Vanilla Transformer language model의 segment 처리

긴 corpus 전체를 한 번에 self-attention으로 처리하면 메모리와 계산량이 감당되지 않는다. 따라서 현실적인 학습은 sequence를 길이 `L`의 segment로 자른다.

```text
corpus:
x1 x2 x3 x4 | x5 x6 x7 x8 | x9 x10 x11 x12 | ...

training:
segment 1만 독립 처리
segment 2만 독립 처리
segment 3만 독립 처리
```

각 segment를 독립적으로 처리하면 forward와 backward 모두 경계를 넘지 않는다. self-attention 자체는 segment 내부에서 모든 위치를 직접 연결하지만, 경계 밖 token은 아예 입력에 존재하지 않는다.

### 문제 1: dependency length의 상한

segment 길이가 `L`이면 한 layer가 볼 수 있는 최대 과거 범위도 기본적으로 `L`에 묶인다. self-attention의 최단 경로 장점은 segment 내부에서만 유효하다.

```text
long-range dependency가 1,000 token 떨어져 있음
segment length = 128

vanilla model:
두 token이 서로 다른 segment에 있으면 직접 연결 불가능
```

### 문제 2: context fragmentation

고정 길이 chunk는 문장, 문단, 문서 경계를 존중하지 않는다. segment가 의미 단위 중간에서 시작하면 첫 token들은 필요한 왼쪽 문맥 없이 예측된다.

```text
... the committee rejected | the proposal because ...
                           ^ segment boundary
```

두 번째 segment의 `the proposal`은 앞 segment의 주어와 동사를 보지 못한다. 이 문제는 장거리 dependency가 없는 데이터에서도 발생한다. 즉, long-context 능력과 별개로 segment 시작부의 조건부 확률 추정이 나빠진다.

### 문제 3: 느린 평가

vanilla Transformer가 매 token마다 길이 `L`의 sliding window를 다시 계산한다고 생각해 보자.

```text
step t:   x[t-L+1 : t] 전체 계산
step t+1: x[t-L+2 : t+1] 전체 계산
```

두 window는 대부분 겹치지만 과거 hidden state를 재사용하지 않는다. 논문은 이 중복 계산도 제거하려 한다.

## 핵심 아이디어 1: Segment-level recurrence

### 이전 hidden state를 memory로 사용

연속된 두 segment를 다음처럼 둔다.

```math
s_\tau=[x_{\tau,1},\ldots,x_{\tau,L}],
\qquad
s_{\tau+1}=[x_{\tau+1,1},\ldots,x_{\tau+1,L}]
```

`n`번째 layer에서 이전 segment의 hidden state는 다음과 같다.

```math
h_\tau^n \in \mathbb{R}^{L\times d}
```

현재 segment를 계산할 때 이전 layer의 memory와 현재 hidden state를 길이 축으로 붙인다.

```math
\tilde h_{\tau+1}^{n-1}
=
[\operatorname{SG}(m_\tau^{n-1})\circ h_{\tau+1}^{n-1}]
```

여기서

- `SG`는 stop-gradient다.
- `\circ`는 length dimension concatenation이다.
- `m_\tau^{n-1}`은 이전 segment들에서 보존한 memory다.

query는 현재 segment에서만 만들고, key와 value는 memory를 포함한 확장 문맥에서 만든다.

```math
Q = h_{\tau+1}^{n-1}W_q
```

```math
K = \tilde h_{\tau+1}^{n-1}W_k,
\qquad
V = \tilde h_{\tau+1}^{n-1}W_v
```

![Transformer-XL의 segment recurrence와 relative attention](https://github.com/user-attachments/assets/e94bdc24-9e1d-4cf7-ad5c-ad255f9f0874)

### Tensor shape

batch와 head 차원을 포함하면 대표적인 shape는 다음과 같다.

```text
B: batch size
H: number of heads
L: current segment length
M: memory length
D: model dimension
Dh: head dimension = D / H

current hidden: [B, L, D]
memory:         [B, M, D]
extended:       [B, M + L, D]

Q: [B, H, L,     Dh]
K: [B, H, M + L, Dh]
V: [B, H, M + L, Dh]

score:  [B, H, L, M + L]
output: [B, H, L, Dh]
```

중요한 비대칭은 `Q` 길이가 `L`, `K/V` 길이가 `M+L`이라는 점이다. memory token에 대한 새 출력은 계산하지 않는다. memory는 현재 token이 읽는 과거 문맥이다.

### 왜 stop-gradient인가

memory를 계산 그래프에 그대로 연결하면 segment 수가 늘어날수록 역전파 그래프도 계속 길어진다. 메모리 사용량이 커지고 사실상 전체 corpus BPTT가 된다.

Transformer-XL은 이전 hidden state의 값은 사용하지만 gradient는 전달하지 않는다.

```text
forward information: crosses segment boundary
backward gradient:    stops at segment boundary
```

이는 truncated BPTT와 닮았지만 RNN의 마지막 state 하나가 아니라 **hidden state sequence 전체**를 cache한다는 차이가 있다.

### 여러 layer를 통해 문맥이 더 멀리 전달되는 방식

현재 `n`번째 layer는 이전 segment의 `n-1`번째 layer memory를 읽는다. 따라서 segment가 하나 넘어갈 때마다 recurrence 경로는 layer 하나 아래로 이동한다.

논문은 최대 dependency path가 layer 수 `N`과 segment 길이 `L`에 대해 대략 다음처럼 증가한다고 설명한다.

```math
O(NL)
```

한 번의 attention이 `M+L` 길이를 보지만, 여러 segment와 layer를 통해 더 오래된 정보가 hidden representation에 요약되어 전달될 수 있다.

## 핵심 아이디어 2: 왜 absolute position을 그대로 쓸 수 없는가

표준 Transformer에서 각 segment가 동일한 absolute position table `U_{1:L}`을 사용한다고 하자.

```math
h_{\tau}=f(h_{\tau-1}, E_{s_\tau}+U_{1:L})
```

```math
h_{\tau+1}=f(h_{\tau}, E_{s_{\tau+1}}+U_{1:L})
```

그러면 다음 두 token이 같은 position embedding을 공유한다.

```text
previous segment position 1 -> U_1
current segment position 1  -> U_1
```

memory를 현재 segment 앞에 붙였을 때 실제 거리는 다르지만 hidden state에 기록된 절대 좌표는 중복된다. 모델은 `x_{\tau,j}`와 `x_{\tau+1,j}`의 시간적 차이를 안정적으로 구분하기 어렵다.

Transformer-XL은 “token이 corpus에서 몇 번째인가”보다 “query에서 key가 얼마나 과거에 있는가”를 score에 넣는다.

```math
\text{relative distance}=i-j
```

## 핵심 아이디어 3: Relative positional attention

### Absolute attention score의 네 항 분해

content embedding을 `E_{x_i}`, absolute position embedding을 `U_i`라고 하면 표준 attention의 query-key score는 네 항으로 전개할 수 있다.

```math
A^{abs}_{i,j}
=
E_{x_i}^{\top}W_q^{\top}W_kE_{x_j}
+E_{x_i}^{\top}W_q^{\top}W_kU_j
+U_i^{\top}W_q^{\top}W_kE_{x_j}
+U_i^{\top}W_q^{\top}W_kU_j
```

각 항의 의미는 다음과 같다.

1. content가 content를 찾는 항
2. content가 key position을 찾는 항
3. query position에 따른 global content bias
4. query와 key position 사이의 positional bias

### Transformer-XL의 재매개화

논문은 절대 위치 `U_i`, `U_j`를 상대 거리 embedding `R_{i-j}`와 두 global bias vector `u`, `v`로 바꾼다.

```math
A^{rel}_{i,j}
=
E_{x_i}^{\top}W_q^{\top}W_{k,E}E_{x_j}
+E_{x_i}^{\top}W_q^{\top}W_{k,R}R_{i-j}
+u^{\top}W_{k,E}E_{x_j}
+v^{\top}W_{k,R}R_{i-j}
```

이를 projection 이후 표기로 단순화하면 다음처럼 읽을 수 있다.

```math
A_{i,j}
=q_i^{\top}k_j
+q_i^{\top}r_{i-j}
+u^{\top}k_j
+v^{\top}r_{i-j}
```

네 항은 다음 역할을 한다.

| 항 | 의미 |
|---|---|
| $q_i^\top k_j$ | content-based addressing |
| $q_i^\top r_{i-j}$ | query content에 따른 relative position bias |
| $u^\top k_j$ | query 위치와 무관한 global content bias |
| $v^\top r_{i-j}$ | global relative position bias |

`W_{k,E}`와 `W_{k,R}`를 분리한 것도 중요하다. content key와 positional key가 서로 다른 projection을 학습한다.

### Shaw et al.과의 차이

05번 논문의 Shaw relative position은 주로 content-content 항과 content-position 항을 사용한다. Transformer-XL은 여기에 `u`, `v` 두 bias 항을 추가하고 sinusoidal relative embedding을 별도 positional projection에 통과시킨다.

```text
Shaw-style:
content-content + content-position

Transformer-XL:
content-content + content-position
+ global content bias + global position bias
```

또한 `R`은 sinusoidal encoding이므로 학습 때 보지 않은 더 긴 상대 거리에도 값을 생성할 수 있다. 이것이 추론 시 memory 길이를 늘리는 데 유리하다.

## Relative shift

### Naive 계산의 문제

모든 `(i,j)` 쌍마다 `R_{i-j}`를 모아 projection하면 position 관련 score를 명시적으로 구성해야 한다. score 자체는 어차피 `L(M+L)` 크기지만, pair별 positional tensor를 만들면 추가 메모리가 매우 커진다.

거리 값은 다음 범위에만 존재한다.

```math
i-j \in \{0,1,\ldots,M+L-1\}
```

따라서 모든 가능한 relative key를 한 번만 만든 뒤 행렬 곱을 수행할 수 있다.

```math
Q_{\mathrm{pos}}=qR_{\mathrm{projected}}^\top,
\qquad
\operatorname{shape}(Q_{\mathrm{pos}})=[L,M+L].
```

문제는 행 `i`마다 필요한 거리 정렬이 한 칸씩 다르다는 점이다. 논문은 행별 left shift 관계를 이용한다. 구현에서는 흔히 dummy column을 붙이고 reshape한 다음 slice하는 `relative_shift` 연산으로 나타난다.

```python
def relative_shift(x):
    # x: [batch, heads, query_len, key_len]
    zero = x.new_zeros((*x.shape[:-1], 1))
    x_padded = torch.cat([zero, x], dim=-1)

    b, h, q, k1 = x_padded.shape
    x_padded = x_padded.view(b, h, k1, q)
    x = x_padded[:, :, 1:, :].view(b, h, q, k1 - 1)
    return x
```

실제 library마다 dimension order와 slice가 다를 수 있다. 핵심은 **행별 relative distance index를 materialize하지 않고 view와 slice로 정렬한다**는 것이다.

## 한 layer의 전체 계산

논문 표기를 구현 순서로 정리하면 다음과 같다.

```text
1. memory와 current hidden을 concat
2. current에서 Q 생성
3. extended context에서 content K,V 생성
4. relative distance table에서 positional K 생성
5. content score와 position score 계산
6. relative shift
7. causal mask 적용
8. softmax와 value aggregation
9. residual + layer normalization
10. position-wise FFN
11. 현재 hidden 일부를 다음 segment memory로 저장
```

수식으로는 다음과 같다.

```math
\tilde h^{n-1}=[\operatorname{SG}(m^{n-1})\circ h^{n-1}]
```

```math
q^n=h^{n-1}W_q^n,
\quad
k^n=\tilde h^{n-1}W_{k,E}^n,
\quad
v^n=\tilde h^{n-1}W_v^n
```

```math
A_{i,j}^n
=(q_i^n+u)^\top k_j^n
+(q_i^n+v)^\top W_{k,R}^nR_{i-j}
```

```math
a^n=\operatorname{MaskedSoftmax}(A^n)V^n
```

```math
o^n=\operatorname{LayerNorm}(a^n+h^{n-1})
```

```math
h^n=\operatorname{PositionwiseFFN}(o^n)
```

논문의 원식에는 multi-head scaling, dropout, output projection 등 구현 세부가 생략되어 있다. 실제 구현에서는 보통 score를 `sqrt(d_h)`로 나누고 residual/dropout을 적용한다.

## Causal mask와 memory

현재 위치 `i`는 다음을 볼 수 있다.

- 모든 memory 위치
- 현재 segment에서 `i`보다 과거인 위치
- 현재 위치 자신

미래 current token은 볼 수 없다.

```text
keys:    [memory ........][current ........]
query i: [all visible.....][<= i visible][future masked]
```

memory는 이미 과거 segment에서 계산되었으므로 별도의 triangular mask가 필요하지 않다. mask를 만들 때 `M` offset을 반영하지 않으면 memory 일부를 잘못 가리거나 미래 token을 노출할 수 있다.

## 작은 예시

segment 길이 `L=4`, memory 길이 `M=4`라고 하자.

```text
memory:  x1 x2 x3 x4
current: x5 x6 x7 x8
```

현재 첫 token `x5`의 query가 보는 key는 다음과 같다.

```text
x1 x2 x3 x4 x5
```

상대 거리는 다음과 같이 계산된다.

```text
key:      x1  x2  x3  x4  x5
distance:  4   3   2   1   0
```

`x8`은 memory와 current의 모든 과거 token을 본다.

```text
key:      x1 x2 x3 x4 x5 x6 x7 x8
distance:  7  6  5  4  3  2  1  0
```

절대 position id를 segment마다 다시 0부터 시작해도 attention score는 `i-j`로 정렬되므로 두 segment의 위치가 충돌하지 않는다.

## 학습과 추론의 차이

### 학습

논문 실험에서는 보통 training memory length를 segment length와 비슷하게 설정한다.

```text
segment length = L
training memory length = M_train
loss: current segment token에 대해서만 계산
memory: stop-gradient
```

gradient 길이는 current segment에 제한되지만 forward context는 memory까지 확장된다.

### 추론

추론에서는 memory length를 학습보다 더 크게 둘 수 있다.

```text
M_eval > M_train
```

Transformer-XL의 relative encoding은 sinusoidal distance를 사용하므로 더 긴 상대 거리에도 정의된다. 논문은 WikiText-103에서 training attention보다 긴 evaluation attention이 perplexity를 개선함을 보였다.

다만 “더 긴 memory를 넣으면 항상 좋아진다”는 뜻은 아니다. positional formulation, 학습 분포, 모델 용량에 따라 너무 오래된 memory는 잡음이 될 수 있다.

## 계산 복잡도와 메모리

현재 segment 길이가 `L`, memory가 `M`이면 attention score 크기는 다음과 같다.

```math
O(L(M+L))
```

표준 `L` 길이 causal attention의 `O(L^2)`보다 memory만큼 늘어난다. 따라서 Transformer-XL은 quadratic attention을 선형 attention으로 바꾸는 방법이 아니다.

```text
vanilla segment attention: O(L^2)
Transformer-XL:            O(L(M + L))
```

추론에서의 장점은 과거 segment를 다시 layer 전체에 통과시키지 않는다는 데 있다. 과거 hidden state를 cache해 새 segment의 query만 계산한다.

### 현대 KV cache와의 차이

Transformer-XL memory는 각 layer의 hidden state sequence를 저장하고 다음 segment에서 다시 key/value projection에 사용한다. 현대 decoder-only LLM의 KV cache는 projection이 끝난 K와 V 자체를 저장하는 경우가 일반적이다.

| 구분 | Transformer-XL memory | 일반적 KV cache |
|---|---|---|
| 저장 대상 | layer hidden states | projected K/V |
| 단위 | segment | token 또는 block |
| positional 처리 | Transformer-XL relative score | RoPE, ALiBi 등 다양 |
| 주요 목적 | segment 경계 연결과 long context | autoregressive 중복 계산 제거 |

개념적으로는 모두 과거 계산을 재사용하지만 자료구조와 positional convention이 다르다.

## 실험 결과

논문은 word-level과 character-level language modeling을 모두 평가했다.

| 데이터셋 | Transformer-XL 결과 | 지표 |
|---|---:|---|
| enwik8 | 0.99 | bpc, 낮을수록 좋음 |
| text8 | 1.08 | bpc |
| WikiText-103 | 18.3 | perplexity |
| One Billion Word | 21.8 | perplexity |
| Penn Treebank | 54.52 | perplexity, 별도 2-step finetuning 없음 |

One Billion Word는 문장 순서가 섞여 long-range dependency가 거의 없다. 그럼에도 recurrence를 제거하면 perplexity가 `25.2`에서 `27.1`로 악화되었다. 이는 성능 개선이 장거리 정보뿐 아니라 context fragmentation 완화에서도 온다는 근거다.

### WikiText-103 ablation

128M 설정에서 주요 비교는 다음과 같다.

| Recurrence | Encoding | PPL best | 최적 attention length |
|---|---|---:|---:|
| 사용 | Transformer-XL | 26.77 | 500 |
| 사용 | Shaw et al. | 27.94 | 256 |
| 제거 | Transformer-XL | 29.02 | 260 |
| 제거 | Shaw et al. | 29.75 | 120 |
| 제거 | Vaswani absolute | 30.97 | 120 |

두 결론이 중요하다.

1. recurrence를 제거하면 성능이 크게 나빠진다.
2. evaluation attention length를 늘렸을 때 개선되는 것은 논문의 relative encoding을 쓴 경우다.

151M 설정에서는 attention length를 `300 -> 450 -> 640`으로 늘릴수록 perplexity가 `23.35 -> 23.16 -> 23.09`로 개선되었다.

### Relative Effective Context Length

기존 Effective Context Length는 이미 짧은 문맥에서 좋은 모델이 상대적으로 불리할 수 있다. 논문은 모델 집합에서 가장 좋은 짧은-context baseline과 비교하는 RECL을 제안한다.

`r=0.1` 기준 결과는 다음과 같다.

| 모델 | RECL |
|---|---:|
| Transformer-XL 151M | 900 |
| QRNN | 500 |
| LSTM | 400 |
| vanilla Transformer | 128 |

논문 표현대로 Transformer-XL의 dependency length는 RNN보다 약 80%, vanilla Transformer보다 약 450% 길었다.

### 평가 속도

attention length별 vanilla Transformer의 상대적 slowdown은 다음과 같다.

| Attention length | vanilla가 더 느린 배수 |
|---:|---:|
| 800 | 363x |
| 1,800 | 773x |
| 2,800 | 1,409x |
| 3,800 | 1,874x |

이 수치는 두 모델의 일반적인 모든 구현을 대표하는 절대 속도가 아니다. 논문이 사용한 per-token evaluation 절차에서 vanilla sliding window가 반복 계산하는 비용과 비교한 결과다.

## 구현 예시

아래 코드는 개념을 보여주는 축약 형태다.

```python
def forward_segment(x, memories):
    # x: [B, L]
    h = token_embedding(x)           # [B, L, D]
    new_memories = []

    for layer_id, layer in enumerate(layers):
        mem = memories[layer_id]     # [B, M, D]
        mem = mem.detach()

        # Q는 current에서, K/V는 memory + current에서 생성
        h_next = layer(
            current=h,
            memory=mem,
            relative_positions=True,
        )

        # 다음 segment용 memory에는 현재 layer 입력/출력 convention을
        # 구현과 checkpoint에 맞춰 저장해야 한다.
        cat = torch.cat([mem, h], dim=1)
        new_mem = cat[:, -memory_length:].detach()
        new_memories.append(new_mem)

        h = h_next

    return h, new_memories
```

실제 구현에서 가장 위험한 부분은 memory가 layer input인지 output인지, pre-LN인지 post-LN인지다. checkpoint와 다른 convention을 쓰면 shape는 맞아도 결과가 달라진다.

## 구현 체크리스트

```text
1. memory에 detach/stop-gradient가 적용되는가?
2. Q length는 L, K/V length는 M+L인가?
3. causal mask가 memory offset M을 올바르게 반영하는가?
4. relative distance의 부호가 i-j인지 j-i인지 일관적인가?
5. sinusoidal relative table의 index 순서가 뒤집혀 있지 않은가?
6. relative shift의 reshape dimension order가 맞는가?
7. memory는 batch의 문서 순서를 따라 이어지는가?
8. 문서 경계에서 memory를 reset하는가?
9. padding token이 memory에 누적되지 않는가?
10. training과 evaluation의 memory length가 명시되어 있는가?
11. mixed precision에서 mask의 음의 큰 값이 overflow를 만들지 않는가?
12. distributed training에서 각 batch stream의 memory가 섞이지 않는가?
```

### 문서 경계 reset

서로 관련 없는 문서 사이에서 memory를 유지하면 데이터 누수가 생긴다.

```text
document A end -> memory reset -> document B start
```

평가에서도 sample 경계, batch 재정렬, beam search 복제에 맞춰 memory를 관리해야 한다.

## 자주 헷갈리는 지점

### “Recurrence가 있으니 RNN인가?”

segment 사이에는 recurrence가 있지만 segment 내부 계산은 self-attention으로 병렬화된다. token 단위 RNN recurrence와는 다르다.

### “Gradient도 과거 segment까지 흐르는가?”

아니다. forward 정보만 memory를 통해 전달되고 gradient는 `SG`에서 끊긴다.

### “Memory를 늘리면 학습 context도 자동으로 길어지는가?”

추론 시 더 긴 memory를 넣을 수 있지만, 모델이 그 거리를 유용하게 활용하는지는 positional encoding과 학습 경험에 달려 있다.

### “Transformer-XL은 attention 복잡도를 선형으로 줄이는가?”

아니다. 현재 query와 `M+L` key 사이의 dense attention을 계산한다. long-context를 가능하게 하는 방법이지 sparse/linear attention 방법은 아니다.

### “Relative position embedding은 학습 가능한가?”

논문의 `R`은 sinusoidal encoding이라 학습 parameter가 아니다. 다만 `W_{k,R}`와 bias `u`, `v`는 학습된다.

### “Memory는 raw token embedding인가?”

아니다. 각 layer의 contextual hidden state sequence다.

## 장점과 핵심 기여

### 1. Segment 경계를 architecture 수준에서 연결했다

고정 길이 학습의 편의성을 유지하면서 이전 segment 정보를 현재 계산에 재사용한다.

### 2. Context fragmentation을 명확히 문제화했다

장거리 dependency뿐 아니라 segment 첫 부분의 정보 부족 자체가 성능 저하 원인임을 ablation으로 보였다.

### 3. State reuse와 맞는 positional encoding을 설계했다

memory recurrence만 추가하지 않고 absolute position 충돌을 분석해 상대 위치 score를 함께 제안했다.

### 4. 학습 길이보다 긴 evaluation context로 일반화했다

sinusoidal relative distance와 score 재매개화 덕분에 evaluation memory를 늘렸을 때 실제 perplexity 개선을 보였다.

### 5. Autoregressive 평가의 중복 계산을 크게 줄였다

과거 hidden state를 재사용해 sliding window 전체를 매 step 다시 계산하는 비효율을 제거했다.

## 한계와 비판적 관점

### Dense attention 비용은 남는다

`M`이 커질수록 score matrix는 `L(M+L)`로 증가한다. 매우 긴 context에서 memory와 bandwidth 병목이 커진다.

### Stop-gradient의 정보 병목

모델은 과거 memory를 읽지만 현재 loss가 과거 segment representation을 직접 수정하지 못한다. 아주 긴 dependency를 end-to-end로 학습하는 데 한계가 있다.

### Hidden state memory의 stale representation

과거 token은 현재 segment의 새로운 문맥을 반영해 다시 계산되지 않는다. 즉, memory는 과거 시점에서 고정된 representation이다.

### Batch stream 관리가 복잡하다

각 batch row가 동일 문서의 연속 segment를 유지해야 한다. shuffle, 문서 경계, padding, distributed sampler와 memory 상태가 함께 관리되어야 한다.

### 논문의 속도 배수는 비교 설정에 의존한다

1,800배 이상 speedup은 비효율적인 vanilla sliding-window 평가와 비교한 결과다. 현대적인 KV cache baseline과 직접 같은 의미로 읽으면 안 된다.

## 후속 연구와의 연결

Transformer-XL의 아이디어는 이후 여러 long-context 연구에 영향을 주었다.

- XLNet은 Transformer-XL의 recurrence와 relative attention을 사용한다.
- T5는 attention logit에 단순한 relative position bias를 더한다.
- DeBERTa는 content와 position을 분리한 disentangled attention을 사용한다.
- RoPE는 Q/K 회전으로 relative phase를 score에 유도한다.
- ALiBi는 relative distance에 선형 bias를 직접 더한다.
- 현대 LLM inference의 KV cache는 state reuse를 더 직접적인 K/V 저장 형태로 발전시켰다.

```text
Transformer-XL:
cached hidden state + relative score

modern decoder inference:
cached K/V + RoPE or another position scheme
```

## 온디바이스 및 비전 관점의 메모

### Streaming inference

Transformer-XL은 전체 과거 입력을 다시 인코딩하지 않고 고정 길이 memory만 유지할 수 있어 streaming 처리와 잘 맞는다.

```text
new chunk arrives
-> attend to fixed-size memory
-> update memory
-> discard oldest states
```

다만 hidden state memory는 layer마다 `M x D`를 저장한다. 작은 장치에서는 layer 수, dtype, memory length에 따라 footprint가 빠르게 커진다.

### Vision sequence 적용

video frame, image patch strip, event-camera stream처럼 긴 시각 sequence를 chunk로 나눌 때 recurrence를 적용할 수 있다. 그러나 문서 경계 대신 scene cut이나 clip boundary에서 memory reset 정책이 필요하다.

2D image에서는 단순한 1D relative distance `i-j`가 실제 공간 관계를 충분히 표현하지 못할 수 있다. row/column relative bias 또는 2D position scheme이 더 자연스럽다.

### Memory compression 가능성

오래된 hidden state를 그대로 저장하지 않고 pooling, quantization, low-rank projection으로 압축할 수 있다. 다만 score가 민감하게 변할 수 있으므로 perplexity와 latency를 함께 측정해야 한다.

## 개인 학습/연구 메모

이 논문에서 기억할 식은 다음 하나로 압축할 수 있다.

```math
A_{i,j}
=(q_i+u)^\top k_j
+(q_i+v)^\top r_{i-j}
```

그리고 구조는 다음 한 줄이다.

```text
Q from current, K/V from stop-gradient memory + current
```

Transformer-XL을 이해할 때는 세 주장을 분리해서 확인해야 한다.

1. recurrence가 장거리 정보를 전달하는가?
2. recurrence가 context fragmentation을 줄이는가?
3. relative encoding이 더 긴 evaluation context에 일반화하는가?

논문의 ablation은 세 질문에 각각 근거를 제공한다. 따라서 Transformer-XL의 기여는 단일 trick이 아니라 다음 결합이다.

```text
segment-level state reuse
+ stop-gradient training
+ recurrence-compatible relative position
+ efficient relative shift
= fixed-length training을 유지한 long-context language model
```

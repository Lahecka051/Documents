# 09. RoFormer: Enhanced Transformer with Rotary Position Embedding

## 논문 정보

- 원본 파일: `09_RoFormer_RoPE.pdf`
- 제목: RoFormer: Enhanced Transformer with Rotary Position Embedding
- 저자: Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu
- 공개: arXiv:2104.09864, 초판 2021, 제공된 PDF는 v5(2023)
- 주제: self-attention의 query와 key를 위치별로 회전해 상대 위치를 내적 안에 포함하는 방법
- 핵심 키워드: RoFormer, RoPE, rotary position embedding, relative position, rotation matrix, complex representation, linear attention

## 한눈에 보는 요약

RoPE(Rotary Position Embedding)는 위치 embedding을 token embedding에 더하지 않는다. 대신 attention score를 계산하기 직전, 각 위치의 query와 key를 위치에 비례하는 각도로 회전한다. 위치 `m`의 query에는 `R_m`, 위치 `n`의 key에는 `R_n`을 적용한다.

```math
\begin{aligned}
\tilde q_m &= R_m q_m,\\
\tilde k_n &= R_n k_n.
\end{aligned}
```

이때 두 벡터의 내적은 다음과 같이 정리된다.

```math
\begin{aligned}
\tilde q_m^\top \tilde k_n
&= (R_m q_m)^\top (R_n k_n)\\
&= q_m^\top R_m^\top R_n k_n\\
&= q_m^\top R_{n-m} k_n.
\end{aligned}
```

마지막 식에는 절대 위치 `m`, `n`이 따로 남지 않고 상대 위치 `n-m`만 남는다. RoPE의 핵심은 이 한 줄이다. 각 벡터는 자신의 절대 위치에 따라 회전하지만, attention score에서는 두 회전의 차이, 즉 상대 위치가 나타난다.

회전은 2차원 feature pair마다 수행된다. 각 pair는 서로 다른 주파수 `theta_i`를 사용하므로 일부 차원은 빠르게 회전해 가까운 위치 변화에 민감하고, 일부 차원은 천천히 회전해 긴 범위의 위치 변화를 표현한다. 주파수 배치는 Transformer의 sinusoidal positional encoding과 같은 계열의 기하급수적 스케줄을 사용하지만, 위치 정보를 더하는 것이 아니라 곱셈적 회전으로 적용한다는 점이 다르다.

이 논문의 지속적인 기여는 특정 RoFormer 모델의 벤치마크 성능보다 RoPE 연산 자체에 있다. 수식이 간결하고, 추가 학습 parameter가 없으며, Q/K의 shape를 바꾸지 않고, 회전의 직교성 때문에 norm도 보존한다. 반면 RoPE는 attention의 `O(n^2)` 비용을 줄이지 않으며, 임의로 긴 position index를 계산할 수 있다는 사실이 학습 길이 밖에서의 성능을 보장하지도 않는다.

이 리뷰는 다음 네 가지에 초점을 둔다.

1. 왜 query-key 내적이 상대 위치만 보도록 만들 수 있는가
2. 2차원 복소수 회전이 일반 `d`차원 구현으로 어떻게 확장되는가
3. 논문이 주장한 long-term decay와 linear attention 호환성을 어디까지 받아들여야 하는가
4. 실험 결과가 실제로 무엇을 지지하며 무엇은 아직 입증하지 못하는가

## 연구 배경과 문제의식

### Self-attention은 위치를 스스로 알지 못한다

입력 sequence를 다음과 같이 두자.

```text
tokens:     w_1, w_2, ..., w_N
embeddings: x_1, x_2, ..., x_N
x_i in R^d
```

Self-attention은 각 위치의 embedding에서 query, key, value를 만든다. 논문은 위치 정보까지 포함하는 일반적인 함수를 다음과 같이 둔다.

```math
\begin{aligned}
q_m &= f_q(x_m,m),\\
k_n &= f_k(x_n,n),\\
v_n &= f_v(x_n,n).
\end{aligned}
```

Attention weight와 출력은 다음과 같다.

```math
\begin{aligned}
\mathrm{score}_{m,n} &= \frac{q_m^\top k_n}{\sqrt d},\\
a_{m,n} &= \frac{\exp(\mathrm{score}_{m,n})}
{\sum_j \exp(\mathrm{score}_{m,j})},\\
o_m &= \sum_n a_{m,n}v_n.
\end{aligned}
```

위치 정보를 `f_q`, `f_k`, `f_v` 어디에도 넣지 않고 단순히 동일한 선형 projection만 사용하면, self-attention 자체는 입력 순서를 구분하지 못한다. 같은 token vector들의 순서를 바꾸면 출력도 같은 방식으로 순열될 뿐이다. 따라서 Transformer에는 순서를 알려 주는 별도 장치가 필요하다.

### Absolute position embedding의 방식

기본적인 절대 위치 방식은 token embedding `x_i`에 위치 vector `p_i`를 더한 뒤 Q/K/V projection을 수행한다.

```math
f_t(x_i,i)=W_t(x_i+p_i),
\qquad t\in\{q,k,v\}.
```

`p_i`는 학습 가능한 embedding일 수도 있고, sine/cosine으로 계산한 고정 vector일 수도 있다. 이 방식은 단순하지만 다음 특성이 있다.

- Content와 position이 입력에서 먼저 더해진 뒤 같은 projection을 통과한다.
- Learned absolute embedding은 보통 정해진 최대 위치까지만 parameter를 가진다.
- Attention score가 상대 거리 `m-n`을 사용하도록 구조적으로 강제하지 않는다.
- 새로운 위치나 더 긴 sequence에서의 동작은 위치 표현과 학습 분포에 의존한다.

다만 여기서 주의할 점이 있다. 고정 sinusoidal encoding은 학습 때보다 큰 위치의 값도 계산할 수 있다. 따라서 모든 absolute encoding이 position index 자체를 생성하지 못하는 것은 아니다. 문제는 값을 계산할 수 있느냐보다 모델이 그 값을 의미 있게 일반화하느냐다.

### 기존 relative position 방법의 흐름

상대 위치 방법은 attention score나 value에 거리 `m-n`에 따른 embedding 또는 bias를 추가한다. 예를 들어 Shaw 계열의 방법은 key와 value에 clipped relative embedding을 결합한다.

```math
\begin{aligned}
q_m &= W_qx_m,\\
k_n &= W_k\left(x_n+\tilde p^k_{m-n}\right),\\
v_n &= W_v\left(x_n+\tilde p^v_{m-n}\right).
\end{aligned}
```

Transformer-XL 계열은 absolute position을 더한 Q/K 내적을 네 항으로 전개한 뒤 일부 항을 상대 위치 표현으로 교체한다.

```math
\begin{aligned}
(x_m+p_m)^\top W_q^\top W_k(x_n+p_n)
&= \text{content-to-content}\\
&\quad+\text{content-to-position}\\
&\quad+\text{position-to-content}\\
&\quad+\text{position-to-position}.
\end{aligned}
```

T5 계열처럼 attention logit에 거리별 scalar bias를 더하는 방식도 있다.

```math
\mathrm{score}_{m,n}
=\frac{q_m^\top k_n}{\sqrt d}+b_{m-n}.
```

이 방법들은 모두 유효하지만 별도 relative table, clipping/bucketing 규칙, 추가 score 항을 사용할 수 있다. RoFormer 논문은 이런 additive decomposition에서 출발하지 않고, query와 key를 만드는 함수 자체가 상대 위치 조건을 만족하도록 구성하려 한다.

### 논문이 세운 정확한 목표

논문의 핵심 문제는 다음 함수 방정식이다.

```math
\left\langle f_q(x_m,m),f_k(x_n,n)\right\rangle
=g(x_m,x_n,m-n).
```

왼쪽에서 query와 key는 각각 자신의 절대 위치 `m`, `n`을 받는다. 그러나 두 벡터의 내적은 오른쪽처럼 content `x_m`, `x_n`과 상대 위치 `m-n`에만 의존해야 한다.

RoPE는 회전 행렬을 사용해 이 조건을 만족하는 간단한 구성적 해를 제시한다.

```math
\begin{aligned}
f_q(x_m,m)&=R_mW_qx_m,\\
f_k(x_n,n)&=R_nW_kx_n.
\end{aligned}
```

## 핵심 아이디어와 2차원 유도

### RoPE를 한 문장으로 설명하면

RoPE는 Q와 K의 feature를 2차원 pair로 묶고, 각 pair를 `position x frequency`만큼 회전시켜 QK 내적이 상대 회전각만 보게 만드는 위치 표현이다.

Value는 원 논문의 RoPE에서 회전하지 않는다.

```text
Q: rotate
K: rotate
V: keep unchanged
```

### 2차원 회전 행렬

2차원 vector `x = [x_1, x_2]^T`를 각도 `phi`만큼 반시계 방향으로 회전하는 행렬은 다음과 같다.

```math
R(\phi)=
\begin{bmatrix}
\cos\phi&-\sin\phi\\
\sin\phi&\cos\phi
\end{bmatrix}.
```

위치 `m`과 주파수 `theta`를 사용하면 회전각은 `m theta`가 된다.

```math
\begin{aligned}
R_m&=R(m\theta),\\
R_mx&=
\begin{bmatrix}
x_1\cos(m\theta)-x_2\sin(m\theta)\\
x_1\sin(m\theta)+x_2\cos(m\theta)
\end{bmatrix}.
\end{aligned}
```

회전 행렬은 직교 행렬이다.

```math
\begin{aligned}
R_m^\top R_m&=I,\\
R_m^{-1}&=R_m^\top=R_{-m},\\
\det(R_m)&=1.
\end{aligned}
```

따라서 회전 전후의 norm은 같다.

```math
\lVert R_mx\rVert_2=\lVert x\rVert_2.
```

위치를 넣기 위해 vector의 크기를 키우거나 줄이지 않고 방향만 바꾼다는 뜻이다. 이 성질은 위치 encoding이 Q/K activation scale을 직접 변화시키지 않게 한다.

### 복소수로 보면 더 간단하다

2차원 vector `[x_1, x_2]`는 복소수 `x_1 + i x_2`로 볼 수 있다. 복소수에 `e^(i phi)`를 곱하면 크기는 유지되고 위상만 `phi`만큼 변한다.

```math
\begin{aligned}
x&=x_1+ix_2,\\
xe^{i\phi}
&=(x_1\cos\phi-x_2\sin\phi)
+i(x_1\sin\phi+x_2\cos\phi).
\end{aligned}
```

따라서 query와 key의 위치 encoding을 다음처럼 쓸 수 있다.

```math
\begin{aligned}
\tilde q_m&=(W_qx_m)e^{im\theta},\\
\tilde k_n&=(W_kx_n)e^{in\theta}.
\end{aligned}
```

두 2차원 vector의 실수 내적은 한 복소수와 다른 복소수의 켤레복소수 곱의 실수부다.

```math
\begin{aligned}
\langle\tilde q_m,\tilde k_n\rangle
&=\mathrm{Re}\!\left[\tilde q_m\mathrm{conj}(\tilde k_n)\right]\\
&=\mathrm{Re}\!\left[q\mathrm{conj}(k)e^{i(m-n)\theta}\right].
\end{aligned}
```

절대 위치는 위상에 각각 들어가지만, 켤레를 곱하면 위상 차이 `m-n`만 남는다.

### 행렬로 보는 상대 위치 성질

같은 내용을 실수 회전 행렬로 전개하면 다음과 같다.

```math
\begin{aligned}
(R_mq)^\top(R_nk)
&=q^\top R_m^\top R_nk\\
&=q^\top R_{-m}R_nk\\
&=q^\top R_{n-m}k.
\end{aligned}
```

회전 행렬은 각도를 더하는 군 성질을 갖는다.

```math
R(a)^\top R(b)=R(b-a).
```

두 위치에 같은 offset `c`를 더해도 내적은 바뀌지 않는다.

```math
\begin{aligned}
(R_{m+c}q)^\top(R_{n+c}k)
&=q^\top R_{-(m+c)}R_{n+c}k\\
&=q^\top R_{n-m}k.
\end{aligned}
```

<p align="center"><img src="https://github.com/user-attachments/assets/f536d8aa-4db1-4987-b4a2-df4fbf9e22e5" alt="RoPE common-shift invariance" width="860"></p>
<p align="center"><sub>보조 도식 — Q/K 위치를 함께 이동해도 상대 offset과 내적이 유지되는 RoPE 성질</sub></p>

세 단계는 원래 query-key 위치쌍, 두 위치를 함께 이동한 쌍, 이동 뒤에도 같은 상대 offset이 남는 관계를 보여 준다. 두 벡터를 같은 크기만큼 더 회전해도 내적은 동일하므로, attention score 관점에서 중요한 것은 좌표계의 절대 원점보다 두 위치 사이의 차이다.

### 작은 수치 예시

다음 값을 사용해 보자.

```math
q=[1,2]^\top,\qquad
k=[3,4]^\top,\qquad
\theta=\frac{\pi}{6},\qquad
m=1,\qquad n=3.
```

위치가 없는 원래 내적은 `11`이다.

```math
q^\top k=1\cdot3+2\cdot4=11.
```

위치별로 직접 회전하면 다음과 같다.

```math
\begin{aligned}
R_mq&=R(\pi/6)q
=[-0.133975,\,2.232051]^\top,\\
R_nk&=R(\pi/2)k
=[-4,\,3]^\top,\\
(R_mq)^\top(R_nk)
&=(-0.133975)(-4)+(2.232051)(3)\\
&=7.232051.
\end{aligned}
```

상대 위치만 사용해도 같은 값이 나온다. `n-m=2`이므로 상대 회전각은 `pi/3`이다.

```math
\begin{aligned}
R_{n-m}k&=R(\pi/3)k
=[-1.964102,\,4.598076]^\top,\\
q^\top R_{n-m}k
&=1(-1.964102)+2(4.598076)\\
&=7.232051.
\end{aligned}
```

RoPE가 원래 content 내적을 그대로 보존하는 것은 아니다. 서로 다른 위치에 서로 다른 회전을 적용하므로 두 vector 사이 각도와 내적은 변한다. 보존되는 것은 각 vector의 norm이며, 변화한 내적이 상대 위치에만 의존한다.

## 짝수 차원으로 일반화

### Feature dimension을 2차원 pair로 나눈다

Head dimension `d`가 짝수라고 하자. RoPE는 `d`차원 vector를 `d/2`개의 2차원 subspace로 나눈다.

```math
(x_0,x_1),(x_2,x_3),\ldots,(x_{d-2},x_{d-1}).
```

각 pair `i`에는 서로 다른 주파수 `theta_i`를 사용한다. 0-based index로 쓰면 논문의 스케줄은 다음과 같다.

```math
\theta_i=10000^{-2i/d},
\qquad i=0,1,\ldots,d/2-1.
```

위치 `m`에서 `i`번째 pair의 회전각은 `m theta_i`다.

```math
\begin{aligned}
\text{pair }0 &: \quad \text{angle}=m\theta_0,\\
\text{pair }1 &: \quad \text{angle}=m\theta_1,\\
&\ \vdots\\
\text{pair }d/2-1 &: \quad \text{angle}=m\theta_{d/2-1}.
\end{aligned}
```

낮은 index에서는 `theta_i`가 커서 빠르게 회전하고, 높은 index에서는 `theta_i`가 작아 천천히 회전한다. 여러 시간 척도를 한 vector 안에 함께 넣는 셈이다.

### Block diagonal rotation matrix

전체 `d`차원 회전 행렬은 2x2 회전 block을 대각선에 놓은 block diagonal matrix다.

```math
R_{\Theta,m}=\mathrm{diag}\!\left(
R(m\theta_0),
R(m\theta_1),
\ldots,
R(m\theta_{d/2-1})
\right).
```

그러면 위치가 포함된 Q/K는 다음과 같다.

```math
\begin{aligned}
\tilde q_m&=R_{\Theta,m}W_qx_m,\\
\tilde k_n&=R_{\Theta,n}W_kx_n.
\end{aligned}
```

각 block에서 상대 회전 성질이 성립하므로 전체 행렬에서도 성립한다.

```math
\begin{aligned}
R_{\Theta,m}^\top R_{\Theta,n}&=R_{\Theta,n-m},\\
\tilde q_m^\top\tilde k_n
&=x_m^\top W_q^\top R_{\Theta,n-m}W_kx_n.
\end{aligned}
```

이 식은 content projection `W_q x_m`, `W_k x_n`과 상대 위치 회전 `R_Theta,n-m`이 하나의 bilinear score 안에서 결합됨을 보여 준다.

### 왜 여러 주파수가 필요한가

모든 pair가 같은 `theta`를 사용하면 전체 feature가 동일한 각도로 회전한다. 그러면 위치 차이를 표현하는 패턴이 한 주기에 묶인다. 서로 다른 주파수를 사용하면 가까운 거리에서 빠르게 변하는 성분과 먼 범위에서 천천히 변하는 성분을 동시에 제공할 수 있다.

```text
large theta_i -> fast phase change -> local position sensitivity
small theta_i -> slow phase change -> long-range position signal
```

이 아이디어는 sinusoidal positional encoding의 다중 주파수 구성과 닮았다. 그러나 적용 위치가 다르다.

```text
Sinusoidal PE:
token representation에 position vector를 더한다.

RoPE:
Q/K의 각 2D pair를 position-dependent angle로 회전한다.
```

<p align="center"><img src="https://github.com/user-attachments/assets/365961d2-dc64-4994-b4a3-33f5455152a5" alt="RoPE rotary position embedding" width="760"></p>
<p align="center"><sub>Figure 1 — 위치별 Q/K feature pair를 여러 주파수로 회전시키는 RoPE</sub></p>

원 논문 Figure 1은 각 2D feature pair가 $\theta_i$와 위치에 따라 서로 다른 속도로 회전하는 과정을 보여 준다. 별도로 논문 Figure 2의 장거리 감쇠 proxy는 전체 추세가 감소하지만 진동이 분명하므로, 이를 모든 token pair의 attention score가 거리와 함께 단조 감소한다는 보장으로 읽으면 안 된다.

## Tensor shape와 실제 구현

### Attention pipeline에서 RoPE가 들어가는 위치

Batch size를 `B`, head 수를 `h`, sequence length를 `N`, head dimension을 `d`라고 하자.

```text
X: [B, N, d_model]

Q: [B, h, N, d]
K: [B, h, N, d]
V: [B, h, N, d_v]
```

RoPE는 Q/K projection 뒤, QK transpose score 계산 전에 들어간다.

```math
\begin{aligned}
Q,K,V&=\mathrm{project}(X),\\
Q_{\mathrm{rope}}&=\mathrm{apply\_rope}(Q,\mathrm{position\_ids}),\\
K_{\mathrm{rope}}&=\mathrm{apply\_rope}(K,\mathrm{position\_ids}),\\
S&=\frac{Q_{\mathrm{rope}}K_{\mathrm{rope}}^\top}{\sqrt d},\\
A&=\mathrm{softmax}(S+\mathrm{mask}),\\
O&=AV.
\end{aligned}
```

<p align="center"><img src="https://github.com/user-attachments/assets/35bcd5fc-8749-4420-8096-0209f2c60714" alt="Attention 계산에서 RoPE가 적용되는 위치" width="860"></p>
<p align="center"><sub>보조 도식 — Q/K projection 뒤 score 계산 전에 RoPE를 적용하고 V는 우회하는 pipeline</sub></p>

RoPE 적용 전후의 Q/K shape는 동일하다.

```text
Q_rope: [B, h, N, d]
K_rope: [B, h, N, d]
score:  [B, h, N, N]
```

따라서 기존 attention interface를 크게 바꿀 필요가 없다.

### 큰 회전 행렬을 만들지 않는다

논문의 block diagonal matrix는 수학적 설명을 위한 것이다. 실제 구현에서 `[d, d]` 행렬을 만들고 곱하면 낭비가 크다. Pair별 원소 연산으로 같은 결과를 얻는다.

`x`의 인접 pair를 다음처럼 90도 회전한 vector를 정의하자.

```math
\begin{aligned}
x&=[x_0,x_1,x_2,x_3,\ldots,x_{d-2},x_{d-1}],\\
\mathrm{rotate\_pairs}(x)
&=[-x_1,x_0,-x_3,x_2,\ldots,-x_{d-1},x_{d-2}].
\end{aligned}
```

각 위치와 pair에 대한 cosine/sine을 dimension마다 복제한다.

```math
\begin{aligned}
\cos_m&=[\cos(m\theta_0),\cos(m\theta_0),
\cos(m\theta_1),\cos(m\theta_1),\ldots],\\
\sin_m&=[\sin(m\theta_0),\sin(m\theta_0),
\sin(m\theta_1),\sin(m\theta_1),\ldots].
\end{aligned}
```

그러면 회전은 element-wise 연산으로 구현된다.

```math
R_mx=x\odot\cos_m+\mathrm{rotate\_pairs}(x)\odot\sin_m.
```

PyTorch 형태의 의사 코드는 다음과 같다.

```python
def rotate_pairs(x):
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    return stack((-x_odd, x_even), dim=-1).flatten(-2)

def apply_rope(x, cos, sin):
    return x * cos + rotate_pairs(x) * sin

q_rope = apply_rope(q, cos[position_ids], sin[position_ids])
k_rope = apply_rope(k, cos[position_ids], sin[position_ids])
```

이 추가 연산은 Q/K tensor 크기에 선형인 `O(B h N d)`다. 원래 full attention score의 `O(B h N^2 d)`보다 낮은 차수이므로 보통 주 병목은 아니다.

### 구현마다 pair layout이 다를 수 있다

원 논문의 설명은 인접 차원 pair를 사용한다.

```text
paper-style interleaved pairs:
(0,1), (2,3), (4,5), ...
```

일부 구현은 vector의 앞 절반과 뒤 절반을 짝지어 `rotate_half`를 구현한다.

```text
half-split layout:
(0,d/2), (1,d/2+1), ...
```

두 방식은 dimension permutation을 적절히 맞추면 같은 종류의 2차원 회전이다. 하지만 이미 학습된 checkpoint의 Q/K weight와 cos/sin layout은 특정 convention에 맞춰져 있다. 구현 convention만 바꾸면 수학적으로 비슷해 보여도 checkpoint와 호환되지 않는다.

### Q/K projection과 회전의 순서

원 논문의 순서는 다음이다.

```text
x -> W_q/W_k projection -> position rotation
```

일반적으로 `R_m W x`와 `W R_m x`는 같지 않다.

```math
R_mW\ne WR_m.
```

따라서 입력 embedding을 먼저 회전한 뒤 projection하는 것은 원 논문의 연산과 동일하지 않다.

### Autoregressive KV cache에서의 처리

Decoder-only 추론에서 현재 token 위치가 `t`라면 현재 Q와 K에 같은 position id `t`를 사용한다.

```math
\tilde q_t=R_tq_t,
\qquad
\tilde k_t=R_tk_t.
```

Cache에는 보통 회전이 끝난 key와 회전하지 않은 value를 저장한다.

```text
K cache: rotated K
V cache: original V
```

미래 위치 `s`의 query는 cached key `t`와 다음 score를 만든다.

```math
(R_sq_s)^\top(R_tk_t)=q_s^\top R_{t-s}k_t.
```

이때 position id가 cache index, padding, prefix, sliding window와 일관되게 맞아야 한다. 단순 off-by-one 오류도 모든 상대 phase를 바꾸므로 generation 품질에 직접 영향을 준다.

## RoPE의 수학적 성질

### 1. Absolute position을 넣고 relative position으로 비교한다

각 Q/K vector에는 자신의 절대 위치 `m`, `n`이 회전각으로 들어간다. 그러나 내적에서는 회전 차이만 남는다.

```text
vector encoding: absolute position dependent
pairwise score:   relative position dependent
```

따라서 RoPE를 단순 absolute embedding 또는 단순 relative bias 중 하나로만 분류하면 핵심을 놓치기 쉽다.

### 2. Norm을 보존한다

회전 행렬의 직교성으로 다음이 성립한다.

```math
\lVert\tilde q_m\rVert=\lVert q_m\rVert,
\qquad
\lVert\tilde k_n\rVert=\lVert k_n\rVert.
```

Position index가 커져도 rotation 자체가 Q/K norm을 폭발시키거나 소실시키지는 않는다. 다만 finite precision에서 큰 각도의 sine/cosine 계산 오차는 별도 문제다.

### 3. Parameter-free frequency schedule

원 논문의 `theta_i`는 고정값이므로 별도 position embedding parameter가 없다.

```text
learned position table: none
fixed inverse frequencies: d/2 values
```

Sin/cos table은 미리 계산해 둘 수 있고 필요할 때 동적으로 만들 수도 있다. 최대 길이에 대한 학습 parameter table은 없지만, 모델이 경험하지 않은 phase pattern까지 잘 해석한다는 보장은 아니다.

### 4. Long-term decay 주장의 정확한 의미

논문은 pair별 complex representation을 사용해 RoPE 내적을 다음 형태로 쓴다.

```math
\begin{gathered}
\sum_i h_i\exp(i\Delta\theta_i),\\
\Delta=m-n,\qquad
h_i=q_{\mathrm{pair},i}\mathrm{conj}(k_{\mathrm{pair},i}).
\end{gathered}
```

여기서 주파수 방향의 부분합을 다음처럼 정의한다.

```math
S_j=\sum_{i=0}^{j-1}\exp(i\Delta\theta_i).
```

Abel transformation을 적용하면 내적 크기의 상계를 대략 다음 구조로 묶을 수 있다.

```math
\left|\sum_i h_i\exp(i\Delta\theta_i)\right|
\le \max_i|h_{i+1}-h_i|\sum_i|S_{i+1}|.
```

논문은 `theta_i = 10000^(-2i/d)`일 때 다음 평균 partial-sum 항이 상대 거리가 커질수록 전반적으로 감소하는 모습을 Figure 2로 제시한다.

```math
\frac{1}{d/2}\sum_j|S_j|.
```

직관은 여러 주파수의 phase가 먼 거리에서 더 많이 어긋나 상쇄될 수 있다는 것이다.

그러나 이 주장을 다음처럼 과도하게 해석하면 안 된다.

- 실제 attention logit이 모든 Q/K에 대해 거리와 함께 단조 감소한다는 정리가 아니다.
- Figure 2의 curve 자체도 진동한다.
- 상계에는 content-dependent 항 `max_i |h_(i+1)-h_i|`가 남아 있다.
- 학습된 Q/K가 특정 먼 거리를 강하게 선택하는 것을 금지하지 않는다.

따라서 RoPE의 long-term decay는 hard distance penalty라기보다 주파수 상쇄가 제공하는 평균적 경향 또는 inductive bias로 읽는 편이 정확하다.

## Linear attention과의 결합

### 일반화된 attention 식

논문은 attention을 similarity function `sim`으로 다음처럼 쓴다.

```math
\mathrm{Attention}(Q,K,V)_m
=\frac{\sum_n\mathrm{sim}(q_m,k_n)v_n}
{\sum_n\mathrm{sim}(q_m,k_n)}.
```

Softmax attention에서는 다음 similarity를 사용한다.

```math
\mathrm{sim}(q_m,k_n)
=\exp\!\left(\frac{q_m^\top k_n}{\sqrt d}\right).
```

모든 query-key pair의 score를 계산하므로 sequence length에 대해 quadratic하다.

### Linear attention의 associative trick

Linear attention은 similarity를 feature map의 내적으로 분해한다.

```math
\mathrm{sim}(q_m,k_n)=\phi(q_m)^\top\psi(k_n).
```

그러면 numerator를 다음처럼 재배열할 수 있다.

```math
\sum_n\left[\phi(q_m)^\top\psi(k_n)\right]v_n
=\phi(q_m)^\top\left[\sum_n\psi(k_n)v_n^\top\right].
```

먼저 key-value summary를 계산하면 모든 `[m,n]` pair를 명시적으로 만들지 않아도 된다. Causal attention에서는 전체 합 대신 prefix sum을 유지한다.

### RoPE를 numerator에 넣는다

RoPE는 norm을 보존하는 회전이므로 feature-mapped Q/K에 회전을 적용할 수 있다.

```math
\bar q_m=R_m\phi(q_m),
\qquad
\bar k_n=R_n\psi(k_n).
```

논문의 RoPE linear attention numerator는 다음과 같다.

```math
\begin{aligned}
&\sum_n\left(R_m\phi(q_m)\right)^\top
\left(R_n\psi(k_n)\right)v_n\\
&\qquad=
\left(R_m\phi(q_m)\right)^\top
\left[\sum_n\left(R_n\psi(k_n)\right)v_n^\top\right].
\end{aligned}
```

회전 후에도 associative reorder가 가능하므로 linear-time 구조를 유지할 수 있다.

### Denominator는 회전하지 않는다

논문은 denominator에 원래의 non-negative feature map을 그대로 사용한다.

```math
\text{denominator}
=\sum_n\phi(q_m)^\top\psi(k_n).
```

전체 식은 다음 형태다.

```math
\mathrm{Attention}_{\mathrm{rope\_linear}}(m)
=\frac{
\sum_n\left[(R_m\phi(q_m))^\top(R_n\psi(k_n))\right]v_n
}{
\sum_n\left[\phi(q_m)^\top\psi(k_n)\right]
}.
```

그 이유는 회전 후 feature가 음수가 될 수 있어 denominator가 0에 가까워지거나 부호가 불안정해지는 위험을 피하기 위해서다. 그러나 이 선택 때문에 numerator coefficient와 denominator normalization이 같은 similarity에서 나오지 않는다. 논문도 value별 weight가 엄밀한 확률적 normalization은 아니라고 인정한다.

따라서 이 부분의 기여는 "RoPE를 linear attention의 associative 계산에 삽입할 수 있다"는 구성 가능성이다. Softmax attention과 완전히 동일한 확률 해석을 보존한다는 주장은 아니다.

## 전체 계산 흐름

RoPE를 사용하는 한 attention layer의 계산을 처음부터 정리하면 다음과 같다.

### 1단계: hidden state projection

```math
Q=XW_Q,
\qquad K=XW_K,
\qquad V=XW_V.
```

Multi-head layout으로 reshape한다.

```text
Q: [B, h, N, d]
K: [B, h, N, d]
V: [B, h, N, d_v]
```

### 2단계: position별 angle 준비

```math
\theta_i=10000^{-2i/d},
\qquad
\mathrm{angle}_{m,i}=\mathrm{position\_id}_m\theta_i.
```

Pair별 cosine과 sine을 만든다.

```math
\cos_{m,i}=\cos(\mathrm{angle}_{m,i}),
\qquad
\sin_{m,i}=\sin(\mathrm{angle}_{m,i}).
```

### 3단계: Q와 K 회전

```math
\begin{aligned}
Q_{\mathrm{rope}}&=Q\odot\cos+\mathrm{rotate\_pairs}(Q)\odot\sin,\\
K_{\mathrm{rope}}&=K\odot\cos+\mathrm{rotate\_pairs}(K)\odot\sin.
\end{aligned}
```

Value에는 RoPE를 적용하지 않는다.

### 4단계: attention score 계산

```math
S=\frac{Q_{\mathrm{rope}}K_{\mathrm{rope}}^\top}{\sqrt d}.
```

필요하면 causal mask와 padding mask를 적용한다.

```math
S=S+\mathrm{causal\_mask}+\mathrm{padding\_mask}.
```

### 5단계: softmax와 value 합산

```math
A=\mathrm{softmax}(S,\mathrm{dim}=-1),
\qquad O=AV.
```

이후의 output projection, residual connection, normalization, FFN은 일반 Transformer와 같다.

### 6단계: autoregressive cache

추론에서는 현재 position id로 회전한 K를 cache하고 V는 그대로 cache한다.

```text
append(K_cache, K_rope_current)
append(V_cache, V_current)
```

## 계산 복잡도와 메모리

| 항목 | RoPE의 영향 |
| --- | --- |
| 학습 parameter | 고정 frequency를 쓰면 추가 parameter 없음 |
| Q/K shape | 변화 없음 |
| V shape | 변화 없음, 원 논문에서는 V를 회전하지 않음 |
| 추가 연산 | Q/K에 대해 `O(B h N d)` element-wise rotation |
| sin/cos 저장 | 미리 저장하면 `O(N d)`; layer/head 간 공유 가능 |
| attention score | 여전히 `O(B h N^2 d)` |
| attention matrix memory | 여전히 `O(B h N^2)` |
| KV cache | K는 회전된 상태로 저장, cache 크기 차수는 동일 |
| 최대 position index | 수학적으로 새 sin/cos를 계산할 수 있으나 품질 보장은 아님 |

RoPE는 위치 표현의 방식이지 efficient attention 알고리즘이 아니다. Sequence length가 두 배가 되면 RoPE 회전 비용은 대략 두 배가 되지만, full attention의 pairwise score 비용은 대략 네 배가 된다.

## 실험 설정과 결과

논문은 machine translation, masked language model pre-training, GLUE fine-tuning, Performer 기반 linear attention, 중국어 long-text task를 평가한다. 아이디어의 범용성을 보여 주려는 구성은 좋지만, 수치의 일관성과 비교 통제에는 주의가 필요하다.

### WMT 2014 English-German 번역

데이터는 약 450만 sentence pair이고, joint BPE vocabulary는 37k다. 논문은 Transformer-base와 같은 계열의 설정을 사용하고, 마지막 5개 checkpoint를 평균하며, beam size 4와 length penalty 0.6으로 평가한다.

| 모델 | BLEU |
| --- | ---: |
| Transformer-base | 27.3 |
| RoFormer | 27.5 |

RoFormer가 `+0.2 BLEU` 높다. 방향은 긍정적이지만 차이가 작고 여러 seed의 분산이나 통계적 유의성은 보고되지 않았다. 이 결과 하나만으로 강한 번역 성능 우위를 주장하기는 어렵다.

### BERT형 MLM pre-training

BookCorpus와 Wikipedia를 8:2로 나누고, `bert-base-uncased` 크기의 baseline과 RoFormer를 비교한다. 논문 본문의 설정은 다음과 같다.

```text
batch size: 64
maximum sequence length: 512
training steps: 100k
optimizer: AdamW
learning rate: 1e-5
```

Figure 3의 loss curve에서는 RoFormer가 더 빠르게 낮은 MLM loss로 수렴하는 모습을 보인다. 다만 재현성 관점에서 두 가지 내부 불일치가 있다.

- Section 4.2는 BERT의 "original sinusoidal position encoding"을 RoPE로 교체했다고 쓰지만, 원래 BERT는 learned absolute position embedding을 사용한다. 문장상의 오류인지 실제 baseline 구현 차이인지 명확하지 않다.
- 본문은 100k step 학습이라고 적지만 Figure 3 왼쪽 x축은 250k step까지 표시된다.

따라서 빠른 수렴이라는 관찰은 흥미롭지만, baseline에서 정확히 무엇을 교체했고 몇 step을 비교했는지 code/config 수준의 확인이 필요하다.

### GLUE fine-tuning

논문은 pre-trained RoFormer를 6개 GLUE task에 3 epoch 동안 fine-tuning한다. 아래는 논문 Table 2의 validation 결과와 단순 차이다.

| Task | BERT | RoFormer | 차이 |
| --- | ---: | ---: | ---: |
| MRPC F1 | 88.9 | 89.5 | +0.6 |
| SST-2 Accuracy | 93.5 | 90.7 | -2.8 |
| QNLI Accuracy | 90.5 | 88.0 | -2.5 |
| STS-B Spearman | 85.8 | 87.0 | +1.2 |
| QQP F1 | 71.2 | 86.4 | +15.2 |
| MNLI m/mm Accuracy | 84.6 / 83.4 | 80.2 / 79.8 | -4.4 / -3.6 |

논문 표현대로 6개 dataset 중 MRPC, STS-B, QQP 세 곳에서는 높고, SST-2, QNLI, MNLI 세 곳에서는 낮다. 즉 일관된 우위라기보다 task별 trade-off가 크다. 특히 QQP의 `+15.2`는 다른 task의 변화보다 비정상적으로 큰 폭인데, 원인 분석이나 반복 실험의 분산이 없다. Test server 결과가 아니라 validation에서 learning rate 후보 중 best-averaged 결과를 보고한다는 점도 감안해야 한다.

### Performer with RoPE

논문은 Enwik8에서 12-layer, hidden dimension 768, 12-head character-level Performer에 RoPE를 결합한다.

```text
maximum sequence length: 1024
batch size: 128
learning rate: 1e-4
```

Figure 3 오른쪽에서 RoPE를 사용한 Performer가 더 빠르게 수렴하고 낮은 LM loss를 보인다. 이는 앞서 제시한 linear attention 결합이 실제 학습에서도 작동할 수 있음을 보여 준다. 그러나 최종 loss 수치, 여러 seed, throughput 또는 memory 측정은 표로 제시되지 않아 효과 크기를 정밀하게 비교하기 어렵다.

### 중국어 pre-training과 long-text 평가

논문은 약 34GB의 중국어 Wikipedia, news, forum data로 RoFormer를 여러 stage에 걸쳐 pre-train한다. Stage마다 maximum sequence length와 batch size가 달라진다.

| Stage | Max length | Batch size | Steps | Loss | Accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 256 | 200k | 1.73 | 65.0% |
| 2 | 1536 | 256 | 12.5k | 1.61 | 66.8% |
| 3 | 256 | 256 | 120k | 1.75 | 64.6% |
| 4 | 128 | 512 | 80k | 1.83 | 63.4% |
| 5 | 1536 | 256 | 10k | 1.58 | 67.4% |
| 6 | 512 | 512 | 30k | 1.66 | 66.2% |

긴 sequence stage에서 높은 accuracy가 관찰되지만, 이 표는 같은 checkpoint를 length만 바꾸어 독립 평가한 ablation이 아니다. Stage가 누적되고 batch size, 학습 step, 입력 분포가 함께 변하므로 accuracy 변화 전체를 sequence length 하나의 효과로 돌릴 수 없다.

Downstream task는 CAIL2019-SCM이다. 세 법률 사건 `(A, B, C)`가 주어졌을 때 `(A,B)`가 `(A,C)`보다 더 유사한지 판단한다.

| 모델 | Validation | Test |
| --- | ---: | ---: |
| BERT-512 | 64.13% | 67.77% |
| WoBERT-512 | 64.07% | 68.10% |
| RoFormer-512 | 64.13% | 68.29% |
| RoFormer-1024 | 66.07% | 69.79% |

같은 길이 512에서 RoFormer는 BERT보다 test `+0.52%p`, WoBERT보다 `+0.19%p` 높다. RoFormer-1024는 WoBERT-512보다 `+1.69%p` 높지만, 이 비교에는 위치 표현뿐 아니라 입력 정보량 증가가 함께 들어 있다.

또한 pre-training 중 최대 길이 1536 stage를 이미 사용했으므로 CAIL의 길이 1024는 학습 길이 밖 extrapolation이 아니다. 이 실험은 RoFormer가 1024-token 입력을 실제로 처리할 수 있고 더 긴 입력이 task에 유용하다는 증거지만, 짧게 학습한 모델이 더 긴 길이에 일반화한다는 증거는 아니다.

### 실험 전체에 대한 평가

실험에서 비교적 일관되게 보이는 신호는 pre-training loss의 빠른 감소다. 반면 downstream 성능은 task에 따라 상승과 하락이 섞여 있고, long-context 결과는 입력 길이와 학습 stage가 완전히 통제되지 않았다.

논문 저자들도 limitation section에서 다음을 인정한다.

- 왜 다른 position encoding보다 더 빨리 수렴하는지 충분히 설명하지 못했다.
- long-term decay 성질과 long-text 성능 우위 사이의 직접적인 인과 설명이 부족하다.
- RoFormer pre-training은 Transformer 기반이므로 상당한 hardware resource를 요구한다.

결론적으로 실험은 RoPE가 작동 가능한 유망한 방법이라는 점은 지지하지만, 모든 task에서 우월하다거나 학습 길이 밖 extrapolation을 해결했다는 강한 결론까지 지지하지는 않는다.

## 기존 위치 표현과 비교

| 방법 | 위치 정보가 들어가는 곳 | 상대 위치가 score에 나타나는 방식 | 추가 학습 parameter | 길이 관련 특징 |
| --- | --- | --- | --- | --- |
| Learned absolute embedding | token embedding에 더함 | 간접 학습 | 위치별 table | table 최대 index 밖 사용이 곤란 |
| Sinusoidal absolute encoding | token embedding에 더함 | 삼각함수 관계를 간접 활용 | 없음 | 새 index 계산 가능, 일반화는 별도 문제 |
| Shaw relative position | K/V 또는 score에 relative vector 결합 | 거리별 vector | 보통 있음 | clipping 범위에 의존 |
| Transformer-XL relative | content/position score 항 분해 | sinusoidal relative term과 bias | 일부 있음 | segment recurrence와 결합 |
| T5 relative bias | attention logit에 scalar bias 추가 | bucketed `b_(m-n)` | 있음 | 먼 거리를 bucket으로 묶음 |
| RoPE | Q/K의 2D pair 회전 | `R_m^T R_n = R_(n-m)` | 원 논문은 없음 | 새 index 계산 가능, phase OOD 문제 존재 |
| ALiBi | attention logit에 거리 선형 bias 추가 | `-slope_h * distance` | 보통 고정 slope | 거리 penalty가 단조적이고 직접적 |

RoPE와 ALiBi의 차이는 특히 선명하다.

```text
RoPE:
content vector 자체를 회전시켜 거리별 bilinear interaction을 만든다.

ALiBi:
content score에 거리만의 scalar penalty를 더한다.
```

RoPE는 거리 효과가 Q/K content와 결합되어 있고 진동할 수 있다. ALiBi는 head별 slope에 따른 단조 거리 penalty다.

## 장점과 핵심 기여

### 1. 상대 위치 성질이 매우 간결하다

핵심 식이 명확하다.

```math
R_m^\top R_n=R_{n-m}.
```

별도 relative table을 조회하거나 additive score 항을 여러 개 전개하지 않아도 상대 위치가 query-key 내적 안에 들어간다.

### 2. Content와 position이 곱셈적으로 상호작용한다

RoPE score는 단순히 content score와 position-only bias를 더한 형태가 아니다.

```math
q_m^\top R_{n-m}k_n.
```

상대 위치 회전이 key의 방향을 바꾸므로 같은 거리라도 query와 key content에 따라 영향이 달라진다.

### 3. Norm-preserving transformation이다

직교 회전은 Q/K의 크기를 보존한다. Position index가 representation scale을 직접 바꾸지 않고 phase만 조절한다.

### 4. Parameter와 interface 부담이 작다

고정 frequency를 사용하면 학습 parameter가 없고 Q/K shape도 유지된다. 기존 attention block에 국소적으로 삽입하기 쉽다.

### 5. Linear attention에도 삽입 가능한 구조를 제시했다

회전된 feature map을 사용해 associative numerator 계산을 유지하는 구성을 제시했다. 엄밀한 확률 normalization은 아니지만, relative position과 linear attention을 연결하려는 이론적 시도라는 가치가 있다.

### 6. 후속 context-extension 연구의 기준점이 되었다

RoPE의 phase와 주파수 스케줄을 어떻게 조정해야 학습 길이보다 긴 context로 확장할 수 있는지가 하나의 독립 연구 축이 되었다.

## 한계와 비판적 관점

### 1. Sequence length flexibility와 extrapolation은 다르다

RoPE는 learned position table 없이 임의 position의 sin/cos를 계산할 수 있다. 이것이 논문이 말하는 length flexibility의 한 부분이다. 그러나 학습 중 보지 못한 큰 `m theta_i` phase에서 모델이 안정적으로 작동하는지는 별도 문제다.

```text
can compute position 100000
!=
can understand position 100000
```

### 2. 회전은 주기적이다

각 2D pair는 주기 `2pi/theta_i`를 가진다. 여러 주파수를 함께 쓰므로 전체 vector에 단순한 짧은 공통 주기가 생기는 것은 아니지만, 학습 범위 밖에서는 각 차원의 phase 조합이 새로운 분포가 될 수 있다. 이것이 후속 RoPE scaling 연구가 필요한 이유다.

### 3. Long-term decay는 단조 보장이 아니다

논문이 그린 것은 partial phase sum의 평균적 상계 proxy다. 실제 Q/K score에는 content-dependent coefficient가 있고 curve도 진동한다. 따라서 RoPE를 거리 증가에 따라 attention을 항상 줄이는 방법으로 설명하면 부정확하다.

### 4. Attention 복잡도는 그대로다

RoPE는 `QK^T`를 계산하는 pair 수를 줄이지 않는다. 긴 context의 계산량과 attention matrix memory 문제는 FlashAttention, sparse attention, linear attention 같은 별도 방법이 해결해야 한다.

### 5. 2차원 유도는 구성적 해이지 유일성 증명이 아니다

논문은 radial component를 position과 무관하게 두는 straightforward solution을 선택하고, angular component를 등차수열 형태로 구성한다. 이는 조건을 만족하는 자연스럽고 해석 가능한 해를 보여 주지만 가능한 모든 `f_q`, `f_k` 중 RoPE만이 유일하다는 증명은 아니다.

### 6. Linear attention normalization은 절충안이다

Numerator에는 회전된 feature를 쓰고 denominator에는 회전 전 non-negative feature를 쓴다. 계산 안정성을 위한 선택이지만 numerator weight와 normalization이 서로 다른 kernel을 사용하므로 softmax attention과 같은 확률 해석은 사라진다.

### 7. 실험 통제가 충분히 강하지 않다

- 번역 향상은 `+0.2 BLEU`로 작고 분산이 없다.
- GLUE는 3개 task 상승, 3개 task 하락이다.
- QQP의 큰 상승과 MNLI의 큰 하락에 대한 설명이 없다.
- BERT position encoding 서술과 training step에 내부 불일치가 있다.
- 중국어 1024 결과는 512 baseline보다 더 많은 입력을 보므로 RoPE 효과만 분리하지 못한다.
- 학습에서 이미 길이 1536을 사용했기 때문에 1024 평가는 length extrapolation이 아니다.

이론적 아이디어의 강도에 비해 empirical section은 제한적이다.

## 자주 헷갈리는 지점

### RoFormer와 RoPE는 같은 말이 아니다

```text
RoPE: 위치 회전 연산
RoFormer: RoPE를 적용한 Transformer 계열 모델
```

후속 모델이 RoPE를 사용한다고 해서 모두 RoFormer architecture를 그대로 사용한다는 뜻은 아니다.

### RoPE는 absolute인가 relative인가

각 vector의 회전은 absolute position으로 계산하고, 두 vector의 attention score에는 relative position만 남는다.

```text
encoding step: absolute
interaction step: relative
```

### 회전하면 정보가 손실되는가

고정 위치에서 `R_m`은 역행렬이 있는 직교 변환이다. Norm과 차원 수를 보존하므로 회전 자체는 정보 손실 변환이 아니다. 다만 attention은 서로 다른 위치의 vector를 서로 다른 각도로 비교하므로 score가 달라진다.

### Sinusoidal PE와 같은 것인가

같은 종류의 주파수와 sine/cosine을 사용하지만 결합 방식이 다르다.

```math
\begin{aligned}
\text{sinusoidal PE}:&\quad x+p_m,\\
\text{RoPE}:&\quad R_m(Wx).
\end{aligned}
```

### V도 회전해야 하는가

원 논문의 RoPE는 Q와 K에 적용한다. V는 회전하지 않는다. Position은 무엇을 얼마나 볼지 결정하는 attention score에 들어가고, 가져올 content인 value에는 직접 회전으로 넣지 않는다.

### RoPE가 causal mask를 대신하는가

아니다. RoPE는 position-dependent score를 만들 뿐 미래 token을 차단하지 않는다. Autoregressive model에는 별도 causal mask가 필요하다.

### RoPE가 context 비용을 줄이는가

아니다. RoPE는 위치 표현을 바꾸며 full attention의 quadratic pairwise computation은 유지된다.

### `m-n`과 `n-m` 중 어느 것이 맞는가

복소수 표현에서 어느 vector에 conjugate를 두는지, 행렬을 row/column 중 어느 방향으로 쓰는지에 따라 표면적인 부호가 달라질 수 있다.

```math
\begin{aligned}
\text{complex phase form}:&\quad \exp(i(m-n)\theta),\\
\text{matrix form}:&\quad q^\top R_{n-m}k.
\end{aligned}
```

둘은 같은 상대 회전 관계를 convention에 맞게 표현한 것이다. 구현에서는 Q와 K에 같은 convention을 일관되게 적용하는 것이 중요하다.

## 후속 논문과의 연결점

RoPE 이후의 긴 context 연구는 주로 학습 범위 밖에서 phase가 너무 빠르게 변하거나 새로운 조합이 되는 문제를 다룬다.

- xPos: 회전에 위치별 exponential scaling을 결합해 length extrapolation을 개선하려 한다.
- Position Interpolation: 긴 target position을 학습 때의 position 범위 안으로 압축한다.
- YaRN: 주파수별 interpolation/extrapolation 전략과 attention scaling을 결합한다.
- LongRoPE: dimension별로 다른 비균일 extension factor를 탐색하고 긴 context 적응을 단계화한다.

이 흐름을 한 줄로 정리하면 다음과 같다.

```text
RoPE defines relative phase
-> later methods redesign how that phase grows beyond the training length
```

ALiBi는 다른 방향의 비교점이다. 회전 phase 대신 attention score에 거리 선형 bias를 넣어 train-short-test-long 성질을 노린다.

## 온디바이스 및 비전 관점의 메모

### 온디바이스 추론

RoPE의 연산 자체는 element-wise multiply/add이므로 full attention에 비해 가볍다. 구현할 때는 compute와 table memory 사이에서 선택할 수 있다.

```text
option 1: position별 sin/cos table을 미리 저장
option 2: 필요한 position의 sin/cos를 동적으로 계산
option 3: recurrence로 다음 phase를 갱신
```

예를 들어 length 4096, rotary dimension 128, FP16에서 cos와 sin을 모두 dimension별로 저장하면 대략 다음 크기다.

```math
4096\times128\times2\text{ arrays}\times2\text{ bytes}
=2{,}097{,}152\text{ bytes}
\approx2\text{ MiB}.
```

이 table은 보통 layer와 head 사이에 공유할 수 있다. 메모리가 더 제한적이면 pair별 `d/2` 값만 저장하거나 position 단위로 생성할 수 있다.

KV cache에는 이미 회전된 K를 저장하면 과거 key를 매 step 다시 회전할 필요가 없다. 다만 긴 context에서 주 병목은 RoPE table보다 KV cache와 attention/KV memory traffic인 경우가 많다.

### Vision Transformer에 적용할 때

원 논문은 1차원 token sequence를 다룬다. Image patch를 단순히 raster order로 펼쳐 1D RoPE를 적용하면 인접한 sequence index와 실제 2D 공간 인접성이 완전히 일치하지 않을 수 있다.

```text
1D text position: one axis
2D image position: row axis + column axis
```

비전에서는 row/column별 회전을 나누는 axial/2D rotary embedding 같은 확장이 자연스럽다. 그러나 이는 원 RoFormer 논문의 실험 범위가 아니라 후속 일반화다. 2D RoPE를 평가할 때는 patch flatten order와 축별 frequency 배치를 명시해야 한다.

## 개인 학습/연구 메모

이 논문에서 반드시 기억할 식은 다음이다.

```math
(R_mq)^\top(R_nk)=q^\top R_{n-m}k.
```

이 식에서 RoPE의 설계 원리가 모두 나온다.

- `R_m q`: 각 query는 자신의 absolute position을 안다.
- `R_n k`: 각 key도 자신의 absolute position을 안다.
- `R_m^T R_n`: 비교할 때 공통 회전은 사라진다.
- `R_(n-m)`: attention score에는 relative position이 남는다.

구현 검증 체크리스트는 다음과 같다.

```text
1. RoPE를 Q/K projection 뒤에 적용했는가?
2. Q와 K가 같은 frequency와 pair convention을 쓰는가?
3. position_ids가 padding/cache index와 일치하는가?
4. 인접-pair 방식인지 half-split 방식인지 checkpoint와 맞는가?
5. V는 원 논문대로 회전하지 않는가?
6. cos/sin broadcast shape가 [B or 1, 1, N, d]와 맞는가?
7. KV cache에 rotated K가 저장되는가?
8. 긴 position에서 sin/cos 계산 precision을 확인했는가?
9. context 확장 성능과 단순 position 계산 가능성을 구분했는가?
10. RoPE가 attention의 O(N^2) 비용을 줄인다고 오해하지 않았는가?
```

RoFormer 논문의 가장 큰 미덕은 위치를 별도 vector로 "더하는 정보"가 아니라 representation의 좌표계를 "회전시키는 변환"으로 다시 보았다는 점이다. 이 관점 덕분에 absolute position encoding과 relative position interaction을 하나의 간단한 연산으로 연결했다.

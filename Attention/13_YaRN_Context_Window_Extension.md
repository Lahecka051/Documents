# 13. YaRN: Efficient Context Window Extension of Large Language Models

## 논문 정보

- 제목: **YaRN: Efficient Context Window Extension of Large Language Models**
- 저자: Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole
- 최초 공개: 2023년
- 현재 제공된 PDF: arXiv v3, 2026-02-06
- 원본 PDF: [13_YaRN_Context_Window_Extension.pdf](./13_YaRN_Context_Window_Extension.pdf)
- 핵심 키워드: `YaRN`, NTK-by-parts, RoPE scaling, attention temperature, Dynamic Scaling

## 한눈에 보는 요약

Position Interpolation(PI)은 RoPE의 모든 주파수를 같은 비율로 압축한다. 안정적이지만 확장 배율 $s$가 커질수록 local position을 구분하는 고주파 성분까지 약해진다.

YaRN(Yet another RoPE extensioN method)은 RoPE dimension을 wavelength에 따라 세 구간으로 나눈다.

```text
high frequency   -> interpolation하지 않고 local resolution 보존
transition band  -> extrapolation과 interpolation을 연속적으로 혼합
low frequency    -> 목표 길이에 맞춰 full interpolation
```

그리고 긴 context에서 달라진 attention entropy를 보정하기 위해 softmax 전 logit temperature를 조절한다.

```math
\operatorname{softmax}\left(
\frac{q_m^\top k_n}{t\sqrt{d_h}}
\right).
```

논문에서 말하는 YaRN은 정확히 다음 두 요소의 결합이다.

```math
\boxed{
\text{YaRN}
=
\text{NTK-by-parts frequency scaling}
+
\text{attention temperature scaling}
}
```

LLaMA/Llama 2에서는 다음 경험식을 권장한다.

```math
\sqrt{\frac1t}=0.1\ln(s)+1.
```

논문은 Llama 2의 4k context를 64k/128k로 확장하면서, PI보다 약 2.5배 적은 steps와 10배 적은 token으로 경쟁력 있는 성능을 얻었다. 또한 64k 길이 data로 학습한 128k 모델이 128k까지 extrapolate하는 결과를 보였다.

![YaRN의 RoPE 주파수 band와 attention temperature 보정](https://github.com/user-attachments/assets/45cd578b-4def-466a-a0f7-3389d119c325)

## 연구 배경과 문제의식

### PI가 해결한 것과 남긴 것

PI는 original context $L$을 target $L'=sL$로 늘릴 때 position을 $m/s$로 줄인다.

```math
m\theta_d
\quad\longrightarrow\quad
\frac{m}{s}\theta_d.
```

Direct extrapolation의 out-of-distribution phase를 피한다는 점은 강력하다. 하지만 모든 $\theta_d$를 $1/s$로 줄이면 high-frequency 성분도 사라진다. 확장 배율이 커질수록 인접 token의 phase 차이가 너무 작아져 모델이 local order를 구분하기 어려워진다.

논문은 PI fine-tuning이 대략 $s\approx8$을 넘어가면 품질이 악화되는 사례를 문제로 든다.

### RoPE dimension은 서로 같은 정보를 담지 않는다

RoPE의 각 complex dimension은 서로 다른 frequency를 갖는다.

```math
\theta_d=b^{-2d/|D|},
\qquad b=10000.
```

이에 대응하는 wavelength는 다음과 같다.

```math
\lambda_d=\frac{2\pi}{\theta_d}
=
2\pi b^{2d/|D|}.
```

작은 wavelength는 original context 안에서 여러 번 회전한다. 이 차원은 동일 phase가 반복되므로 주로 **relative/local position**을 표현한다. 큰 wavelength는 original context 안에서 한 바퀴도 돌지 않을 수 있다. 이 경우 각 position의 phase가 사실상 고유해 **absolute-like position**으로 사용될 수 있다.

따라서 모든 dimension에 같은 interpolation을 적용하는 것은 각 차원의 역할 차이를 무시한다.

### Attention entropy도 바뀐다

Context가 길어지면 softmax에서 경쟁하는 key 수가 늘고, RoPE phase scaling으로 logit 분포도 달라진다. Position frequency만 조정해도 긴 context의 attention concentration이 original model과 같다는 보장은 없다.

YaRN은 이 문제를 별도의 attention temperature로 보정한다.

## RoPE와 통일된 표기

Hidden dimension 집합의 크기를 $|D|$, token position을 $m$이라 하자. RoPE는 complex coordinate에서 다음과 같다.

```math
f_W(x_m,m,\theta)
=
e^{im\theta}Wx_m.
```

논문은 다양한 scaling 방법을 다음 두 함수로 통일한다.

```math
f_W'(x_m,m,\theta)
=
f_W(x_m,g(m),h(\theta)).
```

- $g(m)$: position index를 바꾸는 함수
- $h(\theta)$: frequency를 바꾸는 함수

Position을 $m/s$로 줄이는 것과 모든 frequency를 $\theta/s$로 줄이는 것은 angle $m\theta$ 관점에서 동등하다.

## YaRN으로 이어지는 방법들

### 1. Position Interpolation

PI는 모든 frequency를 균일하게 줄인다.

```math
h_{PI}(\theta_d)=\frac{\theta_d}{s}.
```

장점은 모든 target phase를 pretrained 범위 안으로 넣는 안정성이다. 단점은 local high-frequency 정보 손실이다.

### 2. NTK-aware interpolation

NTK-aware 방법은 RoPE base $b$를 바꿔 interpolation pressure를 dimension 전체에 비균일하게 분배한다.

```math
b'
=
b\cdot s^{|D|/(|D|-2)}.
```

새 frequency는 다음과 같다.

```math
h_{NTK}(\theta_d)
=
b'^{-2d/|D|}.
```

최고 frequency는 거의 유지하고 최저 frequency는 PI에 가깝게 줄인다. Fine-tuning 없이 PI보다 잘 작동할 수 있지만 일부 dimension이 원래 범위 밖으로 extrapolate하고, 원하는 실제 extension에 맞는 base를 경험적으로 찾기 어렵다.

### 3. NTK-by-parts interpolation

YaRN의 핵심 frequency scaling이다. 먼저 original context $L$에서 특정 dimension이 몇 번 회전하는지 계산한다.

```math
r(d)
=
\frac{L}{\lambda_d}.
```

$r(d)$가 크면 wavelength가 짧아 학습 구간 안에서 여러 번 회전한 고주파다. $r(d)$가 작으면 wavelength가 길어 한 번도 충분히 회전하지 않은 저주파다.

두 경계 $\alpha$, $\beta$를 두고 ramp를 정의한다.

```math
\gamma(r)=
\begin{cases}
0,&r<\alpha\\
1,&r>\beta\\
\dfrac{r-\alpha}{\beta-\alpha},&\text{otherwise}.
\end{cases}
```

최종 frequency는 다음과 같다.

```math
h(\theta_d)
=
\left(1-\gamma(r(d))\right)\frac{\theta_d}{s}
+\gamma(r(d))\theta_d.
```

세 구간의 의미는 다음과 같다.

| 조건 | 해석 | 적용 |
| --- | --- | --- |
| $r(d)<\alpha$ | original context에서 회전이 매우 적음 | full PI, $\theta_d/s$ |
| $\alpha\le r(d)\le\beta$ | 중간 wavelength | ramp blend |
| $r(d)>\beta$ | 여러 번 회전하는 고주파 | no interpolation, $\theta_d$ |

LLaMA 계열에서 논문이 제시한 좋은 기본값은 다음이다.

```math
\alpha=1,
\qquad
\beta=32.
```

이 방식은 high frequency의 local resolution을 보존하면서 low frequency의 out-of-distribution absolute-like phase를 target range 안으로 압축한다.

## Attention temperature

### Softmax logit 보정

YaRN은 NTK-by-parts만 쓰지 않고 attention logit의 temperature $t$를 조정한다.

```math
A_{m,n}
=
\operatorname{softmax}_n\left(
\frac{q_m^\top k_n}{t\sqrt{|D|}}
\right).
```

$t<1$이면 logit magnitude가 커져 softmax가 더 sharp해진다. Target context가 길어지면서 분산되는 attention을 보정하려는 목적이다.

### RoPE vector scaling으로 구현

Attention kernel 자체에 새 temperature 인자를 넣지 않고, query와 key 모두에 $\sqrt{1/t}$를 곱할 수 있다.

```math
(\sqrt{1/t}\,q)^\top(\sqrt{1/t}\,k)
=
\frac1t q^\top k.
```

Cos/sin RoPE cache에 이 scale을 미리 포함하면 attention code를 수정하지 않고 같은 효과를 얻는다. 고정 context에서는 cache가 모든 forward에 재사용되므로 추가 runtime overhead가 거의 없다.

### 경험식

LLaMA와 Llama 2에서 extension scale $s$에 따른 추천값은 다음이다.

```math
\sqrt{\frac1t}
=
0.1\ln(s)+1.
```

이 식은 이론적으로 유도한 상수가 아니라 여러 LLaMA model size에서 lowest perplexity를 주는 값을 fitting한 경험식이다. 다른 architecture나 학습 recipe에서는 재검증이 필요하다.

## Dynamic Scaling

### 고정 scale의 문제

Target $L'$를 기준으로 항상 $s=L'/L$을 쓰면 sequence가 아직 짧을 때도 position을 압축한다. 예를 들어 128k 모델이 2k prompt를 처리할 때부터 32배 scaling을 적용하면 original short-context resolution를 불필요하게 잃을 수 있다.

또한 sequence가 target $L'$를 넘으면 다시 abrupt failure가 생길 수 있다.

### 현재 길이에 따른 scale

Dynamic Scaling은 현재 sequence length $\ell$을 사용한다.

```math
s_{dynamic}
=
\max\left(1,\frac{\ell}{L}\right).
```

```math
\begin{cases}
\text{current length}\le L:
&\text{use original RoPE},\\
\text{current length}>L:
&\text{increase scaling only as needed}.
\end{cases}
```

Fine-tuning하지 않은 base Llama 2에서도 Dynamic-YaRN이 Dynamic-PI보다 긴 길이 perplexity를 안정적으로 유지하는 결과를 보인다.

### KV cache의 중요한 함정

Autoregressive decode에서 $s_{dynamic}$은 token이 추가될 때 변한다. 그러면 과거 token의 RoPE angle도 새 scale 기준으로 달라진다.

이미 RoPE를 적용한 key를 cache하면 과거 key는 옛 scale, 새 key는 새 scale을 사용해 dot product가 일관되지 않는다. 논문은 dynamic scaling과 KV cache를 함께 쓸 때 **RoPE 적용 전 K/V representation을 cache**하고 필요할 때 현재 scale로 변환해야 한다고 지적한다.

이는 메모리와 compute trade-off를 만든다. 현대 inference engine에서 dynamic RoPE를 구현할 때 가장 중요한 검증 지점 중 하나다.

## Tensor shape와 구현

```math
\begin{aligned}
Q,K,V&:\ [B,H,N,d_h],\\
\operatorname{inv\_freq}&:\ [d_h/2],\\
\operatorname{wavelength}&:\ [d_h/2],\\
r=L/\operatorname{wavelength}&:\ [d_h/2],\\
\operatorname{ramp}\gamma&:\ [d_h/2],\\
\text{scaled }\operatorname{inv\_freq}&:\ [d_h/2],\\
\cos,\sin&:\ [N,d_h],\\
\text{rotated }Q,K&:\ [B,H,N,d_h].
\end{aligned}
```

### 단순화한 frequency 계산

```python
def yarn_inv_freq(inv_freq, original_len, factor,
                  alpha=1.0, beta=32.0):
    wavelength = 2 * math.pi / inv_freq
    rotations = original_len / wavelength

    ramp = ((rotations - alpha) / (beta - alpha)).clamp(0, 1)
    interpolated = inv_freq / factor
    scaled = (1 - ramp) * interpolated + ramp * inv_freq
    return scaled
```

### Temperature까지 포함

```python
scaled_inv_freq = yarn_inv_freq(inv_freq, L, factor)
angles = position_ids[:, None] * scaled_inv_freq[None, :]

rope_gain = 0.1 * math.log(factor) + 1.0  # sqrt(1/t)
cos = angles.cos() * rope_gain
sin = angles.sin() * rope_gain

q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)
```

실제 library는 `attention_factor`, `beta_fast`, `beta_slow`처럼 다른 이름을 쓸 수 있다. Eq. 15의 scale이 이미 Q/K 양쪽에 적용되는지, 최종 logit에 한 번만 적용되는지 확인해야 한다. 중복 적용하면 $1/t^2$에 가까운 잘못된 scale이 된다.

## 전체 알고리즘 흐름

1. Original context $L$과 target factor $s=L'/L$을 정한다.
2. 각 RoPE dimension의 wavelength $\lambda_d$를 계산한다.
3. Original context 안의 회전 횟수 $r(d)=L/\lambda_d$를 계산한다.
4. $\alpha$, $\beta$ 사이에서 ramp $\gamma(r)$를 만든다.
5. 고주파는 original frequency, 저주파는 $1/s$ frequency, 중간은 두 값을 blend한다.
6. $\sqrt{1/t}=0.1\ln(s)+1$로 attention scale을 정한다.
7. 수정된 cos/sin cache로 Q/K를 회전한다.
8. 일반 causal attention과 next-token prediction fine-tuning을 수행한다.
9. Dynamic mode라면 현재 sequence length에 따라 $s$를 갱신하고 cache 일관성을 보장한다.

## 실험 설정

### 64k와 128k 모델

- Base: Llama 2 7B, 13B
- Original context: 4k
- $s=16$: 64k context
- $s=32$: 128k context
- $s=16$ 모델: PG19를 64k segment로 나눠 400 steps fine-tuning
- $s=32$ 모델: 64k data만 사용하고 $s=16$ checkpoint에서 추가 200 steps
- learning rate $2\times10^{-5}$
- AdamW, $\beta_1=0.9$, $\beta_2=0.95$
- global batch 64, FlashAttention 2와 FSDP

128k 모델이 64k training sequence만 보고 128k에서 잘 동작하는지는 `train short, test long` 성질을 검증하는 핵심이다.

### 32k ablation

LLaMA 7B를 2k에서 32k로 $s=16$ 확장하고 PG19 32k segment로 400 steps 학습한다. PI, NTK-aware, NTK-by-parts, YaRN을 같은 예산으로 비교한다.

### 평가

- Proof-pile과 GovReport sliding-window perplexity
- Passkey retrieval
- ARC-Challenge, HellaSwag, MMLU, TruthfulQA
- Fine-tuning 없는 Dynamic Scaling 평가

## 주요 실험 결과

### 64k training에서 128k evaluation

Proof-pile 10개 128k 문서의 sliding-window PPL은 다음과 같다.

| 모델 | 학습 단계 | 8k | 16k | 32k | 64k | 128k |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Llama 2 7B YaRN $s=16$ | 400 | 3.51 | 2.99 | 2.65 | 2.42 | $>10^1$ |
| Llama 2 7B YaRN $s=32$ | 400+200 | 3.56 | 3.04 | 2.70 | 2.45 | **2.37** |
| Llama 2 13B YaRN $s=16$ | 400 | 3.25 | 2.79 | 2.50 | 2.29 | $>10^1$ |
| Llama 2 13B YaRN $s=32$ | 400+200 | 3.29 | 2.83 | 2.53 | 2.31 | **2.24** |

$s=32$ 모델은 64k data로만 추가 학습했지만 128k에서 낮은 perplexity를 보였다. Progressive transfer와 length extrapolation가 함께 작동한 결과다.

### 동일 400-step budget의 방법 비교

LLaMA 7B를 2k에서 32k로 확장한 경우 32k PPL은 다음과 같다.

| 방법 | Fine-tuning | 32k PPL |
| --- | ---: | ---: |
| PI | 없음 | $>10^2$ |
| NTK-aware | 없음 | $>10^1$ |
| NTK-by-parts | 없음 | $>10^1$ |
| YaRN | 없음 | 3.45 |
| PI | 400 steps | 3.57 |
| NTK-aware | 400 steps | 8.49 |
| NTK-by-parts | 400 steps | 2.81 |
| YaRN | 400 steps | **2.77** |

Temperature를 포함한 YaRN이 fine-tuned/non-fine-tuned 조건 모두에서 강했다. 특히 큰 scaling에서 PI의 local-frequency 손실을 줄였다.

### PI 대비 학습 효율

Llama 2 7B를 4k에서 8k로 확장한 비교다.

| 방법 | Steps | 2k | 4k | 6k | 8k |
| --- | ---: | ---: | ---: | ---: | ---: |
| PI | 1000 | 3.92 | 3.51 | 3.51 | **3.34** |
| YaRN | 400 | **3.91** | **3.50** | 3.51 | 3.35 |

YaRN은 2.5배 적은 steps로 거의 같은 결과를 얻었다. 논문은 전체 fine-tuning token도 이전 PI 연구보다 약 10배 적다고 강조한다.

### Passkey retrieval

| 모델 | Training data context | 평가 context | 평균 정확도 |
| --- | ---: | ---: | ---: |
| YaRN 7B $s=16$ | 64k | 64k | 96.3% |
| YaRN 7B $s=32$ | 64k | 128k | 99.4% |
| YaRN 13B $s=16$ | 64k | 64k | 97.5% |
| YaRN 13B $s=32$ | 64k | 128k | 99.4% |

128k sequence로 fine-tuning하지 않았는데도 128k에서 높은 retrieval 정확도를 기록했다.

논문은 중요한 경고도 한다. Code Llama 13B는 100k 이후 perplexity가 증가하지만 128k passkey는 잘 찾았다. 즉 **perplexity 하나만으로 effective context를 결정할 수 없다.** Generation 품질과 retrieval 능력은 서로 다를 수 있다.

### Original benchmark 보존

| 모델 | ARC-c | HellaSwag | MMLU | TruthfulQA |
| --- | ---: | ---: | ---: | ---: |
| Llama 2 7B | 53.1 | 77.8 | 43.8 | 39.0 |
| YaRN 7B 64k | 52.3 | 78.8 | 42.5 | 38.2 |
| YaRN 7B 128k | 52.1 | 78.4 | 41.7 | 37.3 |
| Llama 2 13B | 59.4 | 82.1 | 55.8 | 37.4 |
| YaRN 13B 128k | 58.0 | 82.2 | 51.9 | 37.3 |

Short-context 성능은 대체로 보존되지만 MMLU 등 일부 지표는 하락한다. 64k에서 128k로 늘릴 때 평균 약 0.49% 추가 하락이 있었다고 논문은 보고한다.

### Compute 비교

| 모델 | 확장 | 학습 A100-hours |
| --- | ---: | ---: |
| LLaMA 7B YaRN | 2k → 32k | 128 |
| Llama 2 7B YaRN | 4k → 64k | 256 |
| Llama 2 7B YaRN | 4k → 128k | 256 + 128 |
| PI 7B 사례 | 2k → 16k | 640 |
| Code Llama NTK | 4k → 약 100k | 6400 |

서로 다른 data와 recipe의 공개 모델을 비교하므로 완전히 통제된 표는 아니다. 그래도 YaRN이 적은 adaptation budget을 목표로 했음을 보여준다.

## 계산 복잡도와 메모리

고정 scale의 YaRN은 cos/sin cache를 미리 만들 수 있어 RoPE 단계의 추가 비용이 거의 없다.

```math
\text{position transform}=O(Nd).
```

그러나 dense attention은 그대로다.

```math
\text{attention}=O(N^2d),
\qquad
\text{KV cache}=O(NHd_h).
```

128k model을 학습·서빙할 수 있게 만드는 핵심은 YaRN만이 아니라 FlashAttention 2, distributed training, 충분한 memory다.

Dynamic Scaling은 scale이 바뀔 때 과거 key를 재회전해야 할 수 있어 고정 scaling보다 시스템 overhead가 커질 수 있다. “YaRN은 zero overhead”라는 표현은 주로 **고정 target scale과 cached RoPE** 조건에서 읽어야 한다.

## Position extension 방법 비교

| 방법 | High frequency | Low frequency | Attention scale | Fine-tuning 없는 동작 |
| --- | --- | --- | --- | --- |
| PI | $1/s$로 압축 | $1/s$로 압축 | 그대로 | 큰 배율 취약 |
| NTK-aware | 거의 보존 | 강하게 압축 | 그대로 | 비교적 강함, 실제 배율 불명확 |
| NTK-by-parts | 보존 | $1/s$로 압축 | 그대로 | PI보다 강함 |
| YaRN | 보존 | $1/s$로 압축 | temperature 보정 | 가장 강한 결과 |
| Dynamic-YaRN | 현재 길이에 따라 변경 | 현재 길이에 따라 변경 | 동적 | base model 즉시 적용 가능 |

## 장점과 핵심 기여

### 1. PI의 resolution 손실을 dimension 역할로 설명했다

RoPE wavelength가 original context 안에서 몇 번 회전하는지를 기준으로 local-relative와 absolute-like 차원을 구분한다.

### 2. 사람의 직관을 연속 ramp로 구현했다

Frequency를 세 hard group으로 잘라 불연속적으로 바꾸지 않고 transition band에서 blend한다.

### 3. Position scaling과 attention entropy를 분리했다

Frequency correction만으로 충분하지 않고 softmax scale도 보정해야 한다는 실험적 관찰을 추가했다.

### 4. 적은 fine-tuning budget

수백 steps와 짧은 long-context data로 64k/128k를 달성해 PI 계열의 실용성을 높였다.

### 5. 학습 길이 밖 extrapolation

64k data로 추가 학습한 $s=32$ 모델이 128k에서 낮은 PPL과 높은 passkey accuracy를 보였다.

### 6. Dynamic inference 가능성

고정 target context의 short-context 손실을 줄이고, fine-tuning 없는 base model에도 적용할 경로를 제시했다.

## 한계와 비판적 관점

### 1. $\alpha$, $\beta$와 temperature가 경험적이다

$\alpha=1$, $\beta=32$, $0.1\ln(s)+1$은 LLaMA 계열에서 찾은 값이다. Architecture, head dimension, RoPE base, pretraining length가 바뀌면 최적값도 달라질 수 있다.

### 2. 세 구간 가정이 최적임을 보장하지 않는다

RoPE dimension의 실제 learned 역할은 layer와 head마다 다를 수 있다. 하나의 global ramp를 모든 layer/head에 적용하는 것은 여전히 coarse heuristic이다. LongRoPE는 이 한계를 search로 다룬다.

### 3. Dynamic Scaling과 KV cache가 복잡하다

Scale 변화가 과거 key의 RoPE를 바꾸기 때문에 일반적인 post-RoPE KV cache와 바로 호환되지 않는다.

### 4. 128k의 dense attention 비용

Position encoding이 안정적이어도 실제 128k prefill과 training은 비싸다. YaRN은 sparse attention이나 KV compression이 아니다.

### 5. Perplexity와 retrieval이 서로 다른 신호를 준다

논문 자체가 이를 보여준다. Long-context 품질은 PPL, passkey, multi-needle, reasoning, original benchmark를 함께 평가해야 한다.

### 6. Fine-tuning data domain의 영향

PG19 book data는 original Llama pretraining distribution 및 downstream benchmark와 다르다. Original capability 하락 중 일부는 position scaling뿐 아니라 data adaptation에서 올 수 있다.

### 7. 매우 큰 extension의 안정성은 제한적이다

128k는 당시 강한 결과지만 million-token 수준에서는 scale heuristic만으로 충분하지 않다. Progressive extension과 더 세밀한 scaling이 필요하다.

## 자주 헷갈리는 지점

### YaRN은 단순히 RoPE base를 키우는가

아니다. Base change는 NTK-aware 방법이다. YaRN은 wavelength 기반 ramp로 original frequency와 interpolated frequency를 섞고 attention temperature를 추가한다.

### High frequency를 왜 interpolation하지 않는가

Original context 안에서 이미 여러 번 반복된 relative phase이므로 긴 position에서도 새로운 unique phase가 아니다. 이를 보존해야 인접 token resolution를 유지할 수 있다.

### Low frequency를 왜 interpolation하는가

Original context에서 한 바퀴도 돌지 않았다면 각 phase가 absolute position처럼 unique할 수 있다. 긴 길이의 새 phase는 OOD이므로 target range 안으로 압축한다.

### Temperature는 RoPE frequency인가

아니다. Frequency는 phase를 바꾸고 temperature는 QK logit magnitude를 바꾼다. 서로 다른 문제를 보정한다.

### YaRN이 attention code를 변경하는가

수학적으로는 temperature를 바꾸지만, Q/K 양쪽 RoPE vector scale에 흡수하면 기존 attention kernel을 유지할 수 있다.

### Dynamic-YaRN에서 회전된 key를 cache해도 되는가

Scale이 계속 바뀐다면 안전하지 않다. 과거 key도 현재 scale에 맞춰야 하므로 pre-RoPE cache 또는 재계산 전략이 필요하다.

## 온디바이스 및 비전 관점

### 온디바이스 LLM

고정 YaRN은 model parameter를 늘리지 않고 cos/sin cache 생성만 바꿔 배포가 쉽다. 하지만 64k/128k KV cache는 edge memory에 매우 크다. GQA, KV quantization, paged cache, sliding-window를 함께 써야 한다.

Dynamic Scaling은 short prompt 품질에 유리하지만 cache 재회전 비용이 모바일 NPU/GPU에서 부담이 될 수 있다. 여러 fixed profile(예: 4k/16k/32k)을 준비하는 방식이 더 단순할 수 있다.

### Vision

Image의 row/column RoPE에서도 spatial frequency마다 interpolation 비율을 다르게 줄 수 있다. High spatial frequency는 local patch order를 보존하고 low frequency는 큰 resolution에 맞춰 압축하는 식이다. 다만 language의 causal direction과 달리 2D bidirectional geometry이므로 row와 column, aspect ratio를 별도로 설계해야 한다.

## 후속 논문과의 연결점

LongRoPE는 YaRN이 수작업으로 만든 frequency group/ramp를 더 일반화한다.

- 각 RoPE dimension의 scale $\lambda_i$를 별도로 탐색한다.
- 초기 몇 token은 interpolation하지 않는 token-position non-uniformity를 추가한다.
- 128k/256k로 fine-tuning한 뒤 다시 search해 2048k로 progressive extension한다.

즉 PI가 “모두 동일 압축”, YaRN이 “주파수 band별 압축”, LongRoPE가 “차원별 탐색 압축”으로 이어진다.

## 개인 학습/연구 메모

YaRN을 다음처럼 기억하면 된다.

```text
Do not compress every RoPE frequency equally.
Keep local high frequencies.
Compress long wavelengths.
Blend the middle.
Then correct the attention temperature.
```

핵심은 RoPE를 하나의 position signal로 보지 않고, 서로 다른 역할을 가진 **frequency bank**로 해석한 데 있다.

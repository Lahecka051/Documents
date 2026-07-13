# 12. Extending Context Window of Large Language Models via Position Interpolation

## 논문 정보

- 제목: **Extending Context Window of Large Language Models via Position Interpolation**
- 저자: Shouyuan Chen, Sherman Wong, Liangjian Chen, Yuandong Tian
- 발표: 2023년 arXiv
- 원본 PDF: [12_Positional_Interpolation_Context_Window_Extension.pdf](./12_Positional_Interpolation_Context_Window_Extension.pdf)
- 핵심 키워드: `Position Interpolation`, `PI`, RoPE scaling, context window extension, passkey retrieval

## 한눈에 보는 요약

Position Interpolation(PI)은 RoPE 기반 pretrained LLM의 context window를 늘릴 때, 새 position을 학습 범위 밖으로 extrapolate하지 않는다. 대신 **긴 position index 전체를 기존 학습 구간 안으로 선형 압축**한다.

원래 context 길이를 $L$, 목표 길이를 $L'>L$이라 하면 RoPE에 위치 $m$을 그대로 넣는 대신 다음 위치를 넣는다.

$$
m' = m\frac{L}{L'}.
$$

따라서 수정된 RoPE는 다음과 같다.

$$
f'(x,m)
=
f\left(x,m\frac{L}{L'}\right).
$$

예를 들어 2048 token으로 pretrain한 LLaMA를 8192로 확장하면 RoPE에는 $m/4$를 사용한다. Target position 8191도 원래 학습 범위의 약 2047.75에 대응하므로, 모델은 보지 못한 phase 구간 대신 이미 학습한 구간 사이를 interpolation한다.

논문은 LLaMA 7B~65B를 최대 32768 context로 확장했고, 대부분 1000 fine-tuning steps 안에서 긴 context를 활용하는 성능을 얻었다. 핵심 절충은 명확하다. **범위 밖 불안정성을 없애는 대신, 인접 token의 positional resolution을 확장 배율만큼 낮춘다.**

![Position Interpolation이 긴 위치 구간을 기존 RoPE 범위로 매핑하는 과정](https://github.com/user-attachments/assets/b7725511-4985-4c83-84ea-194ecaeb305e)

## 연구 배경과 문제의식

### 기존 LLM의 context window는 pretraining 설정에 묶여 있다

LLaMA 1은 최대 2048 token context로 pretrain되었다. 이 길이는 긴 대화, 보고서 요약, 코드 저장소 분석, 장기 planning에는 짧다. 처음부터 더 긴 context로 pretrain하면 dense attention의 비용과 긴 학습 데이터 요구량이 크게 늘어난다.

자연스러운 질문은 다음과 같다.

```text
Can an existing RoPE-pretrained LLM be adapted to a much longer context
without repeating expensive pretraining?
```

### Direct fine-tuning만으로는 적응이 느리다

RoPE 공식을 바꾸지 않고 긴 sequence로 fine-tuning하면 position $m\ge L$에서 학습 때 없던 phase를 사용한다. 논문은 10,000 batches 이상 직접 fine-tuning해도 effective context가 2048에서 약 2560으로만 늘어난 사례를 보고한다.

즉 긴 데이터만 보여준다고 모델이 새로운 위치 함수의 out-of-distribution 영역에 쉽게 적응하는 것은 아니다.

### Direct extrapolation의 catastrophic behavior

RoPE가 상대 위치 $m-n$에만 의존한다는 사실만 보면 긴 길이에서도 동작할 것처럼 보인다. 하지만 학습 범위 안에서 안정적인 trigonometric basis의 선형 결합이 범위 밖에서도 안정적이라는 보장은 없다.

논문은 attention score를 다음 basis expansion으로 본다.

$$
a(s)
=
\operatorname{Re}\left[
\sum_{j=0}^{d/2-1}h_je^{is\theta_j}
\right],
\qquad s=m-n.
$$

$h_j$는 Q/K content에서 나온 complex coefficient다. 충분히 많은 trigonometric basis는 학습 구간 $s\in[0,L]$의 값을 잘 맞추면서도 범위 밖에서 매우 큰 값을 만들 수 있다. 논문의 합성 예시는 $[0,2048]$에서 약 $[-1,1]$로 맞춘 함수가 4096 부근에서 8000을 넘을 수 있음을 보여준다.

이것이 direct extrapolation의 핵심 위험이다. 학습 범위 밖에서 일부 Q/K 조합이 매우 큰 attention logit을 만들면 softmax가 거의 one-hot으로 포화되고 layer 전체가 무너질 수 있다.

## RoPE 배경

Head dimension을 $d$, position을 $m$이라 하자. Real vector의 인접 두 차원을 complex number로 묶으면 RoPE는 다음과 같다.

$$
f(x,m)
=
\left[
(x_0+ix_1)e^{im\theta_0},
\ldots,
(x_{d-2}+ix_{d-1})e^{im\theta_{d/2-1}}
\right]^{\top},
$$

$$
\theta_j=10000^{-2j/d}.
$$

Query 위치 $m$, key 위치 $n$의 attention score는 다음과 같이 상대 위치에 의존한다.

$$
a(m,n)
=
\operatorname{Re}\langle f(q,m),f(k,n)\rangle
=
a(m-n).
$$

Real form으로 전개하면 각 pair에서 cosine과 sine의 합이 된다.

$$
\begin{aligned}
a(m-n)
=\sum_j &
(q_{2j}k_{2j}+q_{2j+1}k_{2j+1})
\cos((m-n)\theta_j)\\
&+(q_{2j}k_{2j+1}-q_{2j+1}k_{2j})
\sin((m-n)\theta_j).
\end{aligned}
$$

PI는 $q$, $k$, $\theta_j$를 학습하거나 새 parameter를 넣지 않고, 오직 $m$과 $n$을 동일 비율로 줄인다.

## 핵심 방법: Position Interpolation

### 위치 index 압축

원래 최대 길이 $L$의 모델을 $L'$로 확장할 때 scale factor를 다음과 같이 둔다.

$$
s=\frac{L'}{L}.
$$

PI 위치는 다음과 같다.

$$
\tilde m=\frac{m}{s}=m\frac{L}{L'}.
$$

RoPE angle은 모든 주파수에서 동일 비율로 줄어든다.

$$
\tilde\phi_{m,j}
=
\frac{m}{s}\theta_j.
$$

Attention의 상대 phase도 목표 거리 $m-n$을 원래 범위로 압축한다.

$$
(m-n)\theta_j
\quad\longrightarrow\quad
\frac{m-n}{s}\theta_j.
$$

### Extrapolation이 interpolation으로 바뀌는 이유

$0\le m<L'$라면 다음이 성립한다.

$$
0\le \tilde m < L.
$$

목표 context의 모든 position이 pretrained interval 안에 놓인다. 대부분 non-integer가 되지만 sine과 cosine은 연속 함수이므로 두 학습 grid 사이의 phase를 자연스럽게 계산할 수 있다.

```text
original RoPE grid:  0, 1, 2, 3, ..., 2047
PI for 4x context:  0, .25, .50, .75, 1, ..., 2047.75
```

모델이 새로 적응해야 하는 것은 “한 번도 보지 못한 phase 범위”가 아니라 “기존 phase 사이에 token이 더 촘촘히 놓이는 현상”이다.

### Position resolution의 손실

압축 전 인접 token의 phase 차이는 $\theta_j$다. 압축 후에는 $\theta_j/s$가 된다.

$$
\Delta\phi_j
=
\frac{\theta_j}{s}.
$$

$s$가 커질수록 인접 위치가 더 비슷하게 보인다. PI가 안정적인 대신 큰 배율에서 short-range order를 희생하는 이유다. 논문의 후속인 YaRN과 LongRoPE는 모든 주파수를 똑같이 압축하는 이 문제를 개선한다.

## Interpolation bound

### 정리의 의미

논문은 학습된 integer grid 사이에서 attention score가 얼마나 휘어질 수 있는지 상한을 제시한다. $s\in[s_1,s_2]$이고 두 끝점이 안정적이라고 하자.

$$
a_{linear}(s)
=
(1-\lambda(s))a(s_1)+\lambda(s)a(s_2),
$$

$$
\lambda(s)=\frac{s-s_1}{s_2-s_1}.
$$

그러면 RoPE base를 $c$라 할 때 다음 bound를 얻는다.

$$
|a(s)-a_{linear}(s)|
\le
d\left(\max_j|h_j|\right)
\frac{(s-s_1)(s_2-s)}{8\ln c}.
$$

인접 integer grid라면 $(s-s_1)(s_2-s)\le1/4$이므로 $c=10000$에서 다음과 같이 단순화된다.

$$
|a(s)-a_{linear}(s)|
\lesssim
\frac{d\max_j|h_j|}{294.73}.
$$

### 왜 extrapolation bound보다 작다고 주장하는가

RoPE의 기존 extrapolation bound에는 다음 누적항이 들어간다.

$$
B(s)
=
\sum_{k=0}^{d/2-1}|A_{k+1}(s)|,
\qquad
A_k(s)=\sum_{j=0}^{k-1}e^{is\theta_j}.
$$

LLaMA-7B의 head dimension $d=128$에서 논문은 수치적으로 $B(s)/d\ge1$이고 여러 위치에서 훨씬 크다고 관찰한다. 이를 비교하면 interpolation bound가 extrapolation bound보다 최소 약 $2\times294.73\approx600$배 작다는 주장이다.

이 정리는 PI가 실제 task 성능을 600배 개선한다는 뜻이 아니다. **Attention score의 최악 편차에 대한 이론적 상한이 훨씬 작다**는 뜻이다.

### 증명의 핵심

Taylor expansion으로 $a(s_1)$과 $a(s_2)$를 $a(s)$ 주변에서 전개한다. Linear interpolation error는 second derivative의 크기로 제한된다.

$$
|a''(s)|
\le
\left(\max_j|h_j|\right)
\sum_{j=0}^{d/2-1}\theta_j^2.
$$

$\theta_j=c^{-2j/d}$의 기하급수 합을 bound하면 다음을 얻는다.

$$
|a''(s)|
\le
\left(\max_j|h_j|\right)
\frac{d}{4\ln c}.
$$

이 값을 표준 linear interpolation error 식에 대입하면 위 정리가 나온다.

## Tensor shape와 구현

PI는 Q/K의 shape를 바꾸지 않는다.

```text
hidden states               [B, N, d_model]
Q, K                        [B, H, N, d_h]
position index              [N]
scaled position             [N]
cos/sin cache               [N, d_h]
rotated Q, K                [B, H, N, d_h]
attention score             [B, H, N, N]
```

### 단순화한 구현

```python
def position_interpolated_rope(q, k, position_ids,
                               original_length, target_length):
    scale = target_length / original_length
    scaled_pos = position_ids.float() / scale
    angles = scaled_pos[:, None] * inv_freq[None, :]
    cos = angles.cos()
    sin = angles.sin()
    return apply_rope(q, cos, sin), apply_rope(k, cos, sin)
```

다른 구현에서는 `rope_scaling={"type": "linear", "factor": s}`처럼 frequency를 $1/s$로 조정한다. Position을 나누는 것과 frequency를 나누는 것은 곱 $m\theta$ 관점에서 동등하다.

### Architecture와 parameter는 그대로

PI는 다음을 바꾸지 않는다.

- Transformer layer 수와 hidden dimension
- Q/K/V projection weight
- Attention mask와 value aggregation
- Vocabulary와 tokenizer
- 학습 objective

변경점은 RoPE angle 계산 한 곳뿐이다. 따라서 FlashAttention, FSDP 같은 기존 인프라를 재사용할 수 있다.

### Fine-tuning

논문은 next-token prediction으로 짧게 adaptation한다.

- LLaMA 7B/13B: learning rate $2\times10^{-5}$
- LLaMA 33B/65B: learning rate $10^{-5}$
- AdamW, $\beta_1=0.9$, $\beta_2=0.95$
- 20-step linear warmup
- weight decay 0
- PI는 기본 1000 steps
- direct fine-tuning baseline은 10,000 steps
- 주로 Pile 사용

Fine-tuning은 새 지식을 대량 학습하기보다 **압축된 position geometry에 기존 attention weight를 적응**시키는 과정으로 해석한다.

## 전체 계산 흐름

1. Pretrained RoPE 모델과 original context $L$을 준비한다.
2. 목표 context $L'$와 scale $s=L'/L$을 정한다.
3. 모든 position index를 $m/s$로 변환한다.
4. 변환한 non-integer position으로 RoPE cos/sin을 계산한다.
5. Q/K를 회전하고 일반 causal attention을 수행한다.
6. 목표 길이 sequence로 next-token prediction fine-tuning한다.
7. Long-context perplexity와 retrieval을 평가한다.
8. 원래 context 길이 benchmark의 성능 보존도 함께 확인한다.

## 작은 수치 예시

2048에서 8192로 확장하면 $s=4$다. 같은 실제 token 거리도 RoPE에서는 1/4로 보인다.

```text
actual query position: 6000
actual key position:   5996
actual relative gap:      4

PI query position: 1500
PI key position:   1499
PI relative gap:      1
```

이 덕분에 phase는 학습 범위 안에 있지만, 원래 4-token 차이가 1-token 차이와 같은 phase 간격을 가진다. Model이 content와 여러 주파수를 이용해 이 압축된 geometry에 적응하는 것이 fine-tuning의 역할이다.

## 실험 설정

### 모델

- LLaMA 7B, 13B, 33B, 65B
- Original context 2048
- 목표 context 8192, 16384, 32768

### 학습 시스템

- 7B/13B/33B의 8192 확장: 32 A100, global batch 64
- 그 외: 128 A100, global batch 128
- PyTorch FSDP와 FlashAttention 사용

GPU 수가 많은 주된 이유는 model parameter보다 long-sequence activation과 attention memory다.

### 평가

- PG19 book language modeling
- Arxiv Math Proof-pile
- Passkey retrieval
- Original LLaMA benchmark subset
- GovReport long document summarization

Perplexity는 sliding window stride 256으로 계산한다. 이 protocol은 각 token이 가능한 긴 left context를 받도록 해 모델이 추가 context를 실제 활용하는지 본다.

## 주요 실험 결과

### PG19 perplexity

| 모델 | 확장 방식 | 2048 | 4096 | 8192 | 16384 | 32768 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LLaMA 7B base | 없음 | 7.20 | $>10^3$ | $>10^3$ | $>10^3$ | $>10^3$ |
| 7B, 8192 | Direct FT | 7.21 | 7.34 | 7.69 | - | - |
| 7B, 8192 | PI | 7.13 | 6.96 | 6.95 | - | - |
| 7B, 16384 | PI | 7.11 | 6.93 | 6.82 | 6.83 | - |
| 7B, 32768 | PI | 7.23 | 7.04 | 6.91 | 6.80 | **6.77** |

Base model은 2048 밖에서 catastrophic failure를 보인다. Direct fine-tuning은 context가 길어질수록 perplexity가 나빠지지만, PI 모델은 긴 context를 실제로 활용해 전반적으로 내려간다.

13B의 32768 PI 모델도 PG19에서 2048의 6.54에서 32768의 6.09로 개선되었다. 단순히 더 긴 tensor를 받아들이는 것이 아니라 추가 history가 next-token prediction에 기여한다.

### 적응 속도

LLaMA-7B PI 모델의 PG19 perplexity는 다음과 같이 빠르게 줄었다.

| 목표 길이 | step 0 | 200 | 400 | 600 | 800 | 1000 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 16.10 | 7.12 | 7.10 | 7.02 | 6.99 | 6.95 |
| 16384 | 112.13 | 7.05 | 6.93 | 6.88 | 6.84 | 6.83 |

Fine-tuning 전에도 8192 확장은 direct extrapolation의 $>10^3$보다 훨씬 안정적이다. 16384는 초기 PPL이 높지만 200 steps 만에 크게 회복한다.

### Effective context: passkey retrieval

Passkey task는 무의미한 긴 text 속 5자리 숫자를 숨기고 마지막에 회수하게 한다. 논문은 성공률 20% 이상인 최대 거리를 effective context의 하한으로 사용한다.

| 모델 | 방식 | 200 steps | 1000 steps | 10000 steps |
| --- | --- | ---: | ---: | ---: |
| 7B → 8192 | Direct FT | 1792 | 2304 | 2560 |
| 7B → 8192 | PI | **8192** | **8192** | - |
| 7B → 16384 | PI | **16384** | **16384** | - |
| 7B → 32768 | PI | **32768** | **32768** | - |

PI 모델은 200 steps 뒤 목표 window 전체에서 passkey를 회수했다. Direct FT는 10,000 steps 뒤에도 2560에 머물렀다.

다만 passkey는 단일 정보 검색 과제다. 복수 근거 통합과 장거리 reasoning까지 같은 수준으로 된다는 증거는 아니다.

### 원래 길이의 성능 보존

7B base와 8192 PI 모델을 비교하면 다음과 같다.

| 모델 | BoolQ | PIQA | RACE-M | RACE-H | WinoGrande |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base 2048 | 76.1 | 78.9 | 55.7 | 42.2 | 69.6 |
| PI 8192 | 73.2 | 78.2 | 53.8 | 41.7 | 69.0 |

대부분의 하락은 작지만 BoolQ처럼 짧은 reference에서 word order가 중요한 task는 더 민감하다. 16384/32768로 배율이 커질수록 원래 길이 benchmark의 저하도 커졌다. Position compression의 비용이 실제로 존재한다.

### Long document summarization

16k로 확장한 LLaMA-7B를 GovReport에 fine-tuning한 결과는 다음과 같다.

| 모델 | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| CoLT5 Base | 58.7 | 29.6 | 31.4 |
| CoLT5 XL | 61.3 | 32.2 | 33.8 |
| PI LLaMA-7B 16K | 60.0 | 28.0 | 29.5 |

R1은 경쟁력 있지만 R2/RL은 강한 baseline보다 낮다. Long context 접근이 summarization 품질 전체를 자동으로 해결하지는 않는다.

## 계산 복잡도와 시스템 비용

PI의 RoPE 계산 자체는 기존과 거의 같다.

$$
\text{RoPE cost}=O(Nd).
$$

그러나 dense attention은 그대로다.

$$
\text{attention time}=O(N^2d),
\qquad
\text{score memory}=O(HN^2).
$$

2048에서 32768로 16배 확장하면 naive attention matrix의 원소 수는 256배가 된다. 논문이 FSDP와 FlashAttention, 많은 A100 GPU를 사용한 이유다.

따라서 PI는 **position extrapolation 문제를 해결**하지만 다음 문제는 해결하지 않는다.

- Prefill latency
- KV cache growth
- Long sequence activation memory
- Long data loading과 batching
- Multi-million-token attention의 시스템 비용

## 기존 방법과 비교

| 방법 | 적용 시점 | 핵심 조작 | 장점 | 주요 절충 |
| --- | --- | --- | --- | --- |
| ALiBi | 처음부터 학습 | logit에 거리 bias | 자연스러운 extrapolation | pretrained RoPE 변환 아님 |
| xPos | 주로 from scratch | RoPE + reciprocal decay | resolution 안정화 | 수치/구현 복잡성 |
| Direct FT | checkpoint 이후 | 긴 data만 사용 | 위치식 변경 없음 | 적응이 매우 느림 |
| PI | checkpoint 이후 | 모든 position/frequency uniform 압축 | 단순하고 안정적 | local resolution 손실 |
| YaRN | checkpoint 이후 | 주파수별 압축 + temperature | 큰 배율에서 PI 개선 | hyperparameter 증가 |
| LongRoPE | checkpoint 이후 | 차원/위치별 scale search | 극단적 길이 확장 | search와 progressive FT 비용 |

## 장점과 핵심 기여

### 1. 매우 작은 변경으로 pretrained LLM을 확장한다

RoPE position index 한 줄을 조정하고 짧게 fine-tuning하면 된다. Architecture와 parameter shape가 그대로다.

### 2. Extrapolation 실패를 함수 근사의 범위 밖 문제로 설명한다

Relative position encoding도 학습 구간 밖에서 안전하지 않을 수 있음을 trigonometric basis 관점으로 명확히 보였다.

### 3. 이론과 실험이 같은 방향을 지지한다

Interpolation bound가 훨씬 작고, 실제 perplexity도 direct extrapolation보다 안정적이다.

### 4. Fine-tuning 효율이 높다

수백~1000 steps 만에 7B~65B 모델을 최대 16배 확장했다.

### 5. Effective context를 retrieval로 확인했다

Perplexity만 보고 “context가 늘었다”고 주장하지 않고, passkey가 먼 위치의 정보를 실제로 회수하는지 측정했다.

## 한계와 비판적 관점

### 1. Uniform compression은 모든 RoPE 주파수를 동일하게 훼손한다

고주파는 local order에, 저주파는 긴 범위에 다른 역할을 할 수 있다. 모든 $\theta_j$를 $1/s$로 줄이는 것은 큰 배율에서 비효율적이다.

### 2. Short-context 성능 저하

확장 모델은 원래 2048 구간 안에서도 position을 좁은 범위로 압축한다. 목표 길이가 클수록 원래 task의 positional resolution가 줄어든다.

### 3. Fine-tuning은 여전히 긴 sequence를 요구한다

Step 수는 적지만 16k/32k sequence의 dense training은 비싸다. “1000 steps”만으로 총 GPU 비용이 작다고 단정하면 안 된다.

### 4. 이론적 bound는 실제 learned distribution을 완전히 설명하지 않는다

Worst-case upper bound의 600배 차이는 accuracy나 perplexity의 배율이 아니다. Q/K coefficient, layer normalization, softmax saturation, content distribution을 단순화한다.

### 5. Passkey는 쉬운 장거리 과제다

정확한 effective context 평가는 multi-needle retrieval, order-sensitive aggregation, long-range QA, generation coherence를 함께 봐야 한다.

### 6. Attention/KV 비용은 해결하지 않는다

위치가 안정적이어도 32k serving은 계산과 메모리 측면에서 훨씬 어렵다.

### 7. Scaling factor는 checkpoint metadata의 일부다

Fine-tuned weight만 저장하고 RoPE factor나 original max length를 잘못 설정하면 성능이 즉시 무너진다. Model config와 tokenizer-side truncation까지 함께 배포해야 한다.

## 자주 헷갈리는 지점

### PI는 position embedding table을 interpolation하는가

아니다. RoPE에는 보통 학습 position table이 없다. 연속적인 position index를 $m/s$로 바꿔 cosine/sine angle을 계산한다.

### Non-integer position을 써도 되는가

된다. Sine과 cosine은 실수 입력에 정의된다. 문제는 계산 가능성보다 모델이 더 촘촘한 phase를 구분하도록 적응하는가다.

### Fine-tuning 없이도 바로 되는가

Direct extrapolation보다 훨씬 안정적이지만 큰 배율에서는 성능이 나쁘다. 논문의 주 결과는 짧은 fine-tuning을 포함한다.

### PI가 context를 늘리면 attention 비용도 줄어드는가

아니다. 위치 angle만 바뀐다. $N^2$ attention과 KV cache는 목표 길이에 맞게 그대로 증가한다.

### 2048 모델을 8192로 확장하면 실제 4 token 차이가 사라지는가

완전히 사라지지는 않지만 RoPE phase 차이는 원래 1 token 차이 수준으로 압축된다. Content feature와 여러 layer가 보완하지만 resolution 손실은 존재한다.

### 평가를 32768에서 성공하면 모든 32768 token을 reasoning에 쓰는가

Perplexity와 단일 passkey retrieval만으로는 보장할 수 없다. 이용 가능한 context와 복잡하게 활용 가능한 context는 다르다.

## 온디바이스 및 비전 관점

### 온디바이스 LLM

PI는 extra parameter가 없고 cos/sin cache만 바꾸므로 model binary 크기에 거의 영향이 없다. 반면 긴 KV cache는 모바일/edge memory를 빠르게 소모한다. 실제 배포에서는 다음을 함께 고려해야 한다.

- GQA/MQA
- KV cache quantization
- sliding-window 또는 chunk attention
- prefill chunking
- target context별 RoPE cache 생성

### Vision Transformer

2D RoPE에서 row와 column position을 각각 scaling할 수 있다.

$$
r' = r\frac{H_{train}}{H_{test}},
\qquad
c' = c\frac{W_{train}}{W_{test}}.
$$

이 방식은 train resolution보다 큰 image에서 spatial position을 interpolation하는 것과 유사하다. 다만 aspect ratio가 바뀌면 row/column scale을 별도로 써야 하고, 아주 큰 배율에서는 인접 patch resolution가 낮아진다.

## 후속 논문과의 연결점

PI는 후속 RoPE context-extension 연구의 출발점이다.

- **YaRN**: 모든 주파수의 uniform scaling 대신 wavelength에 따라 extrapolation, interpolation, transition을 나눈다.
- **LongRoPE**: 차원별 scaling factor와 초기 token 처리까지 search하고, progressive extension으로 2M token을 목표로 한다.

두 방법 모두 PI의 핵심 철학, 즉 **범위 밖 extrapolation보다 범위 안 interpolation이 안정적이다**라는 관찰을 유지하면서 resolution 손실을 줄인다.

## 개인 학습/연구 메모

Position Interpolation은 다음 mapping 하나로 기억할 수 있다.

$$
\boxed{
[0,L')\ni m
\quad\mapsto\quad
m\frac{L}{L'}\in[0,L)
}
$$

이 방법이 성공한 이유는 모델에 더 복잡한 position function을 가르쳤기 때문이 아니라, 이미 학습한 RoPE phase 범위 안에서 문제를 다시 정의했기 때문이다. 이후 방법들의 질문은 “압축할 것인가”가 아니라 **어떤 주파수와 위치를 얼마나 압축해야 하는가**로 바뀐다.

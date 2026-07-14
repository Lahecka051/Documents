# 14. LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens

## 논문 정보

- 제목: **LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens**
- 저자: Yiran Ding, Li Lyna Zhang, Chengruidong Zhang, Yuanyuan Xu, Ning Shang, Jiahang Xu, Fan Yang, Mao Yang
- 발표: 2024년 arXiv
- 원본 PDF: [14_LongRoPE.pdf](./14_LongRoPE.pdf)
- 핵심 키워드: `LongRoPE`, non-uniform position interpolation, evolutionary search, progressive extension, short-context recovery

## 한눈에 보는 요약

LongRoPE는 RoPE 기반 pretrained LLM의 context를 2048k, 즉 $2048\times1024=2{,}097{,}152$ token까지 확장한다. 단순히 512배 Position Interpolation을 적용하지 않고 세 단계로 문제를 나눈다.

1. **두 종류의 non-uniformity를 탐색한다.**
   - RoPE dimension마다 다른 scale $\lambda_i$
   - 초기 $\hat n$ token은 interpolation하지 않는 position threshold
2. **Progressive extension을 사용한다.**
   - 4k → 128k → 256k는 총 1000 steps fine-tuning
   - 256k checkpoint에서 두 번째 scale search로 2048k까지 추가 fine-tuning 없이 확장
3. **Short-context profile을 다시 탐색한다.**
   - 4k/8k 입력에는 덜 압축된 별도 RoPE factor를 적용해 원래 성능을 회복

핵심 수식은 하나의 global factor $s$ 대신 각 RoPE pair에 $\lambda_i$를 적용하는 것이다.

```math
\phi_{n,i}
=I(\lambda_i,\hat n)\frac{n}{\beta_i},
```

```math
I(\lambda_i,\hat n)=
\begin{cases}
1, & n<\hat n\\
1/\lambda_i, & n\ge\hat n.
\end{cases}
```

논문은 LLaMA2-7B와 Mistral-7B에서 2048k context를 구성하고, LLaMA2가 4k~2048k passkey retrieval에서 90% 이상을 유지했다고 보고한다. 다만 position scaling이 성공했다고 해서 2M-token dense attention의 계산·메모리 문제가 해결된 것은 아니다.

![LongRoPE의 차원별 scale 탐색과 progressive context extension](https://github.com/user-attachments/assets/164b91b1-c6d8-4c05-b7d0-c8f6ad278251)

## 연구 배경과 문제의식

### 기존 context extension의 세 장애물

논문은 128k를 넘어 million-token context로 가기 어려운 이유를 세 가지로 정리한다.

#### 1. 새 position의 out-of-distribution 문제

4k 모델을 2048k로 직접 extrapolate하면 전체 position의 99% 이상이 학습 범위 밖이다. RoPE의 새로운 phase가 catastrophic attention logit을 만들 수 있다.

#### 2. Million-token fine-tuning data와 compute의 부족

2M token을 넘는 자연스러운 문서는 드물고, dense attention으로 해당 길이를 직접 학습하는 비용은 매우 크다.

#### 3. Short-context 성능 하락

512배 uniform interpolation은 original 4k 위치도 8-token 상당의 RoPE 구간에 압축한다.

```math
\frac{4096}{512}=8.
```

인접 token의 phase 차이가 극단적으로 작아져 original benchmark 성능이 떨어질 수 있다.

### PI와 YaRN도 충분히 세밀하지 않다

- PI: 모든 dimension에 $\lambda=s$
- NTK-aware: dimension index의 고정 함수로 scale
- YaRN: frequency를 세 group으로 나눠 scale

이 방식들은 모두 사람이 정한 비교적 단순한 rule을 사용한다. LongRoPE는 model과 target length마다 최적 dimension scale이 다를 수 있다고 보고 **perplexity를 objective로 직접 탐색**한다.

## RoPE와 interpolation의 통일된 표현

RoPE head dimension을 $d$, base를 $\theta=10000$이라 하자. Position $n$의 encoding은 단순화하면 다음과 같다.

```math
[
\cos(n\theta_0),\sin(n\theta_0),
\ldots,
\cos(n\theta_{d/2-1}),\sin(n\theta_{d/2-1})
],
```

```math
\theta_i=\theta^{-2i/d}.
```

논문은 $\beta=\theta^{2/d}$를 두고 각 pair의 angle을 대략 $n/\beta^i$로 표기한다. Context extension ratio는 다음과 같다.

```math
s=\frac{L'}{L}.
```

다양한 interpolation은 dimension별 rescale factor $\lambda_i$로 통일할 수 있다.

```math
\left[
\cos\left(\frac{n}{\lambda_0\beta^0}\right),
\sin\left(\frac{n}{\lambda_0\beta^0}\right),
\ldots
\right].
```

PI는 모든 $\lambda_i=s$인 특수한 경우다. $\lambda_i=1$이면 해당 dimension은 전혀 interpolation하지 않고 direct extrapolation한다.

## 발견 1: RoPE dimension non-uniformity

### 모든 차원을 같은 비율로 줄일 필요가 없다

High-frequency dimension은 local position을 구분하는 데 중요하고, low-frequency dimension은 긴 범위의 phase를 담당한다. YaRN도 이 차이를 이용하지만 group boundary와 scale formula가 heuristic이다.

LongRoPE는 LLaMA2-7B의 각 RoPE dimension에 대해 $\lambda_i$를 직접 탐색했다. Fine-tuning 없이 8k/16k로 확장한 결과는 다음과 같다.

| 방법 | PG19 8k | PG19 16k | Proof-pile 8k | Proof-pile 16k |
| --- | ---: | ---: | ---: | ---: |
| PI | 10.65 | 20.49 | 3.65 | 4.93 |
| Dynamic NTK | 10.21 | 23.29 | 3.50 | 3.87 |
| YaRN | 32.64 | 87.89 | 3.49 | 3.25 |
| Dimension-wise search | **9.37** | **11.34** | **3.45** | **3.13** |

Search한 scale이 기존 rule보다 안정적이다. 논문은 이를 original RoPE의 중요한 dimension 정보를 더 잘 보존한 결과로 해석한다.

### Monotonicity prior

완전히 자유로운 $\lambda_i$를 탐색하면 search space가 너무 크다. NTK 관점에서 high-frequency의 낮은 dimension은 덜 interpolation하고 low-frequency의 높은 dimension은 더 interpolation하는 것이 합리적이므로 다음 제약을 둔다.

```math
\lambda_i\le\lambda_{i+1}.
```

이 제약은 탐색을 크게 줄이지만 최적해가 반드시 monotone하다는 증명은 아니다. 효율을 위한 inductive bias다.

## 발견 2: 초기 token position non-uniformity

### 왜 시작 token을 다르게 취급하는가

StreamingLLM과 LM-Infinite 계열 연구는 sequence 시작 token이 높은 attention을 받는 `attention sink` 현상을 관찰한다. BOS와 initial token은 여러 query가 공통적으로 보는 anchor 역할을 할 수 있다.

LongRoPE는 첫 $\hat n$ token에는 original RoPE를 그대로 쓰고, 이후 token에만 dimension-wise factor를 적용한다.

```math
I(\lambda_i,\hat n)=
\begin{cases}
1, & 0\le n<\hat n\\
1/\lambda_i, & n\ge\hat n.
\end{cases}
```

### 실험 관찰

PI와 Dynamic NTK로 8k/16k를 확장하면서 $\hat n$을 0~256으로 바꿨다. PI 16k의 PG19 PPL은 $\hat n=0$에서 20.49였고, $\hat n=16$에서 19.64로 개선됐다. Dynamic NTK 16k도 23.29에서 $\hat n=16$의 22.68로 개선됐다.

하지만 너무 많은 initial token을 보존하면 오히려 나빠진다. PI 16k에서 $\hat n=256$은 34.65였다. Optimal $\hat n$은 target length에 따라 달랐다.

이 결과는 `첫 N token은 항상 보존`이라는 고정 rule보다 threshold도 search해야 한다는 근거다.

### 두 non-uniformity의 결합

64k LLaMA2-7B에서 fine-tuning 전/후 Proof-pile PPL은 다음과 같다.

| 방법 | Fine-tuning 전 | Fine-tuning 후 |
| --- | ---: | ---: |
| PI | 72.54 | 2.44 |
| YaRN | 4.15 | 2.42 |
| Dimension $\lambda_i$ + $\hat n$ search | **3.22** | **2.36** |

좋은 scaling은 fine-tuning 이전 initialization을 크게 개선하고, fine-tuning 이후에도 작은 이득을 남긴다.

## Search problem의 정식화

Target context $L'$ 이상의 긴 document 집합을 $X$라 하자. 각 candidate RoPE로 LLM의 next-token loss를 평가한다.

```math
\arg\min_{\{\lambda_i\},\hat n}
\mathcal{L}
\left(
\operatorname{LLM}(\operatorname{RoPE}_{\lambda,\hat n},X)
\right).
```

### Search space

| 변수 | 범위 |
| --- | --- |
| $\lambda_i$ | 1.0부터 $1.25s$까지, step 0.01 |
| $\hat n$ | $\{0,1,2,4,8,12,16,20,24,28,32,64,128,256\}$ |

$\lambda_i=1$은 direct extrapolation이고 $\lambda_i=s$는 PI와 같은 압축이다. $1.25s$까지 허용해 PI보다 더 강한 interpolation도 후보에 넣는다.

Head dimension 128에서 64개 RoPE pair를 가정하면 $s=4$만 해도 단순 조합 수가 약 $4\times10^{167}$에 달한다. Exhaustive search는 불가능하다.

## Evolutionary search

### 초기 population

Population $P$를 완전 random으로 만들지 않는다. PI, NTK, YaRN의 scale pattern을 seed로 넣고 나머지는 이들을 mutation해 생성한다. 알려진 좋은 영역에서 탐색을 시작하는 warm start다.

### 반복 과정

```text
1. 각 candidate의 validation perplexity 계산
2. 낮은 PPL의 top-k를 parent로 선택
3. parent를 mutation
4. parent끼리 crossover
5. lambda_i <= lambda_i+1 제약을 만족하는 candidate만 유지
6. top-k와 새 candidate로 다음 population 구성
7. T iterations 반복
```

논문의 Algorithm 1을 요약하면 다음과 같다.

```python
population = initialize_with_PI_NTK_YaRN(P)
topk = []

for _ in range(T):
    scores = evaluate_perplexity(model, population, samples)
    topk = update_topk(topk, population, scores)

    mutated = mutate(topk, N1, monotone=True)
    crossed = crossover(topk, N2, monotone=True)
    population = topk + mutated + crossed

return lowest_perplexity_candidate(topk)
```

### Search 설정

256k 이하에서는 다음을 사용한다.

- population $P=64$
- mutation $N_1=16$
- crossover $N_2=16$
- mutation probability $p=0.3$
- iterations $T=40$
- 매 iteration top-32를 parent로 사용
- PG19 validation의 target length 이상 문서 5개로 PPL 평가

512k 이상에서는 population/mutation/crossover를 절반으로 줄이고 Books3 validation 문서 3개를 사용한다. 극단적 길이의 한 번 PPL 평가가 매우 비싸기 때문이다.

## Fine-tuning 없는 8배 확장

Dimension과 initial position non-uniformity를 search하면 LLaMA2-7B의 4k context를 32k로 fine-tuning 없이 확장할 수 있었다. PI, NTK, YaRN은 이 설정에서 약 2배 이후 PPL이 급격히 악화됐지만 LongRoPE search는 8배까지 안정적이었다.

이 관찰이 progressive extension의 핵심 연결 고리다. 한 번 256k까지 학습한 모델을 다시 최대 8배 확장하면 2048k에 도달할 수 있다.

## Progressive extension to 2048k

### 왜 4k에서 2048k로 바로 fine-tuning하지 않는가

512배 scale은 초기 loss가 매우 크고, 2M-token training sample과 attention compute가 필요하다. Search로 좋은 initialization을 찾아도 직접 fine-tuning은 비현실적이다.

LongRoPE는 다음 세 단계로 나눈다.

### 1단계: 128k와 256k scale search

Base LLaMA2-7B에서 target 128k($32\times$)와 256k($64\times$)의 $\lambda_i,\hat n$을 각각 탐색한다.

### 2단계: 256k까지 progressive fine-tuning

```text
4k base
  -> searched 128k RoPE로 400 steps fine-tuning
  -> searched 256k RoPE로 교체
  -> 128k checkpoint에서 추가 600 steps fine-tuning
  -> 256k model
```

총 fine-tuning은 1000 steps다. 256k를 base model에서 바로 학습하는 것보다 128k checkpoint를 거치는 편이 loss와 PPL이 훨씬 안정적이었다.

### 3단계: 256k에서 2048k로 두 번째 search

Fine-tuned 256k 모델을 새 base로 보고 $8\times$ extension factor를 탐색한다.

```text
256k fine-tuned model
  -> second LongRoPE search
  -> 2048k model
  -> no additional 2048k fine-tuning
```

이 단계는 2M-token training data를 요구하지 않는다. 다만 search evaluation에는 2M-token 문서와 긴 inference가 필요하다.

### 128k checkpoint에서 바로 2048k로 가는 대안

논문은 128k에서 $16\times$ 확장한 모델도 만든다. LLaMA2에서는 256k에서 8배 확장한 쪽이 2048k PPL 7.08로, 128k에서 16배 확장한 7.80보다 좋았다. Secondary extension ratio가 작을수록 안정적이라는 예상과 맞는다.

## Short-context recovery

### 문제

2048k profile의 큰 $\lambda_i$를 4k prompt에도 쓰면 original position이 지나치게 압축된다. Original benchmark accuracy와 short PPL이 떨어진다.

### 해결

Extended model에서 4k/8k용 RoPE factor를 다시 evolutionary search한다. Search의 maximum $\lambda$를 줄여 덜 interpolation하도록 유도한다. Inference에서는 sequence length가 8k 이하일 때 short profile을 사용한다.

이 접근은 하나의 smooth scaling function이 아니라 **길이 구간별 RoPE profile**을 배포하는 방식에 가깝다.

### 효과

| 모델 | Recovery | Proof-pile 4k | Proof-pile 8k | Benchmark avg. |
| --- | ---: | ---: | ---: | ---: |
| LLaMA2 2048k, ft=128k | 없음 | 4.16 | 3.72 | 49.3 |
| LLaMA2 2048k, ft=128k | 적용 | **3.71** | **3.50** | **52.9** |
| LLaMA2 2048k, ft=256k | 없음 | 4.51 | 3.82 | 47.9 |
| LLaMA2 2048k, ft=256k | 적용 | **3.85** | **3.65** | **50.8** |

Recovery가 short-context 품질을 크게 회복하지만 original model과 완전히 같아지는 것은 아니다.

## Tensor shape와 구현

```text
lambda per RoPE pair       [d_h / 2]
start threshold n_hat      scalar
position ids               [N]
position-dependent scale   [N, d_h / 2]
angles                     [N, d_h / 2]
cos/sin                    [N, d_h]
Q, K                       [B, H, N, d_h]
```

### Angle 생성

```python
def longrope_angles(position_ids, inv_freq, lambdas, n_hat):
    # position_ids: [N]
    # inv_freq, lambdas: [d_h/2]
    pos = position_ids.float()[:, None]
    use_original = pos < n_hat

    scaled_freq = inv_freq[None, :] / lambdas[None, :]
    freq = torch.where(use_original, inv_freq[None, :], scaled_freq)
    return pos * freq
```

실제 implementation에서는 abrupt threshold에서 phase가 불연속이 될 수 있다. 논문의 식은 initial token과 이후 token에 다른 factor를 쓰므로 query-key pair가 threshold를 가로지를 때 주의해야 한다.

### 여러 profile

```text
short profile:  sequence <= 8k
long profile:   sequence > 8k and up to 2048k
```

Model weight는 같지만 RoPE metadata가 다르다. Checkpoint 배포 시 다음을 함께 저장해야 한다.

- Original context length
- Target context length
- $\lambda_i$ vector
- $\hat n$
- Length별 recovery profile
- RoPE base와 pair layout

### KV cache

한 generation 동안 profile을 고정하면 회전된 key를 cache할 수 있다. 그러나 sequence가 threshold를 넘어 profile을 바꾸면 과거 key도 새 factor에 맞춰 다시 회전해야 한다. 안전한 serving은 generation 시작 시 예상 target profile을 고정하거나 pre-RoPE key cache를 관리해야 한다.

## 실험 설정

### Base models

- LLaMA2-7B, original 4k
- Mistral-7B, original 8k

### LLaMA2 fine-tuning

- learning rate $2\times10^{-5}$, linear decay
- global batch 32
- RedPajama 128k chunks
- 128k: 8 A100 GPUs, 400 steps, 약 1주
- 256k: 16 A100 GPUs, 추가 600 steps, 약 2주

### Mistral fine-tuning

- constant learning rate $10^{-6}$
- global batch 64
- 128k와 256k 모두 16k training sequence
- 각 400 steps
- 4 A100 GPUs, 약 2일

Mistral은 target context보다 훨씬 짧은 16k data로 학습했다. Compute는 줄지만 million-token language modeling 품질에는 한계가 나타난다.

### 평가

- Proof-pile, PG19, Books3 perplexity
- Passkey retrieval
- ARC-Challenge, HellaSwag, MMLU, TruthfulQA
- PI, NTK, YaRN, LongLoRA, Code Llama 등 비교

## 주요 실험 결과

### 256k 이내 Proof-pile

| LLaMA2-7B 계열 | Context | 4k | 32k | 64k | 128k | 256k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 4k | 3.58 | $>10^4$ | $>10^4$ | $>10^4$ | $>10^4$ |
| LongLoRA | 100k | 3.83 | 2.68 | 2.44 | 9.89 | $>10^3$ |
| Code Llama | 100k | 3.95 | 2.74 | 2.55 | 2.71 | 49.33 |
| YaRN $s=32$ | 128k | 3.75 | 2.70 | 2.45 | 2.37 | 99.64 |
| LongRoPE ft=128k | 2048k | 3.71 | **2.60** | **2.36** | **2.26** | 1.88 |
| LongRoPE ft=256k | 2048k | 3.85 | 2.63 | 2.38 | **2.26** | **1.87** |

2048k profile이 더 짧은 256k 평가에서도 기존 장문 모델보다 안정적이다. Context가 늘수록 PPL이 전반적으로 감소해 추가 history를 활용한다.

### 2048k Books3

LLaMA2 LongRoPE 결과는 다음과 같다.

| 모델 | 8k | 64k | 128k | 256k | 512k | 1024k | 2048k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ft=128k | 6.55 | 6.18 | 6.17 | 6.17 | 6.36 | 6.83 | 7.80 |
| ft=256k | 6.81 | 6.27 | 6.21 | **6.17** | **6.17** | **6.35** | **7.08** |

2M까지 catastrophic failure는 없지만 512k 이후 PPL이 다시 증가한다. `2048k context window`는 2048k에서도 short-length와 같은 품질이라는 뜻이 아니다.

Mistral LongRoPE는 2048k에서 PPL 12.78 또는 13.71로 더 크게 악화됐다. 16k training data만 사용한 설정이 extreme extension에 충분하지 않았을 가능성을 논문도 지적한다.

### Progressive secondary search ablation

Fine-tuned 256k LLaMA2를 512k/1024k/2048k로 다시 확장한 Books3 PPL이다.

| Secondary 방법 | 512k | 1024k | 2048k |
| --- | ---: | ---: | ---: |
| PI | 6.60 | 8.73 | 20.17 |
| YaRN | 6.39 | 6.79 | 8.27 |
| LongRoPE search | **6.17** | **6.35** | **7.08** |

Progressive extension의 마지막 단계에서도 dimension-wise search가 의미 있는 이득을 준다.

### Passkey retrieval

- LLaMA2 LongRoPE ft=256k: 4k~2048k에서 90% 이상
- Mistral LongRoPE ft=128k: 1800k까지 100%, 2048k에서 60%

Million-token sea에서 단일 5자리 key를 회수했다는 점은 인상적이다. 하지만 passkey는 복잡한 reasoning보다 position access를 측정한다.

### Original benchmark

| 모델 | ARC-c | HellaSwag | MMLU | TruthfulQA |
| --- | ---: | ---: | ---: | ---: |
| LLaMA2-7B base | 53.1 | 78.6 | 46.6 | 39.0 |
| LongRoPE 2048k ft=128k | 52.9 | 76.5 | 43.4 | 38.8 |
| LongRoPE 2048k ft=256k | 51.0 | 75.3 | 39.6 | 37.3 |
| Mistral-7B base | 60.6 | 83.2 | 63.6 | 42.6 |
| LongRoPE Mistral ft=128k | 59.0 | 81.2 | 61.3 | 43.1 |

Short recovery 후에도 일부 성능 하락은 남는다. LLaMA2 256k fine-tuning은 128k보다 original MMLU 손실이 크다.

### 두 non-uniformity ablation

| 방법 | PG19 16k | PG19 32k | Books3 2048k |
| --- | ---: | ---: | ---: |
| PI | 14.88 | 136.30 | 20.17 |
| Dimension search | 7.28 | 13.00 | **7.08** |
| Dimension + initial token search | **7.22** | **11.51** | **7.08** |

Dimension non-uniformity가 대부분의 이득을 만든다. Initial token threshold는 16k/32k에서 추가 개선하지만 2048k에서는 차이가 없다. 극단적 길이에서는 몇 initial token을 보존하는 효과가 상대적으로 작아진다는 해석이다.

## Search와 fine-tuning 비용

### Search cost

- 256k 이하: single A100으로 최대 약 3일
- 512k: 2 A100
- 1024k: 4 A100
- 2048k: 8 A100, 전체 약 5일 이내
- 2048k candidate 한 번의 PPL 평가: 약 50분

이는 fine-tuning보다 저렴할 수 있지만 무료는 아니다. Target length와 model마다 다시 search하면 비용이 반복된다.

### Fine-tuning cost

- LLaMA2 128k: 8 A100 × 약 1주
- LLaMA2 256k: 16 A100 × 약 2주
- 총 1000 steps라도 각 step이 매우 긴 sequence라 큰 비용이다.

`only 1k fine-tuning steps`는 pretraining과 비교하면 작지만, 일반 연구 환경에서 가벼운 작업이라는 뜻은 아니다.

## 계산 복잡도와 시스템 현실

LongRoPE의 position 계산은 다음 정도다.

```math
O(Nd)
```

하지만 dense attention은 여전히 다음과 같다.

```math
O(N^2d).
```

KV cache는 token 수에 선형 증가한다.

```math
O(N\times n_{layers}\times n_{kvheads}\times d_h).
```

2M context에서 naive dense attention을 일반 GPU에 그대로 실행하기는 매우 어렵다. 논문은 FlashAttention 2와 internal distributed platform CUBE를 사용해 512k 이상의 serving 비용을 줄였다.

LongRoPE가 해결한 것은 **positional extrapolation의 안정성**이다. 실제 product에서 2M context를 효율적으로 쓰려면 다음 기술이 함께 필요하다.

- Ring/sequence parallel attention
- Sparse or block attention
- Retrieval과 context selection
- GQA/MQA
- KV cache quantization/offloading
- Chunked prefill과 paged memory

## 방법 비교

| 방법 | Scale 단위 | Token-position 분기 | 확장 전략 | Short recovery |
| --- | --- | --- | --- | --- |
| PI | 모든 dimension 동일 | 없음 | 한 번에 fine-tune | 없음 |
| YaRN | frequency 3구간/ramp | 없음 | 64k → 128k transfer | temperature 보정 |
| LongRoPE | RoPE pair별 탐색 | 첫 $\hat n$ token | 128k → 256k → 2048k | 별도 4k/8k profile |

LongRoPE는 더 좋은 closed-form formula를 제안했다기보다 **scale selection을 optimization problem으로 바꿨다.**

## 장점과 핵심 기여

### 1. 두 종류의 non-uniformity를 실험적으로 분리했다

Dimension별 scale과 initial token threshold의 효과를 각각 검증했다.

### 2. Heuristic scaling을 data-driven search로 일반화했다

PI/NTK/YaRN을 seed로 활용하면서 model/target-specific factor를 찾는다.

### 3. Progressive extension으로 million-token training을 피했다

최대 256k까지만 fine-tuning하고 2M은 두 번째 search로 도달한다.

### 4. Short-context 성능을 별도 objective로 다뤘다

긴 context PPL만 최적화하면 original capability가 손상된다는 문제를 recovery profile로 해결한다.

### 5. 여러 평가 축을 제시했다

PPL, passkey, original benchmark와 ablation을 함께 보고했다.

### 6. Architecture를 유지한다

Transformer weight shape를 바꾸지 않고 RoPE metadata만 수정하므로 기존 optimization을 재사용할 수 있다.

## 한계와 비판적 관점

### 1. Search가 비싸고 checkpoint별로 종속적이다

Model, RoPE base, target length가 바뀌면 $\lambda_i$를 다시 찾아야 할 수 있다. Closed-form rule보다 배포와 재현이 복잡하다.

### 2. 매우 적은 validation sample에 과적합될 수 있다

256k search는 PG19 5개, 극장문 search는 Books3 3개 sample의 PPL을 objective로 사용한다. Search factor가 특정 문서/domain에 맞춰질 가능성이 있다.

### 3. Evolutionary search는 global optimum을 보장하지 않는다

Monotonic constraint와 제한된 iterations는 비용을 줄이지만 더 좋은 non-monotone solution을 배제할 수 있다.

### 4. Initial-token threshold는 discontinuity를 만든다

$n<\hat n$과 $n\ge\hat n$에서 다른 frequency를 사용한다. Boundary를 가로지르는 relative position geometry가 매끄럽지 않다.

### 5. 2M에서 품질이 완전히 유지되지는 않는다

LLaMA2 Books3 PPL은 256k의 6.17에서 2048k의 7.08로, Mistral은 6.43 부근에서 13.71까지 악화된다. Context를 “받을 수 있음”과 같은 품질로 “활용함”은 다르다.

### 6. Passkey가 장거리 reasoning을 대표하지 않는다

단일 숫자 retrieval은 million-token context의 선택적 접근을 보이지만 여러 문서의 evidence synthesis나 distractor-robust reasoning을 증명하지 않는다.

### 7. 실용적 attention 비용이 매우 크다

논문의 internal platform 없이는 동일 길이 evaluation 자체가 어려울 수 있다. Position solution과 system solution을 분리해 평가해야 한다.

### 8. Multiple profile은 cache와 serving을 복잡하게 만든다

Short/long profile 전환 시 past key의 rotation 일관성을 보장해야 한다. 단순 config 변경만으로 안전하지 않을 수 있다.

### 9. 2M 자연 데이터의 품질과 희소성

아주 긴 Books3 sample은 book concatenation, tokenization, document boundary에 민감하다. 실제 응용의 long conversation, code graph, multimodal stream과 distribution이 다르다.

## 자주 헷갈리는 지점

### 2048k는 정확히 몇 token인가

$2048\times1024=2{,}097{,}152$ token이다. 보통 “약 2 million”으로 표현한다.

### LongRoPE는 2M token으로 fine-tuning했는가

아니다. LLaMA2는 최대 256k training length에서 총 1000 steps fine-tuning하고, 256k checkpoint를 두 번째 search로 2048k까지 확장한다.

### $\lambda_i$는 학습 parameter인가

Gradient로 학습하는 model weight가 아니라 evolutionary search로 선택한 RoPE configuration이다.

### 모든 layer/head에 다른 $\lambda_i$를 쓰는가

논문은 RoPE dimension별 factor를 탐색하며, 기본 설명에서는 이를 layer/head별 별도 parameter로 확장하지 않는다.

### Initial token을 interpolation하지 않으면 relative property가 완전히 유지되는가

같은 regime 안의 두 token은 일관되지만 threshold 양쪽의 token은 서로 다른 scaling을 사용한다. Global하게 단일 $m-n$ 함수만으로 표현되는 순수 translation invariance는 약해질 수 있다.

### LongRoPE가 attention을 효율화하는가

아니다. RoPE angle을 바꾼다. Dense attention과 KV cache의 크기는 그대로다.

### 2M passkey 성공이 2M reasoning 성공을 의미하는가

아니다. Access와 reasoning은 다른 능력이다.

## 온디바이스 및 비전 관점

### 온디바이스 LLM

Dimension별 $\lambda_i$ vector 자체는 매우 작아 model size 부담이 없다. 그러나 2M KV cache는 온디바이스에서 사실상 불가능한 크기다. LongRoPE의 현실적인 edge 활용은 8k~32k 수준에서 short/long profile을 조절하거나, retrieval로 선택한 작은 context에 적용하는 쪽이다.

### Vision과 video

Resolution extrapolation에서는 row/column frequency별 scale을 search할 수 있다. Video에서는 temporal RoPE의 frame-frequency마다 다른 scale을 적용하고, 초기 frame을 anchor로 보존하는 아이디어도 가능하다.

하지만 language PPL을 objective로 한 $\lambda_i$는 vision에 그대로 쓸 수 없다. Image classification loss, detection AP, temporal retrieval 등 target task의 validation objective로 다시 탐색해야 한다.

## 후속 연구 방향

- Layer/head별 scale을 허용하되 search space를 효율적으로 줄이는 방법
- Smooth token-position transition으로 $\hat n$ discontinuity 완화
- Retrieval/reasoning-aware search objective
- Sparse attention과 jointly optimized RoPE scaling
- Dynamic profile 전환과 KV cache를 일관되게 처리하는 inference kernel
- Domain-robust scale을 찾기 위한 다중 corpus validation

## 개인 학습/연구 메모

LongRoPE는 다음 progression으로 기억하면 된다.

```text
PI:       one scale for every RoPE dimension
YaRN:     a few frequency regimes
LongRoPE: search a scale for every RoPE pair
          + optionally protect initial tokens
          + extend in stages
          + recover short-context profile
```

가장 중요한 통찰은 million-token context를 한 번의 거대한 fine-tuning 문제로 보지 않고, **좋은 positional initialization을 찾는 search와 감당 가능한 길이의 progressive adaptation**으로 분해한 것이다.

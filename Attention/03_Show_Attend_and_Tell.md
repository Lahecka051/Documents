# 03. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention

## 논문 정보

- 원본 파일: `03_Show_Attend_and_Tell.pdf`
- 제목: Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
- 저자: Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio
- 발표: ICML 2015
- 주제: spatial CNN feature에 soft/hard attention을 적용해 caption word마다 다른 image region을 보는 모델
- 핵심 키워드: visual attention, image captioning, soft attention, hard attention, REINFORCE, doubly stochastic regularization, CNN-LSTM

## 한눈에 보는 요약

이전 image captioning model은 CNN이 image 전체를 하나의 global vector로 압축하고, LSTM이 그 vector를 조건으로 sentence를 생성하는 경우가 많았다. 이 구조는 Bahdanau 이전 NMT의 fixed-length bottleneck과 닮았다.

이 논문은 convolutional feature map의 spatial structure를 유지한다.

```text
image
-> CNN feature map [14, 14, 512]
-> 196 spatial annotations a_1, ..., a_196
```

Caption word $y_t$를 생성할 때마다 이전 LSTM hidden state $h_{t-1}$로 각 image location의 score를 계산한다.

```math
\begin{aligned}
e_{t,i} &= f_{\mathrm{att}}(a_i,h_{t-1}),\\
\alpha_{t,:} &= \mathrm{softmax}(e_{t,:}).
\end{aligned}
```

이후 두 가지 attention을 비교한다.

```math
\begin{aligned}
\text{soft attention:}\quad
z_t &= \sum_i\alpha_{t,i}a_i,\\
\text{hard attention:}\quad
s_t &\sim \mathrm{Categorical}(\alpha_t),\\
z_t &= a_{s_t}.
\end{aligned}
```

Soft attention은 모든 location의 expected feature를 사용하므로 standard backpropagation으로 학습한다. Hard attention은 location 하나를 sampling하므로 discrete latent variable과 REINFORCE가 필요하다.

Caption이 진행되면서 attention map도 바뀐다. 예를 들어 `bird`, `water`, `stop sign` 같은 word를 생성할 때 해당 image region에 높은 weight가 나타난다. 이 논문은 attention이 language translation뿐 아니라 spatial vision-language alignment에도 적용될 수 있음을 보여 준 대표적인 초기 연구다.

## 연구 배경과 문제의식

### Global image vector의 병목

초기 neural image captioning은 보통 마지막 fully connected CNN feature를 image representation으로 사용했다.

```text
image -> one vector v -> LSTM -> caption
```

이 방식에서는 image의 모든 object, attribute, relation, background를 하나의 vector에 압축해야 한다. 또한 caption의 모든 word가 동일한 image representation을 받는다.

```text
word "dog"   sees v
word "floor" sees v
word "red"   sees v
```

하지만 서로 다른 word는 서로 다른 visual evidence를 필요로 한다.

- Noun은 object region이 중요하다.
- Adjective는 object의 local attribute가 중요하다.
- Preposition과 relation은 여러 region의 배치가 중요하다.
- Background word는 넓은 scene context가 중요하다.

논문의 핵심 질문은 다음과 같다.

```text
Caption을 생성하는 동안 model이 word마다 image의 다른 부분을 볼 수 있는가?
```

### Object detector 없이 latent alignment를 학습한다

당시 일부 captioning system은 object detector의 region proposal이나 visual concept detector를 사용했다. 이 논문은 별도 object label이나 region-word alignment supervision 없이 caption likelihood만으로 spatial attention을 학습한다.

Attention target은 반드시 object box일 필요도 없다. Texture, background, 관계 영역처럼 명확한 object가 아닌 region도 볼 수 있다.

## 전체 모델 구조

### Encoder: convolutional annotations

Pre-trained Oxford VGGNet의 convolutional layer에서 spatial feature map을 추출한다. 논문 설정은 다음과 같다.

```text
feature map: 14 x 14 x 512
L = 196 locations
D = 512 feature dimension
```

Spatial axis를 flatten해 annotation set을 만든다.

```math
\begin{aligned}
A &= \{a_1,a_2,\ldots,a_L\},\\
a_i &\in \mathbb{R}^D.
\end{aligned}
```

Fully connected feature와 달리 각 $a_i$는 image의 특정 receptive field에 대응한다. 이 대응 관계가 attention map을 다시 image 위에 시각화할 수 있게 한다.

### Decoder: LSTM

Caption은 vocabulary size $K$의 one-hot word sequence다.

```math
\begin{aligned}
y &= \{y_1,\ldots,y_C\},\\
y_t &\in\mathbb{R}^K.
\end{aligned}
```

LSTM은 세 입력을 사용한다.

```text
previous word embedding E y_(t-1)
previous hidden state h_(t-1)
current visual context z_t_hat
```

Gate 계산을 단순화해 쓰면 다음과 같다.

```math
\begin{aligned}
[i_t,f_t,o_t,g_t]
&=[\mathrm{sigmoid},\mathrm{sigmoid},\mathrm{sigmoid},\tanh]
\!\left(W[E y_{t-1};h_{t-1};\hat z_t]\right),\\
c_t &= f_t\odot c_{t-1}+i_t\odot g_t,\\
h_t &= o_t\odot\tanh(c_t).
\end{aligned}
```

Visual context가 매 step LSTM gate 계산에 직접 들어간다.

### Initial state

LSTM의 초기 memory와 hidden state는 annotation의 평균에서 별도 MLP로 예측한다.

```math
\begin{aligned}
a_{\mathrm{mean}} &= \frac{1}{L}\sum_i a_i,\\
c_0 &= f_{\mathrm{init},c}(a_{\mathrm{mean}}),\\
h_0 &= f_{\mathrm{init},h}(a_{\mathrm{mean}}).
\end{aligned}
```

즉 attention 이전에도 image 전체의 global summary가 decoder initialization에 사용된다.

### Output word probability

다음 word는 이전 word embedding, current LSTM state, visual context를 결합한 deep output layer에서 예측한다.

```math
p(y_t\mid y_{<t},A)
\propto
\exp\!\left(L_o(Ey_{t-1}+L_hh_t+L_z\hat z_t)\right)
```

Visual context는 state update뿐 아니라 output prediction에도 직접 연결된다.

## Attention score

### Location별 energy

각 location $i$의 annotation과 previous hidden state를 MLP $f_{\mathrm{att}}$로 scoring한다.

```math
e_{t,i}=f_{\mathrm{att}}(a_i,h_{t-1})
```

Bahdanau additive attention 형태로 쓰면 다음과 같다.

```math
e_{t,i}=v_a^\top\tanh\!\left(W_aa_i+W_hh_{t-1}+b\right)
```

Previous hidden state는 지금까지 생성한 caption history를 요약한다. 따라서 다음에 어디를 볼지는 이미 생성한 word sequence에 따라 달라진다.

### Spatial distribution

Location 축으로 softmax한다.

```math
\alpha_{t,i}
=\frac{\exp(e_{t,i})}{\sum_k\exp(e_{t,k})}
```

각 time step에서 다음이 성립한다.

```math
\begin{aligned}
\alpha_{t,i}&\ge 0,\\
\sum_i\alpha_{t,i}&=1.
\end{aligned}
```

`Alpha` 전체를 모으면 caption time과 image location 사이 alignment matrix가 된다.

```text
Alpha: [caption_length, 196]
```

## Deterministic soft attention

### Expected context

Soft attention은 annotation의 weighted average를 사용한다.

```math
\hat z_t=\sum_i\alpha_{t,i}a_i
```

이는 categorical location variable 아래 expected annotation이다.

```math
\hat z_t=\mathbb{E}_{s_t\sim\alpha_t}[a_{s_t}]
```

모든 연산이 smooth하고 differentiable하므로 caption negative log-likelihood를 standard backpropagation으로 최적화한다.

### 장점

- Sampling이 없어 gradient variance가 낮다.
- 한 word가 여러 region을 동시에 참고할 수 있다.
- GPU에서 weighted sum으로 효율적으로 구현할 수 있다.
- Attention map을 probability-like heatmap으로 시각화할 수 있다.

### 한계

- 서로 다른 object feature가 평균되어 흐려질 수 있다.
- 실제로 location 하나를 선택하는 hard glimpse보다 계산량이 크다.
- Expected feature를 nonlinear LSTM에 넣는 것이 모든 hard trajectory의 exact marginal과 같지는 않다.

논문은 first-order Taylor approximation과 normalized weighted geometric mean을 통해 soft model이 hard location marginal을 근사한다고 설명한다. 그러나 이는 exact equality가 아니라 nonlinear network에 대한 근사적 해석이다.

## Stochastic hard attention

### Discrete location variable

Hard attention은 one-hot location variable $s_t$를 sampling한다.

```math
\begin{aligned}
s_t &\sim \mathrm{Multinoulli}(\alpha_t),\\
s_{t,i} &\in \{0,1\},\\
\sum_i s_{t,i} &= 1.
\end{aligned}
```

Context는 선택된 annotation 하나다.

```math
\hat z_t
=\sum_i s_{t,i}a_i
=a_{\text{selected location}}
```

Inference 한 step에서 image region 하나만 읽는다는 의미에서 hard하다.

### Variational lower bound

Location trajectory $s$를 latent variable로 보면 caption marginal likelihood는 모든 trajectory를 합해야 한다. 논문은 다음 lower bound를 최적화한다.

```math
\mathcal{L}_s
=\sum_s p(s\mid A)\log p(y\mid s,A)
\le \log p(y\mid A)
```

가능한 trajectory 수가 $L^C$이므로 정확한 합은 불가능하다. Monte Carlo sample로 gradient를 근사한다.

### REINFORCE gradient

Gradient에는 두 부분이 있다.

```text
1. sampled path에서 caption model의 ordinary gradient
2. log p(y | s, A)를 reward로 사용하는 attention policy gradient
```

개념적으로 다음 형태다.

```math
\nabla\mathcal{L}
\approx
\nabla\log p(y\mid s_{\mathrm{sampled}},A)
+(\mathrm{reward}-\mathrm{baseline})
\nabla\log p(s_{\mathrm{sampled}}\mid A)
```

논문은 variance reduction을 위해 다음을 사용한다.

- Moving-average reward baseline
- Attention distribution entropy regularization
- 확률 0.5로 sampled location 대신 expected alpha 사용

이 학습 규칙은 REINFORCE와 동등한 형태다.

### Hard attention의 trade-off

```text
advantage:
selective computation and discrete glimpse

cost:
high-variance gradient and more difficult optimization
```

Hard attention은 이름과 달리 학습 과정에서 완전히 deterministic한 one-hot decision을 직접 미분하는 것이 아니다. Sampling과 policy gradient를 통해 expected reward를 최적화한다.

## Soft와 hard attention 비교

| 항목 | Soft attention | Hard attention |
| --- | --- | --- |
| Context | 모든 annotation weighted sum | sampled annotation 하나 |
| Forward | deterministic | stochastic during training |
| Gradient | standard backpropagation | REINFORCE/Monte Carlo |
| Variance | 낮음 | 높음 |
| 한 step 계산 | 모든 location 사용 | 선택 location 사용 가능 |
| Multi-region 결합 | 자연스러움 | 한 sample에서는 불가능 |
| 해석 | continuous heatmap | discrete glimpse trajectory |

두 방식은 동일한 attention score network와 spatial annotations를 공유하고 context 함수 $\phi$만 다르다.

```math
\begin{aligned}
\phi_{\mathrm{soft}}(A,\alpha) &= \sum_i\alpha_i a_i,\\
\phi_{\mathrm{hard}}(A,\alpha) &= a_s,\qquad
s\sim\mathrm{Categorical}(\alpha).
\end{aligned}
```

## Doubly stochastic regularization

### Row normalization

Softmax 때문에 각 caption step에서는 location weight 합이 1이다.

```math
\sum_i\alpha_{t,i}=1
```

### Column coverage도 장려한다

논문은 caption 전체에 걸쳐 각 image location이 대략 한 번씩 attention을 받도록 regularization한다.

```math
\sum_t\alpha_{t,i}\approx 1
```

Loss는 다음과 같다.

```math
\mathcal{L}_d
=-\log p(y\mid A)
+\lambda\sum_i\left(1-\sum_t\alpha_{t,i}\right)^2
```

행과 열 양쪽 합을 제어한다는 의미에서 doubly stochastic attention이라 부른다.

다만 엄밀히 말하면 column sum이 정확히 1이 되도록 projection하는 것이 아니라 penalty로 장려할 뿐이다. Caption length와 location 수가 다르므로 모든 row와 column 합을 동시에 정확히 1로 만들 수도 없다.

### Visual sentinel과 비슷한 gate

Soft model은 previous hidden state에서 scalar gate $\beta_t$도 예측한다.

```math
\begin{aligned}
\beta_t &= \mathrm{sigmoid}(f_\beta(h_{t-1})),\\
\hat z_t &= \beta_t\sum_i\alpha_{t,i}a_i.
\end{aligned}
```

모든 word가 동일한 정도의 visual evidence를 필요로 하지 않는다는 점을 반영한다. Function word에서는 image context를 줄이고 object word에서는 높일 수 있다.

## Tensor shape와 계산

논문 설정을 batch 형태로 쓰면 다음과 같다.

```text
A: [B, L=196, D=512]
h_(t-1): [B, H]
```

Additive attention hidden dimension을 $d_a$라 하면:

```math
\begin{aligned}
A_{\mathrm{proj}}&=AW_a,
&\mathrm{shape}&=[B,196,d_a],\\
h_{\mathrm{proj}}&=h_{t-1}W_h,
&\mathrm{shape}&=[B,d_a],\\
E_{\mathrm{hidden}}
&=\tanh(A_{\mathrm{proj}}+h_{\mathrm{proj}}[:,\mathrm{None},:]),
&\mathrm{shape}&=[B,196,d_a],\\
e_t&=E_{\mathrm{hidden}}v_a,
&\mathrm{shape}&=[B,196],\\
\alpha_t&=\mathrm{softmax}(e_t,\mathrm{dim}=\mathrm{location}),
&\mathrm{shape}&=[B,196].
\end{aligned}
```

Soft context:

```math
\begin{aligned}
z_t&=\alpha_t[:,\mathrm{None},:]\,A,\\
\mathrm{shape}(z_t)&=[B,512].
\end{aligned}
```

Hard context:

```math
\begin{aligned}
\mathrm{index}_t&\sim\mathrm{Categorical}(\alpha_t),\\
z_t&=\mathrm{gather}(A,\mathrm{index}_t),\\
\mathrm{shape}(z_t)&=[B,512].
\end{aligned}
```

전체 caption의 attention map은 다음 shape다.

```text
Alpha: [B, C, 196]
-> reshape each row to [14,14]
-> upsample to image resolution
```

## 전체 알고리즘 흐름

### Soft attention

```text
1. CNN extracts A = [196,512]
2. mean(A) initializes LSTM h_0 and c_0
3. for caption step t:
   a. score every spatial annotation with h_(t-1)
   b. softmax -> alpha_t
   c. z_t = weighted sum of annotations
   d. update LSTM with previous word and z_t
   e. predict next word
4. minimize caption NLL + coverage penalty
```

### Hard attention

```text
1. CNN extracts spatial annotations
2. for caption step t:
   a. compute alpha_t
   b. sample one location s_t
   c. use selected annotation as z_t
   d. update LSTM and predict word
3. optimize variational lower bound with REINFORCE
4. use baseline and entropy terms to reduce variance
```

## Attention visualization

<p align="center"><img src="https://github.com/user-attachments/assets/36dfdb3b-a7e8-41b7-8c11-c1d734b528ab" alt="Caption word별 soft hard visual attention과 object alignment" width="760"></p>
<p align="center"><sub>Figures 2–3 — 단어 생성에 따라 이동하는 visual attention과 객체 정렬</sub></p>

Caption word가 바뀔 때 bright region도 이동한다.

- `bird`에서는 bird body 주변
- `water`에서는 image 아래 수면
- `dog`에서는 curtain 아래 dog
- `stop sign`에서는 표지판
- `trees`에서는 giraffe 주변 background

Hard attention은 한 location의 bright spot으로 나타나고 soft attention은 여러 overlapping receptive field가 섞인 부드러운 heatmap으로 나타난다.

Visual attention은 오류 분석에도 도움이 된다. Model이 `clock`을 잘못 생성했을 때 attention이 실제로 손 주변의 비슷한 pattern을 보고 있었는지, language model prior만으로 word를 낸 것인지 단서를 제공한다.

그러나 heatmap은 receptive field가 크게 겹치는 14x14 feature를 upsample한 것이다. Pixel-level segmentation이나 인과 설명과 동일하게 해석하면 안 된다.

## 실험 설정

### 데이터

- Flickr8k: 8,000 image
- Flickr30k: 30,000 image
- MS COCO: 82,783 image
- Image당 reference caption: Flickr는 5개, COCO도 비교 일관성을 위해 5개만 사용
- Vocabulary size: 10,000

### Visual encoder와 학습

- Oxford VGGNet pre-trained on ImageNet
- Fourth convolutional layer before max pooling
- `14 x 14 x 512` annotations
- CNN은 fine-tuning하지 않음
- Flickr8k: RMSProp
- Flickr30k/COCO: Adam
- Minibatch 64, caption length별로 bucket
- Dropout과 BLEU-based early stopping
- COCO soft attention: Titan Black 한 장에서 3일 미만

Length bucket은 batch에서 가장 긴 caption에 맞춘 padding 계산 낭비를 줄인다.

## 실험 결과

<p align="center"><img src="https://github.com/user-attachments/assets/1edd5288-487a-42be-8d03-858413bfc431" alt="Flickr8k Flickr30k COCO captioning 결과" width="820"></p>
<p align="center"><sub>Table 1 — Flickr8k·Flickr30k·COCO captioning 성능 비교</sub></p>

### Flickr8k

| 모델 | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Soft attention | 67.0 | 44.8 | 29.9 | 19.5 | 18.93 |
| Hard attention | 67.0 | 45.7 | 31.4 | 21.3 | 20.30 |

Hard attention이 BLEU-2 이후와 METEOR에서 높다.

### Flickr30k

| 모델 | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Soft attention | 66.7 | 43.4 | 28.8 | 19.1 | 18.49 |
| Hard attention | 66.9 | 43.9 | 29.6 | 19.9 | 18.46 |

Hard attention이 BLEU에서는 조금 높지만 METEOR는 soft가 `0.03` 높다. 차이가 작아 한 방식의 일관된 우위라고 보기 어렵다.

### MS COCO

| 모델 | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Log Bilinear | 70.8 | 48.9 | 34.4 | 24.3 | 20.03 |
| Soft attention | 70.7 | 49.2 | 34.4 | 24.3 | 23.90 |
| Hard attention | 71.8 | 50.4 | 35.7 | 25.0 | 23.04 |

Hard attention이 BLEU 전반에서 높고 soft attention이 METEOR에서 높다. Soft attention의 METEOR 향상은 doubly stochastic regularization과 lower convolutional feature의 영향도 함께 포함한다.

### 결과를 읽을 때 주의할 점

논문도 당시 비교의 어려움을 명시한다.

- CNN backbone이 AlexNet, GoogLeNet, VGG로 다르다.
- Single model과 ensemble 결과가 섞여 있다.
- Flickr30k와 COCO split이 완전히 표준화되지 않았다.
- BLEU 구현과 tokenization 차이가 있다.
- 이 논문 결과는 single model이지만 일부 baseline은 ensemble이다.

따라서 attention의 정확한 기여는 같은 encoder와 decoder에서 soft/hard variant를 비교한 부분이 가장 신뢰할 만하다.

## 장점과 핵심 기여

### 1. Spatial feature map을 differentiable memory로 만들었다

CNN의 각 location feature를 annotation으로 보고 language decoder가 필요할 때 조회한다.

### 2. Visual-language latent alignment를 end-to-end 학습했다

Region label이나 word-region alignment 없이 caption loss만으로 attention이 학습된다.

### 3. Soft와 hard attention을 한 framework에서 비교했다

Expected context와 sampled glimpse라는 두 선택을 동일한 score model 위에서 이론적으로 정리했다.

### 4. Hard attention에 policy gradient를 적용했다

Discrete visual glimpse 선택을 variational lower bound와 REINFORCE로 학습하는 구체적 방법을 제시했다.

### 5. Attention visualization을 model diagnosis에 사용했다

단순 성능 metric 외에 word 생성과 visual region 사이의 관계를 관찰하는 방법을 보여 줬다.

### 6. NMT attention을 vision-language로 확장했다

Sequence position alignment를 spatial region alignment로 일반화해 attention의 범용성을 입증했다.

## 한계와 비판적 관점

### 1. Spatial resolution이 낮다

14x14 grid는 작은 object와 정밀한 boundary를 표현하기 어렵다. Upsampled heatmap은 원래 feature의 overlapping receptive field를 부드럽게 확대한 것이다.

### 2. CNN을 fine-tuning하지 않았다

ImageNet classification용 feature가 captioning에 최적화되지 않는다. Attention decoder의 이득과 visual encoder의 한계를 분리해야 한다.

### 3. Soft attention은 feature averaging을 일으킨다

멀리 떨어진 두 region에 weight가 있으면 서로 다른 visual evidence가 한 vector에 섞인다. Multi-head 또는 object-query 구조가 후속 대안이 된다.

### 4. Hard attention gradient variance가 높다

Moving baseline, entropy, soft substitution을 사용해도 REINFORCE optimization은 불안정하고 sample 효율이 낮다.

### 5. Doubly stochastic prior가 모든 caption에 적절하지 않다

Image의 모든 region을 한 번씩 볼 필요는 없다. Background가 넓거나 salient object가 하나인 image에서는 균등 coverage가 불필요할 수 있다.

### 6. Attention map과 설명은 동일하지 않다

높은 weight는 해당 location이 weighted sum에 크게 기여했음을 보여 주지만, 그 region이 word 생성의 유일한 원인이라는 반사실적 증거는 아니다.

### 7. Autoregressive language prior가 강하다

Caption dataset의 빈번한 문장 pattern 때문에 image를 충분히 보지 않고도 그럴듯한 caption을 생성할 수 있다. Attention visualization만으로 grounding이 완전하다고 결론 내릴 수 없다.

## Bahdanau attention과의 대응

| NMT | Image captioning |
| --- | --- |
| source annotations $h_j$ | spatial CNN annotations $a_i$ |
| decoder state $s_{t-1}$ | LSTM hidden state $h_{t-1}$ |
| source word position | image grid location |
| alignment weight $\alpha_{t,j}$ | spatial attention $\alpha_{t,i}$ |
| context $\sum\alpha h$ | visual context $\sum\alpha a$ |
| target translation word | caption word |

핵심 연산은 같다.

```text
query-conditioned scoring
-> softmax over memory locations
-> weighted retrieval
```

차이는 memory가 1D source sequence가 아니라 2D image feature grid라는 점이다.

## 후속 연구와의 연결

이 논문의 spatial attention은 다음 흐름으로 이어진다.

- Bottom-up attention: grid 대신 object region feature를 memory로 사용
- Visual sentinel/adaptive attention: image를 볼지 language state만 쓸지 더 명시적으로 결정
- Transformer captioning: LSTM 대신 self-attention/cross-attention을 사용
- ViT: image patch 자체를 token sequence로 만들어 global self-attention 수행
- DETR: learned object query가 image feature memory를 cross-attention으로 검색
- Vision-language model: text token과 image patch/region 사이 대규모 cross-attention

Show, Attend and Tell의 `caption query -> spatial memory` 구조는 DETR의 `object query -> image memory`와 목적은 다르지만 memory retrieval 관점에서 연결된다.

## 구현 체크리스트

```text
1. CNN feature가 [B,14,14,512]에서 [B,196,512]로 올바르게 flatten되는가?
2. Flatten order와 heatmap reshape order가 일치하는가?
3. Attention query가 h_t가 아니라 논문대로 h_(t-1)인가?
4. Softmax 축이 196 location 축인가?
5. Soft attention alpha 합이 step마다 1인가?
6. Initial h_0/c_0가 annotation mean의 서로 다른 MLP에서 계산되는가?
7. beta gate가 context 전체에 scalar로 적용되는가?
8. Coverage penalty에서 time과 location 축을 뒤집지 않았는가?
9. Hard attention log-probability가 REINFORCE loss에 포함되는가?
10. Reward baseline이 gradient를 받지 않도록 처리하는가?
11. Entropy regularization 부호가 exploration을 장려하는 방향인가?
12. Attention visualization이 receptive field center와 맞는가?
```

## 개인 학습/연구 메모

이 논문은 Bahdanau attention의 추상 구조가 modality에 독립적임을 보여 준다.

```text
memory item = source word annotation
or
memory item = image region annotation
```

기억해야 할 식은 두 개다.

```math
\begin{aligned}
\text{soft:}\quad
z_t &= \sum_i\alpha_{t,i}a_i,\\
\text{hard:}\quad
s_t &\sim \mathrm{Categorical}(\alpha_t),\\
z_t &= a_{s_t}.
\end{aligned}
```

두 식의 차이가 deterministic backpropagation과 stochastic policy gradient라는 전혀 다른 학습 문제를 만든다.

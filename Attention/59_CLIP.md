# 59. Learning Transferable Visual Models From Natural Language Supervision

## 논문 정보

- 저자: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever
- 소속: OpenAI
- 공개: arXiv:2103.00020v1, 2021년 2월 26일
- 원본 파일: `59_CLIP.pdf`
- 논문 링크: https://arxiv.org/abs/2103.00020
- 공개 코드: https://github.com/OpenAI/CLIP

## 한눈에 보는 요약

CLIP은 이미지와 자연어를 각각 인코딩한 뒤 같은 임베딩 공간에서 정답 이미지-텍스트 쌍의 cosine similarity는 높이고, 같은 배치의 나머지 조합은 낮추는 대조 학습 모델이다. 핵심은 400M개의 웹 이미지-텍스트 쌍으로 사전학습한 표현을 고정한 채, 클래스 이름을 문장으로 바꾸어 새로운 분류기를 즉석에서 구성한다는 점이다.

논문의 가장 중요한 결론은 다음과 같다.

1. 고정된 one-hot 클래스가 아니라 자연어를 supervision으로 사용하면 하나의 모델이 다수의 시각 개념과 과제를 학습할 수 있다.
2. 배치가 $N$이면 $N$개의 정답 쌍과 $N^2-N$개의 배치 내 오답 쌍을 동시에 비교하는 symmetric contrastive loss가 효율적인 학습 신호를 만든다.
3. zero-shot 분류는 클래스별 prompt를 text encoder로 임베딩하여 classifier weight처럼 사용하는 과정이다. downstream gradient update가 필요 없다.
4. 최고 모델 ViT-L/14@336px는 ImageNet zero-shot top-1 76.2%를 기록해 원래의 supervised ResNet-50과 비슷한 수준에 도달한다.
5. CLIP의 장점은 범용성과 분포 변화에 대한 강건성이지, 세밀한 객체 수 세기, 작은 객체 검출, 모든 OCR, 진정한 out-of-distribution 일반화를 해결했다는 뜻은 아니다.
6. 온디바이스에서는 학습 시 거대한 $N\times N$ 행렬보다 추론 시 vision encoder가 주 병목이다. 고정 클래스의 text embedding을 캐시하면 text encoder 비용은 초기 1회로 상각할 수 있다.

## 문제 설정과 핵심 아이디어

기존 이미지 분류는 사람이 미리 정한 클래스 집합과 대규모 수동 라벨에 의존한다. 이 방식은 강력하지만 새 과제가 생길 때마다 라벨 수집과 분류기 재학습이 필요하다. 반면 웹의 이미지 주변에는 제목, 설명, alt text와 같은 자연어가 이미 존재한다. CLIP은 이 자연어를 예측 대상 전체로 생성하려 하지 않고, 어떤 텍스트가 어떤 이미지와 짝인지 식별하는 문제로 바꾼다.

논문은 500,000개의 text query로 웹을 검색하고 query당 최대 20,000개의 이미지-텍스트 쌍을 수집해 400M 쌍의 WIT 데이터셋을 구성한다. 이 데이터 규모는 결과 해석에서 매우 중요하다. CLIP의 성능은 objective만의 효과가 아니라 데이터의 크기, 다양성, 모델 규모, 학습 compute가 결합된 결과다.

### 생성식보다 대조식이 효율적인 이유

초기 실험에서 저자들은 이미지로부터 정확한 caption을 생성하는 transformer language model과 bag-of-words 예측을 비교했다. Figure 2에서 caption transformer는 같은 zero-shot ImageNet 정확도에 도달하는 데 bag-of-words 방식보다 약 3배 느렸다. 이후 bag-of-words 예측을 CLIP의 contrastive objective로 바꾸면 추가로 약 4배의 학습 효율 향상이 나타났다.

여기서 효율은 모바일 추론 latency가 아니라, 사전학습 중 처리한 이미지 수 대비 zero-shot 성능을 뜻한다. 논문이 주장하는 3배와 4배를 디바이스 속도 향상으로 해석하면 안 된다.

## 전체 아키텍처

CLIP은 두 개의 독립 encoder와 하나의 단순한 유사도 연산으로 구성된다.

```text
image I [B, 3, H, W]
  -> image encoder (modified ResNet 또는 ViT)
  -> image feature [B, D_i]
  -> linear projection + L2 normalization
  -> image embedding E_i [B, D_e]

text T [B, L]
  -> text Transformer
  -> [EOS] hidden state [B, D_t]
  -> linear projection + L2 normalization
  -> text embedding E_t [B, D_e]

E_i @ E_t^T / temperature
  -> pairwise logits [B, B]
```

두 modality 사이에는 cross-attention이나 fusion transformer가 없다. 학습 중 상호작용은 정규화된 두 벡터의 dot product뿐이다. 이 단순성 덕분에 image embedding과 text embedding을 독립적으로 계산하고 캐시할 수 있다.

### Image encoder

논문은 두 계열을 비교한다.

- Modified ResNet: ResNet-D 계열의 stem, strided convolution 앞의 average pooling을 이용한 anti-aliasing, 마지막 global average pooling 대신 multi-head attention pooling을 사용한다.
- Vision Transformer: ViT 구조를 따르되 patch embedding과 position embedding을 더한 뒤 transformer에 넣기 전에 layer normalization을 추가하고 initialization을 조정한다.

학습한 모델은 RN50, RN101, RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14이며, ViT-L/14는 마지막 한 epoch를 336 해상도로 추가 학습한 버전도 평가한다.

### Text encoder

기본 text encoder는 12-layer, width 512, 8-head Transformer이며 약 63M parameter다. 텍스트는 lower-cased BPE로 토큰화되고 논문 본문은 vocabulary 49,152, 최대 sequence length 76을 설명한다. Appendix의 공통 hyperparameter 표에는 vocabulary size 49,408이 기재되어 있으므로 재현 시 tokenizer special token을 포함한 정의 차이를 확인해야 한다.

입력은 `[SOS]`와 `[EOS]`로 감싸며, 마지막 layer의 `[EOS]` activation을 문장 표현으로 사용한다. 이 벡터에 layer normalization과 linear projection을 적용해 image embedding과 같은 차원으로 맞춘다. masked self-attention을 사용했기 때문에 향후 language modeling objective를 함께 붙일 여지도 남겼지만, 본 논문의 CLIP loss에는 생성 loss가 없다.

## Contrastive objective

배치 크기를 $N$, 정규화된 image embedding을 $\hat{I}\in\mathbb{R}^{N\times d}$, text embedding을 $\hat{T}\in\mathbb{R}^{N\times d}$라 하자. 학습 가능한 temperature를 $\tau$라 하면 logit은 다음과 같다.

$$
S_{ij}=\frac{\hat{I}_i^\top\hat{T}_j}{\tau}.
$$

$i$번째 이미지의 정답은 $i$번째 텍스트이고, $j\ne i$인 텍스트는 같은 배치에서 negative로 작동한다. image-to-text와 text-to-image 방향의 cross entropy를 평균한다.

$$
\mathcal{L}_{i2t}
=-\frac{1}{N}\sum_{i=1}^{N}
\log\frac{\exp(S_{ii})}{\sum_{j=1}^{N}\exp(S_{ij})},
$$

$$
\mathcal{L}_{t2i}
=-\frac{1}{N}\sum_{j=1}^{N}
\log\frac{\exp(S_{jj})}{\sum_{i=1}^{N}\exp(S_{ij})},
\qquad
\mathcal{L}=\frac{\mathcal{L}_{i2t}+\mathcal{L}_{t2i}}{2}.
$$

L2 normalization 후 dot product는 cosine similarity다. temperature가 작을수록 logit 차이가 증폭된다. 저자들은 temperature를 학습시키되 초기값을 0.07에 해당하게 설정하고, logit scale이 100을 넘지 않도록 clip했다.

### 논문의 핵심 pseudocode를 shape 중심으로 다시 쓰기

```python
 # images: [N, 3, H, W]
 # texts:  [N, L]
 # Wi: [Di, De], Wt: [Dt, De]

image_features = image_encoder(images)       # [N, Di]
text_features = text_encoder(texts)          # [N, Dt], EOS state

image_embed = l2_normalize(image_features @ Wi, dim=1)  # [N, De]
text_embed = l2_normalize(text_features @ Wt, dim=1)    # [N, De]

logits = exp(logit_scale) * image_embed @ text_embed.T  # [N, N]
labels = arange(N)

loss_i = cross_entropy(logits, labels)       # image -> text
loss_t = cross_entropy(logits.T, labels)     # text -> image
loss = 0.5 * (loss_i + loss_t)
```

중요한 구현 조건은 동일 이미지의 caption이 batch diagonal에 오도록 정렬하는 것이다. distributed training에서는 global batch의 negative를 공유하되, 각 장치가 필요한 logit shard만 계산하도록 해야 논문 수준의 batch를 감당할 수 있다.

## Tensor shape와 activation memory

### 학습 시 공통 shape

| Tensor | Shape | 역할 |
|---|---:|---|
| 입력 이미지 | $[N,3,H,W]$ | image encoder 입력 |
| 입력 토큰 | $[N,L]$ | text encoder 입력 |
| image feature | $[N,D_i]$ | projection 이전 특징 |
| text feature | $[N,D_t]$ | `[EOS]` 특징 |
| 두 embedding | 각각 $[N,D_e]$ | L2 normalized joint representation |
| similarity logits | $[N,N]$ | 모든 배치 내 image-text 조합 |
| target | $[N]$ | diagonal index $0,1,\ldots,N-1$ |

논문의 batch size는 32,768이다. $N^2=1,073,741,824$이므로 similarity matrix를 한 번에 물리적으로 materialize하면 payload만 다음과 같다.

- FP16/BF16: $1,073,741,824\times2$ bytes = 2.00 GiB
- FP32: $1,073,741,824\times4$ bytes = 4.00 GiB

이는 reviewer 계산이며 optimizer, gradient, softmax workspace를 제외한 logit 하나의 하한이다. 논문이 half-precision Adam statistics, gradient checkpointing, mixed precision, GPU 간 similarity sharding을 사용한 이유를 수치로 보여준다. 이 학습 메모리는 batch=1 모바일 추론 비용과는 별개다.

### ViT별 batch=1 token 계산

Appendix Table 20의 해상도와 width를 이용한 reviewer 계산이다. 표의 메모리는 한 지점의 token hidden state를 FP16으로 저장한 payload이며, 실제 peak memory에는 QKV, MLP 중간값, residual, kernel workspace가 추가된다.

| 모델 | 입력 | patch grid | CLS 포함 token $N_v$ | width | hidden payload |
|---|---:|---:|---:|---:|---:|
| ViT-B/32 | 224 | $7\times7$ | 50 | 768 | 약 75 KiB |
| ViT-B/16 | 224 | $14\times14$ | 197 | 768 | 약 295.5 KiB |
| ViT-L/14 | 224 | $16\times16$ | 257 | 1024 | 약 514 KiB |
| ViT-L/14@336 | 336 | $24\times24$ | 577 | 1024 | 약 1.13 MiB |

계산식은 $N_v\times D\times2$ bytes다. ViT-L/14의 attention score를 naive하게 저장하면 한 layer에서 head당 $257^2=66,049$개, 16 heads 전체 약 2.02 MiB다. 336 입력은 head당 $577^2=332,929$개, 16 heads 전체 약 10.16 MiB다. 해상도를 224에서 336으로 1.5배 높였지만 attention score 원소 수는 약 5.04배가 된다.

이 값 역시 논문 보고 peak memory가 아니라 shape에서 계산한 상한 성격의 중간값이다. fused attention은 score 전체를 저장하지 않아 peak를 줄일 수 있지만, 논문은 모바일 kernel이나 fused attention을 평가하지 않는다.

### Zero-shot classifier cache

ViT-L 계열의 joint embedding dimension은 768이다. ImageNet 1,000개 클래스의 text prototype을 FP16으로 캐시하면 다음과 같다.

$$
1000\times768\times2\ \text{bytes}=1,536,000\ \text{bytes}\approx1.46\ \text{MiB}.
$$

따라서 고정된 클래스 집합에서는 text encoder를 앱 시작 시 한 번 실행하거나 prototype을 모델 asset으로 저장할 수 있다. 프레임마다 필요한 연산은 image encoder, projection, normalization, $[1,768]\times[768,1000]$ dot product다. 반대로 사용자가 매번 자유 문장을 입력하는 retrieval에서는 text encoder latency가 query당 다시 추가된다.

## Zero-shot 분류가 동작하는 방식

클래스 $k$의 이름을 prompt template에 넣고 text encoder로 임베딩한다.

$$
c_k=\operatorname{norm}(f_T(\text{prompt}(y_k))).
$$

입력 이미지 embedding $v$와 모든 $c_k$의 cosine similarity를 구한 뒤 temperature-scaled softmax를 적용한다.

$$
p(y=k\mid x)=
\frac{\exp(v^\top c_k/\tau)}
{\sum_j\exp(v^\top c_j/\tau)}.
$$

이 과정은 learned linear classifier와 형태가 같다. 차이는 classifier weight가 labeled training set에서 최적화된 값이 아니라 자연어로부터 생성된다는 점이다.

### Prompt engineering과 ensembling

단순히 클래스 이름만 넣으면 단어의 다의성과 train-time 문장 분포 차이가 생긴다. `A photo of a {label}.` template은 ImageNet에서 1.3 percentage point를 개선했다. 여러 문맥 template을 사용하고 class별 embedding을 평균하는 prompt ensemble은 추가로 3.5 point를 개선했다. 전체적으로 prompt engineering과 ensembling은 36개 데이터셋 평균에서 거의 5 point를 올렸고, 저자들은 이를 모델 compute를 약 4배 늘린 것과 비슷한 향상으로 해석한다.

실제 구현에서는 다음을 구분해야 한다.

- template 수 증가: 초기 text encoding 비용과 prototype 저장량은 증가하지만 image당 연산은 거의 그대로다.
- class 수 증가: similarity matmul과 prototype memory가 선형 증가한다.
- 자유 형식 query: prototype을 미리 고정할 수 없으므로 text encoder를 런타임에 실행한다.

## 학습 recipe

논문이 공개한 주요 조건은 다음과 같다.

- 전체 모델을 32 epochs 학습
- Adam optimizer와 decoupled weight decay 사용
- cosine learning-rate schedule, warm-up 2,000 iterations
- bias와 normalization gain에는 weight decay를 적용하지 않음
- batch size 32,768
- weight decay 0.2
- temperature 초기값 0.07 상당, maximum logit scale 100
- Adam $\beta_1=0.9$
- ResNet은 $\beta_2=0.999$, $\epsilon=10^{-8}$
- ViT는 $\beta_2=0.98$, $\epsilon=10^{-6}$
- mixed precision, gradient checkpointing, half-precision optimizer statistics, text weight의 stochastic rounding 사용

가장 큰 ResNet인 RN50x64는 592개의 V100 GPU에서 18일, ViT-L/14는 256개의 V100 GPU에서 12일 학습했다. ViT-L/14@336px는 224 사전학습 뒤 336 해상도로 한 epoch를 추가 학습했다. 이는 CLIP을 단순한 architecture 아이디어가 아니라 대규모 데이터와 시스템 최적화까지 포함한 결과로 읽어야 하는 이유다.

### 모델별 hyperparameter

| 모델 | 입력 | joint dim | vision layers/width/heads | text layers/width/heads |
|---|---:|---:|---|---|
| ViT-B/32 | 224 | 512 | 12 / 768 / 12 | 12 / 512 / 8 |
| ViT-B/16 | 224 | 512 | 12 / 768 / 12 | 12 / 512 / 8 |
| ViT-L/14 | 224 | 768 | 24 / 1024 / 16 | 12 / 768 / 12 |
| ViT-L/14@336 | 336 | 768 | 24 / 1024 / 16 | 12 / 768 / 12 |

ResNet 계열은 규모가 커질수록 image resolution, backbone width/depth, text encoder width도 함께 키운다. ViT 계열의 B와 L도 vision 및 text width가 달라진다. 동일한 이름의 CLIP이라도 latency와 memory는 variant에 따라 크게 다르므로 실험 표에는 정확한 variant와 입력 해상도를 함께 적어야 한다.

## 핵심 실험 결과

### ImageNet zero-shot와 전이

- ViT-L/14@336px의 ImageNet zero-shot top-1은 76.2%, top-5는 95%다.
- 이 결과는 원래의 supervised ResNet-50과 비슷하지만, ImageNet label로 CLIP을 학습하지 않았다는 점이 핵심이다.
- zero-shot CLIP은 같은 CLIP feature에서 학습한 평균 4-shot linear probe와 비슷하고, ImageNet에서는 16-shot linear probe와 비슷하다.
- 27개 데이터셋에서 zero-shot과 supervised linear probe의 상관은 $r=0.82$, $p<10^{-6}$였다. 즉 representation이 좋은 과제는 zero-shot도 대체로 좋지만, 대부분의 과제에서 zero-shot은 linear probe보다 10에서 25 point 낮았다.
- zero-shot CLIP은 supervised ResNet-50 linear probe보다 27개 중 16개 데이터셋에서 높았다.

이 결과는 zero-shot이 항상 supervised adaptation을 이긴다는 뜻이 아니다. 과제별 train split을 활용한 최상위 supervised 모델과 비교하면 CLIP이 약한 데이터셋이 많다.

### Scaling과 architecture 비교

ResNet CLIP의 image compute는 RN50부터 RN50x64까지 6.1, 9.9, 21.5, 75.3, 265.9 GFLOPs로 증가한다. 44배 범위에서 zero-shot error는 compute에 대해 매끄러운 log-log scaling을 보였다. CLIP ViT는 CLIP ResNet보다 약 3배 compute-efficient했으며, 최고 ViT-L/14@336px는 12개 데이터셋 평균에서 기존 최고 모델보다 2.6 point, 더 넓은 27개 데이터셋 평균에서는 약 5 point 높았다.

여기서도 GFLOPs는 실제 모바일 latency와 동일하지 않다. ViT의 reshape, layer normalization, attention kernel, memory bandwidth와 backend 지원 여부가 실제 시간을 결정한다.

### Image-text retrieval

논문의 Appendix Table 13은 fine-tuning 없이 수행한 retrieval 결과를 제공한다.

| 데이터셋/방향 | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Flickr30k text retrieval | 88.0 | 98.7 | 99.4 |
| Flickr30k image retrieval | 68.7 | 90.6 | 95.2 |
| MSCOCO text retrieval | 58.4 | 81.5 | 88.1 |
| MSCOCO image retrieval | 37.8 | 62.4 | 72.2 |

description 앞에 `a photo of`를 붙이면 zero-shot R@1이 1에서 2 point 향상됐다. Flickr30k text retrieval은 당시 fine-tuned 최고 결과에 경쟁력이 있었지만, 더 큰 MSCOCO에서는 fine-tuned 최신 모델보다 낮았다. 또한 text retrieval보다 image retrieval 격차가 더 컸다.

### 자연 분포 변화에 대한 강건성

ImageNet classifier를 기준으로 한 7개 자연 분포 변화 데이터에서 zero-shot CLIP은 robustness gap을 최대 75% 줄였다. Appendix의 상세 결과는 다음과 같다.

| 모델/설정 | ImageNet | IN-V2 | IN-A | IN-R | ObjectNet | IN-Sketch |
|---|---:|---:|---:|---:|---:|---:|
| CLIP linear probe | 85.4 | 75.9 | 75.3 | 84.2 | 66.2 | 57.4 |
| zero-shot CLIP | 76.2 | 70.1 | 77.2 | 88.9 | 72.3 | 60.2 |

ImageNet linear probe는 in-distribution 성능을 9.2 point 올려 85.4%가 되지만, 일부 shift에서는 오히려 성능이 감소했다. zero-shot classifier가 특정 ImageNet train distribution에 적응하지 않은 것이 robustness 측면에서 장점이 된다는 해석이다.

### 데이터 중복 분석

웹 규모 사전학습에서는 downstream test 이미지가 train set에 섞였는지 확인해야 한다. 저자들은 35개 데이터셋에 duplicate detector를 적용했다.

- 9개 데이터셋은 검출된 overlap이 없었다.
- overlap 비율의 median은 2.2%, mean은 3.2%였다.
- 전체 정확도 변화는 대부분 0.1 point 이하였고, 최대 증가는 Birdsnap의 0.6 point였다.
- Country211은 overlap이 21.5%로 가장 높았지만 전체 정확도 증가는 0.2 point였다.

저자들 스스로 detector recall을 400M 전체에서 완전히 검증할 수 없고, overlap/clean subset 간 난이도 차이가 confounder가 될 수 있다고 지적한다. 따라서 이 분석은 leakage가 전혀 없다는 증명이 아니라, 검출 가능한 overlap이 보고된 평균 성능을 크게 부풀렸다는 증거는 약하다는 주장이다.

## Ablation에서 읽어야 할 것

### 1. Objective ablation

caption language model에서 bag-of-words로 바꾸며 약 3배, 다시 contrastive objective로 바꾸며 약 4배의 zero-shot 학습 효율 향상을 얻었다. CLIP의 결정적 선택은 더 복잡한 fusion이 아니라 대규모 negative를 활용하는 단순한 식별 objective다.

### 2. Prompt ablation

`A photo of a {label}.`만으로 ImageNet 1.3 point, 80개 prompt ensemble로 추가 3.5 point를 얻었다. 이는 평가 protocol이 모델 weight만큼 중요하다는 뜻이다. 서로 다른 논문의 zero-shot 수치를 비교할 때 template와 ensemble 수를 통제해야 한다.

### 3. Dataset ablation

같은 크기의 YFCC100M subset과 WIT subset으로 RN50을 32 epochs 학습하면 전체 평균과 dataset win 수는 비슷했다. 그러나 개별 과제에서는 10 point가 넘는 차이가 나타났다. 예를 들어 YFCC는 Birdsnap과 Flowers102에서, WIT는 Stanford Cars와 UCF101 등에서 더 강했다. objective가 같아도 사전학습 데이터에 어떤 개념이 조밀하게 존재하는지가 downstream capability를 결정한다.

### 4. Model scale와 resolution

ViT가 ResNet보다 평균적으로 compute-efficient했고, L/14를 마지막 epoch에 336으로 올리면 linear probe ImageNet이 83.9에서 85.4로 높아졌다. 그러나 앞서 계산했듯 224에서 336은 attention score 원소 수를 약 5배 늘린다. 정확도 개선과 모바일 비용을 별도의 축으로 측정해야 한다.

## 실패 사례와 한계

### Fine-grained, counting, distance estimation

CLIP은 자동차 모델, 꽃 종, 항공기 variant 같은 fine-grained 분류에서 약하고, 이미지 속 객체 수 세기와 같은 추상적이고 체계적인 과제에서도 약하다. 학습 데이터에 거의 없을 법한 가장 가까운 자동차까지의 거리 분류는 random에 가까울 수 있다. 언어로 클래스를 표현할 수 있다는 사실이 시각적 관계 추론을 자동으로 보장하지 않는다.

### 진정한 OOD 일반화는 해결되지 않음

Rendered SST2 같은 웹과 유사한 디지털 텍스트에서는 의미 있는 OCR 표현을 보이지만, handwritten MNIST zero-shot 정확도는 88.4%로 raw-pixel logistic regression 92.5%보다 낮았다. full-number SVHN은 51.0%였다. 큰 웹 데이터가 다양한 자연 분포를 포괄해도 학습 분포에 사실상 없는 modality까지 강건하게 일반화하는 것은 아니다.

### Zero-shot과 few-shot의 단절

Oxford-IIIT Pets에서 사람은 zero-shot 53.7%에서 one-shot 75.7%로 크게 향상됐지만, CLIP feature 위의 단순 linear classifier는 사전지식을 효율적으로 활용하지 못한다. 저자들은 few-shot adaptation이 zero-shot보다 오히려 robustness를 낮출 수 있음을 보였다. CLIP은 few-shot 성능을 직접 최적화한 모델이 아니다.

### 분류 후보 집합의 폐쇄성

자연어로 class를 만들 수 있어도 출력은 주어진 후보 중 하나다. captioning처럼 새로운 설명을 생성하지 못한다. 후보 class 설계, prompt wording, threshold가 결과와 위해를 크게 바꾼다.

### 평가 co-adaptation

개발 중 전체 validation set을 반복적으로 조회했으며, 27개 데이터셋 모음도 CLIP 개발 과정과 완전히 독립적이지 않다. 논문은 이를 true zero-shot protocol의 한계로 인정한다. 새 benchmark에서 prompt를 고정한 blind evaluation이 필요하다.

### 데이터와 compute 비용

400M 쌍을 32 epochs 보면 12.8B image presentations다. 저자들의 비유대로 초당 한 장이면 405년이 걸린다. 당시 scaling trend만으로 전체 과제의 state of the art에 도달하려면 약 1,000배 compute가 필요하다고 추정했으며, 이는 실행 가능하지 않다고 명시한다.

## Bias, 안전성, 배포 시 주의점

CLIP은 필터링되지 않은 웹 이미지-텍스트 쌍의 사회적 편향을 학습한다. FairFace probe에서 label set에 범죄 관련 및 비인간 범주를 추가했을 때 전체의 4.9%가 비인간 class로 잘못 분류되었고, 논문이 사용한 FairFace의 `Black` 범주 이미지는 약 14%로 가장 높았다. 남성 이미지는 범죄 관련 class로 16.5%, 여성 이미지는 9.8% 잘못 분류됐다. `child` class 하나를 추가하는 것만으로 20세 미만의 유해 오분류가 크게 줄어 class design이 bias 발현을 좌우함을 보였다.

이는 얼굴의 인종, 성별, 나이를 자동 분류하는 행위를 정당화하는 결과가 아니다. 논문도 benchmark accuracy가 실제 impact fairness를 보장하지 않으며, 해당 범주 자체가 환원적이라는 점을 명시한다.

surveillance probe에서 CCTV 장면의 coarse classification은 91.8%였지만, 매우 비슷한 distractor caption을 추가하면 51.1%로 떨어졌고 작은 객체의 존재 여부는 random에 가까웠다. CelebA celebrity identification은 100 classes에서 59.2%, 1,000 classes에서 43.3%였다. task-specific 학습 없이도 identity 정보를 회수할 수 있다는 사실 자체가 privacy risk다.

배포 전에는 최소한 다음을 별도 검증해야 한다.

- 실제 label set과 prompt wording별 subgroup 오류
- top-1뿐 아니라 score calibration과 reject threshold
- 얼굴, 감시, 고용, 치안 등 민감한 사용 맥락의 적합성
- gallery나 prompt에 개인정보가 포함될 때 embedding 저장 및 삭제 정책
- open-set 입력에서 무조건 후보 하나를 고르는 failure mode

## 온디바이스 관점

### 논문이 실제로 보고한 것

- GPU 학습 시간과 GPU 수
- variant별 입력 해상도, layer, width, head, embedding dimension
- ResNet 계열 GFLOPs/image
- zero-shot accuracy, linear probe, retrieval, robustness 결과

### 논문이 보고하지 않은 것

- 스마트폰 CPU, GPU, NPU latency
- batch=1 p50/p95 latency
- peak resident memory와 activation peak
- model file size, INT8/FP16 결과
- 전력, 온도, sustained workload throttling
- text query 입력부터 첫 retrieval 결과까지의 TTFT
- image/text encoder를 분리한 latency breakdown

따라서 이 논문만으로 특정 모바일 장치의 실시간성을 주장할 수 없다. 앞의 token 및 cache 메모리 값은 architecture table에서 계산한 reviewer estimate이고 측정치가 아니다.

### 실제 모바일 파이프라인 제안

고정 taxonomy 분류는 다음처럼 구성하는 것이 효율적이다.

```text
앱 설치 또는 class 변경 시:
  prompts -> text encoder -> class prototypes -> FP16/INT8 cache

각 카메라 frame:
  resize/crop -> image encoder -> normalized image embedding
  -> cached prototypes와 matmul -> top-k / reject

자유 text query 시:
  query -> text encoder -> normalized query embedding
  -> image embedding index 검색
```

카메라가 30 FPS라고 해서 모든 프레임에 큰 CLIP encoder를 실행할 필요는 없다. 경량 detector가 이벤트를 찾을 때만 CLIP을 실행하거나, 동일 장면의 image embedding을 캐시하고 실행 주기를 동적으로 조절할 수 있다. retrieval gallery가 커지면 encoder보다 nearest-neighbor index 검색과 embedding memory가 병목이 될 수 있다.

### 측정해야 할 latency 정의

- 고정 class 분류 latency: preprocess + image encoder + projection + class matmul
- 자유 text query TTFT: tokenizer + text encoder + index search + top-k 반환
- image 등록 latency: image encoder + embedding 저장 + index update
- cold start: 모델 load, graph compile, prototype load 포함
- warm p50/p95: 최소 수백 회, 동일 thermal state에서 측정

CLIP은 autoregressive decoder가 없으므로 VLM의 token generation TTFT나 tokens/s와 직접 비교할 모델은 아니다. 여기서 TTFT는 사용자가 query를 제출한 시점부터 첫 retrieval 결과가 반환되는 시간으로 정의해야 한다.

### 압축 우선순위

1. target backend에서 지원되는 작은 image encoder variant를 우선 고른다.
2. class가 고정이면 text encoder를 runtime graph에서 제거하고 prototype만 배포한다.
3. image encoder INT8 또는 mixed precision을 적용하되 cosine ranking과 calibration 변화를 측정한다.
4. embedding dimension 축소는 gallery memory와 search latency를 줄이지만 retrieval R@K 손실을 함께 측정한다.
5. resolution을 줄일 때 token 수와 attention 비용은 크게 감소하지만 작은 글자와 fine-grained class가 먼저 손상될 수 있다.

## 재현용 구현 체크리스트

### 데이터와 objective

- [ ] image와 text의 batch 순서를 동일하게 유지했다.
- [ ] embedding projection 뒤 L2 normalization을 적용했다.
- [ ] image-to-text와 text-to-image cross entropy를 모두 사용했다.
- [ ] temperature를 양수로 parameterize하고 logit scale 상한을 적용했다.
- [ ] multi-GPU에서 global negatives와 label offset을 올바르게 처리했다.
- [ ] 중복 caption이나 동일 이미지가 false negative가 되는 비율을 기록했다.

### Zero-shot 평가

- [ ] class name 전처리와 동의어 mapping을 기록했다.
- [ ] prompt template 전체를 공개했다.
- [ ] prompt embedding 평균 전후 normalization 순서를 고정했다.
- [ ] class prototype을 validation label로 직접 튜닝하지 않았다.
- [ ] top-1뿐 아니라 per-class, calibration, reject 성능을 측정했다.
- [ ] data overlap을 가능한 범위에서 검사했다.

### 온디바이스 평가

- [ ] 입력 해상도와 resize/crop 방식을 고정했다.
- [ ] batch=1, cold/warm을 분리했다.
- [ ] image encoder, text encoder, similarity search를 따로 profile했다.
- [ ] p50/p95, peak memory, model size, 전력, 온도, throttling을 기록했다.
- [ ] FP16, vision INT8, prototype INT8/FP16 조합을 비교했다.
- [ ] 동일 정확도 조건에서 실행 주기와 cache hit rate를 기록했다.

## 로드맵에서의 위치

이 논문은 VLM 단계의 출발점이다. CLIP이 제공하는 것은 생성형 대화 능력이 아니라, image와 text를 연결하는 강한 dual-encoder representation과 open-vocabulary interface다.

- 다음 MobileCLIP에서는 같은 contrastive 목표를 모바일 친화적 encoder와 knowledge reinforcement로 얼마나 작게 만들 수 있는지 본다.
- BLIP-2에서는 고정된 vision encoder와 고정된 LLM 사이를 Q-Former가 어떻게 연결하는지 본다.
- LLaVA에서는 visual feature를 projection하여 instruction-tuned LLM으로 보내는 생성형 인터페이스를 본다.
- open-vocabulary detector에서는 CLIP text embedding이 box-level visual representation의 classifier 역할을 한다.

따라서 첫 구현 산출물은 MobileCLIP 또는 작은 CLIP variant로 만든 image-text retrieval이 적합하다. 같은 데이터와 장치에서 image encoder latency, text encoder latency, gallery 크기별 search latency, R@1/R@5, prototype cache memory를 분리해 표로 남겨야 이후 VLM의 visual token 및 decoder 비용과 공정하게 비교할 수 있다.

## 최종 평가

CLIP의 가장 큰 기여는 자연어 supervision을 이용해 이미지 분류를 고정된 label space에서 벗어나게 한 것이다. 단순한 dual encoder와 symmetric contrastive loss만으로 zero-shot classifier, retrieval, 강건한 visual representation을 하나의 모델에 담았고, 이후 open-vocabulary vision과 VLM의 공통 기반을 만들었다.

동시에 결과는 400M 웹 쌍, 매우 큰 batch, 대규모 GPU 학습에 의존하며, fine-grained 인식, counting, 진정한 OOD, few-shot adaptation, 사회적 편향을 해결하지 못했다. 온디바이스 관점에서는 text prototype cache가 큰 장점이지만 vision encoder와 해상도별 token 비용이 여전히 핵심 병목이다. 이 논문을 재현할 때는 정확도 하나보다 `variant + resolution + prompt protocol + cache policy + p50/p95 + peak memory`를 하나의 실험 단위로 다루는 것이 중요하다.

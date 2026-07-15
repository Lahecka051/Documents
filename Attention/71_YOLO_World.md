# 71. YOLO-World

## 논문 정보

- 원본 파일: `71_YOLO_World.pdf`
- 제목: **YOLO-World: Real-Time Open-Vocabulary Object Detection**
- 저자: Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, Ying Shan
- 공개: arXiv:2401.17270, PDF 기준 v3, 2024-02-22
- 공식 링크: [arXiv:2401.17270](https://arxiv.org/abs/2401.17270)
- 태스크: open-vocabulary object detection, phrase grounding, open-vocabulary instance segmentation
- 핵심 키워드: YOLOv8, vision-language pre-training, RepVL-PAN, prompt-then-detect, re-parameterization, region-text contrastive learning

## 한눈에 보는 요약

YOLO-World는 YOLOv8의 빠른 one-stage detector 구조에 CLIP text encoder와 vision-language fusion을 붙여, 입력 문장에 들어 있는 임의의 범주를 찾는 open-vocabulary detector다. 연구 질문은 명확하다. Grounding DINO나 GLIP처럼 강하지만 무거운 모델 대신, YOLO의 낮은 지연 시간을 유지하면서 텍스트가 정의하는 vocabulary로 box를 예측할 수 있는가?

```text
text prompts -> frozen CLIP text encoder -> text embeddings W [C, D]
                                               |
image -> YOLOv8 backbone -> RepVL-PAN <---------+
                         -> object embeddings E [N, D]
                         -> box regression [N, 4]
                         -> E x W^T -> class logits [N, C]
                         -> confidence filtering + NMS -> boxes and text labels
```

핵심 기여는 세 가지다.

1. YOLO의 고정 class classifier를 region-text similarity classifier로 바꾼다.
2. RepVL-PAN에서 text가 image feature를 조절하고, pooled image token이 text embedding을 다시 보정한다.
3. 배포 시 vocabulary가 고정되어 있다면 text embedding과 detection head를 재매개변수화해 CLIP text encoder와 일부 fusion 비용을 제거한다.

논문이 보고한 LVIS minival fixed AP에서 YOLO-World-L은 `35.0 AP`, CC3M pseudo-label 데이터를 더하면 `35.4 AP`다. 재매개변수화된 L 모델은 V100에서 `52.0 FPS`이고, 같은 표의 DetCLIP은 `2.3 FPS`다. 다만 이 수치는 V100 처리량이며 모바일 실측이 아니다. 실제 온디바이스 적용에서는 vocabulary 크기에 따른 출력 activation, NMS, 지원 operator, prompt 변경 시 재설정 비용을 별도로 측정해야 한다.

## 문제 설정

### Closed-set detector의 제약

일반 YOLO classifier는 각 후보 위치의 feature `h_k`를 고정된 class weight와 비교한다.

```math
z_k = h_k W_{cls}^{\top},
\qquad
h_k\in\mathbb{R}^{D'},\quad
W_{cls}\in\mathbb{R}^{C_{train}\times D'}
```

`W_cls`의 행은 학습 때 정한 class에 묶인다. 새로운 범주를 추가하려면 보통 annotation과 재학습이 필요하다. Open-vocabulary detector는 class weight를 text encoder가 만든 embedding으로 대체한다.

```math
W = \mathrm{TextEncoder}(t_1,\ldots,t_C)
\in\mathbb{R}^{C\times D}
```

이제 `C`는 학습 class 수가 아니라 현재 prompt vocabulary의 크기다. `cat`, `forklift`, `red safety helmet`처럼 사용자가 지정한 phrase가 classifier를 정의한다.

### 논문이 해결하려는 속도 문제

기존 grounded pre-training 계열은 대개 다음 비용을 가진다.

- 큰 language encoder를 매 이미지마다 실행한다.
- 많은 image token과 text token 사이에 여러 층의 cross-attention을 수행한다.
- DETR decoder의 query가 여러 층을 통과한다.
- end-to-end detector라도 실제 파이프라인에서 prompt parsing과 후처리 비용이 붙는다.

YOLO-World의 방향은 text 조건을 YOLO neck과 head에 최소한으로 넣고, 반복되는 vocabulary는 미리 계산하는 것이다. 따라서 이 논문에서 `real-time`이라는 말은 단순히 경량 backbone을 뜻하지 않는다. text branch를 frame loop 밖으로 꺼낼 수 있는 구조까지 포함한다.

## 전체 아키텍처

### 학습 경로

배치 크기를 `B`, 세 feature level의 크기를 `(H_l,W_l)`, text category 수를 `C`, 공통 embedding 차원을 `D`라 하자.

```text
image I [B, 3, H, W]
  -> YOLOv8 backbone
  -> {X3, X4, X5}
     Xl [B, Dl, Hl, Wl]
  -> RepVL-PAN
  -> multi-scale object features
  -> decoupled detection head
       box distribution / objectness
       object embeddings E [B, N, D]

text list {t1, ..., tC}
  -> tokenizer
  -> frozen CLIP text encoder
  -> W [B, C, D]

normalized E and W
  -> scaled cosine similarity
  -> S [B, N, C]
```

YOLO 계열의 typical 640 입력을 예로 들면 stride 8, 16, 32 feature map은 각각 `80 x 80`, `40 x 40`, `20 x 20`이다. anchor-free prediction 위치 수는 다음과 같다.

```math
N = 80^2 + 40^2 + 20^2 = 8{,}400
```

이 `N=8,400`은 논문이 특정 문장으로 고정한 상수가 아니라 640 입력과 일반적인 세 pyramid level을 둔 리뷰어 계산이다. 실제 export graph의 입력 크기와 head 구성으로 다시 확인해야 한다.

### 추론 경로: online과 offline vocabulary

논문은 두 사용법을 구분한다.

```text
online vocabulary:
  prompt changes -> run CLIP -> fuse text and image -> detect

offline vocabulary:
  prompt fixed -> encode/fuse once -> cache or re-parameterize
              -> image-only detector path for every frame
```

Offline vocabulary는 카메라가 늘 같은 80개 안전 범주를 찾는 상황에 적합하다. 반면 사용자가 매번 새로운 문장을 입력하면 text encoding과 재구성이 필요하다. 재매개변수화된 FPS를 임의 prompt가 매 frame 바뀌는 시스템의 latency로 해석하면 안 된다.

## Region-text contrastive head

각 prediction 위치 `k`의 object embedding을 `e_k`, text `j`의 embedding을 `w_j`라 하자. 논문의 similarity는 L2 normalization 뒤 scaled cosine similarity를 사용한다.

```math
s_{k,j}
=
\alpha\,
\mathrm{L2Norm}(e_k)
\mathrm{L2Norm}(w_j)^{\top}
+\beta
```

`alpha`와 `beta`는 학습 가능한 affine parameter다. normalization 덕분에 logit은 feature norm보다 방향 정렬에 집중한다. `alpha`는 softmax 또는 binary classification의 sharpness를 조절하고 `beta`는 전체 logit 기준점을 이동한다.

Shape은 다음과 같다.

```text
E: [B, N, D]
W: [B, C, D]
transpose(W): [B, D, C]
S = E @ transpose(W): [B, N, C]
```

고정 linear classifier와 모양은 같지만 weight가 text에서 왔다는 차이가 있다. 이 구조는 새 phrase를 즉시 class prototype으로 넣을 수 있지만, phrase embedding이 시각적으로 분리 가능한 class boundary가 된다는 보장은 없다. 동의어, 복합 명사, 속성 표현, 문맥 의존 표현의 prompt engineering이 결과에 영향을 준다.

### Vocabulary 크기와 activation memory

`N=8,400`, FP16 logit을 가정하면 COCO 80 class의 similarity tensor는 다음 크기다.

```math
8{,}400\times80=672{,}000\text{ logits}
```

```math
672{,}000\times2\text{ bytes}
=1{,}344{,}000\text{ bytes}
\approx1.28\text{ MiB}
```

LVIS 1,203 class라면:

```math
8{,}400\times1{,}203=10{,}105{,}200\text{ logits}
```

```math
10{,}105{,}200\times2
\approx19.27\text{ MiB}
```

이는 similarity 출력 하나만 계산한 값이며 object embedding, box branch, neck activation, kernel workspace, NMS buffer는 포함하지 않는다. 특히 batch=1 모바일에서는 parameter 수보다 이 순간적인 `[N,C]` activation이 peak memory와 메모리 bandwidth를 좌우할 수 있다.

Embedding 차원을 예시로 `D=512`라고 두면 LVIS dense dot product는 약 `8,400 x 1,203 x 512 = 5.17G` multiply-accumulate다. 이 `D=512`는 비용 감도를 보여 주는 예시이며 실제 배포 graph의 projection 차원을 확인해야 한다. 재매개변수화가 중요한 이유가 여기 있다.

## RepVL-PAN

RepVL-PAN은 YOLOv8의 Path Aggregation Network에 text-image interaction을 삽입한 neck이다. 두 모듈이 서로 반대 방향으로 정보를 보낸다.

```text
text -> image: Text-guided CSPLayer, T-CSPLayer
image -> text: Image-Pooling Attention
```

### Text-guided CSPLayer

각 spatial image feature가 text vocabulary 중 가장 강하게 대응하는 항목을 찾아 channel feature를 gate한다. 논문의 표현을 shape 중심으로 쓰면 다음과 같다.

```math
A_l = X_l W^{\top}
```

```text
X_l: spatial feature, conceptually [B, H_l W_l, D]
W:   text embeddings [B, C, D]
A_l: region-text affinities [B, H_l W_l, C]
```

Class 축 maximum을 취하고 sigmoid gate로 image feature를 조절한다.

```math
g_l=\sigma\left(\max_j(X_lW_j^{\top})\right)
```

```math
X'_l=X_l\odot g_l
```

Max-sigmoid는 모든 class를 weighted average하지 않고 가장 관련 있는 text 신호 하나를 gate로 사용한다. 장점은 text 조건을 간단히 넣는다는 점이다. 단점은 max가 선택한 class 외의 미세한 관계 정보가 사라지고, 큰 vocabulary에서 dense affinity 계산이 선행될 수 있다는 점이다.

### Image-Pooling Attention

Image가 text를 보정하는 방향에서는 multi-scale feature를 각 level당 `3 x 3`로 pooling한다. 세 level이면 image token은 `3 x 9=27`개다.

```text
P3 -> 3 x 3 -> 9 tokens
P4 -> 3 x 3 -> 9 tokens
P5 -> 3 x 3 -> 9 tokens
concat         27 tokens
```

```math
\widetilde X\in\mathbb{R}^{27\times D}
```

Text embedding은 query, pooled image token은 key와 value가 된다.

```math
W'=W+\mathrm{MultiHeadAttention}(W,\widetilde X,\widetilde X)
```

Attention score는 `[C,27]`이므로 전체 high-resolution image token과 text를 직접 cross-attention하는 것보다 작다. 대신 27-token pooling이 작은 물체나 세밀한 위치 정보를 잃을 수 있다. 이 모듈의 역할은 정확한 box localization이 아니라 현재 이미지 문맥에 맞게 text prototype을 조정하는 것으로 보는 편이 맞다.

### 왜 두 방향이 모두 필요한가

Text-to-image gate만 있으면 image feature는 vocabulary에 맞게 강조되지만 text embedding은 이미지 문맥을 모른다. Image-to-text attention을 더하면 같은 `bat`라는 단어라도 현재 이미지의 visual evidence에 따라 embedding이 보정될 수 있다. 반대로 image pooling만 쓰면 high-resolution neck 자체가 text-conditioned되지 않는다.

논문의 ablation은 Objects365만 사용했을 때 baseline `22.4 AP`, text-to-image 추가 `23.2 AP`, 두 방향 모두 `23.5 AP`를 보고한다. Objects365와 GQA를 함께 쓴 더 큰 데이터 설정에서는 baseline `29.7 AP`, 두 방향 `31.9 AP`다. 모듈 이득이 데이터 다양성과 상호작용한다는 점을 보여 준다.

## Prompt-then-detect와 재매개변수화

### 핵심 아이디어

고정 vocabulary의 text embedding을 매 frame 다시 만들 필요는 없다. 더 나아가 object embedding projection과 text matrix multiplication이 모두 linear이면 두 weight를 합성할 수 있다.

단순화한 식은 다음과 같다.

```math
E=HW_e^{\top},\qquad S=EW^{\top}
```

```math
S=H(W W_e)^{\top}
```

따라서 합성된 class kernel을 미리 만들 수 있다.

```math
W_{rep}=WW_e
```

이후 추론은 일반 YOLO의 `C`-channel 1x1 convolution처럼 실행된다.

```text
before re-parameterization:
feature -> object embedding -> normalized similarity with text

after re-parameterization:
feature -> vocabulary-specific class convolution
```

실제 RepVL-PAN에는 비선형 gate와 image-conditioned text update가 있어 전체 graph를 이 한 식으로만 설명할 수는 없다. 논문이 제공하는 re-parameterized model은 배포용 경로로 정리된 결과이며, 구현할 때는 공식 export 코드가 어떤 branch를 제거하고 어떤 weight를 합성하는지 확인해야 한다.

### 운영상 의미

- vocabulary가 하루 동안 고정: 앱 시작 시 한 번 text를 encode하고 cache 가능
- 사용자가 prompt를 바꿈: text encode와 head 재구성 후 다음 frame부터 사용
- frame마다 prompt가 바뀜: re-parameterized FPS를 그대로 기대하기 어려움
- 여러 vocabulary preset: preset별 kernel을 저장하면 저장 공간과 전환 시간을 교환 가능

즉 이 논문의 가장 실용적인 시스템 아이디어는 `prompt latency`와 `frame latency`를 분리한 것이다.

## 학습 데이터 구성

### Detection과 grounding 데이터

논문은 detection, grounding, image-text 데이터를 한 pre-training 과정에 섞는다. PDF 표에 제시된 규모는 다음과 같다.

| 데이터 | 이미지 | annotation | 역할 |
|---|---:|---:|---|
| Objects365 | 609K | 9.621M | detection box와 category |
| GQA | 621K | 3.681M | grounded phrase-region |
| Flickr | 149K | 641K | phrase grounding |
| CC3M pseudo labels | 246K | 821K | image-text에서 생성한 box-text |

Image-text 데이터에는 원래 box가 없다. 논문은 CC3M caption에서 noun phrase를 추출하고 GLIP으로 candidate box를 만든 뒤 CLIP으로 filtering하고 NMS를 적용한다. 이 pseudo annotation은 vocabulary를 넓히지만 teacher의 오류와 bias도 전달한다.

```text
caption
  -> n-gram or noun phrase extraction
  -> GLIP candidate localization
  -> CLIP image-region/text consistency filtering
  -> NMS
  -> pseudo region-text pairs
```

### Online vocabulary sampling

학습마다 전체 데이터 vocabulary를 모두 비교하면 비용이 너무 크다. 논문은 4-image mosaic 단위로 positive nouns를 모으고 dataset vocabulary에서 negative nouns를 무작위로 추가해 최대 `M=80`개의 text를 사용한다.

```text
positives from current mosaic
  + sampled negatives
  -> at most 80 text categories
```

이 방법은 classifier cost를 제한하고 negative contrast를 제공한다. 다만 sampled softmax와 비슷하게, 어떤 negative가 뽑히는지에 따라 학습 난도가 달라진다. 의미적으로 가까운 hard negative보다 무작위 쉬운 negative가 많으면 fine-grained 구분을 충분히 배우지 못할 수 있다.

## 손실 함수

이미지 `I`에 대한 손실은 region-text contrastive classification과 YOLO box regression을 결합한다.

```math
\mathcal L(I)
=
\mathcal L_{con}
+\lambda_I
\left(
\mathcal L_{iou}+\mathcal L_{dfl}
\right)
```

- `L_con`: region과 text를 정렬하는 classification/contrastive loss
- `L_iou`: box overlap regression
- `L_dfl`: Distribution Focal Loss 기반 좌표 분포 학습
- `lambda_I=1`: detection 또는 grounding처럼 신뢰할 수 있는 box가 있는 데이터
- `lambda_I=0`: noisy image-text pseudo-label 설정에서 box regression을 끄는 경우

이 분리는 중요한 설계다. 언어 다양성을 얻기 위해 noisy pair를 쓰되, 부정확한 pseudo box가 localization head를 망가뜨리는 영향을 줄인다. 하지만 `lambda=0`이어도 잘못된 region-text pair는 classification representation에 영향을 준다.

## 학습 설정

논문이 기술한 pre-training 설정은 다음과 같다.

- optimizer: AdamW
- initial learning rate: `0.002`
- weight decay: `0.05`
- epochs: `100`
- hardware: NVIDIA V100 32개
- global batch size: `512`
- text encoder: frozen
- augmentation: color, affine, flip, mosaic

CLIP text encoder를 freeze하면 대규모 language model의 gradient memory를 피하고 기존 semantic space를 유지한다. Ablation에서도 BERT frozen `14.6 AP`, BERT fine-tune `18.3 AP`, CLIP frozen `22.4 AP`, CLIP fine-tune `19.3 AP`로 frozen CLIP이 가장 좋다. 이 결과는 `fine-tune은 항상 유리하다`는 직관과 반대이며, detector 데이터의 제한된 vocabulary가 CLIP의 넓은 semantic space를 좁힐 수 있음을 시사한다.

## 주요 결과

### LVIS zero-shot fixed AP

PDF Table 2의 V100 측정 결과를 요약하면 다음과 같다. FPS는 TensorRT 없이 측정되었고, 괄호 안 원 모델 수치는 text/fusion branch가 남은 상태다.

| 모델 | 재매개변수화 params | 원 모델 params | 재매개변수화 FPS | 원 모델 FPS | AP | APr |
|---|---:|---:|---:|---:|---:|---:|
| YOLO-World-S | 13M | 77M | 74.1 | 19.9 | 26.2 | 19.1 |
| YOLO-World-M | 29M | 92M | 58.1 | 18.5 | 31.0 | - |
| YOLO-World-L | 48M | 110M | 52.0 | 17.6 | 35.0 | - |
| YOLO-World-L + CC3M | 48M | 110M | 52.0 | 17.6 | 35.4 | 27.6 |
| DetCLIP | 155M | - | 2.3 | - | 34.4 | - |

CC3M을 더한 L 모델의 frequency별 AP는 `APr 27.6`, `APc 34.1`, `APf 38.0`이다. 희귀 class 개선은 open-vocabulary 학습의 핵심 목표와 맞지만 frequent class와는 여전히 차이가 있다.

### 데이터 ablation

| 사전학습 데이터 | LVIS AP |
|---|---:|
| Objects365 | 23.5 |
| + GQA | 31.9 |
| + GoldG | 32.5 |
| + CC3M pseudo labels | 33.0 |

가장 큰 증가는 Objects365에 GQA를 더할 때 나타난다. 단순 class name detection보다 phrase-region grounding 데이터가 open-vocabulary transfer에 중요하다는 근거다. CC3M은 규모에 비해 추가 이득이 작으므로 pseudo-label quality와 중복 vocabulary를 함께 고려해야 한다.

### COCO fine-tuning과 LVIS base fine-tuning

- COCO zero-shot YOLO-World-L: `45.1 AP`
- COCO fine-tuned YOLO-World-L, RepVL 제거: `53.3 AP`, V100 TensorRT `156 FPS`
- 비교 YOLOv8-L: `52.9 AP`, `159 FPS`
- LVIS base fine-tuned YOLO-World-L: `34.1 AP`, rare `20.4 AP`
- 비교 YOLOv8-L: `26.9 AP`, rare `10.2 AP`

COCO fine-tuning 후 일반 YOLOv8과 거의 같은 속도와 약간 높은 AP를 얻는 결과는 pre-training initialization의 가치로 해석할 수 있다. 그러나 open-vocabulary 기능을 제거한 설정과 유지한 설정을 같은 표에서 혼동하지 않아야 한다.

### Open-vocabulary instance segmentation

YOLO-World에 segmentation head를 붙인 결과도 보고한다. M/L 모델의 mask AP는 다음 경향을 보인다.

| segmentation head 학습 범위 | M mask AP | L mask AP |
|---|---:|---:|
| COCO | 12.3 | 16.2 |
| LVIS base | 16.7 | 19.1 |
| LVIS all | 25.9 | 28.7 |

논문은 full fine-tuning이 box rare AP를 떨어뜨릴 수 있음도 보여 준다. Mask supervision을 늘리면 segmentation은 좋아지지만 기존 language-aligned detection space를 보존하는 별도 전략이 필요하다.

## 파라미터와 메모리 계산

모델 weight만 FP16과 INT8로 단순 환산하면 다음과 같다. optimizer, activation, alignment padding은 제외한다.

| 모델 | params | FP16 weight | INT8 weight |
|---|---:|---:|---:|
| reparam S | 13M | 약 24.8 MiB | 약 12.4 MiB |
| reparam M | 29M | 약 55.3 MiB | 약 27.7 MiB |
| reparam L | 48M | 약 91.6 MiB | 약 45.8 MiB |
| original L | 110M | 약 209.8 MiB | 약 104.9 MiB |

계산식 예시는 다음과 같다.

```math
48\times10^6\times2 / 2^{20}\approx91.6\text{ MiB}
```

INT8 weight가 곧 전체 peak memory 절반을 의미하지는 않는다. Neck activation과 `[N,C]` logit이 FP16/FP32로 남거나 backend가 quantize/dequantize를 삽입하면 절감이 작아진다. 실제 측정은 resident set, accelerator heap, CPU heap을 모두 포함해야 한다.

## 구현 의사코드

### 학습용 dynamic vocabulary

```python
def train_step(images, targets, dataset_vocab, text_encoder, detector):
    # Four-image mosaic labels provide positive phrases.
    positives = unique_phrases(targets)
    negatives = sample_negatives(
        dataset_vocab,
        exclude=positives,
        max_total=80,
    )
    prompts = positives + negatives

    with no_grad():
        text = text_encoder(prompts)       # [C, D], frozen CLIP
        text = l2_normalize(text, dim=-1)

    boxes, obj_embed = detector(images, text)
    obj_embed = l2_normalize(obj_embed, dim=-1)  # [B, N, D]
    logits = alpha * matmul(obj_embed, text.T) + beta

    loss_con = region_text_loss(logits, targets, prompts)
    loss_box = iou_loss(boxes, targets) + dfl_loss(boxes, targets)
    loss = loss_con + targets.box_weight * loss_box
    loss.backward()
```

### 배포용 prompt cache

```python
class PromptThenDetect:
    def set_vocabulary(self, prompts):
        text = clip_text_encoder(prompts)
        text = normalize(text)
        self.class_kernel = model.reparameterize(text)
        self.prompts = prompts

    def infer_frame(self, image):
        raw = image_only_yolo(image, class_kernel=self.class_kernel)
        keep = confidence_filter(raw)
        return nms(keep)
```

실제 구현에서는 `set_vocabulary` 시간, cache 파일 크기, kernel swap 동기화, 첫 추론 warm-up을 별도 기록해야 한다.

## 온디바이스 배포 분석

### 적합한 실행 구조

```text
app start or vocabulary change
  -> tokenize prompts
  -> text encode
  -> re-parameterize and warm up

camera loop
  -> resize and normalize
  -> image-only YOLO path
  -> confidence filtering
  -> NMS
  -> track or trigger downstream segmenter/VLM
```

이 구조라면 text encoder는 상시 메모리에 둘 필요가 없다. vocabulary preset의 kernel을 파일로 저장하거나 앱 서버에서 생성해 전달하는 방식도 가능하다. 다만 모델 배포와 prompt 생성 환경의 tokenizer, CLIP checkpoint, normalization이 정확히 같아야 한다.

### 측정해야 할 지표

- batch=1 p50, p95 latency
- cold start와 warm latency
- prompt encode 및 re-parameterization latency
- vocabulary 20, 80, 1,203개에서 latency와 peak memory
- NMS 포함 end-to-end latency
- CPU, GPU, NPU별 operator fallback
- 5분 이상 연속 실행 시 전력, 온도, throttling
- detector AP와 rare/common/frequent class AP
- prompt 동의어와 template에 따른 정확도 편차

### 양자화 주의점

Vision backbone과 neck은 INT8 후보지만 cosine similarity 전후 normalization은 backend 지원을 확인해야 한다. Reparameterized classifier는 일반 convolution으로 바뀌므로 INT8 적용이 더 쉽다. 그러나 class별 kernel range가 크게 다르면 per-tensor quantization 오차가 커질 수 있다. 가능하면 per-channel weight quantization을 쓰고 다음을 비교한다.

```text
FP16 full model
FP16 reparameterized
INT8 backbone + FP16 head
full INT8 reparameterized
```

Open-vocabulary 평가에서는 양자화 전후 전체 AP뿐 아니라 rare AP와 prompt별 logit ranking 변화를 봐야 한다. 작은 cosine margin이 INT8 rounding으로 뒤집히기 쉽기 때문이다.

## 강점

1. YOLO의 실용적인 one-stage 경로와 open-vocabulary learning을 직접 결합했다.
2. Fixed vocabulary에서 text branch를 frame loop 밖으로 옮기는 시스템 설계가 명확하다.
3. Detection, grounding, image-text pseudo data를 한 region-text objective로 통합한다.
4. LVIS rare class와 속도 사이의 강한 trade-off를 보여 준다.
5. Reparameterization 전후 parameter와 FPS를 함께 보고해 비용의 출처를 드러낸다.
6. Fine-tuning하면 일반 closed-set YOLO로도 경쟁력 있는 initialization을 제공한다.

## 한계와 해석 시 주의점

### 논문이 직접 보여 주지 않은 것

- V100 FPS는 모바일 latency, p95, 전력, 발열을 대신하지 않는다.
- 실제 prompt 변경 latency와 re-parameterization latency가 주요 표에 없다.
- 매우 큰 vocabulary에서 logit memory와 NMS 비용을 충분히 분석하지 않는다.
- pseudo-label pipeline의 오류 유형과 teacher bias 분석이 제한적이다.
- text phrase가 모호하거나 관계 중심일 때 실패하는 정성적 범위가 충분하지 않다.

### `open-vocabulary`의 경계

모델이 어떤 단어든 text embedding으로 받을 수 있다는 것과 그 대상을 안정적으로 검출한다는 것은 다르다. 성능은 CLIP의 language-vision prior, pre-training 데이터에 등장한 visual concept, prompt template에 의존한다. 완전히 새로운 fine-grained industrial class는 few-shot visual adaptation이나 추가 distillation이 필요할 수 있다.

### Reparameterization의 경계

재매개변수화는 arbitrary prompt를 무료로 처리한다는 뜻이 아니다. 특정 vocabulary에 대해 만들어진 classifier를 빠르게 실행한다는 뜻이다. 동적 질의 시스템에서는 vocabulary update를 별도 stage로 설계해야 한다.

## 재현 체크리스트

### 데이터와 prompt

- [ ] Objects365, GQA, Flickr, GoldG의 실제 사용 split을 기록한다.
- [ ] CC3M pseudo-label 생성 checkpoint와 threshold를 고정한다.
- [ ] noun phrase extraction 규칙과 중복 제거 규칙을 저장한다.
- [ ] negative sampling seed와 `M=80` 제한을 확인한다.
- [ ] LVIS fixed AP의 prompt template를 동일하게 사용한다.
- [ ] 동의어 병합과 plural 처리 규칙을 기록한다.

### 모델과 학습

- [ ] YOLOv8 S/M/L backbone 및 neck 폭을 공식 설정과 맞춘다.
- [ ] CLIP text encoder를 freeze했는지 확인한다.
- [ ] alpha, beta와 feature normalization 위치를 확인한다.
- [ ] T-CSPLayer와 Image-Pooling Attention을 각각 ablate한다.
- [ ] image-text 데이터의 `lambda_I=0` 적용 범위를 확인한다.
- [ ] AdamW, lr 0.002, wd 0.05, 100 epochs, batch 512를 기록한다.

### 평가와 배포

- [ ] original과 reparameterized model의 AP 동등성을 검증한다.
- [ ] text encoder 포함 latency와 제외 latency를 모두 측정한다.
- [ ] NMS 포함 여부를 명확히 표시한다.
- [ ] 입력 640, batch=1, 동일 precision으로 비교한다.
- [ ] vocabulary 크기를 바꾸며 peak activation을 측정한다.
- [ ] mobile backend의 fallback log를 검사한다.
- [ ] p50/p95, 전력, 온도, throttling을 기록한다.

## 추천 ablation

온디바이스 연구에서 가장 유용한 실험은 `vocabulary size x precision x update frequency` 3축 ablation이다.

| 조건 | 권장 값 |
|---|---|
| vocabulary size | 20, 80, 300, 1,203 |
| precision | FP16, mixed INT8/FP16, full INT8 가능 범위 |
| prompt update | 시작 시 1회, 10초마다, 매 frame |
| 측정 | AP, rare AP, p50/p95, peak memory, update latency, power |

이 실험은 논문의 V100 FPS만으로 알 수 없는 실제 사용 경계를 보여 준다. 특히 매 frame 업데이트 조건에서 text encoder와 kernel rebuild가 병목이 되고, 큰 vocabulary에서는 `[N,C]` activation과 후처리가 병목이 될 가능성이 높다.

## 최종 평가

YOLO-World의 핵심 성과는 open-vocabulary detection을 단순히 `빠른 backbone + text classifier`로 만든 것이 아니다. Text-guided neck, region-text 학습, offline vocabulary 재매개변수화를 함께 설계해 detector의 semantic 확장성과 frame latency를 분리했다. 그 결과 Grounding DINO류를 teacher/reference로 두고, 상시 실행 가능한 student detector를 만들려는 연구에 매우 좋은 출발점이 된다.

온디바이스 관점에서 가장 중요한 교훈은 세 가지다.

1. 반복되는 text embedding은 cache하고 frame loop에서 제거한다.
2. FLOPs와 parameter뿐 아니라 vocabulary에 비례하는 `[N,C]` activation을 계산한다.
3. 실제 비교는 NMS, prompt update, p95, peak memory, 전력까지 포함한다.

이 조건을 지키면 YOLO-World는 `텍스트 질의 -> box -> mask -> VLM 응답` 파이프라인에서 상시 또는 이벤트 기반 box proposal 모듈로 실용적인 가치가 크다.

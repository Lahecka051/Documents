# 72. YOLOE

## 논문 정보

- 원본 파일: `72_YOLOE.pdf`
- 제목: **YOLOE: Real-Time Seeing Anything**
- 저자: Ao Wang, Lihao Liu, Hui Chen, Zijia Lin, Jungong Han, Guiguang Ding
- 공개: arXiv:2503.07465, PDF 기준 v2, 2025-10-17
- 공식 링크: [arXiv:2503.07465](https://arxiv.org/abs/2503.07465)
- 태스크: text-prompt detection, visual-prompt detection, prompt-free open-vocabulary detection, instance segmentation
- 핵심 키워드: RepRTA, SAVPE, LRPC, YOLOv8, YOLO11, re-parameterization, visual prompting, lazy region-prompt contrast

## 한눈에 보는 요약

YOLOE는 하나의 실시간 YOLO 계열 모델이 세 가지 질의 방식을 모두 처리하도록 확장한다.

```text
text prompt       -> "person, helmet, forklift" -> detection + masks
visual prompt     -> exemplar image or region    -> detection + masks
prompt-free mode  -> all recognizable objects    -> detection + masks
```

이를 위해 서로 다른 병목에 대응하는 세 모듈을 제안한다.

1. **RepRTA**: text embedding을 경량 auxiliary network로 보정하고 detection kernel에 합성한다. 배포 시 일반 YOLO class convolution과 같은 경로가 된다.
2. **SAVPE**: binary mask로 지정한 visual exemplar에서 semantic feature를 뽑되, 하나의 평균 vector가 아니라 channel-group별 공간 가중치를 사용한다.
3. **LRPC**: prompt-free 모드에서 4,585개 built-in category와 모든 후보를 곧바로 비교하지 않는다. 먼저 class-agnostic special prompt로 후보를 거른 뒤 살아남은 region만 vocabulary와 비교한다.

LVIS minival zero-shot fixed AP에서 YOLOE-v8-L의 text/visual prompt 성능은 각각 `35.9/34.2 AP`다. TensorRT를 사용한 T4 처리량은 `102.5 FPS`, CoreML iPhone 12 측정은 `27.2 FPS`다. 논문은 실제 iPhone 수치를 제공한다는 점에서 의미가 크지만 p95 latency, peak memory, 전력, 발열은 보고하지 않는다.

## YOLO-World에서 무엇을 더 해결하는가

YOLO-World는 text vocabulary를 빠른 YOLO head로 바꾸는 데 성공했지만 다음 문제는 남는다.

- Text prompt는 편하지만 이름을 모르거나 말로 설명하기 어려운 물체를 지정하기 어렵다.
- Visual exemplar를 넣으려면 별도 prompt encoder가 필요하다.
- `모든 것을 찾아라`라는 prompt-free 요구에서는 거대한 vocabulary와 모든 region을 비교해야 한다.
- Vision-language fusion 모듈이 남아 있으면 mobile latency가 증가한다.

YOLOE는 이 세 사용 모드를 단일 detection/segmentation framework로 묶는다. 여기서 `seeing anything`은 무한한 semantic understanding을 뜻하지 않는다. Text와 visual prompt는 주어진 대상을 찾고, prompt-free mode는 내장 vocabulary에 mapping 가능한 대상을 찾는 구조다.

## 기본 아키텍처와 출력

Backbone은 YOLOv8 또는 YOLO11이고 PAN neck과 decoupled head를 사용한다. Detection 외에 YOLACT 계열 prototype mask 방식을 붙인다.

```text
image [B, 3, H, W]
  -> YOLO backbone
  -> PAN multi-scale features
  -> box head
  -> object embedding head O [B, N, D]
  -> mask coefficient head A [B, N, K]
  -> prototype branch Pm [B, K, Hm, Wm]

mask for region n = sigmoid(sum_k A[n,k] * Pm[k,:,:])
```

Text prompt matrix를 `P in R^{C x D}`, object embedding을 `O in R^{N x D}`라 하면 class logit은 다음과 같다.

```math
\operatorname{Label}=OP^{\top}
\in\mathbb{R}^{N\times C}
```

배치와 pyramid level을 포함하면 object embedding은 `[B,N,D]`, class logit은 `[B,N,C]`다. 640 입력과 stride 8, 16, 32를 가정한 리뷰어 계산에서는 `N=8,400`이다.

Instance mask는 box마다 full mask tensor를 직접 출력하지 않고, 공통 prototype과 box별 coefficient를 결합한다. 메모리를 줄이는 장점이 있지만 prototype resolution, coefficient 수 `K`, crop/resize 후처리가 latency와 작은 객체 mask 품질에 영향을 준다.

## RepRTA: Re-parameterizable Region-Text Alignment

### 학습 경로

Text prompt `P`는 MobileCLIP-B(LT)가 만든 embedding이다. 논문은 작은 SwiGLU FFN `f_theta`를 추가해 detection task에 맞게 prompt를 보정한다.

Object embedding head의 마지막 convolution kernel을 다음처럼 두자.

```text
input feature I: [B, D', H, W]
kernel K:        [D, D', 1, 1]
object map:      [B, D, H, W]
```

Training-time score는 개념적으로 다음과 같다.

```math
\operatorname{Label}
=
\operatorname{reshape}(I*K)
f_{\theta}(P)^{\top}
```

`f_theta`가 text embedding을 detection feature space에 맞춘다. YOLO-World처럼 image-text fusion block을 여러 곳에 유지하는 대신 text-side adaptation에 집중한다.

### 추론 시 kernel 합성

두 linear operation은 합성할 수 있다. 보정된 prompt matrix와 object projection kernel을 미리 곱해 category-specific 1x1 convolution을 만든다.

```math
K'
=
\operatorname{Compose}(f_{\theta}(P),K)
\in\mathbb{R}^{C\times D'\times1\times1}
```

이후 추론은 다음 한 줄이 된다.

```math
\operatorname{Label}=I*K'
```

```text
training:
  feature -> object embedding -> similarity with adapted text

inference after re-parameterization:
  feature -> ordinary C-channel 1x1 convolution
```

이 방식의 장점은 명확하다.

- Text encoder와 SwiGLU를 frame loop 밖으로 이동한다.
- Runtime graph에서 custom region-text matrix multiplication을 없앤다.
- 일반 YOLO class head와 같은 operator pattern이 되어 TensorRT/CoreML 최적화가 쉽다.
- Fixed vocabulary에서는 추가 fusion latency 없이 open-vocabulary pre-training의 이점을 쓸 수 있다.

단, 새로운 prompt가 들어오면 `P`, `f_theta(P)`, `K'`를 다시 계산해야 한다. 논문 FPS는 이미 준비된 vocabulary의 frame inference에 해당한다고 해석해야 한다.

### RepRTA ablation의 의미

논문의 development path는 다음과 같다.

| 설정 | LVIS AP | T4 FPS | iPhone 12 FPS |
|---|---:|---:|---:|
| YOLO-Worldv2-L, 100 epochs | 33.0 | 80.0 | 22.1 |
| 30 epochs | 31.0 | - | - |
| global negatives | 31.9 | - | - |
| fusion 제거 | 30.0 | 102.5 | 27.2 |
| MobileCLIP | 31.5 | 102.5 | 27.2 |
| RepRTA | 33.5 | 102.5 | 27.2 |
| segmentation 추가 | 33.3 | 102.5 | 27.2 |

Fusion을 제거하면 AP가 떨어지지만 속도가 크게 오른다. MobileCLIP과 RepRTA가 그 정확도를 회복한다. 따라서 RepRTA는 단순 추가 module이 아니라 `fusion 없이도 정확도를 유지하기 위한 train-time adapter`다.

## SAVPE: Semantic-Activated Visual Prompt Encoder

### Visual prompt의 입력

사용자는 exemplar image에서 대상 영역을 binary mask로 지정할 수 있다.

```text
reference image -> semantic feature map S
binary mask     -> prompt activation
                 -> visual prompt embedding P_v
```

가장 단순한 방법은 mask 안 feature를 평균 pooling하는 것이다. 그러나 물체 내부에는 배경, texture, part, boundary가 섞이고 channel마다 중요한 위치가 다를 수 있다. SAVPE는 하나의 공간 weight를 모든 channel에 공유하지 않고 `A`개 group으로 나눈다.

### Semantic feature branch

P3, P4, P5 feature에 각각 두 개의 `3 x 3` convolution을 적용하고 동일 resolution으로 upsample한 뒤 concatenate/project한다.

```text
P3 -- 3x3 -- 3x3 --+
P4 -- 3x3 -- 3x3 --+-> resize + concat + projection -> S [B, D, Hs, Ws]
P5 -- 3x3 -- 3x3 --+
```

이 branch는 localization에 쓰는 PAN feature와 별도로 visual prompt를 위한 semantic activation map을 만든다.

### Group-wise spatial aggregation

Mask activation feature와 image low-dimensional feature를 다음처럼 둔다.

```math
F_V,F_I\in\mathbb{R}^{A\times H_s\times W_s}
```

둘을 결합해 group별 spatial weight `W`를 만들고, prompt 영역 안에서 softmax한다.

```math
W\in\mathbb{R}^{A\times H_s\times W_s}
```

Semantic map `S in R^{D x H_s x W_s}`의 channel을 `A`개 group으로 나눈다. 각 group 크기는 `D/A`다.

```math
G_i
=
W_i
S_{iD/A:(i+1)D/A}^{\top}
```

그룹 결과를 이어 붙여 prompt embedding을 만든다.

```math
P_v=\operatorname{Concat}(G_1,\ldots,G_A)
\in\mathbb{R}^{D}
```

논문의 기본값은 `A=16`이다. `A=1`이면 일반적인 하나의 spatial attention과 유사하고, `A=D`에 가까워질수록 channel별 attention이 되어 비싸고 불안정할 수 있다. `A=16`은 표현력과 비용의 절충이다.

### Shape 예시

예를 들어 `D=512`, `A=16`, `H_s=W_s=80`이라면:

```text
semantic S:       [512, 80, 80]
spatial weights:  [16, 80, 80]
channels/group:   32
output prompt:    [512]
```

FP16 `S` 하나는 `512 x 80 x 80 x 2 = 6,553,600 bytes`, 약 `6.25 MiB`다. 이는 설명을 위한 가정이며 실제 YOLOE S/M/L의 `D`, prompt resolution과 activation reuse를 export graph에서 확인해야 한다. Visual prompt mode에서는 이 branch의 peak activation이 text-only reparameterized path보다 클 수 있다.

### SAVPE ablation

| visual prompt encoder | AP |
|---|---:|
| mask pooling | 30.4 |
| SAVPE | 31.9 |

Group 수 ablation은 `A=1`에서 `30.9 AP`, `A=16`에서 `31.9 AP`, `A=32`에서도 `31.9 AP`다. 16 이후 전체 AP는 포화하고 rare AP가 오히려 낮아질 수 있어 `A=16`이 합리적이다.

Visual evaluation은 category마다 기본 16개의 training image를 사용하며 GT box에서 얻은 embedding을 평균한다. 이는 strict one-shot이 아니다. 배포에서 exemplar 한 장만 받는 경우와 16장 prototype bank를 쓰는 경우를 분리해 평가해야 한다.

## LRPC: Lazy Region-Prompt Contrast

### Prompt-free 모드의 계산 문제

Prompt-free라고 해도 모델이 label을 붙이려면 reference vocabulary가 필요하다. 논문은 4,585개 built-in category name을 사용한다. 모든 `N` region을 모든 category와 비교하면 logit은 `[N,4585]`다.

640 입력에서 `N=8,400`을 가정하면:

```math
8{,}400\times4{,}585=38{,}514{,}000\text{ logits}
```

FP16이면:

```math
38{,}514{,}000\times2
\approx73.46\text{ MiB}
```

이는 logit tensor만의 크기다. 모바일에서는 이 dense 비교가 prompt-free 기능을 사실상 막을 수 있다.

### Lazy filtering

LRPC는 `모든 object`를 나타내도록 학습한 special prompt `P_s`를 먼저 사용한다.

```math
O'
=
\{o\in O\mid oP_s^{\top}>\delta\}
```

Threshold를 넘은 region만 4,585개 vocabulary와 비교한다.

```text
all N regions
  -> one special-prompt score per region
  -> threshold delta
  -> M active regions, M << N
  -> M x 4585 vocabulary similarity
```

예를 들어 `M=100`개만 남는다면 dense logit은 `458,500`개, FP16 약 `0.875 MiB`다. 이 `M=100`은 논문 보고값이 아니라 lazy 구조의 효과를 보이는 예시다. 실제 `M`은 장면, threshold, object density에 따라 달라지므로 p95 active-region 수를 측정해야 한다.

### Threshold ablation

YOLOE-v8-S에서 LRPC 없이 `21.0 AP`, `56.5 FPS`다. `delta=0.001`은 같은 `21.0 AP`에서 `95.8 FPS`를 낸다. `0.0001`은 `66.1 FPS`, `0.01`은 `20.8 AP`, `106 FPS`다.

```text
lower delta -> more regions -> higher recall, slower
higher delta -> fewer regions -> faster, possible recall loss
```

v8-L은 LRPC 없이 `19.9 FPS`, LRPC 적용 시 `25.3 FPS`이며 AP는 유지된다. Default `delta=0.001`은 정확도와 속도의 균형점이다. 하지만 평균 FPS만으로는 복잡한 crowd 장면의 latency tail을 알 수 없다.

## 학습 데이터와 단계

Text prompt pre-training에는 Objects365와 GoldG를 사용한다. Segmentation mask는 ground-truth box를 SAM 2.1에 넣어 pseudo mask를 만들고 filtering 및 polygon simplification을 수행한다.

```text
GT boxes
  -> SAM 2.1 box prompts
  -> candidate masks
  -> quality filtering
  -> contour simplification
  -> instance segmentation supervision
```

학습은 mode별로 나뉜다.

1. Text prompt 학습: `30 epochs`
2. Visual prompt 학습: 나머지 모델을 freeze하고 SAVPE를 `2 epochs`
3. Prompt-free 학습: special embedding을 `1 epoch`

YOLOE-v8 S/M/L을 RTX 4090 8장으로 학습한 시간은 각각 `12`, `17`, `22.5`시간이다. 짧은 단계별 학습은 mode 추가 비용을 줄이지만, 서로 독립적으로 freeze한 branch가 joint optimum을 보장하지는 않는다.

COCO transfer에서는 linear probing과 full fine-tuning을 모두 수행한다.

- linear probe: 10 epochs
- full fine-tuning S: 160 epochs
- full fine-tuning M/L: 80 epochs

## 주요 결과

### LVIS text 및 visual prompt detection

PDF Table 1의 zero-shot fixed AP와 장치 처리량을 요약한다. Params는 text mode와 visual mode가 다르다.

| 모델 | text/visual params | T4 TensorRT FPS | iPhone 12 CoreML FPS | text AP | visual AP | text/visual APr |
|---|---:|---:|---:|---:|---:|---:|
| YOLOE-v8-S | 12M/13M | 305.8 | 64.3 | 27.9 | 26.2 | 22.3/21.3 |
| YOLOE-v8-M | 27M/30M | 156.7 | 41.7 | 32.6 | 31.0 | - |
| YOLOE-v8-L | 45M/50M | 102.5 | 27.2 | 35.9 | 34.2 | 33.2/33.2 |
| YOLOE-11-L | 26M/32M | 130.5 | 35.1 | 35.2 | 33.7 | - |

비교 YOLO-Worldv2 S/M/L의 AP는 `24.4/32.4/35.5`, T4 FPS는 `216.4/117.9/80.0`, iPhone FPS는 `48.9/34.2/22.1`이다. 특히 S와 L에서 YOLOE가 정확도와 처리량을 함께 개선한다.

### Zero-shot instance segmentation

LVIS val의 mask AP는 다음과 같다.

| YOLOE-v8 | text mask AP | visual mask AP |
|---|---:|---:|
| S | 17.7 | 16.8 |
| M | 20.8 | 20.3 |
| L | 23.5 | 22.0 |

Visual prompt가 text보다 약간 낮지만 차이는 크지 않다. 다만 이 결과는 category당 16개 exemplar 평균이라는 평가 조건을 함께 적어야 한다.

### Prompt-free detection

| 모델 | params | AP | PyTorch T4 FPS |
|---|---:|---:|---:|
| YOLOE-v8-S | 13M | 21.0 | 95.8 |
| YOLOE-v8-M | 29M | 24.7 | 45.9 |
| YOLOE-v8-L | 47M | 27.2 | 25.3 |
| GenerateU Swin-T | 297M | 26.8 | 0.48 |

YOLOE-L은 GenerateU와 비슷하거나 높은 AP를 훨씬 작은 모델과 빠른 경로로 달성한다. 단, YOLOE의 built-in vocabulary와 비교 모델의 label generation protocol이 같은지 확인해야 공정한 시스템 비교가 된다.

### COCO fine-tuning

YOLOE-v8-L은 80 epoch full fine-tuning에서 box `53.0 AP`, mask `42.7 AP`를 보고한다. Scratch YOLOv8-L 300 epoch의 `52.4/42.3 AP`보다 약간 높다. Open-vocabulary pre-training이 closed-set transfer initialization으로도 유효하고 학습 epoch를 줄였다는 결과다.

## 모델 크기 계산

논문 parameter 수를 weight 저장량으로 단순 환산하면 다음과 같다.

| 모델/모드 | params | FP16 | INT8 |
|---|---:|---:|---:|
| v8-S text | 12M | 22.9 MiB | 11.4 MiB |
| v8-M text | 27M | 51.5 MiB | 25.7 MiB |
| v8-L text | 45M | 85.8 MiB | 42.9 MiB |
| v8-L visual | 50M | 95.4 MiB | 47.7 MiB |
| 11-L text | 26M | 49.6 MiB | 24.8 MiB |

예를 들어 45M FP16은:

```math
45\times10^6\times2/2^{20}\approx85.8\text{ MiB}
```

Visual mode의 추가 parameter보다 semantic feature activation이 peak memory에 더 큰 영향을 줄 수 있다. Prompt-free mode에서는 parameter보다 active region 수와 vocabulary logit이 더 중요하다.

## 구현 의사코드

### Text prompt kernel 생성

```python
def build_text_kernel(prompts, mobileclip, adapter, object_kernel):
    # Run when the vocabulary changes, not for every frame.
    p = mobileclip.encode_text(prompts)       # [C, D]
    p = l2_normalize(p, dim=-1)
    p_adapt = adapter(p)                      # SwiGLU, [C, D]
    # Compose [C, D] with [D, D', 1, 1].
    k_rep = compose_1x1(p_adapt, object_kernel)
    return k_rep                              # [C, D', 1, 1]
```

### SAVPE

```python
def savpe(pyramid, binary_mask, groups=16):
    semantic = semantic_branch(pyramid)       # [B, D, Hs, Ws]
    mask_f = mask_activation(binary_mask)     # [B, A, Hs, Ws]
    image_f = image_activation(semantic)      # [B, A, Hs, Ws]
    weights = masked_softmax(mask_f + image_f, binary_mask)

    chunks = semantic.chunk(groups, dim=1)
    vectors = []
    for i, chunk in enumerate(chunks):
        vectors.append(spatial_sum(chunk * weights[:, i:i+1]))
    return concat(vectors, dim=1)              # [B, D]
```

### LRPC

```python
def prompt_free_classify(object_embeddings, special_prompt,
                         builtin_vocab, delta=1e-3):
    objectness = object_embeddings @ special_prompt.T  # [N, 1]
    active = objectness[:, 0] > delta
    selected = object_embeddings[active]               # [M, D]
    labels = selected @ builtin_vocab.T                 # [M, 4585]
    return active, labels
```

Production 구현에서는 dynamic shape `M`이 accelerator에서 비효율적일 수 있다. Top-K 고정 buffer, prefix-sum gather, CPU fallback 여부를 비교해야 한다.

## 온디바이스 관점

### Text prompt 모드

가장 배포하기 쉽다. Vocabulary가 고정되면 RepRTA를 합성한 일반 convolution graph로 export할 수 있다. MobileCLIP은 앱 시작 또는 prompt 변경 때만 실행하고 해제할 수 있다.

```text
prompt update path: text encoder -> adapter -> kernel compose -> warm-up
frame path: image -> YOLO -> NMS/mask decode
```

측정 시 prompt update latency를 숨기지 말고 별도로 보고해야 한다.

### Visual prompt 모드

SAVPE가 reference feature를 만들 때는 detector image branch와 다른 비용이 생긴다. 같은 exemplar를 반복 사용하면 prompt embedding을 cache해야 한다. Video에서 첫 frame의 mask를 prompt로 쓰는 경우 embedding drift와 appearance change를 다룰 tracking/update 정책이 필요하다.

### Prompt-free 모드

LRPC의 평균 속도는 빠르지만 장면 복잡도에 따라 `M`이 달라진다. 빈 방, 거리 crowd, 상품 진열대를 따로 측정해 `M`의 median과 p95를 기록해야 한다. Dynamic gather와 4,585-way classifier가 NPU에서 지원되지 않으면 CPU fallback으로 논문 기대치가 무너질 수 있다.

### iPhone 수치 해석

논문은 CoreML iPhone 12 FPS를 제공하지만 다음 정보는 별도로 필요하다.

- ANE, GPU, CPU 중 실제 실행 unit
- batch=1 latency distribution
- preprocessing, NMS, mask decoding 포함 여부
- cold start와 model load time
- device temperature와 sustained throughput
- text/visual prompt generation 포함 여부

`27.2 FPS`를 단순 역수로 바꾸면 frame당 약 `36.8 ms`지만, 이는 평균 처리량의 역수일 뿐 p95 end-to-end latency가 아니다.

## 양자화와 operator 지원

RepRTA 후 text mode는 convolution 중심이므로 INT8 친화적이다. 반면 다음 연산은 backend를 확인해야 한다.

- Visual prompt의 masked softmax와 group-wise reduction
- Prompt-free gather/scatter와 dynamic `M`
- Mask prototype과 coefficient matmul
- NMS 및 mask crop/upsample
- Reparameterized class kernel의 per-channel scale

권장 precision 분할은 다음과 같다.

```text
backbone and PAN: INT8
box and class convolution: INT8 or FP16 comparison
mask prototype: FP16 if boundary quality drops
visual prompt softmax/reduction: FP16
text encoder: offline FP16, or server/cache
```

Rare class는 logit margin이 작을 수 있으므로 전체 AP뿐 아니라 APr와 visual prompt retrieval ranking을 양자화 전후 비교해야 한다.

## 강점

1. Text, visual, prompt-free 세 인터페이스를 하나의 빠른 detector에 통합한다.
2. RepRTA가 정확도 회복과 deployment graph 단순화를 동시에 해결한다.
3. LRPC는 거대 vocabulary의 activation memory와 연산 문제를 직접 다룬다.
4. T4뿐 아니라 iPhone 12 CoreML 처리량을 보고한다.
5. Detection과 instance segmentation을 같은 prompt representation으로 수행한다.
6. 단계별 ablation이 각 모듈의 정확도와 속도 기여를 비교적 명확히 보여 준다.

## 한계와 해석 주의점

### Visual prompt의 평가 조건

기본 평가가 category당 16개 training exemplar를 평균하므로 single-click 또는 one-shot 성능과 다르다. Exemplar selection bias, mask 품질, 배경 변화에 대한 분석이 더 필요하다.

### Prompt-free는 vocabulary-free가 아니다

4,585개 category name이 label space를 제공한다. 내장 목록 밖 개념은 special prompt가 object region을 살려도 최종 label을 제대로 붙이지 못할 수 있다. Unknown 처리 또는 embedding 반환이 필요하다.

### Pseudo mask 의존

SAM 2.1이 만든 mask를 supervision으로 쓰므로 teacher의 boundary 오류, 작은 객체 누락, box 안 다중 객체 혼합이 student에 전달될 수 있다. SAM teacher와 YOLOE student를 같은 데이터에서 비교할 때 leakage와 annotation protocol을 명시해야 한다.

### 실시간 지표의 빈칸

평균 FPS는 있지만 p95, peak memory, prompt update time, energy, sustained thermal behavior가 없다. 특히 visual mode와 prompt-free dynamic region 수는 tail latency 분석이 중요하다.

### 버전과 비교 공정성

PDF는 arXiv v2이며 일부 비교 모델의 공개 checkpoint, TensorRT 설정, prompt template가 다를 수 있다. 동일 input, precision, postprocessing으로 다시 benchmark해야 한다.

## 재현 체크리스트

### 모델

- [ ] YOLOv8 또는 YOLO11 variant와 exact width/depth를 기록한다.
- [ ] Object embedding dimension `D`와 input channel `D'`를 확인한다.
- [ ] MobileCLIP-B(LT) checkpoint와 tokenizer를 고정한다.
- [ ] RepRTA SwiGLU 구조와 normalization 순서를 확인한다.
- [ ] SAVPE group `A=16`과 prompt feature resolution을 기록한다.
- [ ] LRPC built-in vocabulary 4,585개와 `delta=0.001`을 확인한다.

### 데이터

- [ ] Objects365와 GoldG split을 동일하게 사용한다.
- [ ] SAM 2.1 pseudo-mask 생성 및 filtering 기준을 저장한다.
- [ ] Visual prompt의 category당 exemplar 수를 명시한다.
- [ ] Text prompt template와 동의어 처리 규칙을 고정한다.
- [ ] COCO linear probe와 full fine-tuning epoch를 구분한다.

### 동등성 검증

- [ ] RepRTA 합성 전후 logit max error를 측정한다.
- [ ] 합성 전후 AP가 같은지 확인한다.
- [ ] PyTorch, TensorRT, CoreML box 좌표와 NMS 차이를 검사한다.
- [ ] Mask prototype decode의 resize/crop 순서를 맞춘다.
- [ ] Prompt-free active region 수 `M`의 분포를 기록한다.

### 온디바이스

- [ ] batch=1, 고정 해상도에서 p50/p95를 측정한다.
- [ ] preprocessing, NMS, mask decode 포함 end-to-end 수치를 낸다.
- [ ] text prompt update와 visual prompt encode 시간을 별도 측정한다.
- [ ] peak CPU/GPU/NPU memory를 기록한다.
- [ ] 5분 이상 전력, 온도, throttling을 측정한다.
- [ ] fallback operator와 execution provider를 확인한다.

## 추천 ablation

YOLOE를 실제 제품에 평가할 때는 mode별 cache 전략을 비교하는 것이 좋다.

| 모드 | 조건 | 측정할 항목 |
|---|---|---|
| Text | prompt 매 시작 시 1회, 10초마다, 매 frame | update latency, FPS, AP |
| Visual | exemplar 1개, 4개, 16개 | prompt encode time, memory, visual AP |
| Prompt-free | delta 1e-4, 1e-3, 1e-2 | active M, recall, p95, memory |
| Segmentation | prototype FP16/INT8 | mask AP, boundary quality, latency |

이 실험은 세 mode가 같은 parameter budget 안에 있다는 사실보다 중요한 `실행 시 실제로 어느 branch가 켜지는가`를 보여 준다.

## 최종 평가

YOLOE는 YOLO-World의 text-conditioned detector를 한 단계 확장해, 이름으로 지정하기 어려운 대상은 visual exemplar로 찾고, prompt가 없는 경우에는 lazy vocabulary matching으로 가능한 대상을 찾는다. 세 모듈은 각각 실제 배포 병목을 겨냥한다. RepRTA는 runtime fusion을 없애고, SAVPE는 visual prompt의 표현력을 높이며, LRPC는 큰 vocabulary의 dense 비교를 줄인다.

온디바이스 연구에서 특히 가치 있는 부분은 LRPC다. `N x C` 비용을 무조건 감수하지 않고 먼저 object-like region을 선별하는 구조는 detection 뒤 VLM으로 넘길 visual token을 줄이는 설계와도 연결된다. 다만 논문 결과를 제품 수준 실시간성으로 받아들이려면 prompt preparation, mask decoding, dynamic region tail, peak memory, 전력까지 포함한 재측정이 필요하다.

따라서 YOLOE는 `텍스트 또는 예시 영역 -> box와 mask -> 선택 ROI만 VLM` 파이프라인의 경량 front-end로 유망하며, Grounding DINO/SAM 같은 강한 teacher에서 증류할 student 구조로도 적합하다.

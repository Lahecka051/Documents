# 73. Florence-2

## 논문 정보

- 원본 파일: `73_Florence_2.pdf`
- 제목: **Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks**
- 저자: Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, Lu Yuan
- 소속: Microsoft Azure AI
- 공개: arXiv:2311.06242, PDF 기준 v1, 2023-11-10
- 공식 링크: [arXiv:2311.06242](https://arxiv.org/abs/2311.06242)
- 태스크: captioning, object detection, phrase grounding, referring expression, segmentation, OCR, region-to-text
- 핵심 키워드: unified sequence-to-sequence, FLD-5B, DaViT, spatial tokens, task prompt, multi-task pre-training

## 한눈에 보는 요약

Florence-2는 caption, detection, grounding, segmentation, OCR처럼 출력 형식이 서로 다른 vision task를 하나의 `image + task prompt -> token sequence` 문제로 바꾼 vision foundation model이다. 별도 detection head, segmentation head, caption decoder를 두는 대신 text token과 location token으로 모든 정답을 직렬화한다.

```text
image -> DaViT vision encoder -> visual tokens ----+
                                                   +-> Transformer encoder
task prompt -> text embeddings --------------------+
                                                        |
                                                        v
                                                Transformer decoder
                                                        |
                           text tokens + location tokens + EOS
```

예를 들면 같은 이미지에 prompt만 바꿔 다음 출력을 만든다.

```text
<CAPTION>              -> "a person riding a bicycle on a road"
<OD>                   -> label + <loc_x1><loc_y1><loc_x2><loc_y2> + ...
<CAPTION_TO_PHRASE>    -> phrase + corresponding location tokens
<REFERRING_SEGMENTATION> -> polygon location-token sequence
<OCR>                  -> recognized text + quadrilateral tokens
```

두 번째 핵심은 학습 데이터다. FLD-5B는 `126M` images와 `5.4B` annotations를 포함하고, image-level text, region-text, text-phrase-region을 함께 구성한다. Specialist model과 filtering을 조합해 대규모 synthetic annotation을 만든다.

모델은 Florence-2-B `0.23B`와 Florence-2-L `0.77B` 두 크기다. Zero-shot에서 B/L은 COCO caption `133.0/135.6 CIDEr`, COCO detection `34.7/37.5 AP`를 보고한다. 하나의 작은 generalist가 여러 task를 처리한다는 점은 강하지만, autoregressive location decoding과 232M 이상의 parameter 때문에 상시 모바일 실행에는 여전히 무겁다.

## 통합 표현이 필요한 이유

전통적인 multi-task vision 시스템은 task마다 출력 head가 다르다.

```text
shared image encoder
  + classification head
  + box regression head
  + mask decoder
  + caption language decoder
  + OCR recognition head
```

이 구조는 각 task에 최적화하기 쉽지만 task 수가 늘수록 head와 annotation interface가 복잡해진다. Detection은 unordered box set, segmentation은 dense mask, caption은 text sequence이므로 loss와 postprocessing도 다르다.

Florence-2는 모든 출력을 token sequence로 바꾼다.

```math
p(y\mid x,t)
=
\prod_{i=1}^{L_y}
p(y_i\mid y_{\lt i},x,t)
```

여기서 `x`는 image, `t`는 task prompt, `y`는 text와 location token의 혼합 sequence다. Task가 달라도 같은 decoder와 cross-entropy를 쓴다. 장점은 architecture와 interface 통합이고, 대가는 spatial output까지 autoregressive하게 생성해야 한다는 점이다.

## 지원 task와 입출력 형식

PDF Appendix Table 13을 기준으로 지원 범주를 정리하면 다음과 같다.

| Task | annotation type | prompt input | output |
|---|---|---|---|
| Caption | Text | image, text | text |
| Detailed caption | Text | image, text | text |
| More detailed caption | Text | image, text | text |
| Region proposal | Region | image, text | region |
| Object detection | Region-Text | image, text | text, region |
| Dense region caption | Region-Text | image, text | text, region |
| Phrase grounding | Text-Phrase-Region | image, text | text, region |
| Referring expression comprehension | Region-Text | image, text | text, region |
| Open-vocabulary detection | Region-Text | image, text | text, region |
| Referring segmentation | Region-Text | image, text | text, region |
| Region to text | Region-Text | image, text, region | text |
| Text detection and recognition | Region-Text | image, text | text, region |

`prompt input`의 text는 자연어 질문일 수도 있고 task를 지정하는 special prompt일 수도 있다. Region-to-text는 입력에 location token도 들어간다. 즉 coordinate token은 출력 전용이 아니라 query interface로도 쓰인다.

## Spatial tokenization

### 1,000개 coordinate bin

연속 좌표를 그대로 regression하지 않고 각 축을 1,000개 bin으로 양자화한다. 이미지 폭 `W`, 높이 `H`에서 정규화 좌표를 다음처럼 생각할 수 있다.

```math
u_x=\mathrm{round}\left(999\frac{x}{W}\right),
\qquad
u_y=\mathrm{round}\left(999\frac{y}{H}\right)
```

```text
x coordinate -> <loc_0> ... <loc_999>
y coordinate -> same 1000-location vocabulary
```

정확한 clipping과 rounding은 구현을 따라야 하지만 핵심은 normalized location을 discrete token으로 바꾸는 것이다. 동일 vocabulary를 여러 spatial format에 재사용한다.

- Bounding box: 4 tokens, `(x1,y1,x2,y2)`
- Quadrilateral: 8 tokens, 네 꼭짓점
- Polygon: 가변 길이, 꼭짓점마다 2 tokens

1,000 bin의 normalized 양자화 step은 축 길이의 약 `1/999`다. 768 pixel 축에서는 약 `0.77 pixel`, 384에서는 약 `0.38 pixel` 수준이다. 따라서 bin 자체의 이론적 해상도보다 decoder의 token 오류, image encoder stride, polygon 단순화가 실제 boundary 오차를 더 크게 만들 가능성이 높다.

### Sequence 직렬화의 장단점

Detection 출력 예시는 다음처럼 직렬화할 수 있다.

```text
person <loc_41> <loc_15> <loc_63> <loc_73>
car    <loc_58> <loc_26> <loc_89> <loc_61> <eos>
```

장점:

- Category와 coordinate를 같은 language model loss로 학습한다.
- Variable number of objects를 EOS로 표현한다.
- Box, quadrilateral, polygon을 하나의 token vocabulary로 다룬다.
- Text와 region 관계가 sequence 안에서 명시된다.

단점:

- Object 순서가 본질적으로 unordered인데 sequence 순서를 정해야 한다.
- 앞 token 오류가 뒤 생성에 누적될 수 있다.
- Polygon이 복잡할수록 decoder step이 길어진다.
- Parallel dense head보다 latency가 output length에 비례한다.
- Invalid token order나 닫히지 않은 polygon을 parser가 처리해야 한다.

## 모델 아키텍처

### Image encoder

Vision backbone은 DaViT다. 입력 image를 multi-stage hierarchical feature로 바꾼 뒤 visual token sequence를 얻는다.

```math
V=\mathrm{DaViT}(I)
\in\mathbb{R}^{N_v\times D_v}
```

Linear projection과 LayerNorm으로 language hidden dimension `D`에 맞춘다.

```math
V'=\mathrm{LN}(VW_p)
\in\mathbb{R}^{N_v\times D}
```

Task prompt embedding을 `T_prompt in R^{N_t x D}`라 하면 encoder 입력은 concatenate된다.

```math
X_{enc}=[V';T_{prompt}]
\in\mathbb{R}^{(N_v+N_t)\times D}
```

논문은 이 visual token과 prompt token을 standard Transformer encoder-decoder에 넣는다. Figure 2에서 image embedding과 text/location embedding을 이어 붙이고, decoder가 text/location token을 생성하는 흐름을 확인할 수 있다.

### 모델 설정

PDF Appendix Table 15를 직접 렌더링해 확인한 설정은 다음과 같다.

| 모델 | DaViT dimensions | blocks | heads/groups | vision params | encoder/decoder layers | hidden D | Transformer params | total reported |
|---|---|---|---|---:|---:|---:|---:|---:|
| Florence-2-B | [128,256,512,1024] | [1,1,9,1] | [4,8,16,32] | 90M | 6/6 | 768 | 140M | 0.23B |
| Florence-2-L | [256,512,1024,2048] | [1,1,9,1] | [8,16,32,64] | 360M | 12/12 | 1024 | 410M | 0.77B |

표의 vision과 Transformer parameter를 단순 합치면 B는 230M, L은 770M으로 main table의 0.23B/0.77B와 맞는다. 일부 설명은 B 232M, L 771M처럼 embedding과 projection을 포함한 더 정확한 값을 쓸 수 있으므로 저장량 계산에서는 main parameter 수의 반올림 오차를 명시해야 한다.

### Decoder loss

모든 task는 teacher forcing 기반 token cross-entropy로 학습한다.

```math
\mathcal L
=
-\sum_{i=1}^{L_y}
\log p_{\theta}
(y_i\mid y_{\lt i},x)
```

Task별 별도 detection loss, IoU loss, mask BCE가 핵심 objective에 직접 등장하지 않는다. Spatial quality도 location-token likelihood를 통해 학습된다. 이는 통합의 원천인 동시에 box overlap을 직접 최적화하지 않는 한계다.

## FLD-5B 데이터

### 전체 규모

FLD-5B는 `126M images`, `5.4B annotations`를 포함한다. Annotation은 세 계층으로 구성된다.

```text
image-level text          약 0.5B
region-text               약 1.3B
text-phrase-region        약 3.6B
total                     약 5.4B
```

Source image는 ImageNet-22K, Objects365, OpenImages, Conceptual Captions, filtered LAION 등을 포함한다. 모든 annotation을 사람이 새로 단 것이 아니라 specialist model과 filtering pipeline으로 확장한다.

### Annotation 통계

PDF Table 2를 렌더링해 열 정렬을 확인한 수치는 다음과 같다.

| Annotation type | Text type | image annotations | avg tokens | regions | avg regions | avg regional tokens |
|---|---|---:|---:|---:|---:|---:|
| Text | Brief | 235M | 7.95 | - | - | - |
| Text | Detailed | 126M | 31.65 | - | - | - |
| Text | More detailed | 126M | 70.53 | - | - | - |
| Region-Text | Phrase | 126M | - | 681M | 5.42 | 1.19 |
| Region-Text | Brief | 126M | - | 681M | 5.42 | 2.55 |
| Text-Phrase-Region | Brief | 235M | 7.95 | 1,007M | 4.27 | 1.93 |
| Text-Phrase-Region | Detailed | 126M | 31.65 | 1,289M | 10.25 | 1.49 |
| Text-Phrase-Region | More detailed | 126M | 70.53 | 1,278M | 10.17 | 1.35 |

`126M images`와 `235M image annotations`가 동시에 나타나는 이유는 이미지당 여러 annotation type이나 version이 존재하기 때문이다. Row 수를 서로 더해 unique image 수로 해석하면 안 된다.

Detailed caption의 semantic statistics도 scale을 보여 준다.

| Text type | avg objects | avg attributes | avg actions | avg proper nouns |
|---|---:|---:|---:|---:|
| Brief | 3.23 | 2.80 | 0.58 | 1.10 |
| Detailed | 13.31 | 7.27 | 4.21 | 2.40 |
| More detailed | 28.06 | 16.25 | 8.76 | 2.41 |

길어진 caption이 단순 문장 반복이 아니라 object, attribute, action coverage를 늘리는 방향임을 보이려는 분석이다.

### Annotation 생성 pipeline

논문은 task별 specialist를 조합한다.

- Detection: DINO 계열 specialist
- OCR: Azure OCR API
- Grounding: Grounding DINO
- Segmentation: SAM
- Caption/detail expansion: caption model, LMM, LLM
- Filtering/scoring: Florence-1 계열과 task-specific consistency check

```text
source images
  -> specialist annotations
  -> format normalization
  -> confidence and consistency filtering
  -> unified text/location-token serialization
  -> train initial Florence-2
  -> use improved model for refinement
  -> repeat
```

이 구조는 annotation 비용을 줄이고 scale을 키우지만 teacher bias가 한 모델에 모인다. DINO가 놓친 object, SAM boundary 오류, OCR 언어 편향, caption hallucination이 downstream pre-training target이 될 수 있다. `5.4B`는 정답의 양이지 독립적인 human-verified fact의 수가 아니다.

## 학습 설정

- Vision initialization: UniCL
- Encoder-decoder initialization: BART
- Optimizer: AdamW
- Schedule: cosine decay, warm-up 5,000 steps
- Maximum learning rate B: `1e-4`
- Maximum learning rate L: `1e-5`
- Batch size B: `2,048`
- Batch size L: `3,072`
- Main pre-training resolution: `384 x 384`
- Main phase: effective samples `3B`
- High-resolution phase: `768 x 768`
- High-resolution B: effective samples `0.5B`
- High-resolution L: effective samples `0.1B`

L 모델의 batch가 B보다 크다는 수치는 직관과 다르지만 PDF에 그렇게 적혀 있으므로 그대로 기록한다. Gradient accumulation, hardware 수, sample 정의를 재현 코드에서 추가 확인해야 한다.

384에서 768로 올리면 pixel 수는 4배다. Vision token 수도 stride가 같다면 대략 4배이고 attention/workspace 부담이 크게 증가한다. L 모델의 high-resolution sample 수를 줄인 것은 이 비용을 반영한 것으로 볼 수 있다.

## Tensor shape와 메모리 분석

### Visual token 수

논문 configuration table은 DaViT stage를 주지만 최종 token 선택과 stride를 상세한 숫자로 고정하지 않는다. 아래는 최종 stride 32 feature를 그대로 token화한다고 가정한 리뷰어 계산이다.

```text
384 / 32 = 12 -> N_v = 12 x 12 = 144
768 / 32 = 24 -> N_v = 24 x 24 = 576
```

768 입력의 projected visual token FP16 메모리는:

```math
\text{B}: 576\times768\times2
\approx0.844\text{ MiB}
```

```math
\text{L}: 576\times1024\times2
=1.125\text{ MiB}
```

실제 encoder가 여러 stage token이나 다른 sampling을 쓰면 이 값은 달라진다. 이 계산은 output feature 하나의 lower-bound 성격이며 backbone 중간 activation은 포함하지 않는다.

### Encoder attention

Prompt를 32 token, visual token을 576개라 두면 encoder sequence length는 `608`이다. 한 attention head의 FP16 score matrix는:

```math
608^2\times2
=739{,}328\text{ bytes}
\approx0.705\text{ MiB}
```

Head와 layer 전체 score를 동시에 보존하면 더 커지지만 fused attention kernel은 materialization을 줄일 수 있다. 실제 peak memory는 backend에 따라 큰 차이가 난다.

### Decoder KV cache

Autoregressive self-attention의 layer별 key/value를 FP16으로 cache한다고 하자. 생성 token 하나당 cache 증가는 대략 다음과 같다.

```math
\text{B}:2\times6\times768\times2
=18{,}432\text{ bytes}=18\text{ KiB/token}
```

```math
\text{L}:2\times12\times1024\times2
=49{,}152\text{ bytes}=48\text{ KiB/token}
```

512 token을 생성하면 B 약 `9 MiB`, L 약 `24 MiB`다. Polygon이나 dense region caption처럼 출력이 길면 KV cache와 serial decoding latency가 함께 증가한다.

Encoder output에 대한 cross-attention key/value도 layer별 cache한다고 가정하면 `N_enc=608`에서:

```math
\text{B}:2\times6\times608\times768\times2
\approx10.69\text{ MiB}
```

```math
\text{L}:2\times12\times608\times1024\times2
\approx28.5\text{ MiB}
```

이는 구현 가정에 따른 계산이며 projection 공유나 cache layout에 따라 다르다.

### Weight memory

| 모델 | params | FP16 weight | INT8 weight |
|---|---:|---:|---:|
| Florence-2-B | 232M 근사 | 약 442.5 MiB | 약 221.3 MiB |
| Florence-2-L | 771M 근사 | 약 1.436 GiB | 약 735.3 MiB |

4-bit weight-only라면 이상적인 raw weight는 B 약 110.6 MiB, L 약 367.6 MiB지만 scale, zero-point, packing, FP16 layer, KV cache가 추가된다. L은 W4A16이어도 저사양 모바일에 가볍다고 보기 어렵다.

## 주요 결과

### Zero-shot generalist

PDF Table 4의 수치를 요약한다. 평가 task의 training data를 사용하지 않은 설정이다.

| 모델 | params | COCO cap CIDEr | NoCaps | TextCaps | COCO det AP | Flickr R@1 | RefCOCO RES mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Florence-2-B | 0.23B | 133.0 | 118.7 | 70.1 | 34.7 | 83.6 | 34.6 |
| Florence-2-L | 0.77B | 135.6 | 120.8 | 72.8 | 37.5 | 84.4 | 35.8 |

Scale을 3.3배 키워도 task별 이득은 대체로 수 AP 또는 수 CIDEr 안쪽이다. 작은 B가 상당히 강하다는 점이 이 논문의 실용적 결과다.

`zero-shot`은 FLD-5B에 평가 train split을 넣지 않았다는 의미다. Source dataset, specialist model, web image 중 평가 이미지나 매우 유사한 내용이 섞였는지 완전한 독립성을 보장하는 용어는 아니다.

### Supervised single generalist

Task-specific specialist가 아니라 하나의 model을 여러 supervised task에 함께 fine-tune한 결과다.

| 모델 | COCO cap | NoCaps | TextCaps | VQAv2 | TextVQA | VizWiz VQA |
|---|---:|---:|---:|---:|---:|---:|
| Florence-2-B | 140.0 | 116.7 | 143.9 | 79.7 | 63.6 | 63.6 |
| Florence-2-L | 143.3 | 124.9 | 151.1 | 81.7 | 73.5 | 72.6 |

L의 TextVQA와 VizWiz 개선이 특히 크다. Language decoder capacity와 high-resolution OCR representation의 영향을 시사한다.

다른 supervised spatial task에서 Florence-2-L은 COCO detection `43.4 AP`, Flickr grounding `85.2 R@1`, RefCOCO val `93.4`, referring segmentation `80.5 mIoU`를 보고한다.

### Vision backbone transfer

DaViT-B backbone을 downstream detector/segmenter에 연결한 표는 pre-training representation 자체를 평가한다.

| Pre-training | Mask R-CNN APb | Mask R-CNN APm | DINO AP | UperNet ADE mIoU |
|---|---:|---:|---:|---:|
| supervised ImageNet-1K | 46.7 | 42.0 | 53.7 | 49.0 |
| UniCL | 50.4 | 45.0 | 57.3 | 53.6 |
| Florence-2 | 53.6 | 46.4 | 59.2 | 54.9 |

Multi-task token generation을 통해 학습한 vision encoder가 closed-set downstream backbone으로도 강하다는 근거다. Multi-scale ADE20K 평가에서는 `55.5 ms-mIoU`다.

## Ablation 해석

### Task granularity 조합

세 pre-training 구성은 다음과 같다.

- Image: image-level task만
- Image-Region: image-level + region-level
- Image-Region-Pixel: image + region + pixel

주요 결과는 다음과 같다.

| Pre-training | caption CIDEr | COCO det AP | Flickr grounding R@1 | RES mIoU |
|---|---:|---:|---:|---:|
| Image | 134.6 | 0.1 | 62.0 | 28.4 |
| Image-Region | - | 29.7 | 79.1 | 18.2 |
| Image-Region-Pixel | 133.4 | 28.3 | 78.1 | 31.6 |

Region task가 detection과 grounding을 가능하게 하고 pixel task가 referring segmentation을 크게 높인다. 모든 task를 섞은 모델이 각 단일 task의 최고값은 아니지만 전 영역에서 경쟁력 있는 균형을 만든다. 이것이 generalist의 핵심 trade-off다.

### Model scale

Ablation subset에서 B/L은 caption `118.7/124.4`, detection `19.7/22.6`, grounding `76.3/78.2`, RES mIoU `18.6/21.5`를 보인다. Scale이 모든 task를 올리지만 data와 training configuration에 따라 main table보다 절대값이 낮으므로 서로 다른 실험 block의 숫자를 직접 섞지 않아야 한다.

### Data scale

0.12M, 0.36M, 1.2M, 12M image scale에서 caption은 `102.8, 114.3, 118.1, 118.7`, detection은 `16.1, 18.7, 18.9, 19.7`이다. Grounding은 `74.0, 75.8, 76.3, 76.3`으로 일찍 포화한다. RES mIoU는 `15.9, 16.6, 19.3, 18.6`으로 단조 증가하지 않는다.

Scale만 늘리면 항상 모든 task가 좋아지는 것이 아니다. Annotation quality, task sampling weight, noisy pixel target의 영향을 함께 조정해야 한다.

### Vision/language initialization과 freezing

PDF Table 12에서 vision encoder를 freeze하고 vision/language pre-training을 모두 사용한 경우 caption `120.0`, detection `6.9`, grounding `66.3`, RES `9.9/13.6`이다. Vision encoder를 unfreeze하면 best row가 caption `118.7`, detection `19.7`, grounding `76.3`, RES `18.6/17.8`을 낸다.

Region/pixel task는 frozen vision representation 위에 decoder만 학습해서는 부족하다. 반면 caption처럼 language 이해 비중이 큰 task는 BART initialization이 도움이 된다. 논문 원문도 vision unfreezing이 region/pixel 학습에 중요하다고 해석한다.

## 구현 의사코드

### Target serialization

```python
def serialize_detection(objects, width, height):
    tokens = []
    for obj in canonical_order(objects):
        x1, y1, x2, y2 = obj.box
        loc = [
            quantize_1000(x1 / width),
            quantize_1000(y1 / height),
            quantize_1000(x2 / width),
            quantize_1000(y2 / height),
        ]
        tokens += tokenize(obj.label)
        tokens += [location_token(v) for v in loc]
    tokens += [EOS]
    return tokens
```

`canonical_order`는 매우 중요하다. 공식 dataset serializer의 object ordering과 separator token을 그대로 따라야 한다.

### Unified forward

```python
def florence_forward(image, task_prompt, target_tokens=None):
    visual = davit(image)                    # [B, Nv, Dv]
    visual = layer_norm(project(visual))     # [B, Nv, D]
    prompt = embed(tokenize(task_prompt))    # [B, Nt, D]
    memory = transformer_encoder(concat(visual, prompt, dim=1))

    if target_tokens is not None:
        logits = decoder_teacher_force(target_tokens[:, :-1], memory)
        return cross_entropy(logits, target_tokens[:, 1:])
    return autoregressive_decode(memory, stop=EOS)
```

### Output parser

```python
def parse_spatial_sequence(tokens, task):
    validate_allowed_tokens(tokens, task)
    if task == "detection":
        return parse_label_box_groups(tokens, coords_per_region=4)
    if task == "ocr":
        return parse_text_quadrilateral_groups(tokens, coords_per_region=8)
    if task == "segmentation":
        polygon = parse_location_pairs_until_separator(tokens)
        return rasterize_and_clip(polygon)
```

Malformed output를 버릴지, repair할지, confidence를 어떻게 만들지 평가 protocol에 포함해야 한다.

## 온디바이스 시스템 분석

### 상시 실행보다는 요청형 모델

Florence-2-B도 FP16 weight가 약 442 MiB이고 autoregressive decoder가 있다. 카메라 매 frame에서 detection만 수행하려면 YOLO 계열보다 비효율적이다. 다음 구조가 현실적이다.

```text
every frame:
  lightweight detector or tracker

on event or user query:
  selected full image or ROI
  -> Florence-2 caption/grounding/OCR
  -> cache vision features if multiple prompts follow
```

### Vision feature cache

같은 이미지에 `caption`, `OCR`, `region-to-text`를 연속 요청한다면 DaViT output을 재사용할 수 있다.

```text
image -> DaViT once -> cached visual tokens
                         + caption prompt -> decode
                         + OCR prompt -> decode
                         + region prompt -> decode
```

단, encoder가 visual과 prompt를 concatenate한 뒤 jointly 처리하므로 prompt별 Transformer encoder 계산은 남을 수 있다. Cache 가능한 경계를 실제 구현으로 확인해야 한다.

### ROI와 dynamic resolution

Detector box로 crop한 ROI를 384에 넣으면 작은 text/object의 유효 pixel을 늘리고 full 768 입력보다 vision token을 줄일 수 있다. 비교할 조건은 다음과 같다.

- full image 384
- full image 768
- detector ROI 384
- multi-crop ROI batch

Caption은 full context가 필요할 수 있고 OCR/region-to-text는 ROI가 유리하다. Task별 resolution policy가 하나의 고정 resolution보다 효율적이다.

### 생성 길이와 TTFT

측정해야 할 지표는 frame FPS보다 VLM 계열 지표에 가깝다.

- Image encoding latency
- TTFT, first output token까지 시간
- Location/text tokens per second
- Total output length와 end-to-end latency
- Decoder KV cache peak
- Polygon vertex 수에 따른 latency
- Task prompt별 p50/p95

Detection에서도 object 수가 많으면 sequence가 길어진다. 같은 모델이라도 빈 이미지와 crowd 장면의 latency가 달라진다.

## 압축 방향

### Weight-only quantization

Language encoder-decoder linear layer는 W4A16 후보지만 DaViT convolution/attention과 output embedding을 함께 검증해야 한다. Location token은 인접 bin 간 logit 차이가 작을 수 있어 output projection 양자화가 coordinate jitter를 유발할 수 있다.

### Vision INT8 + decoder mixed precision

```text
DaViT: INT8 calibration or QAT
visual projection and encoder: INT8/FP16 comparison
decoder attention and LayerNorm: FP16
large linear weights: W4A16
output head: FP16 if location accuracy degrades
```

평가는 CIDEr나 AP만 보지 말고 다음을 추가해야 한다.

- box coordinate absolute error
- invalid sequence rate
- polygon self-intersection rate
- OCR character accuracy
- rare location token calibration

### Distillation

Florence-2-L을 teacher, B 또는 더 작은 decoder를 student로 두고 sequence distribution을 distill할 수 있다. 하지만 text token과 location token의 temperature를 동일하게 쓰면 spatial sharpness가 지나치게 부드러워질 수 있다. Task별 또는 token-type별 loss weight가 필요하다.

## 강점

1. 서로 다른 vision output을 text/location token으로 실제 통합했다.
2. 0.23B 모델이 caption, detection, grounding, segmentation에서 강한 zero-shot 균형을 보인다.
3. FLD-5B가 image, region, pixel granularity를 한 dataset schema에 연결한다.
4. Generalist 결과뿐 아니라 vision backbone transfer도 강하다.
5. Data scale, task composition, initialization, freezing ablation이 풍부하다.
6. Region token을 입력과 출력 모두에 사용해 grounded interaction interface를 만든다.

## 한계와 주의점

### Synthetic annotation bias

5.4B annotation의 규모는 강점이지만 specialist model의 오류가 통합된다. Human annotation과 pseudo annotation 비율, category별 precision, long-tail quality를 더 세밀하게 분석할 필요가 있다.

### Autoregressive spatial output

Box와 polygon은 parallel head보다 느리고 error propagation이 있다. Object가 많은 장면에서 sequence length가 커지며 latency upper bound가 불명확하다.

### Generalist trade-off

모든 task에서 specialist 최고 성능을 이기는 모델은 아니다. Task mixing이 특정 task를 손해 보게 할 수 있고, ablation에서도 Image-Region이 detection/grounding에서 전체 model보다 약간 높다.

### 배포 지표 부족

Parameter는 작다고 표현하지만 mobile latency, peak memory, TTFT, tokens/s, power를 보고하지 않는다. 0.77B는 모바일 기준으로 작지 않다.

### 데이터 중복과 zero-shot 해석

Web-scale source, specialist pre-training data, evaluation benchmark 사이의 semantic 또는 image near-duplicate 가능성을 완전히 배제하기 어렵다. Zero-shot 결과를 완전한 unseen-world 일반화로 확대 해석해서는 안 된다.

## 재현 체크리스트

### 데이터

- [ ] FLD-5B annotation type별 sampling ratio를 기록한다.
- [ ] Specialist checkpoint와 filtering threshold를 고정한다.
- [ ] Image duplicate와 benchmark overlap filtering을 확인한다.
- [ ] Object serialization order와 separator token을 저장한다.
- [ ] 1,000-bin coordinate quantization 규칙을 정확히 맞춘다.
- [ ] Polygon simplification과 invalid target 제거 규칙을 기록한다.

### 모델

- [ ] Florence-2-B/L의 DaViT dimensions, blocks, heads를 확인한다.
- [ ] BART tokenizer에 location token을 추가한 방식과 vocabulary ID를 고정한다.
- [ ] Visual projection과 LayerNorm 위치를 확인한다.
- [ ] Encoder와 decoder layer 수, hidden dimension을 표와 맞춘다.
- [ ] UniCL/BART initialization과 unfreeze schedule을 기록한다.

### 학습

- [ ] AdamW, cosine, warm-up 5,000 step을 재현한다.
- [ ] B/L learning rate 1e-4/1e-5를 구분한다.
- [ ] Effective sample과 optimizer step을 혼동하지 않는다.
- [ ] 384 phase와 768 high-resolution phase를 별도 기록한다.
- [ ] Task별 batch mixing과 loss masking을 확인한다.

### 평가

- [ ] Zero-shot과 supervised generalist checkpoint를 구분한다.
- [ ] COCO caption Karpathy split을 사용한다.
- [ ] Detection parser와 coordinate dequantization을 고정한다.
- [ ] Beam search, greedy decoding, max length를 기록한다.
- [ ] Invalid output 처리 규칙을 공개한다.
- [ ] AP, mIoU, CIDEr 외에 latency와 memory를 측정한다.

## 추천 온디바이스 ablation

| 축 | 설정 | 관찰할 값 |
|---|---|---|
| Input | full 384, full 768, ROI 384 | 정확도, vision latency, peak memory |
| Precision | FP16, vision INT8, W4A16 decoder | task별 metric, invalid rate |
| Cache | 없음, visual cache, encoder cache 가능 범위 | 반복 prompt TTFT |
| Output | caption short/long, object 1/10/50, polygon vertices | tokens/s, p95 latency |
| Model | B, L teacher, distilled student | quality-memory frontier |

이 표의 목적은 단일 평균 latency가 감추는 task별 비용 차이를 드러내는 것이다.

## 최종 평가

Florence-2의 가장 큰 의미는 많은 vision task를 지원한다는 목록 자체보다, 그 task를 실제로 동일한 sequence-to-sequence parameter와 동일한 spatial token vocabulary로 학습했다는 데 있다. FLD-5B의 image-region-pixel annotation은 unified interface가 작동하도록 만드는 데이터 기반이고, DaViT+BART 구조는 기존 vision과 language pre-training을 실용적으로 연결한다.

온디바이스에서는 상시 semantic segmenter나 detector를 직접 대체하기보다 요청형 generalist로 배치하는 편이 합리적이다. 경량 detector가 frame을 감시하고, 이벤트가 생기면 ROI를 Florence-2에 넣어 caption, OCR, grounding을 수행하며, 같은 image의 vision feature를 가능한 범위에서 cache하는 구조가 적합하다.

연구 과제로는 `ROI 기반 visual token 축소`, `task별 dynamic resolution`, `location-token-aware distillation`, `vision INT8 + decoder W4A16 mixed precision`이 특히 자연스럽다. 최종 평가는 task accuracy와 함께 TTFT, tokens/s, output length별 p95, peak memory, 전력과 온도를 반드시 포함해야 한다.

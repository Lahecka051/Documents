# 69. Grounded Language-Image Pre-training

## 논문 정보

- 원본 파일: `69_GLIP.pdf`
- 제목: **Grounded Language-Image Pre-training**
- 저자: Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, Jianfeng Gao
- 발표: CVPR 2022
- 공식 링크: [arXiv:2112.03857](https://arxiv.org/abs/2112.03857)
- 태스크: phrase grounding, open-vocabulary object detection, zero/few-shot transfer
- 핵심 키워드: GLIP, detection-as-grounding, region-word alignment, DyHead, BERT, deep fusion, pseudo grounding data, prompt tuning

## 한눈에 보는 요약

GLIP은 고정 class classifier를 가진 object detector를 **region과 text phrase를 정렬하는 grounding model**로 다시 쓴다. 전통 detector가 region feature `O`와 학습된 class weight `W`의 내적을 계산했다면, GLIP은 `W` 자리에 language encoder가 만든 prompt token feature `P`를 넣는다.

```text
traditional detection: region O x learned class weights W
GLIP:                  region O x prompt token features P
```

이 reformulation으로 detection data와 phrase-grounding data를 같은 loss로 학습할 수 있고, web image-caption에서 noun phrase를 뽑아 teacher가 pseudo box를 붙인 대규모 data도 사용할 수 있다. 최종 GLIP-L은 4개 detection dataset, human-grounded data, 24M pseudo-grounded image-text pair를 사용한다.

Architecture는 가볍지 않다.

```text
image -> Swin backbone -> DyHead visual features --+
                                                   +-> region-word scores -> boxes/NMS
prompt -> BERT text features ----------------------+
             ^              |
             +-- bidirectional cross-attention ----+
                 repeated inside late encoder layers
```

핵심 결과는 다음과 같다.

- COCO를 보지 않은 GLIP-L zero-shot: `49.8 AP`; COCO fine-tuning: `60.8 AP`, test-dev `61.0`.
- LVIS zero-shot GLIP-L: rare `17.1`, common `23.3`, frequent `35.4`, 전체 `26.9 AP`.
- Flickr30K Entities test Recall@1: `87.1`.
- ODinW 13개 task에서 zero/few-shot와 prompt tuning의 높은 data efficiency를 보인다.

반면 language-aware deep fusion은 P100 batch 1에서 GLIP-T inference를 `4.84 -> 2.52 FPS`, memory를 `1.0 -> 2.4 GB`로 바꾼다. GLIP-L은 `0.54 -> 0.32 FPS`, `4.8 -> 7.7 GB`다. GLIP은 강한 teacher/reference지만 그대로 모바일 실시간 detector로 배포하기에는 무겁다.

## 연구 배경과 문제의식

전통 object detection dataset은 annotation 비용 때문에 vocabulary가 제한된다. COCO는 80 class, Objects365는 365 class이고, 더 큰 dataset도 보통 수천 class를 넘기 어렵다. 실제 사용자는 `pothole`, `stingray`, 특정 부품처럼 학습 class 밖의 개념을 요청한다.

Image-text pair는 web에 많지만 box annotation이 없다. CLIP 같은 global alignment만으로는 phrase가 image의 어디에 있는지 직접 알려 주지 못한다. Phrase grounding data는 object-level language supervision을 주지만 규모가 작다.

GLIP의 목표는 세 data source를 한 objective 아래 결합하는 것이다.

- Detection: class name과 box
- Grounding: natural-language phrase와 box
- Image-text: caption phrase와 teacher-generated pseudo box

이 통합이 가능하려면 detector의 고정 class classifier를 language-conditioned interface로 바꿔야 한다.

## Detection을 grounding으로 다시 쓰기

### 전통 detector

Image encoder가 `N`개 region/box feature를 만든다고 하자.

```math
O=\mathrm{Enc}_I(I)\in\mathbb R^{N\times d}
```

`c`개 class에 대한 linear classifier weight는 다음과 같다.

```math
W\in\mathbb R^{c\times d}
```

Classification score는

```math
S_{cls}=OW^\top\in\mathbb R^{N\times c}
```

이고 assignment target `T`로 cross entropy나 focal loss를 계산한다.

### Phrase grounding

Class name을 prompt로 만들고 language encoder를 통과시킨다.

```text
person. bicycle. car. ... . toothbrush.
```

```math
P=\mathrm{Enc}_L(\mathrm{Prompt})\in\mathbb R^{M\times d}
```

Region-token alignment score는

```math
S_{ground}=OP^\top\in\mathbb R^{N\times M}
```

이다. `P`가 고정 class weight `W`의 동적 language-conditioned 버전 역할을 한다. Prompt만 바꾸면 classifier parameter를 바꾸지 않고 새 concept를 query할 수 있다.

### Phrase와 subword target

Phrase 수 `c`와 token 수 `M`은 같지 않다.

- `traffic light`처럼 phrase가 여러 word를 포함한다.
- 한 word가 여러 subword로 split될 수 있다.
- Separator와 special token이 있다.
- `[NoObj]` token이 추가된다.

GLIP이 사용하는 sigmoid/focal 형태에서는 positive phrase에 속한 모든 subword를 positive로 확장해 `T' in {0,1}^{N x M}`을 만든다. Inference에서는 phrase에 속한 token probability를 평균한다. Tokenization이 달라지면 phrase score도 달라질 수 있으므로 tokenizer와 span mapping은 model definition의 일부다.

## Detection과 grounding은 정말 동등한가

한 prompt에 모든 class가 들어가고 target expansion이 정확하다면 learned class weight를 contextual text embedding으로 대체한 형태이므로 학습/inference interface가 대응한다. 논문 appendix는 COCO 80 class를 한 prompt에 넣은 DyHead에서 reformulation 전후 성능이 같음을 확인한다.

그러나 실제 open-vocabulary inference에서는 조건이 달라진다.

- Prompt 길이 제한 때문에 vocabulary를 chunk로 나눌 수 있다.
- Contextual BERT embedding은 class의 순서와 주변 word에 의존할 수 있다.
- 여러 prompt run 사이의 score calibration이 필요하다.
- Deep fusion에서는 visual feature 자체가 prompt에 따라 바뀐다.

따라서 수학적 correspondence가 대규모 vocabulary의 계산량까지 동일하게 만든다는 뜻은 아니다.

## Language-aware deep fusion

Late fusion은 image와 text를 따로 encode한 뒤 마지막 내적에서만 만난다. GLIP은 DyHead의 마지막 layer들과 추가 BERT layer 사이에서 양방향 cross-attention을 반복한다.

```math
O^i_{t2i},P^i_{i2t}=\text{X-MHA}(O^i,P^i)
```

```math
O^{i+1}=\mathrm{DyHeadModule}(O^i+O^i_{t2i})
```

```math
P^{i+1}=\mathrm{BERTLayer}(P^i+P^i_{i2t})
```

한 attention head의 image-to-text 방향은 다음처럼 쓸 수 있다.

```math
A=\frac{(O W_q^I)(P W_q^L)^\top}{\sqrt d}
```

```math
O_{t2i}=\mathrm{softmax}(A)(P W_v^L)W_o^I
```

반대 방향은 `A^T`로 visual value를 모은다.

```math
P_{i2t}=\mathrm{softmax}(A^\top)(O W_v^I)W_o^L
```

이 구조에서 visual representation은 prompt에 조건화된다. `red car`와 `blue bicycle` prompt는 같은 image에서도 다른 visual emphasis를 만들 수 있다. 반면 prompt embedding만 cache하고 image feature를 한 번 계산하는 late-fusion 최적화는 어려워진다.

## Scalable grounded pretraining

### Human-annotated data

Gold grounding data는 Flickr30K, Visual Genome Caption, GQA 등을 합친 `GoldG` 약 0.8M이다. Detection vocabulary가 수천 class 수준인 데 비해 Flickr30K는 44,518 unique phrase, VG Caption은 110,689 unique phrase를 포함한다고 논문은 보고한다.

### Pseudo grounding

1. Gold detection/grounding data로 teacher GLIP을 학습한다.
2. Web caption에서 NLP parser로 noun phrase를 추출한다.
3. Teacher가 각 phrase의 box를 예측한다.
4. Image, caption, pseudo box를 student 학습 data로 사용한다.

저자들은 language context가 teacher의 "educated guess"를 가능하게 한다고 설명한다. 예를 들어 직접 학습하지 않은 `vaccine`도 `small vial`, `syringe` 문맥을 이용해 localize할 수 있고, 그 pseudo label이 student의 supervision이 된다.

이 방법은 vocabulary를 넓히지만 teacher bias와 caption noise도 함께 증류한다. Pseudo box confidence threshold, noun phrase parsing 오류, 중복 phrase가 long-tail 성능과 false positive에 미치는 영향을 별도로 봐야 한다.

## Model variants와 data 규모

| Model | Backbone | Deep fusion | Detection | Grounding | Pseudo caption |
| --- | --- | :---: | --- | --- | --- |
| GLIP-T (A) | Swin-T | - | Objects365 | - | - |
| GLIP-T (B) | Swin-T | O | Objects365 | - | - |
| GLIP-T (C) | Swin-T | O | Objects365 | GoldG | - |
| GLIP-T | Swin-T | O | Objects365 | GoldG | Cap4M |
| GLIP-L | Swin-L | O | FourODs | GoldG | Cap24M |

GLIP-L의 FourODs는 Objects365, OpenImages, Visual Genome, ImageNetBoxes를 합친 2.66M detection data다. Abstract는 전체 pretraining을 3M human-annotated와 24M web image-text의 총 27M grounded data로 요약한다.

## Tensor shape와 activation memory

논문은 standard input resolution별 내부 tensor shape를 완전하게 표로 주지 않는다. 따라서 아래는 구조의 scaling을 이해하기 위한 reviewer example이며 paper-reported memory와 구분한다.

`800x1280`, batch 1, stride `8/16/32/64`, channel 256의 multi-scale feature를 flatten한다고 가정하면 visual token 수는 대략 다음과 같다.

```math
N=100\cdot160+50\cdot80+25\cdot40+13\cdot20
=21{,}260
```

Text prompt가 `M=128`, attention head가 8개이면 X-MHA score 원소 수는

```math
hNM=8\cdot21{,}260\cdot128=21{,}770{,}240
```

FP16 score만 약 `41.5 MiB`다. 양방향 softmax, projected Q/V, gradient, 여러 fusion layer를 포함하면 훨씬 커진다. 실제 DyHead가 cross-attention에 투입하는 representation과 token filtering은 구현에서 확인해야 하므로 이 숫자는 논문 재현값이 아니다.

핵심 scaling은 명확하다.

```math
\text{X-MHA cost}=O(NMd)
```

Input resolution로 `N`이 늘고 prompt 길이로 `M`이 늘면 memory와 latency가 곱으로 증가한다. Open-vocabulary class를 한 번에 많이 query할수록 text cost뿐 아니라 cross-modal attention cost도 커진다.

## 최소 구현 의사코드

```python
def build_prompt(class_names):
    # Keep deterministic ordering and token-to-phrase spans.
    text = ". ".join(class_names) + "."
    token_ids, phrase_spans = tokenizer_with_spans(text)
    return token_ids, phrase_spans

def deep_fused_encode(image, prompt_ids):
    visual = swin_backbone(image)
    text = bert_backbone(prompt_ids)
    for dyhead, bert_layer, cross_attn in fusion_layers:
        text_to_image, image_to_text = cross_attn(visual, text)
        visual = dyhead(visual + text_to_image)
        text = bert_layer(text + image_to_text)
    return visual, text

def forward(image, class_names, targets=None):
    prompt_ids, spans = build_prompt(class_names)
    visual, token_features = deep_fused_encode(image, prompt_ids)
    regions, box_deltas = dense_detection_head(visual)
    token_logits = regions @ token_features.transpose(-1, -2)
    phrase_logits = aggregate_subwords(token_logits, spans, reduce="mean")

    if targets is not None:
        token_targets = expand_phrase_targets_to_tokens(targets, spans)
        return focal_loss(token_logits, token_targets) + box_loss(box_deltas, targets)

    boxes = decode_boxes(box_deltas)
    return classwise_nms(boxes, sigmoid(phrase_logits))
```

실제 GLIP은 DyHead의 dense candidate, multi-scale feature, focal classification/localization loss와 postprocessing을 상속한다. Language conditioning을 추가했지만 DETR처럼 NMS-free set prediction으로 바뀐 것은 아니다.

## COCO 결과

| Model | Pretraining | Zero-shot AP | Fine-tune val/test-dev AP |
| --- | --- | ---: | ---: |
| DyHead-T | Objects365 | 43.6 | 53.3 / - |
| GLIP-T (A) | Objects365 | 42.9 | 52.9 / - |
| GLIP-T (B) | Objects365 | 44.9 | 53.8 / - |
| GLIP-T (C) | Objects365+GoldG | 46.7 | 55.1 / - |
| GLIP-T | Objects365+GoldG+Cap4M | 46.3 | 54.9 / - |
| GLIP-L | FourODs+GoldG+Cap24M | **49.8** | **60.8 / 61.0** |

COCO class가 Objects365에 모두 포함되므로 `zero-shot`은 완전히 보지 못한 concept가 아니라 **COCO image/domain을 보지 않은 transfer**다. Novel category generalization의 증거는 LVIS rare와 ODinW의 novel concepts를 함께 봐야 한다.

Cap4M GLIP-T의 COCO zero-shot `46.3`은 GoldG-only `46.7`보다 낮다. Web data scaling이 모든 benchmark를 단조롭게 높이는 것은 아니며, long-tail/grounding 이득과 common-category 성능을 분리해야 한다.

## LVIS와 long-tail 결과

LVIS zero-shot 결과다.

| Model | APr | APc | APf | AP |
| --- | ---: | ---: | ---: | ---: |
| GLIP-T (A) | 6.0 | 8.0 | 19.4 | 12.3 |
| GLIP-T (C) | 7.5 | 11.6 | 26.1 | 16.5 |
| GLIP-T | 10.1 | 12.5 | 25.5 | 17.2 |
| GLIP-L | **17.1** | **23.3** | **35.4** | **26.9** |

Grounding data와 scaling은 rare category를 크게 개선한다. Data ablation에서 Objects365만 쓴 모델의 LVIS rare AP `13.5`가 GoldG 추가 후 `17.7`, Cap4M 추가 후 `20.8`로 오른다. 반면 frequent AP는 GoldG 추가 시 `22.2 -> 31.0`으로도 크게 오르므로 이득이 rare에만 제한되지는 않는다.

## Phrase grounding 결과

Flickr30K Entities test Recall@1은 다음과 같다.

- MDETR-RN101: `83.4`
- MDETR-ENB5: `84.3`
- GLIP-T, GoldG: `84.4`
- GLIP-T, O365+GoldG+Cap4M: `85.7`
- GLIP-L: **87.1**

Detection data를 추가해도 grounding이 좋아지고 image-text data도 추가 이득을 준다. Detection과 grounding을 같은 alignment objective로 학습한 시너지를 보여 준다.

## Deep fusion ablation과 실제 비용

Appendix Table 7의 batch 1 inference 결과다.

| Model | Deep fusion | P100 FPS | Inference memory | V100 train FPS | Train memory |
| --- | :---: | ---: | ---: | ---: | ---: |
| GLIP-T | - | 4.84 | 1.0 GB | 2.79 | 11.5 GB |
| GLIP-T | O | 2.52 | 2.4 GB | 1.62 | 16.0 GB |
| GLIP-L | - | 0.54 | 4.8 GB | 1.27 | 19.7 GB |
| GLIP-L | O | 0.32 | 7.7 GB | 0.88 | 23.4 GB |

Reviewer conversion으로 GLIP-T latency는 약 `207 -> 397 ms`, GLIP-L은 `1.85 -> 3.13 s`다. GLIP-T inference memory는 2.4배가 된다. 저자 표현인 "1x 미만의 추가 계산"은 100% 미만의 속도 overhead라는 의미에 가깝고 memory overhead가 작다는 뜻은 아니다.

Deep fusion의 성능 효과도 균일하지 않다.

- O365 only: COCO `42.9 -> 44.9`, 그러나 LVIS AP `18.5 -> 17.8`, Flickr R@1 `46.4 -> 41.4`.
- O365+GoldG: COCO `41.6 -> 46.7`, Flickr R@1 `82.4 -> 84.8`, 그러나 LVIS AP `26.1 -> 24.9`.
- Grounding data가 있을 때 LVIS rare AP는 `15.8 -> 17.7`로 오르지만 common/frequent가 내려 전체 AP는 낮아진다.

Language conditioning은 common transfer와 long-tail에 서로 다른 영향을 줄 수 있다. 단일 COCO AP로 deep fusion을 선택하면 안 된다.

## Object Detection in the Wild

ODinW는 PascalVOC, Aquarium, Rabbits, EgoHands, Pothole, Thermal 등 13개 서로 다른 domain을 포함한다. 저자들은 다음을 보고한다.

- Zero-shot GLIP-T가 5-shot DyHead-T보다 강하다.
- 1-shot GLIP-L이 fully supervised DyHead-T와 경쟁한다.
- Grounding data가 Pothole, EgoHands처럼 Objects365 밖의 concept에 특히 중요하다.
- GLIP-L의 13-dataset zero-shot 평균은 `52.1 AP`다.

Manual prompt에 attribute를 추가해 `stingray`를 `flat and round`로 설명하면 해당 Aquarium AP50이 `4.6 -> 9.7`로 오른다. Appendix는 EgoHands에서 prompt로 AP가 `22.1 -> 50.0`이 되는 사례도 보고한다. Prompt는 무료가 아니라 model behavior를 바꾸는 task specification이다.

## Prompt tuning

Task별 prompt가 고정이라면 language encoder가 만든 초기 prompt embedding `P0`를 저장하고 BERT를 제거한 뒤 이 embedding만 학습할 수 있다. Deep-fused GLIP-T/L에서는 prompt tuning이 full fine-tuning에 가까운 성능을 보인다.

장점은 task별 model copy 대신 작은 embedding만 저장하는 것이다. 그러나 다음 제한이 있다.

- 학습한 embedding은 사람이 읽을 수 있는 prompt가 아니다.
- Task마다 class ordering과 length가 달라지면 tensor shape가 달라진다.
- Prompt embedding만 작을 뿐 base GLIP model은 그대로 크다.
- Deep fusion 때문에 prompt가 달라지면 image feature 재계산이 필요할 수 있다.

## 대규모 vocabulary inference

LVIS 1,000개 이상 class는 한 prompt에 들어가지 않아 저자들은 40 category씩 chunk하여 model을 여러 번 실행한다. 단순 계산으로 약 25개 chunk다.

Late fusion이면 image feature를 cache하고 text/score만 반복할 수 있다. 그러나 deep fusion은 prompt가 visual feature를 바꾸므로 전체 fusion stack을 반복할 가능성이 크다. 논문의 LVIS AP는 강하지만 모바일 latency 관점에서는 매우 비싼 protocol이다. 반드시 query class 수와 prompt chunk 수를 latency 표에 포함해야 한다.

## 장점과 핵심 기여

1. **Detection과 grounding의 interface를 통일했다.** 학습 data 종류를 크게 확장한다.
2. **고정 classifier를 language-conditioned classifier로 바꿨다.** Prompt로 novel concept를 query할 수 있다.
3. **Object-level web supervision을 scale했다.** Global image-text pair에 pseudo box를 부여한다.
4. **Rare/open-domain transfer가 강하다.** LVIS와 ODinW에서 grounding data의 이득이 뚜렷하다.
5. **Prompt tuning이라는 deployment adapter를 제시했다.** Task별 full model copy를 피할 수 있다.
6. **비용을 appendix에 공개했다.** Deep fusion의 FPS와 GPU memory trade-off를 직접 확인할 수 있다.

## 한계와 비판적 관점

### 1. 매우 무겁다

Swin, DyHead, BERT, 반복 bidirectional cross-attention을 결합한다. GLIP-L deep fusion은 P100 batch 1에서 0.32 FPS다. 실시간 edge model이 아니다.

### 2. Deep fusion의 비용과 이득이 일관되지 않는다

COCO와 grounding은 좋아지지만 LVIS 전체 AP가 낮아지는 조합이 있다. Inference memory는 GLIP-T에서 2.4배다.

### 3. Vocabulary가 커지면 반복 실행한다

LVIS는 40 class prompt chunk를 여러 번 query한다. Open vocabulary의 semantic 범위가 넓어져도 runtime cost가 고정되는 것은 아니다.

### 4. Pseudo label이 teacher bias를 확대할 수 있다

Web caption의 noun phrase와 teacher box가 틀리면 student가 그 오류를 학습한다. Confidence와 data filtering ablation이 더 필요하다.

### 5. Zero-shot 용어가 조건별로 다르다

COCO zero-shot은 COCO image를 보지 않았지만 category는 Objects365에 포함된다. 완전 novel concept transfer와 구분해야 한다.

### 6. Prompt sensitivity가 있다

Class delimiter, order, descriptive attribute가 score를 바꾼다. 사람이 prompt를 수동 최적화하면 benchmark 비교의 재현성이 낮아질 수 있다.

### 7. NMS-free가 아니다

Underlying dense detector의 candidate decode와 NMS를 유지한다. Language grounding이 postprocessing을 없애지 않는다.

## 자주 헷갈리는 지점

### GLIP은 CLIP에 detector head를 붙인 것인가

아니다. Global image-text contrastive model이 아니라 region-token alignment detector이며 DyHead와 BERT를 end-to-end deep fusion한다.

### Text phrase 하나가 logit 하나인가

실제 score는 token 단위 `N x M`이고 multi-word/subword probability를 phrase 단위로 aggregate한다.

### Open vocabulary면 한 번에 무한한 class를 검출하는가

아니다. Prompt length와 memory 제한이 있고 LVIS에서는 40 class씩 chunk해 반복 실행한다.

### Prompt를 cache하면 BERT 비용만 없어지는가

Late fusion에서는 유리하지만 deep fusion visual feature가 prompt-conditioned이므로 전체 image-path cache가 제한된다.

### GLIP이 mask도 출력하는가

기본 논문은 box detection과 phrase grounding이 중심이다. Segmentation mask가 필요하면 SAM/Mask2Former 같은 별도 model 또는 확장이 필요하다.

## 온디바이스 관점

GLIP은 직접 배포 대상보다 **teacher와 semantic reference**로 보는 편이 적절하다.

### 주요 병목

- Swin-L/T high-resolution feature
- DyHead multi-scale processing
- BERT tokenization/encoding
- Bidirectional cross-attention의 `N x M` matrix
- Prompt chunk별 반복 실행
- Dense box decode와 class-wise NMS
- Dynamic prompt length와 NPU graph recompilation

### 현실적인 경량화 경로

1. GLIP/Grounding DINO를 offline teacher로 사용한다.
2. 제품 vocabulary와 target domain의 pseudo box를 생성한다.
3. YOLO-World/YOLOE 또는 anchor-free small detector로 distill한다.
4. Text embedding을 사전 계산하고 device에는 compact embedding table만 둔다.
5. Mask는 선택된 box에 EdgeSAM을 요청형으로 실행한다.

### 측정해야 할 system 지표

- Vision-only, text-only, fusion, NMS 시간을 분해
- Prompt class 수 `1/10/40/80/1000`별 p50/p95
- Prompt chunk 수와 image feature cache hit
- Peak memory와 attention workspace
- Detector AP, rare AP, false-positive rate
- 전력, 온도, 지속 실행 throttling

## 재현 계획

### 핵심 block 재현

기존 FCOS/RetinaNet류 detector의 class weight를 BERT class-name embedding으로 교체해 late-fusion baseline을 만든다. 같은 assignment와 focal loss에서 learned classifier와 text classifier를 비교한다.

### Ablation 하나

Objects365 subset에서 다음 네 구성을 같은 backbone으로 비교한다.

1. Learned classifier
2. Text late fusion
3. Text deep fusion
4. Deep fusion + grounding data

COCO AP, LVIS APr/APc/APf, latency, peak memory를 함께 기록하면 GLIP의 핵심 주장과 비용을 동시에 재현할 수 있다.

### 온디바이스 확장

- Text encoder on-device 대 offline embedding
- Deep fusion 제거/distillation
- Query class 수에 따른 latency curve
- INT8 vision/detection, FP16 text projection
- NMS CPU fallback과 p95

## 구현 체크리스트

- [ ] Class name order와 delimiter를 고정했는가?
- [ ] Token-to-phrase span을 정확히 저장했는가?
- [ ] Multi-subword target expansion이 맞는가?
- [ ] `[NoObj]`와 special token target을 처리했는가?
- [ ] Phrase score의 mean/sum convention이 학습 loss와 맞는가?
- [ ] Image와 text feature dimension/normalization이 일치하는가?
- [ ] Deep fusion 양방향 attention mask가 맞는가?
- [ ] Prompt chunk 사이 score calibration과 NMS를 검증했는가?
- [ ] COCO image/category exposure를 zero-shot 표에 명시했는가?
- [ ] Web pseudo box confidence와 filtering을 기록했는가?
- [ ] Text encoding, fusion, NMS latency를 분리했는가?
- [ ] Query class 수별 peak memory와 p95를 측정했는가?

## 로드맵에서의 위치

OWL-ViT는 pretrained vision-language representation을 open-vocabulary detector로 전환하는 단순한 baseline이다. GLIP은 detection/grounding data를 unified pretraining하고 visual feature까지 language-conditioned하게 만든다. Grounding DINO는 강한 DETR계 detection backbone과 language fusion을 결합한다. YOLO-World와 YOLOE는 실시간 deployment를 위해 text embedding과 detector 구조를 더 효율적으로 설계한다.

통합 pipeline에서는 다음처럼 쓰는 것이 합리적이다.

```text
offline GLIP/Grounding DINO teacher
  -> domain pseudo boxes / phrase labels
  -> small open-vocabulary detector student
  -> selected box
  -> prompt segmenter
  -> ROI-focused VLM response
```

## 최종 평가

GLIP의 가장 큰 기여는 detector의 class head를 text로 바꾼 것만이 아니라 **detection, grounding, web caption을 object-level alignment라는 한 학습 문제로 통합한 것**이다. 그 결과 rare concept와 low-data domain transfer가 강해지고 prompt를 task interface로 사용할 수 있다.

그러나 semantic breadth는 무료가 아니다. Deep fusion은 memory와 latency를 크게 늘리고, vocabulary가 길면 prompt chunk마다 반복 실행한다. 따라서 온디바이스 연구에서 GLIP은 최종 model보다 teacher/reference로 두고, language-aware knowledge를 smaller detector에 distill하는 방향이 더 현실적이다.

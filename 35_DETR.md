# 35. End-to-End Object Detection with Transformers

## 논문 정보

- 원본 파일: `C:\Users\lkm\Documents\Codex\Embbeded_OnDevice_ComputerVision\attention_papers_pdf\35_DETR.pdf`
- 제목: End-to-End Object Detection with Transformers
- 저자: Nicolas Carion et al.
- 발표: ECCV 2020
- 주제: DETR의 set prediction object detection
- 핵심 키워드: DETR, object detection, set prediction, Hungarian matching, transformer decoder

## 한눈에 보는 요약

DETR은 object detection을 anchor, proposal, NMS가 필요한 pipeline이 아니라 set prediction 문제로 재정의한다.

CNN backbone feature를 Transformer encoder-decoder에 넣고, learnable object queries가 객체 후보를 직접 예측한다.

Hungarian matching loss로 예측 set과 ground-truth set을 일대일 매칭해 end-to-end로 학습한다.

## 연구 배경과 문제의식

전통적인 object detector는 anchor generation, proposal selection, non-maximum suppression 같은 hand-designed component가 많다.

Object detection을 fixed set prediction으로 보고 Transformer decoder query와 Hungarian matching으로 end-to-end 학습한다.

Detection output은 순서가 없는 객체 set이다. 따라서 sequence loss보다 set loss가 자연스럽다.

## 이 논문에서 먼저 잡아야 할 이론적 축

이 논문의 중심축은 `DETR의 set prediction object detection`이다.

따라서 리뷰의 초점은 단순히 모델이 성능을 올렸다는 사실이 아니라, 논문이 기존 방법의 어떤 가정을 바꾸었고 그 변화가 수식과 계산 절차에서 어떻게 나타나는지에 둔다.

아래 섹션에서는 핵심 개념을 먼저 분해하고, 그 다음 실제 계산 흐름을 단계별로 따라간다.

## 기존 방식과 무엇이 다른가

DETR이 기존 detector와 다른 점은 detection을 후보 box 후처리 문제가 아니라 set prediction 문제로 본다는 것이다. Faster R-CNN이나 YOLO 계열은 anchor/proposal 또는 dense candidate를 만들고 NMS로 중복을 제거한다.

```text
traditional detector:
image -> dense candidates / anchors -> class + box
     -> non-maximum suppression -> final detections

DETR:
image -> transformer encoder memory
object queries -> transformer decoder
     -> fixed-size prediction set
     -> Hungarian matching loss
```

즉, DETR은 중복 제거를 inference 후처리로 해결하지 않고, 학습 중 bipartite matching으로 각 ground-truth object가 하나의 query와만 대응되도록 만든다.

## 핵심 개념 상세 해설

### Vision attention의 핵심 수식

DETR의 object query는 anchor가 아니라 learnable object slot이다. 각 query가 이미지 memory를 cross-attention해서 하나의 객체 후보를 만든다.

```text
image features -> encoder memory
object queries -> decoder
decoder output_i -> class_i, box_i
Hungarian matching assigns predictions to ground-truth objects
```

Hungarian matching은 prediction set과 target set을 permutation-invariant하게 비교한다. 이 때문에 NMS 없이 end-to-end detection이 가능해진다.

## Tensor shape와 자료구조

Vision attention은 입력을 어떤 sequence로 보느냐에 따라 shape가 달라진다. CNN feature map은 `[B, C, H, W]`이고, ViT/DETR에서는 이를 token sequence로 펼친다.

```text
CNN feature: [B, C, H, W]
spatial tokens: [B, H*W, C]
image patches: [B, N, P*P*C]
patch embeddings: [B, N, D]
object queries: [B, num_queries, D]
```

SE/CBAM은 token-token attention이 아니라 channel/spatial gate를 만든다. ViT/DETR은 patch 또는 object query를 Transformer token처럼 다룬다. 따라서 같은 `attention`이라는 말이지만 계산 대상이 상당히 다르다.

## 수식과 계산 전개

### Backbone feature flattening

```text
Backbone feature `[C, H, W]`를 `HW`개의 token으로 flatten하고 positional encoding을 더한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Encoder global context

```text
Encoder self-attention은 image 위치들 사이의 global context를 계산한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Object query decoding

```text
Decoder object query `q_i`는 cross-attention으로 image memory에서 자신의 객체 정보를 찾는다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Class-box prediction

```text
각 query는 `class_i = softmax(W_c h_i)`, `box_i = sigmoid(MLP(h_i))`를 출력한다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

### Hungarian matching loss

```text
Hungarian matching은 예측과 ground truth 사이 비용을 최소화하는 permutation을 찾는다.
```

이 단계는 위 이론적 아이디어가 실제 forward 계산에서 구현되는 지점이다.

## 전체 알고리즘 흐름

1. 이미지 또는 CNN feature를 patch, spatial token, channel descriptor, object query 입력으로 바꾼다.
2. Attention 또는 gate score를 계산한다.
3. Feature를 재가중하거나 token representation을 갱신한다.
4. 분류, captioning, detection 등 task head로 output을 만든다.

## 작은 예시로 보는 직관

이미지 안에 사람, 자전거, 강아지 세 객체가 있고 DETR query가 100개 있다고 하자. 모델은 100개의 prediction slot을 내지만, Hungarian matching은 그중 3개만 ground-truth 객체와 매칭하고 나머지는 `no-object`로 학습시킨다.

```text
ground truth objects: 3
model queries: 100
matched queries: 3
unmatched queries: 97 -> no-object class
```

이 구조 덕분에 query 12와 query 47이 같은 강아지를 동시에 맞히는 중복 예측은 loss에서 불리해진다. 그래서 DETR은 NMS 없이도 set 형태의 detection을 학습할 수 있다.

## 계산 복잡도와 병목

- 원래 DETR은 학습 수렴이 느리고 small object 성능이 약하다는 문제가 있다.
- 이 논문에서 말하는 효율 또는 성능 개선이 어느 병목을 줄이는지 분리해서 봐야 한다.
- 수식상 복잡도, 실제 메모리 사용량, GPU 실행 효율, 학습 안정성, 추론 latency는 서로 다른 축이다.

예를 들어 같은 `더 효율적이다`라는 주장도 Linformer에서는 low-rank 근사, FlashAttention에서는 IO 절감, PagedAttention에서는 KV cache memory management, ViT에서는 대규모 pretraining을 통한 inductive bias 보완을 뜻한다.

## 구현할 때 확인할 부분

- 수식에서 normalization이 어느 축에 적용되는지 확인한다.
- Mask, position index, cache index, spatial flatten order처럼 off-by-one 오류가 나기 쉬운 부분을 별도로 검증한다.
- 논문이 exact 계산을 유지하는지, 근사 또는 sparse 제한을 쓰는지 구분한다.
- 학습 시 이득과 추론 시 이득이 같은지 분리해서 본다.

## 자주 헷갈리는 지점

- Attention weight가 항상 사람이 해석하는 원인 설명과 일치한다고 보면 안 된다. 모델 내부의 정보 routing 가중치로 보는 편이 안전하다.
- Score에 추가되는 bias나 position term은 value에 직접 더해지는 정보와 역할이 다르다. Score는 무엇을 볼지, value는 무엇을 가져올지를 정한다.
- Vision attention이 항상 CNN보다 낫다는 뜻은 아니다. 데이터 크기와 local inductive bias가 큰 영향을 준다.
- Channel attention, spatial attention, token self-attention은 모두 attention이라고 불리지만 계산 대상이 다르다.

## 관련 방법과 비교

| Detector | 후보 생성 | 중복 제거 | 학습 관점 |
| --- | --- | --- | --- |
| Faster R-CNN | region proposal | NMS 필요 | proposal classification/regression |
| YOLO/SSD | dense anchors/grid | NMS 필요 | dense prediction |
| DETR | learned object queries | NMS 없음 | set prediction + Hungarian matching |
| Deformable DETR | sparse multi-scale sampling | NMS 없음 | DETR의 수렴/소형 객체 문제 개선 |

## 실험 결과를 읽는 관점

Vision 논문에서는 attention이 CNN 대비 어떤 inductive bias를 잃고 어떤 global modeling 능력을 얻는지 봐야 한다.

ImageNet classification, detection, segmentation, video understanding 결과를 볼 때는 input resolution, pretraining data scale, backbone 차이를 함께 확인해야 공정한 비교가 된다.

## 장점과 기여

- 기존 방법의 한계를 명확한 이론적 문제로 재정의한다.
- 그 문제를 해결하기 위한 계산 구조 또는 학습/추론 절차를 제안한다.
- 후속 연구가 비교할 수 있는 기준점과 용어를 제공한다.

## 한계와 비판적 관점

- 원래 DETR은 학습 수렴이 느리고 small object 성능이 약하다는 문제가 있다.
- Feature map을 단일 scale로 사용하면 다양한 크기의 객체 처리에 한계가 있다.
- 이후 Deformable DETR 등은 sparse multi-scale attention으로 이러한 문제를 개선한다.

## 후속 논문과의 연결점

DETR은 Deformable DETR, DINO, Conditional DETR 등 transformer-based detector의 출발점이 되었다.

## 개인 학습/연구 메모

DETR은 `object detection = 고정 개수 query가 만드는 set prediction + Hungarian matching`으로 이해하면 된다.

